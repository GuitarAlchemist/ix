//! Plan admission: resolve a proposal into an exact plan, admit it under a mandate as an
//! immutable digest-named document, and apply only that document, after checking that what it
//! names has not moved since.
//!
//! ```text
//! proposal (JSON) --resolve(world)--> Plan --admit(mandate)--> Admitted --apply(mandate, world)--> Runner
//!                       \ Vec<Finding>              \ Vec<Finding>               \ Vec<Finding>, nothing run
//! ```
//!
//! It is pure but for the [`Store`] it reads. It keeps ixctl's vocabulary: `admit` is the
//! instant a document is accepted, `apply` acts on an accepted one. What it adds is the content
//! digest of the admitted document, `ix_recover::canonical_sha256`, the format ix-recover's
//! manifests already use.
//!
//! What it checks, by [`Code`]:
//! - [`resolve`]: the proposal's shape, an exact artifact version present in the store for this
//!   platform, the consumer contract the plane announces, every declared variable present and
//!   of its declared type, a bounded budget;
//! - [`admit`]: every scope the plan needs granted, the budget within the mandate's ceiling;
//!   nothing is widened or clamped;
//! - [`apply`]: the document's digest, the same mandate, and the artifact, contract, platform
//!   and variables the plan names, read again. Any difference refuses; the runner is never
//!   called with a document whose world has moved.
//!
//! A secret variable (`secret: true`) is checked for presence and type only. Neither its value
//! nor a digest of it enters the plan, a finding or the admitted document, so a changed secret
//! is not detected. Variables the plan does not declare are not pinned at all: ixctl's
//! supervisor passes its own environment to the worker.

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

/// The schema a proposal declares.
pub const PROPOSAL_SCHEMA: &str = "ix-admission-proposal/0";
/// The schema of a resolved plan.
pub const PLAN_SCHEMA: &str = "ix-admission-plan/0";

/// Why a proposal, a plan or an admitted document is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Code {
    /// Not the document's schema, or a field missing, unknown or of the wrong type.
    Schema,
    /// An artifact version that is not one exact version (`latest`, `^1.2`, `1.x`, ...).
    FloatingVersion,
    /// No artifact of that name, version and platform in the store.
    ArtifactNotFound,
    /// The artifact is for another platform than the one it would run on.
    PlatformMismatch,
    /// The plane keeps another consumer contract than the one required.
    ContractMismatch,
    /// A declared variable is absent.
    MissingConfig,
    /// A declared variable does not parse as its declared type.
    ConfigType,
    /// A budget that is zero (unbounded in ixctl) or otherwise unusable.
    InvalidBudget,
    /// The plan needs a scope the mandate does not grant.
    MandateInsufficient,
    /// The plan's budget is past the mandate's ceiling.
    BudgetExceedsMandate,
    /// The admitted document no longer matches its digest.
    RevisionTampered,
    /// `apply` was given another mandate than the one the document was admitted under.
    MandateChanged,
    /// The artifact's bytes differ from the admitted digest.
    ArtifactChanged,
    /// A non-secret variable's value differs from the admitted one.
    ConfigChanged,
    /// The plane's contract differs from the admitted one.
    ContractChanged,
}

impl Code {
    pub fn as_str(self) -> &'static str {
        match self {
            Code::Schema => "Schema",
            Code::FloatingVersion => "FloatingVersion",
            Code::ArtifactNotFound => "ArtifactNotFound",
            Code::PlatformMismatch => "PlatformMismatch",
            Code::ContractMismatch => "ContractMismatch",
            Code::MissingConfig => "MissingConfig",
            Code::ConfigType => "ConfigType",
            Code::InvalidBudget => "InvalidBudget",
            Code::MandateInsufficient => "MandateInsufficient",
            Code::BudgetExceedsMandate => "BudgetExceedsMandate",
            Code::RevisionTampered => "RevisionTampered",
            Code::MandateChanged => "MandateChanged",
            Code::ArtifactChanged => "ArtifactChanged",
            Code::ConfigChanged => "ConfigChanged",
            Code::ContractChanged => "ContractChanged",
        }
    }
}

/// One refusal: its code, the JSON path it is about (`$.config.PORT`), and what was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: Code,
    pub path: String,
    pub message: String,
}

impl Finding {
    pub fn to_json(&self) -> Value {
        json!({ "code": self.code.as_str(), "path": self.path, "message": self.message })
    }
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at {}: {}",
            self.code.as_str(),
            self.path,
            self.message
        )
    }
}

/// A declared variable's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    String,
    Int,
    Bool,
}

/// The budget, by ixctl's names (`go/ixctl/spec.go` `Limits`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub wall_ms: u64,
    pub memory_mb: u64,
}

/// What a caller may do: the scopes it grants and the budget ceiling.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mandate {
    pub scopes: BTreeSet<String>,
    pub max: Limits,
}

impl Mandate {
    pub fn to_json(&self) -> Value {
        json!({
            "scopes": self.scopes,
            "max": { "wall_ms": self.max.wall_ms, "memory_mb": self.max.memory_mb },
        })
    }
}

/// An artifact by name, exact version and platform.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub name: String,
    pub version: String,
    pub platform: String,
}

/// Where artifacts are found: their path and `sha256:<hex>` digest, read now.
pub trait Store {
    fn find(&self, artifact: &ArtifactRef) -> Option<(PathBuf, String)>;
}

/// Versioned binaries on disk: `<root>/<name>/<version>/<platform>/` holds exactly one file,
/// the artifact, digested each time it is looked up.
pub struct DirStore {
    pub root: PathBuf,
}

impl Store for DirStore {
    fn find(&self, a: &ArtifactRef) -> Option<(PathBuf, String)> {
        // Names are path segments: a separator or `..` in one would leave the store.
        let segment = |s: &str| !s.is_empty() && s != ".." && !s.contains(['/', '\\', ':']);
        if ![&a.name, &a.version, &a.platform]
            .iter()
            .all(|s| segment(s))
        {
            return None;
        }
        let dir = self.root.join(&a.name).join(&a.version).join(&a.platform);
        let mut files = std::fs::read_dir(dir)
            .ok()?
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()));
        let (Some(only), None) = (files.next(), files.next()) else {
            return None;
        };
        let bytes = std::fs::read(only.path()).ok()?;
        Some((only.path(), ix_recover::sha256_hex(&bytes)))
    }
}

/// What the plan is resolved against: the platform it would run on, the contract the plane
/// announces, the environment and the artifact store.
pub struct World<'a> {
    pub platform: String,
    pub contract: String,
    pub env: BTreeMap<String, String>,
    pub store: &'a dyn Store,
}

/// A resolved plan: every reference exact, every variable read and typed.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan(Value);

impl Plan {
    pub fn to_json(&self) -> &Value {
        &self.0
    }
}

/// A plan admitted under a mandate, named by its digest. Built only by [`admit`] or read back
/// with [`Admitted::from_json`]; [`apply`] checks the digest either way.
#[derive(Debug, Clone, PartialEq)]
pub struct Admitted {
    digest: String,
    mandate_sha256: String,
    plan: Value,
}

impl Admitted {
    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn plan(&self) -> &Value {
        &self.plan
    }

    /// `{digest, mandate_sha256, plan}`: what crosses a process boundary.
    pub fn to_json(&self) -> Value {
        json!({ "digest": self.digest, "mandate_sha256": self.mandate_sha256, "plan": self.plan })
    }

    /// Reads a document [`Admitted::to_json`] wrote. Only its shape is checked here: whether
    /// the plan still matches its digest is [`apply`]'s first check.
    pub fn from_json(v: &Value) -> Result<Admitted, Vec<Finding>> {
        let text = |name: &str| {
            v.get(name)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| {
                    vec![finding(
                        Code::Schema,
                        &format!("$.{name}"),
                        "missing or not a string",
                    )]
                })
        };
        let plan = v
            .get("plan")
            .filter(|p| p.is_object())
            .cloned()
            .ok_or_else(|| vec![finding(Code::Schema, "$.plan", "missing or not an object")])?;
        Ok(Admitted {
            digest: text("digest")?,
            mandate_sha256: text("mandate_sha256")?,
            plan,
        })
    }
}

/// Starts the work an admitted document describes. The test runner records; nothing in this
/// module starts a process.
pub trait Runner {
    fn launch(&mut self, digest: &str, artifact: &Path, plan: &Value);
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Proposal {
    schema: String,
    service: String,
    artifact: ArtifactRef,
    contract: String,
    #[serde(default)]
    config: BTreeMap<String, ConfigDecl>,
    limits: Limits,
    #[serde(default)]
    needs: BTreeSet<String>,
}

/// A declared variable: the environment variable it is read from, its type, and whether its
/// value may be written down. Every declared variable is required.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigDecl {
    env: String,
    #[serde(rename = "type")]
    kind: Kind,
    #[serde(default)]
    secret: bool,
}

fn finding(code: Code, path: &str, message: &str) -> Finding {
    Finding {
        code,
        path: path.to_string(),
        message: message.to_string(),
    }
}

/// A proposal, resolved against `world`. Every finding is reported, not only the first.
pub fn resolve(proposal: &Value, world: &World) -> Result<Plan, Vec<Finding>> {
    let p: Proposal = serde_json::from_value(proposal.clone())
        .map_err(|e| vec![finding(Code::Schema, "$", &e.to_string())])?;
    if p.schema != PROPOSAL_SCHEMA {
        return Err(vec![finding(
            Code::Schema,
            "$.schema",
            &format!("`{}` is not `{PROPOSAL_SCHEMA}`", p.schema),
        )]);
    }
    let mut out = Vec::new();

    let a = &p.artifact;
    let mut sha256 = None;
    if !exact_version(&a.version) {
        out.push(finding(
            Code::FloatingVersion,
            "$.artifact.version",
            &format!(
                "`{}` is not one exact version (MAJOR.MINOR.PATCH)",
                a.version
            ),
        ));
    } else if a.platform != world.platform {
        out.push(finding(
            Code::PlatformMismatch,
            "$.artifact.platform",
            &format!(
                "built for `{}`, would run on `{}`",
                a.platform, world.platform
            ),
        ));
    } else {
        match world.store.find(a) {
            Some((_, digest)) => sha256 = Some(digest),
            None => out.push(finding(
                Code::ArtifactNotFound,
                "$.artifact",
                &format!(
                    "no single `{}` {} for {} in the store",
                    a.name, a.version, a.platform
                ),
            )),
        }
    }
    if p.contract != world.contract {
        out.push(finding(
            Code::ContractMismatch,
            "$.contract",
            &format!(
                "requires `{}`, the plane keeps `{}`",
                p.contract, world.contract
            ),
        ));
    }

    let mut config = serde_json::Map::new();
    for (name, decl) in &p.config {
        match read_config(name, decl, &world.env) {
            Ok(entry) => {
                config.insert(name.clone(), entry);
            }
            Err(f) => out.push(f),
        }
    }

    for (field, value) in [
        ("wall_ms", p.limits.wall_ms),
        ("memory_mb", p.limits.memory_mb),
    ] {
        if value == 0 {
            out.push(finding(
                Code::InvalidBudget,
                &format!("$.limits.{field}"),
                "0 is no limit in ixctl: an admitted plan names a bound",
            ));
        }
    }

    if !out.is_empty() {
        return Err(out);
    }
    Ok(Plan(json!({
        "schema": PLAN_SCHEMA,
        "service": p.service,
        "artifact": {
            "name": a.name, "version": a.version, "platform": a.platform, "sha256": sha256,
        },
        "contract": p.contract,
        "config": config,
        "limits": { "wall_ms": p.limits.wall_ms, "memory_mb": p.limits.memory_mb },
        "needs": p.needs,
    })))
}

/// One declared variable read from `env`: `{env, type, value}`, or `{env, type, secret: true}`
/// for a secret, whose value is checked and dropped.
fn read_config(
    name: &str,
    decl: &ConfigDecl,
    env: &BTreeMap<String, String>,
) -> Result<Value, Finding> {
    let path = format!("$.config.{name}");
    let kind = match decl.kind {
        Kind::String => "string",
        Kind::Int => "int",
        Kind::Bool => "bool",
    };
    let raw = env.get(&decl.env).ok_or_else(|| {
        finding(
            Code::MissingConfig,
            &path,
            &format!("`{}` is not set", decl.env),
        )
    })?;
    let value = match decl.kind {
        Kind::String => Some(json!(raw)),
        Kind::Int => raw.trim().parse::<i64>().ok().map(|n| json!(n)),
        Kind::Bool => match raw.trim() {
            "true" => Some(json!(true)),
            "false" => Some(json!(false)),
            _ => None,
        },
    };
    let Some(value) = value else {
        // The value itself stays out of the message: it may be a secret.
        return Err(finding(
            Code::ConfigType,
            &path,
            &format!("`{}` does not parse as {kind}", decl.env),
        ));
    };
    Ok(if decl.secret {
        json!({ "env": decl.env, "type": kind, "secret": true })
    } else {
        json!({ "env": decl.env, "type": kind, "value": value })
    })
}

/// `MAJOR.MINOR.PATCH`, with an optional `-pre` or `+build` suffix: one version, not a range.
fn exact_version(v: &str) -> bool {
    let core = v.split(['-', '+']).next().unwrap_or("");
    let suffix_ok = v[core.len()..]
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+'));
    let parts: Vec<&str> = core.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        && suffix_ok
}

/// `plan`, admitted under `mandate`: every scope it needs granted and its budget within the
/// mandate's ceiling. A shortfall is refused, never carved down or widened.
pub fn admit(plan: &Plan, mandate: &Mandate) -> Result<Admitted, Vec<Finding>> {
    let p = &plan.0;
    let mut out = Vec::new();
    let missing: Vec<&str> = p["needs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|s| !mandate.scopes.contains(*s))
        .collect();
    if !missing.is_empty() {
        out.push(finding(
            Code::MandateInsufficient,
            "$.needs",
            &format!("not granted: {}", missing.join(", ")),
        ));
    }
    for (field, max) in [
        ("wall_ms", mandate.max.wall_ms),
        ("memory_mb", mandate.max.memory_mb),
    ] {
        let want = p["limits"][field].as_u64().unwrap_or(u64::MAX);
        if want > max {
            out.push(finding(
                Code::BudgetExceedsMandate,
                &format!("$.limits.{field}"),
                &format!("{want} is past the mandate's {max}"),
            ));
        }
    }
    if !out.is_empty() {
        return Err(out);
    }
    Ok(Admitted {
        digest: ix_recover::canonical_sha256(p),
        mandate_sha256: ix_recover::canonical_sha256(&mandate.to_json()),
        plan: p.clone(),
    })
}

/// Hands `admitted` to `runner` once, if its digest, its mandate and the world it names still
/// hold; answers the digest launched. Every difference is reported and nothing runs.
pub fn apply(
    admitted: &Admitted,
    mandate: &Mandate,
    world: &World,
    runner: &mut dyn Runner,
) -> Result<String, Vec<Finding>> {
    let p = &admitted.plan;
    if ix_recover::canonical_sha256(p) != admitted.digest {
        return Err(vec![finding(
            Code::RevisionTampered,
            "$.plan",
            "the plan no longer matches the digest it was admitted under",
        )]);
    }
    if ix_recover::canonical_sha256(&mandate.to_json()) != admitted.mandate_sha256 {
        return Err(vec![finding(
            Code::MandateChanged,
            "$.mandate",
            "not the mandate this plan was admitted under",
        )]);
    }
    let mut out = Vec::new();

    let artifact: Result<ArtifactRef, _> = serde_json::from_value(json!({
        "name": p["artifact"]["name"],
        "version": p["artifact"]["version"],
        "platform": p["artifact"]["platform"],
    }));
    let mut path = None;
    match artifact {
        Err(e) => out.push(finding(Code::Schema, "$.plan.artifact", &e.to_string())),
        Ok(a) if a.platform != world.platform => out.push(finding(
            Code::PlatformMismatch,
            "$.artifact.platform",
            &format!(
                "admitted for `{}`, would run on `{}`",
                a.platform, world.platform
            ),
        )),
        Ok(a) => match world.store.find(&a) {
            None => out.push(finding(
                Code::ArtifactNotFound,
                "$.artifact",
                &format!("`{}` {} is no longer in the store", a.name, a.version),
            )),
            Some((found, digest)) if Some(digest.as_str()) == p["artifact"]["sha256"].as_str() => {
                path = Some(found)
            }
            Some(_) => out.push(finding(
                Code::ArtifactChanged,
                "$.artifact",
                &format!(
                    "`{}` {}'s bytes differ from the admitted digest",
                    a.name, a.version
                ),
            )),
        },
    }

    if p["contract"].as_str() != Some(world.contract.as_str()) {
        out.push(finding(
            Code::ContractChanged,
            "$.contract",
            &format!(
                "admitted for `{}`, the plane keeps `{}`",
                p["contract"], world.contract
            ),
        ));
    }

    for (name, entry) in p["config"].as_object().into_iter().flatten() {
        let decl: Result<ConfigDecl, _> = serde_json::from_value(json!({
            "env": entry["env"], "type": entry["type"], "secret": entry.get("secret").is_some(),
        }));
        let Ok(decl) = decl else {
            out.push(finding(
                Code::Schema,
                &format!("$.plan.config.{name}"),
                "unreadable entry",
            ));
            continue;
        };
        match read_config(name, &decl, &world.env) {
            Err(f) => out.push(f),
            Ok(now) if now != *entry => out.push(finding(
                Code::ConfigChanged,
                &format!("$.config.{name}"),
                &format!("`{}` changed since admission", decl.env),
            )),
            Ok(_) => {}
        }
    }

    match (out.is_empty(), path) {
        (true, Some(path)) => {
            runner.launch(&admitted.digest, &path, p);
            Ok(admitted.digest.clone())
        }
        _ => Err(out),
    }
}
