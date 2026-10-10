//! Pure scope-refinement preview on the existing IXQL capability seam.
//! No subprocess, permission decision, resume, persistent state or effect port.
use std::collections::BTreeSet;
use std::sync::Arc;
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::{CallArgs, Capability, Host, Produced};

pub const CONTRACT_VERSION: &str = "0.1.0";
pub const DSL_VERSION: &str = "1";
pub const FUNCTION_VERSION: &str = "0.1.0";
pub const FUNCTION_NAME: &str = "mission.refine_scope";
pub const INPUT_SCHEMA: &str = include_str!("mission-scope.schema.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnginePin { pub version: String, pub source_sha: String }

/// Canonical typed request; its JSON Schema is exported by input_schema().
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeProposal {
    pub contract_version: String,
    pub dsl_version: String,
    pub engine: EnginePin,
    pub function_version: String,
    pub implementation_sha: String,
    pub mission_id: String,
    pub attempt_id: String,
    pub revision: String,
    pub refusal_id: String,
    pub owner: String,
    pub paths: Vec<String>,
}

/// Legacy Go JSONL fields, carried unchanged inside a host-correlated envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeRecord {
    pub ts: String,
    pub metric: String,
    pub value: f64,
    pub unit: Option<String>,
    pub probe: String,
    pub build: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnershipClaim { pub path: String, pub owners: Vec<String> }

/// Trusted host composition supplies this; proposal/task text cannot replace it.
/// complete/physical_paths_verified concern ownership only, never permissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeObservation {
    pub mission_id: String,
    pub attempt_id: String,
    pub revision: String,
    pub refusal_id: String,
    pub refusal_reason: String,
    pub original_paths: Vec<String>,
    pub observed_at_ms: i64,
    pub expires_at_ms: i64,
    pub complete: bool,
    pub physical_paths_verified: bool,
    pub claims: Vec<OwnershipClaim>,
    pub probe: Option<ProbeRecord>,
}

pub fn input_schema() -> Value {
    serde_json::from_str(INPUT_SCHEMA).expect("embedded input schema")
}

pub fn output_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object", "additionalProperties": false,
        "required": ["contract_version", "dsl_version", "engine", "function_version",
            "implementation_sha", "mission_id", "attempt_id", "revision", "refusal_id",
            "owner", "paths", "argv", "status", "requires_runner_admission",
            "expires_at_ms", "observation"],
        "properties": {
            "contract_version": {"const": CONTRACT_VERSION},
            "dsl_version": {"const": DSL_VERSION},
            "engine": input_schema()["properties"]["engine"],
            "function_version": {"const": FUNCTION_VERSION},
            "implementation_sha": {"type":"string", "pattern":"^[0-9a-f]{40}$"},
            "mission_id":{"type":"string"}, "attempt_id":{"type":"string"},
            "revision":{"type":"string"}, "refusal_id":{"type":"string"},
            "owner":{"type":"string"}, "paths":{"type":"array", "items":{"type":"string"}},
            "argv":{"type":"array", "items":{"type":"string"}},
            "status":{"const":"proposal"}, "requires_runner_admission":{"const":true},
            "expires_at_ms":{"type":"integer"}, "observation":{"type":"object"}
        }
    })
}

pub struct ScopeRefinement {
    host: Arc<dyn Host>,
    implementation_sha: String,
    engine: EnginePin,
    observation: Option<ScopeObservation>,
}

impl ScopeRefinement {
    /// Only the host injects evidence and code identities; no default live binding.
    pub fn new(
        host: Arc<dyn Host>, implementation_sha: String, engine: EnginePin,
        observation: Option<ScopeObservation>,
    ) -> Result<Self, String> {
        if !commit_sha(&implementation_sha) || engine.version != env!("CARGO_PKG_VERSION") ||
            !commit_sha(&engine.source_sha) {
            return Err("ContractPinMismatch".into());
        }
        Ok(Self { host, implementation_sha, engine, observation })
    }

    fn preview(&self, request: ScopeProposal) -> Result<Value, String> {
        if request.contract_version != CONTRACT_VERSION || request.dsl_version != DSL_VERSION ||
            request.function_version != FUNCTION_VERSION ||
            request.implementation_sha != self.implementation_sha || request.engine != self.engine {
            return Err("ContractPinMismatch".into());
        }
        let proof = self.observation.as_ref().ok_or("ObservationMissing")?;
        if request.mission_id != proof.mission_id || request.attempt_id != proof.attempt_id ||
            request.revision != proof.revision || request.refusal_id != proof.refusal_id {
            return Err("ObservationBindingMismatch".into());
        }
        if proof.refusal_reason != "scope_too_broad" || proof.original_paths != ["."] {
            return Err("RefusalUnknown".into());
        }
        let now = self.host.now().timestamp_millis();
        if proof.observed_at_ms > now || proof.expires_at_ms <= proof.observed_at_ms ||
            !matches!(proof.expires_at_ms.checked_sub(proof.observed_at_ms), Some(1..=30_000)) {
            return Err("ObservationTimeUnknown".into());
        }
        if now >= proof.expires_at_ms {
            return Err("ObservationExpired".into());
        }
        let sample = proof.probe.as_ref().ok_or("ProbeMissing")?;
        let at = DateTime::parse_from_rfc3339(&sample.ts)
            .map_err(|_| "ProbeTimeUnknown")?.timestamp_millis();
        if at > now || at > proof.observed_at_ms || now.saturating_sub(at) >= 30_000 {
            return Err("ProbeExpiredOrFuture".into());
        }
        if !sample.value.is_finite() || sample.metric.trim().is_empty() ||
            sample.probe.trim().is_empty() || sample.build.len() != 12 ||
            !sample.build.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("ProbeUnknown".into());
        }
        if !proof.complete || !proof.physical_paths_verified || proof.claims.is_empty() ||
            proof.claims.len() > 256 {
            return Err("OwnershipUnknown".into());
        }
        if request.paths.is_empty() || request.paths.len() > 32 {
            return Err("ScopeUnknown".into());
        }
        let mut paths = BTreeSet::new();
        for path in &request.paths {
            if path == "." { return Err("ScopeTooBroad".into()); }
            canonical_path(path)?;
            if !paths.insert(path) { return Err("DuplicateScope".into()); }
        }
        if request.paths.len() != 1 { return Err("SingleScopeRequired".into()); }
        let candidate = &request.paths[0];
        let mut claim_paths = BTreeSet::new();
        let mut owned = false;
        for claim in &proof.claims {
            canonical_path(&claim.path)?;
            if !claim_paths.insert(&claim.path) { return Err("DuplicateOwnership".into()); }
            if overlaps(candidate, &claim.path) {
                if claim.owners.len() != 1 || claim.owners[0] != request.owner {
                    return Err("OwnershipAmbiguous".into());
                }
                if candidate == &claim.path || candidate.starts_with(&(claim.path.clone() + "/")) {
                    owned = true;
                }
            }
        }
        if !owned { return Err("ScopeNotOwned".into()); }
        let mut result = serde_json::to_value(&request).map_err(|e| e.to_string())?;
        let map = result.as_object_mut().expect("serialized request object");
        map.insert("argv".into(), json!(["git", "add", "--", format!("{candidate}/")]));
        map.insert("status".into(), json!("proposal"));
        map.insert("requires_runner_admission".into(), json!(true));
        map.insert("expires_at_ms".into(), json!(proof.expires_at_ms.min(at.saturating_add(30_000))));
        map.insert("observation".into(), serde_json::to_value(sample).map_err(|e| e.to_string())?);
        Ok(result)
    }
}

fn commit_sha(sha: &str) -> bool {
    sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn canonical_path(path: &str) -> Result<(), String> {
    if path.len() > 512 || crate::path::normalize(path).map_err(|_| "ScopePathInvalid")? != path ||
        path.split('/').any(|segment| segment.eq_ignore_ascii_case(".git") || segment.starts_with('-')) ||
        !path.bytes().all(|b| b.is_ascii_alphanumeric() || b"./_-".contains(&b)) {
        return Err("ScopePathInvalid".into());
    }
    Ok(())
}

fn overlaps(a: &str, b: &str) -> bool {
    a == b || a.starts_with(&(b.to_string() + "/")) || b.starts_with(&(a.to_string() + "/"))
}

impl Capability for ScopeRefinement {
    fn call(&self, args: CallArgs) -> Result<Produced, String> {
        if !args.positional.is_empty() || !args.named.is_empty() {
            return Err("PipedContractRequired".into());
        }
        let value = args.piped.ok_or("PipedContractRequired")?;
        let validator = jsonschema::draft202012::new(&input_schema()).map_err(|e| e.to_string())?;
        if !validator.is_valid(&value) {
            return Err("ProposalContractInvalid".into());
        }
        let request = serde_json::from_value(value).map_err(|e| format!("ProposalContractInvalid: {e}"))?;
        self.preview(request).map(Produced::plain)
    }
}
