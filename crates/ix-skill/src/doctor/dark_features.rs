//! Dark-feature check — source that belongs to no compiled configuration.
//!
//! A cargo feature that **no workspace member enables** is never turned on by
//! `cargo build --workspace`, `cargo clippy --workspace` or `cargo test
//! --workspace`. Every module behind it is uncompiled: it is not type-checked,
//! not linted, and its tests are neither passed nor failed — they are absent.
//! A skipped crate contributes zero to the pass count and zero to the fail
//! count, so nothing in the repository's output distinguishes "these five tests
//! pass" from "these five tests were never run" (ix#315).
//!
//! The motivating case is `ix-code`'s `topology` feature: 296 lines and five
//! tests, merged, reviewed, and never once compiled by CI. `rust-analyzer`
//! already says so by hand — *"this file is not included in any crates"*. This
//! is the machine-readable version.
//!
//! # This is not an argument against feature gating
//!
//! Keeping DuckDB, arrow and ONNX Runtime out of the default build is
//! deliberate and correct, and the root `Cargo.toml` says so. The defect is not
//! that the code is gated — it is that **gated code is indistinguishable from
//! compiled code in every signal the repository produces**. This check restores
//! the distinction and says nothing about whether a gate should exist. A
//! feature that must stay dark belongs in the allowlist with a written reason,
//! and stays there.
//!
//! # Where the answer comes from
//!
//! `cargo metadata` resolves the workspace exactly as the default build does,
//! and reports the feature set each package ends up with. Each `Cargo.toml`
//! declares the features that *exist*. The difference is the dark set — derived
//! per run, never a list to maintain. Cargo is the only authority on feature
//! resolution and re-deriving it here would be a second one.
//!
//! # Reporting size, not just existence
//!
//! A dark feature gating two lines is noise; one gating 95 tests is the defect
//! this check exists for. So each finding carries the modules behind it, their
//! line count, and their test count, and only findings above
//! [`SIGNIFICANT_LOC`] lines *or* with at least one test can fail the check.
//! Meta-features (`full = ["a", "b"]`) and implicit optional-dependency
//! features gate no source of their own, measure zero, and fall out as noise
//! without being special-cased.
//!
//! # Deliberate under-reporting
//!
//! Like [`super::orphan_traits`], every ambiguity resolves toward *not*
//! reporting: an item whose `cfg` also names an enabled feature is treated as
//! reachable, an unresolvable `mod` declaration counts as a bare item rather
//! than a module, and a `cfg` attribute split across lines is missed entirely
//! (there are none in this workspace today). The check under-states how much
//! code is dark rather than crying wolf.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Path of the allowlist, relative to the repo root.
pub const ALLOWLIST_PATH: &str = "state/registry/dark-features.allow.json";

/// Gated lines at or above which a dark feature fails the check on size alone.
///
/// Below this and with no tests behind it, a dark feature is reported but does
/// not block: the point of the check is uncompiled *substance*.
pub const SIGNIFICANT_LOC: usize = 100;

// ---------------------------------------------------------------- cargo metadata

/// The slice of `cargo metadata` output this check reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    /// Every package in the graph, workspace members and dependencies alike.
    pub packages: Vec<MetaPackage>,
    /// Package ids belonging to this workspace.
    pub workspace_members: Vec<String>,
    /// The resolved graph, including each node's enabled feature set.
    pub resolve: MetaResolve,
}

/// One package as `cargo metadata` reports it.
#[derive(Debug, Clone, Deserialize)]
pub struct MetaPackage {
    /// Opaque package id, matching `resolve.nodes[].id`.
    pub id: String,
    /// Crate name.
    pub name: String,
    /// Absolute path of the package's `Cargo.toml`.
    pub manifest_path: String,
    /// Declared features: name -> what it enables.
    pub features: std::collections::BTreeMap<String, Vec<String>>,
}

/// The resolved dependency graph.
#[derive(Debug, Clone, Deserialize)]
pub struct MetaResolve {
    /// One entry per package in the resolve.
    pub nodes: Vec<MetaNode>,
}

/// A resolved package and the features the default workspace build gives it.
#[derive(Debug, Clone, Deserialize)]
pub struct MetaNode {
    /// Package id, matching [`MetaPackage::id`].
    pub id: String,
    /// Features enabled for this package in the resolve. Includes features
    /// pulled in transitively by `default` and by dependents.
    pub features: Vec<String>,
}

impl Metadata {
    /// Parse `cargo metadata --format-version 1` output.
    pub fn from_json(raw: &str) -> Result<Self, String> {
        serde_json::from_str(raw).map_err(|e| format!("parsing cargo metadata: {e}"))
    }

    /// Run `cargo metadata` against `root` and parse it.
    ///
    /// No `--all-features`, no `--no-default-features`: the resolve has to be
    /// the one the default build uses, because that is the configuration whose
    /// blind spot this check measures.
    pub fn from_cargo(root: &Path) -> Result<Self, String> {
        let out = std::process::Command::new("cargo")
            .current_dir(root)
            .args(["metadata", "--format-version", "1"])
            .output()
            .map_err(|e| format!("running `cargo metadata`: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "`cargo metadata` failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Self::from_json(&String::from_utf8_lossy(&out.stdout))
    }
}

/// A workspace feature that the default build never enables, before its gated
/// source has been measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// Crate declaring the feature.
    pub krate: String,
    /// Feature name.
    pub feature: String,
    /// Directory containing the crate's `Cargo.toml`.
    pub dir: PathBuf,
    /// Features the resolve *does* enable for this crate. Used to spare any
    /// `cfg` that also names one of them.
    pub enabled: BTreeSet<String>,
}

/// Every declared, non-default feature of a workspace member that the resolve
/// does not enable.
///
/// `default` itself is skipped, and so is anything the resolve lists — the
/// resolve already contains features reached through `default`, so no separate
/// closure over the crate's own feature table is needed.
pub fn candidates(meta: &Metadata) -> Vec<Candidate> {
    let members: BTreeSet<&str> = meta.workspace_members.iter().map(String::as_str).collect();
    let mut out = Vec::new();

    for pkg in &meta.packages {
        if !members.contains(pkg.id.as_str()) {
            continue;
        }
        let enabled: BTreeSet<String> = meta
            .resolve
            .nodes
            .iter()
            .find(|n| n.id == pkg.id)
            .map(|n| n.features.iter().cloned().collect())
            .unwrap_or_default();

        let dir = Path::new(&pkg.manifest_path)
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();

        for feature in pkg.features.keys() {
            if feature == "default" || enabled.contains(feature) {
                continue;
            }
            out.push(Candidate {
                krate: pkg.name.clone(),
                feature: feature.clone(),
                dir: dir.clone(),
                enabled: enabled.clone(),
            });
        }
    }
    out.sort_by(|a, b| a.krate.cmp(&b.krate).then(a.feature.cmp(&b.feature)));
    out
}

// ---------------------------------------------------------------- measurement

/// A dark feature and the source behind it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DarkFeature {
    /// Crate declaring the feature.
    #[serde(rename = "crate")]
    pub krate: String,
    /// Feature name.
    pub feature: String,
    /// Repo-relative module files reachable only through this feature.
    pub modules: Vec<String>,
    /// Total lines across [`Self::modules`].
    pub loc: usize,
    /// `#[test]` / `#[tokio::test]` attributes across [`Self::modules`]. These
    /// are the tests that can neither pass nor fail.
    pub tests: usize,
    /// Gated items that are not whole modules — a `fn`, an `impl`, an inline
    /// `mod { .. }`. Counted, not sized: they are usually small, and measuring
    /// their extent would need a parser.
    pub items: usize,
}

impl DarkFeature {
    /// `crate/feature`, the identity used by the allowlist.
    pub fn key(&self) -> String {
        format!("{}/{}", self.krate, self.feature)
    }

    /// Whether this is substance rather than noise.
    ///
    /// One uncompiled test is enough on its own: a test that cannot fail is
    /// the exact defect ix#315 describes, at any size.
    pub fn is_significant(&self) -> bool {
        self.tests > 0 || self.loc >= SIGNIFICANT_LOC
    }
}

/// Measure the source behind each candidate.
pub fn measure(root: &Path, candidates: &[Candidate]) -> Vec<DarkFeature> {
    candidates.iter().map(|c| measure_one(root, c)).collect()
}

fn measure_one(root: &Path, cand: &Candidate) -> DarkFeature {
    let mut files = Vec::new();
    collect_rs(&cand.dir, &mut files);

    let mut roots: BTreeSet<PathBuf> = BTreeSet::new();
    let mut items = 0usize;

    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains("#[cfg") {
                continue;
            }
            let feats = cfg_features(line);
            if !feats.iter().any(|f| f == &cand.feature) {
                continue;
            }
            // Also reachable through a feature the build does enable, so the
            // code is compiled and this is not a finding.
            if feats.iter().any(|f| cand.enabled.contains(f.as_str())) {
                continue;
            }
            match gated_module_name(&lines, i).and_then(|n| resolve_module(path, &n)) {
                Some(file) => {
                    roots.insert(file);
                }
                None => items += 1,
            }
        }
    }

    // A gated module's own submodules are gated too, whatever their own `cfg`
    // says: the parent never exists, so neither do they.
    let modules = expand_submodules(roots);

    let mut loc = 0usize;
    let mut tests = 0usize;
    for file in &modules {
        let Ok(src) = std::fs::read_to_string(file) else {
            continue;
        };
        loc += src.lines().count();
        tests += count_tests(&src);
    }

    DarkFeature {
        krate: cand.krate.clone(),
        feature: cand.feature.clone(),
        modules: modules.iter().map(|p| rel(root, p)).collect(),
        loc,
        tests,
        items,
    }
}

/// Transitively pull in the modules a gated module declares.
fn expand_submodules(roots: BTreeSet<PathBuf>) -> BTreeSet<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut queue: Vec<PathBuf> = roots.into_iter().collect();
    while let Some(path) = queue.pop() {
        if !seen.insert(path.clone()) {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in src.lines() {
            let Some(name) = parse_mod_decl(line) else {
                continue;
            };
            if let Some(child) = resolve_module(&path, &name) {
                if !seen.contains(&child) {
                    queue.push(child);
                }
            }
        }
    }
    seen
}

/// Feature names appearing in a `#[cfg(..)]` attribute line.
///
/// Structure is ignored: `all(..)`, `any(..)` and `not(..)` all yield the same
/// name set. That is deliberate — the only use of the set is "does this also
/// name something enabled?", and treating an `all()` as reachable when one of
/// its features is on under-reports rather than over-reports.
fn cfg_features(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(at) = rest.find("feature") {
        rest = &rest[at + "feature".len()..];
        let Some(eq) = rest.find('=') else { break };
        let after = rest[eq + 1..].trim_start();
        let Some(body) = after.strip_prefix('"') else {
            continue;
        };
        let Some(end) = body.find('"') else { break };
        out.push(body[..end].to_string());
        rest = &body[end..];
    }
    out
}

/// If the attribute at `idx` decorates a `mod name;` declaration, return the
/// name. Intervening attributes, blank lines and comments are skipped.
fn gated_module_name(lines: &[&str], idx: usize) -> Option<String> {
    for line in lines.iter().skip(idx + 1).take(4) {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with("#[") {
            continue;
        }
        return parse_mod_decl(line);
    }
    None
}

/// Return the module name if `line` is a file-backed `mod name;` declaration.
///
/// An inline `mod name { .. }` is not one: its body is already in this file and
/// counted by the enclosing module.
fn parse_mod_decl(line: &str) -> Option<String> {
    let mut t = line.trim();
    if let Some(r) = t.strip_prefix("pub") {
        t = r.trim_start();
        if let Some(r) = t.strip_prefix('(') {
            t = r.split_once(')')?.1.trim_start();
        }
    }
    let rest = t.strip_prefix("mod ")?.trim_start();
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    rest[name.len()..]
        .trim_start()
        .starts_with(';')
        .then_some(name)
}

/// Resolve `mod name;` declared inside `decl_file` to the file backing it.
///
/// Both layouts are tried, and for a non-`mod.rs`/`lib.rs` parent the
/// `foo.rs` + `foo/` sibling-directory layout as well. `#[path = ".."]`
/// overrides are not followed — an unresolvable declaration is counted as a
/// bare item instead, which under-reports.
fn resolve_module(decl_file: &Path, name: &str) -> Option<PathBuf> {
    let dir = decl_file.parent()?;
    let mut roots = vec![dir.to_path_buf()];
    let stem = decl_file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if !matches!(stem, "mod" | "lib" | "main") {
        roots.push(dir.join(stem));
    }
    for base in roots {
        for cand in [
            base.join(format!("{name}.rs")),
            base.join(name).join("mod.rs"),
        ] {
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

/// Count `#[test]` / `#[tokio::test]` attributes.
fn count_tests(src: &str) -> usize {
    src.lines()
        .filter(|l| {
            let t = l.trim();
            t.starts_with("#[test]") || t.starts_with("#[tokio::test") || t.starts_with("#[test(")
        })
        .count()
}

// ---------------------------------------------------------------- allowlist

/// Why a feature is allowed to stay dark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// Enabling it here is blocked or unreasonably costly for reasons outside
    /// the code: a native toolchain, an external binary, GPU hardware, a
    /// toolchain newer than the workspace MSRV. These are expected to stay.
    Environment,
    /// Nothing environmental prevents compiling it — it simply is not wired up
    /// yet. Requires an issue, and is reported as outstanding debt so it does
    /// not become a permanent parking space.
    Tracked,
}

/// One human-signed-off exemption.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AllowEntry {
    /// Crate declaring the feature.
    #[serde(rename = "crate")]
    pub krate: String,
    /// Feature name.
    pub feature: String,
    /// Which sort of exemption this is.
    pub kind: Kind,
    /// Why this feature may stay uncompiled. Must be non-empty: the whole
    /// point of the allowlist is that a human said why.
    #[serde(default)]
    pub reason: String,
    /// Tracking issue. Required for [`Kind::Tracked`].
    #[serde(default)]
    pub issue: Option<String>,
}

impl AllowEntry {
    /// `crate/feature`, matching [`DarkFeature::key`].
    pub fn key(&self) -> String {
        format!("{}/{}", self.krate, self.feature)
    }
}

/// Parsed allowlist file.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Allowlist {
    /// Free-text header explaining the file to whoever opens it next.
    #[serde(default)]
    pub note: String,
    /// The exemptions.
    #[serde(default)]
    pub allow: Vec<AllowEntry>,
}

impl Allowlist {
    /// Load the allowlist, treating an absent file as empty.
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join(ALLOWLIST_PATH);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        serde_json::from_str(&raw).map_err(|e| format!("parsing {}: {e}", path.display()))
    }
}

// ---------------------------------------------------------------- reconciliation

/// Outcome of the scan, before it is turned into a check result.
#[derive(Debug, Clone, Serialize)]
pub struct Census {
    /// Every dark feature found, significant or not.
    pub dark: Vec<DarkFeature>,
    /// Significant dark features with no allowlist entry — these fail.
    pub unlisted: Vec<DarkFeature>,
    /// Significant dark features covered by the allowlist.
    pub allowed: Vec<String>,
    /// Allowlist entries for a feature that is no longer dark, or no longer
    /// declared. Stale exemptions warn, so the allowlist cannot quietly rot.
    pub stale_allowlist: Vec<String>,
    /// Entries with an empty `reason`. These fail: an exemption without a
    /// stated reason is not a decision, just a silencer.
    pub reasonless_allowlist: Vec<String>,
    /// [`Kind::Tracked`] entries with no issue. These fail for the same reason:
    /// "we will get to it" with no owner is indistinguishable from silence.
    pub unattributed_allowlist: Vec<String>,
    /// Lines behind every dark feature, significant or not.
    pub dark_loc: usize,
    /// Tests behind every dark feature that can neither pass nor fail.
    pub dark_tests: usize,
    /// How many exemptions are [`Kind::Tracked`] — outstanding debt, as
    /// opposed to the environmental ones that are expected to stay.
    pub tracked_debt: usize,
}

/// Reconcile measured dark features against the allowlist.
pub fn reconcile(dark: Vec<DarkFeature>, allowlist: &Allowlist) -> Census {
    let dark_keys: BTreeSet<String> = dark.iter().map(DarkFeature::key).collect();

    let mut unlisted = Vec::new();
    let mut allowed = Vec::new();
    for feature in &dark {
        if !feature.is_significant() {
            continue;
        }
        let key = feature.key();
        if allowlist.allow.iter().any(|e| e.key() == key) {
            allowed.push(key);
        } else {
            unlisted.push(feature.clone());
        }
    }

    let stale_allowlist = allowlist
        .allow
        .iter()
        .filter(|e| !dark_keys.contains(&e.key()))
        .map(AllowEntry::key)
        .collect();
    let reasonless_allowlist = allowlist
        .allow
        .iter()
        .filter(|e| e.reason.trim().is_empty())
        .map(AllowEntry::key)
        .collect();
    let unattributed_allowlist = allowlist
        .allow
        .iter()
        .filter(|e| {
            e.kind == Kind::Tracked && e.issue.as_deref().map(str::trim).unwrap_or("").is_empty()
        })
        .map(AllowEntry::key)
        .collect();
    let tracked_debt = allowlist
        .allow
        .iter()
        .filter(|e| e.kind == Kind::Tracked)
        .count();

    Census {
        dark_loc: dark.iter().map(|d| d.loc).sum(),
        dark_tests: dark.iter().map(|d| d.tests).sum(),
        dark,
        unlisted,
        allowed,
        stale_allowlist,
        reasonless_allowlist,
        unattributed_allowlist,
        tracked_debt,
    }
}

/// Run the whole check against `root`: resolve, measure, reconcile.
pub fn scan(root: &Path) -> Result<Census, String> {
    let meta = Metadata::from_cargo(root)?;
    let allowlist = Allowlist::load(root)?;
    Ok(reconcile(measure(root, &candidates(&meta)), &allowlist))
}

// ---------------------------------------------------------------- filesystem

/// Recursively collect `.rs` files under `dir`.
fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "target" || name == ".git" || name == "node_modules" {
                continue;
            }
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Repo-relative, forward-slashed path for stable output across platforms.
fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
