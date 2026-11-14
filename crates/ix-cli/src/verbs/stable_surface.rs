//! `ix stable-surface` — print the maturity-tier Stable crates and a
//! deterministic hash of each crate's public API surface.
//!
//! ## Source of truth
//!
//! Crate maturity is read from `crate-maturity.toml` at the workspace root.
//! Locate the workspace by walking up from `IX_ROOT` (or CWD) until we find
//! that file.
//!
//! ## API hash
//!
//! We hash the **sorted, deduplicated set of public declarations** found by
//! scanning all `.rs` files under `crates/<crate>/src/`. A "public
//! declaration" is a non-blank, non-comment line whose first non-whitespace
//! token starts with `pub ` or `pub(crate)` (the latter is intentionally
//! excluded — only true `pub` items, plus `pub use` re-exports, count).
//!
//! This is intentionally simple: it catches added / removed / renamed
//! exported symbols, which is exactly the scope this guard cares about
//! (per the PR brief). It does NOT catch:
//!   - changes to function signatures (param types, return type)
//!   - changes to struct field types
//!   - trait bound additions / removals
//!   - variance changes
//!
//! Those finer-grained checks are out of scope for v1. The escape hatch
//! when a real breaking change ships is: demote the crate to "beta" in
//! `crate-maturity.toml` in the same PR (which silences the warn-only
//! guard for that crate).
//!
//! ## Determinism
//!
//! We sort the per-file lines, deduplicate them, then hash with BLAKE3.
//! Whitespace inside each captured line is preserved verbatim (renaming a
//! function changes the line). Comments and blank lines are skipped.

use crate::output::{self, Format};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Locate the workspace root by walking up from `start` until we find a
/// directory containing `crate-maturity.toml`.
pub fn locate_workspace_root() -> Result<PathBuf, String> {
    if let Ok(env_root) = std::env::var("IX_ROOT") {
        let p = PathBuf::from(env_root);
        if p.join("crate-maturity.toml").is_file() {
            return Ok(p);
        }
    }
    let mut cur = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    loop {
        if cur.join("crate-maturity.toml").is_file() {
            return Ok(cur);
        }
        if !cur.pop() {
            return Err("crate-maturity.toml not found in any parent directory".into());
        }
    }
}

/// Parse `crate-maturity.toml` into a name→tier map.
pub fn load_maturity(root: &Path) -> Result<BTreeMap<String, String>, String> {
    let path = root.join("crate-maturity.toml");
    let body = fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let parsed: toml::Value =
        toml::from_str(&body).map_err(|e| format!("parsing {}: {e}", path.display()))?;
    let crates = parsed
        .get("crates")
        .and_then(|v| v.as_table())
        .ok_or_else(|| "missing [crates] table".to_string())?;
    let mut out = BTreeMap::new();
    for (k, v) in crates {
        let tier = v
            .as_str()
            .ok_or_else(|| format!("crate `{k}` tier must be a string"))?;
        out.insert(k.clone(), tier.to_string());
    }
    Ok(out)
}

/// Extract the set of public-API declarations from a single `.rs` file.
///
/// A "public declaration" is a line whose first non-whitespace token starts
/// with `pub ` or `pub(`-something-other-than-`crate`/`super`/`self`. We
/// preserve the line verbatim so renaming a function changes the captured
/// string. Doc-comments and `//` comments are skipped.
fn extract_pub_lines(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in source.lines() {
        let trimmed = raw.trim_start();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        // Match `pub ` and `pub(<vis>) ` where <vis> != crate|super|self.
        // Strip any leading attribute marker `#[...]` continuation by just
        // requiring the line itself to start with `pub`.
        if let Some(rest) = trimmed.strip_prefix("pub") {
            // Next character determines whether this is a public item.
            let c = rest.chars().next();
            match c {
                Some(' ') | Some('\t') => out.push(trimmed.to_string()),
                Some('(') => {
                    // pub(crate), pub(super), pub(self) → not public surface.
                    // pub(in path) → also not the unqualified public surface.
                    // The only "true" public form with parens is pub itself,
                    // which never has parens. So skip all `pub(...)` forms.
                    let after = &rest[1..];
                    let restricted = after.starts_with("crate")
                        || after.starts_with("super")
                        || after.starts_with("self")
                        || after.starts_with("in ");
                    if !restricted {
                        // Unknown form — be conservative and include it.
                        out.push(trimmed.to_string());
                    }
                }
                _ => {} // e.g. `pubfn` (identifier collision) — ignore
            }
        }
    }
    out
}

/// Walk a `crates/<name>/src/` tree, collecting all unique public-decl lines.
fn collect_pub_decls(crate_src: &Path) -> Result<BTreeSet<String>, String> {
    let mut decls = BTreeSet::new();
    let mut stack = vec![crate_src.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("entry in {}: {e}", dir.display()))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|e| format!("file_type of {}: {e}", path.display()))?;
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs")
            {
                let body = fs::read_to_string(&path)
                    .map_err(|e| format!("reading {}: {e}", path.display()))?;
                for line in extract_pub_lines(&body) {
                    decls.insert(line);
                }
            }
        }
    }
    Ok(decls)
}

/// Compute the BLAKE3 hash (hex, first 16 chars) of a crate's public surface.
///
/// Returns `(hash, decl_count)`. If the crate src dir is missing, returns
/// `("missing", 0)` rather than failing — that's a real-world signal worth
/// surfacing in the report instead of crashing.
pub fn compute_api_hash(workspace_root: &Path, crate_name: &str) -> (String, usize) {
    let src = workspace_root.join("crates").join(crate_name).join("src");
    if !src.is_dir() {
        return ("missing".into(), 0);
    }
    let decls = match collect_pub_decls(&src) {
        Ok(d) => d,
        Err(_) => return ("error".into(), 0),
    };
    let mut hasher = blake3::Hasher::new();
    for line in &decls {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    let hash = hasher.finalize().to_hex();
    // First 16 hex chars = 64 bits — plenty for change detection.
    (hash.as_str()[..16].to_string(), decls.len())
}

/// Top-level entry point for the `stable-surface` subcommand.
///
/// Emits a JSON-or-table report:
/// ```text
/// {
///   "schema_version": 1,
///   "workspace_root": "<abs path>",
///   "crates": [
///     { "name": "ix-math", "tier": "stable", "api_hash": "deadbeef...", "decl_count": 123 },
///     ...
///   ]
/// }
/// ```
pub fn run(format: Format, all_tiers: bool) -> Result<(), String> {
    let root = locate_workspace_root()?;
    let maturity = load_maturity(&root)?;

    let mut rows: Vec<Value> = Vec::new();
    for (name, tier) in &maturity {
        if !all_tiers && tier != "stable" {
            continue;
        }
        let (hash, count) = compute_api_hash(&root, name);
        rows.push(json!({
            "name": name,
            "tier": tier,
            "api_hash": hash,
            "decl_count": count,
        }));
    }
    rows.sort_by(|a, b| {
        let an = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let bn = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
        an.cmp(bn)
    });

    let payload = json!({
        "schema_version": 1,
        "workspace_root": root.to_string_lossy(),
        "crates": rows,
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}

/// Diff two stable-surface reports. Used by the CI guard and by tests.
///
/// Returns a structured `DiffResult` listing crates whose hash changed,
/// crates added, and crates removed — partitioned by tier (stable vs other).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffResult {
    /// Stable-tier crates whose hash changed (these FAIL CI).
    pub stable_changed: Vec<HashChange>,
    /// Non-stable crates whose hash changed (these WARN only).
    pub other_changed: Vec<HashChange>,
    /// Crates present on PR but not on base.
    pub added: Vec<String>,
    /// Crates present on base but not on PR.
    pub removed: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashChange {
    pub name: String,
    pub tier_before: String,
    pub tier_after: String,
    pub hash_before: String,
    pub hash_after: String,
}

/// Parse a JSON document produced by `ix stable-surface --format=json` into
/// a name → (tier, hash) map.
fn parse_report(doc: &Value) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    if let Some(arr) = doc.get("crates").and_then(|v| v.as_array()) {
        for item in arr {
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let tier = item
                .get("tier")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let hash = item
                .get("api_hash")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !name.is_empty() {
                out.insert(name, (tier, hash));
            }
        }
    }
    out
}

/// Compute the diff between two stable-surface reports.
///
/// `base` is the report from `main`; `head` is the report from the PR.
/// A crate is "stable_changed" if it is tier=stable in EITHER report and the
/// hash differs.
pub fn diff_reports(base: &Value, head: &Value) -> DiffResult {
    let base_map = parse_report(base);
    let head_map = parse_report(head);

    let mut stable_changed = Vec::new();
    let mut other_changed = Vec::new();
    let mut added = Vec::new();
    let mut removed = Vec::new();

    let all_names: BTreeSet<&String> = base_map.keys().chain(head_map.keys()).collect();
    for name in all_names {
        match (base_map.get(name), head_map.get(name)) {
            (None, Some(_)) => added.push(name.clone()),
            (Some(_), None) => removed.push(name.clone()),
            (Some((tb, hb)), Some((ta, ha))) => {
                if hb != ha {
                    let change = HashChange {
                        name: name.clone(),
                        tier_before: tb.clone(),
                        tier_after: ta.clone(),
                        hash_before: hb.clone(),
                        hash_after: ha.clone(),
                    };
                    if tb == "stable" || ta == "stable" {
                        stable_changed.push(change);
                    } else {
                        other_changed.push(change);
                    }
                }
            }
            (None, None) => unreachable!(),
        }
    }

    DiffResult {
        stable_changed,
        other_changed,
        added,
        removed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconciliation guard (issue #243): every workspace-member crate MUST
    /// appear as a key in `crate-maturity.toml`'s `[crates]` table. The
    /// stable-surface machinery is otherwise SILENT about members it doesn't
    /// know — this is the companion signal to `compute_api_hash`'s
    /// missing-src-dir case (a *listed* crate whose dir is gone → `("missing",
    /// 0)`); this asserts the other direction (a *member* that isn't listed).
    ///
    /// Hermetic: parses the root `Cargo.toml` and each member's own
    /// `Cargo.toml` directly (no `cargo metadata` subprocess), reads the real
    /// `[package].name` (never assumes dir basename == crate name), and
    /// respects `[workspace].exclude`.
    #[test]
    fn every_workspace_member_is_listed_in_crate_maturity() {
        let root = locate_workspace_root().expect("locate workspace root");
        let maturity = load_maturity(&root).expect("load crate-maturity.toml");

        let cargo_toml = fs::read_to_string(root.join("Cargo.toml")).expect("read root Cargo.toml");
        let parsed: toml::Value = toml::from_str(&cargo_toml).expect("parse root Cargo.toml");
        let workspace = parsed
            .get("workspace")
            .and_then(|v| v.as_table())
            .expect("[workspace] table");
        let members = workspace
            .get("members")
            .and_then(|v| v.as_array())
            .expect("[workspace] members array");
        let exclude: BTreeSet<String> = workspace
            .get("exclude")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let mut unlisted: Vec<String> = Vec::new();
        for member in members {
            let Some(member_path) = member.as_str() else {
                continue;
            };
            if exclude.contains(member_path) {
                continue;
            }
            // Read the member's OWN Cargo.toml [package].name — robust against
            // dir basename != crate name (happens to hold here, but don't rely
            // on it).
            let member_cargo = root.join(member_path).join("Cargo.toml");
            let body = fs::read_to_string(&member_cargo)
                .unwrap_or_else(|e| panic!("reading {}: {e}", member_cargo.display()));
            let member_toml: toml::Value = toml::from_str(&body)
                .unwrap_or_else(|e| panic!("parsing {}: {e}", member_cargo.display()));
            let name = member_toml
                .get("package")
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("no [package].name in {}", member_cargo.display()));
            if !maturity.contains_key(name) {
                unlisted.push(name.to_string());
            }
        }
        unlisted.sort();

        assert!(
            unlisted.is_empty(),
            "crate-maturity.toml is missing {} workspace member(s): [{}]. \
             Add each to the [crates] table with an honest tier (issue #243).",
            unlisted.len(),
            unlisted.join(", ")
        );
    }

    #[test]
    fn extract_pub_lines_basic() {
        let src = r#"
//! doc comment
// regular comment
use std::collections::HashMap;

pub fn foo() {}
pub(crate) fn hidden() {}
pub(super) fn also_hidden() {}
pub struct Bar { pub field: u32, pub(crate) inner: u32 }
pub use crate::baz::Quux;

fn private() {}
"#;
        let lines = extract_pub_lines(src);
        // Should capture: pub fn foo(), pub struct Bar..., pub use ...
        // The pub field inside the struct is on the same line as `pub struct`
        // so it's only captured once via the struct line.
        assert!(lines.iter().any(|l| l.starts_with("pub fn foo")));
        assert!(lines.iter().any(|l| l.starts_with("pub struct Bar")));
        assert!(lines.iter().any(|l| l.starts_with("pub use crate::baz")));
        assert!(!lines.iter().any(|l| l.starts_with("pub(crate) fn hidden")));
        assert!(!lines
            .iter()
            .any(|l| l.starts_with("pub(super) fn also_hidden")));
        assert!(!lines.iter().any(|l| l.starts_with("fn private")));
    }

    #[test]
    fn hash_changes_when_pub_added() {
        // Build two fake crate directories on the fly and check hashes differ.
        use std::fs;
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let crate_src = root.join("crates").join("fake-crate").join("src");
        fs::create_dir_all(&crate_src).expect("mkdir");
        fs::write(crate_src.join("lib.rs"), "pub fn alpha() {}\n").expect("write 1");
        let (h1, n1) = compute_api_hash(root, "fake-crate");
        assert_eq!(n1, 1);

        // Add a new pub fn — hash MUST change.
        fs::write(
            crate_src.join("lib.rs"),
            "pub fn alpha() {}\npub fn beta() {}\n",
        )
        .expect("write 2");
        let (h2, n2) = compute_api_hash(root, "fake-crate");
        assert_eq!(n2, 2);
        assert_ne!(h1, h2, "adding a pub fn must change the hash");

        // Add an internal-only change — hash MUST NOT change.
        fs::write(
            crate_src.join("lib.rs"),
            "pub fn alpha() {}\npub fn beta() {}\n\nfn internal() {}\n",
        )
        .expect("write 3");
        let (h3, n3) = compute_api_hash(root, "fake-crate");
        assert_eq!(n3, 2, "internal fn must not affect decl_count");
        assert_eq!(h2, h3, "internal-only changes must not change the hash");
    }

    #[test]
    fn diff_reports_flags_stable_break() {
        let base = json!({
            "crates": [
                { "name": "ix-math", "tier": "stable", "api_hash": "aaaa", "decl_count": 10 },
                { "name": "ix-evolution", "tier": "experimental", "api_hash": "bbbb", "decl_count": 5 },
            ]
        });
        let head = json!({
            "crates": [
                // Pretend ix-math's API changed.
                { "name": "ix-math", "tier": "stable", "api_hash": "cccc", "decl_count": 11 },
                // Pretend ix-evolution also changed — should be warn-only.
                { "name": "ix-evolution", "tier": "experimental", "api_hash": "dddd", "decl_count": 4 },
            ]
        });
        let d = diff_reports(&base, &head);
        assert_eq!(d.stable_changed.len(), 1);
        assert_eq!(d.stable_changed[0].name, "ix-math");
        assert_eq!(d.other_changed.len(), 1);
        assert_eq!(d.other_changed[0].name, "ix-evolution");
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
    }

    #[test]
    fn diff_reports_handles_demotion_escape_hatch() {
        // Stable break + simultaneous demotion to beta in the SAME PR is the
        // documented escape hatch. The diff still reports it as
        // stable_changed (CI will then look at the head tier to decide).
        // For this v1 we simply flag it; CI fails. Future: allow the
        // override via PR label or commit trailer.
        let base = json!({
            "crates": [
                { "name": "ix-game", "tier": "stable", "api_hash": "aa", "decl_count": 8 }
            ]
        });
        let head = json!({
            "crates": [
                { "name": "ix-game", "tier": "beta", "api_hash": "bb", "decl_count": 7 }
            ]
        });
        let d = diff_reports(&base, &head);
        assert_eq!(d.stable_changed.len(), 1);
        assert_eq!(d.stable_changed[0].tier_before, "stable");
        assert_eq!(d.stable_changed[0].tier_after, "beta");
    }
}
