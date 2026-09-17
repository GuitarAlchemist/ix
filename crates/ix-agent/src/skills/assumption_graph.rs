//! Assumption-graph MCP skills — the agent-facing navigation surface for the
//! temporal assumption graph (`crates/ix-assumption-graph`).
//!
//! - `assumption.query` (→ `ix_assumption_query`): scan the workspace's `@ai:`
//!   annotations (optionally folding in a research-claims file), fuse, and
//!   return the faceted view — counts by namespace / kind / domain, plus the
//!   escalated (Contradictory) claims. This is the navigable structure a UI
//!   (Prime Radiant, a dashboard) or an agent reads.
//! - `assumption.belief_at` (→ `ix_assumption_belief_at`): reconstruct the
//!   belief state at a point in time from a `belief-events.jsonl` log.
//! - `assumption.drift` (→ `ix_assumption_drift`): compare current claims to a
//!   committed snapshot — the agent-facing counterpart of the drift CLI/CI gate.
//! - `assumption.claims` (→ `ix_assumption_claims`): the `@ai:` claims anchored
//!   under a file/dir prefix (per-file drill-down), so an agent sees the
//!   invariants in code it is about to edit.
//!
//! All are stateless reads (a full scan / log replay per call), like
//! `governance.graph`. They run auto-approved (Tier 1), so every path a caller
//! names is confined to the workspace root by [`confine`] before it is read or
//! walked.

use std::path::{Component, Path, PathBuf};

use chrono::{DateTime, Utc};
use ix_assumption_graph::{AssumptionGraph, BeliefLog, ResearchClaim};
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

fn workspace_root() -> PathBuf {
    if let Ok(root) = std::env::var("IX_ROOT") {
        return PathBuf::from(root);
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return Path::new(&manifest).join("../..");
    }
    PathBuf::from(".")
}

/// Resolve a caller-supplied path against `root` and refuse anything that is not
/// an existing path inside it.
///
/// `..` is rejected lexically; everything else (absolute paths elsewhere,
/// symlinks or junctions leading out) is caught by canonicalizing and requiring
/// the result to sit under the canonical root. A missing path and a path outside
/// the root get the same message, so the error does not reveal whether a file
/// exists elsewhere on disk. Confinement also bounds a `workspace` walk to the
/// tree the default already scans. Returns the joined (non-canonical) path, like
/// `ix_ixql::FsHost`, so relative paths reported by the walker stay readable.
fn confine(root: &Path, param: &str, raw: &str) -> Result<PathBuf, String> {
    if Path::new(raw)
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(format!("`{param}`: `..` is not allowed in {raw}"));
    }
    let canonical_root = root
        .canonicalize()
        .map_err(|_| format!("`{param}`: the workspace root is not accessible"))?;
    let full = root.join(raw);
    match full.canonicalize() {
        Ok(resolved) if resolved.starts_with(&canonical_root) => Ok(full),
        _ => Err(format!(
            "`{param}`: {raw} is not an existing path inside the workspace root"
        )),
    }
}

/// serde_json's message can quote the offending value, which would echo file
/// contents back to the caller; report only where parsing failed.
fn parse_error(param: &str, raw: &str, e: &serde_json::Error) -> String {
    format!(
        "`{param}`: {raw} has the wrong shape ({:?} error at line {}, column {})",
        e.classify(),
        e.line(),
        e.column()
    )
}

/// The `workspace` parameter confined to the root, or the root itself.
fn workspace_param(root: &Path, params: &Value) -> Result<PathBuf, String> {
    match params.get("workspace").and_then(|v| v.as_str()) {
        Some(w) => confine(root, "workspace", w),
        None => Ok(root.to_path_buf()),
    }
}

fn assumption_query_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": "Workspace dir to scan for @ai: annotations; must lie inside the workspace root (default: the root)"
            },
            "research": {
                "type": "string",
                "description": "Optional path, inside the workspace root, to a research-claims.json file to fold into the graph"
            },
            "format": {
                "type": "string",
                "enum": ["view", "prime-radiant"],
                "description": "`view` (default) = faceted navigation; `prime-radiant` = node+edge graph payload for the Prime Radiant 3D renderer"
            }
        }
    })
}

/// Build the unified assumption graph for a workspace and return either its
/// faceted navigation view (default) or a Prime-Radiant-compatible node+edge
/// graph payload (`format = "prime-radiant"`).
#[ix_skill(
    domain = "assumption",
    name = "assumption.query",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::assumption_graph::assumption_query_schema"
)]
pub fn assumption_query(params: Value) -> Result<Value, String> {
    let root = workspace_root();
    let workspace = workspace_param(&root, &params)?;

    let research: Vec<ResearchClaim> = match params.get("research").and_then(|v| v.as_str()) {
        Some(p) => {
            let path = confine(&root, "research", p)?;
            let text = std::fs::read_to_string(path).map_err(|e| format!("read {p}: {e}"))?;
            serde_json::from_str(&text).map_err(|e| parse_error("research", p, &e))?
        }
        None => Vec::new(),
    };

    let graph = AssumptionGraph::from_workspace_with_research(&workspace, research)
        .map_err(|e| e.to_string())?;

    match params.get("format").and_then(|v| v.as_str()) {
        Some("prime-radiant") => Ok(graph.prime_radiant_graph()),
        _ => {
            let view = graph.view().map_err(|e| e.to_string())?;
            serde_json::to_value(view).map_err(|e| e.to_string())
        }
    }
}

fn assumption_belief_at_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "log": {
                "type": "string",
                "description": "Path to belief-events.jsonl, relative to and inside the workspace root (default: state/assumptions/belief-events.jsonl)"
            },
            "at": {
                "type": "string",
                "description": "RFC3339 timestamp; default = now"
            }
        }
    })
}

/// Reconstruct the belief state at a point in belief-time from a belief-event
/// log: `{ at, beliefs: { <claim-id>: { truth_value, opinion, at } } }`.
#[ix_skill(
    domain = "assumption",
    name = "assumption.belief_at",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::assumption_graph::assumption_belief_at_schema"
)]
pub fn assumption_belief_at(params: Value) -> Result<Value, String> {
    let log_path = params
        .get("log")
        .and_then(|v| v.as_str())
        .unwrap_or("state/assumptions/belief-events.jsonl");

    let path = confine(&workspace_root(), "log", log_path)?;
    let contents = std::fs::read_to_string(path).map_err(|e| format!("read {log_path}: {e}"))?;
    let log = BeliefLog::from_jsonl(&contents).map_err(|e| parse_error("log", log_path, &e))?;

    let at = match params.get("at").and_then(|v| v.as_str()) {
        Some(ts) => DateTime::parse_from_rfc3339(ts)
            .map_err(|e| format!("invalid `at` timestamp: {e}"))?
            .with_timezone(&Utc),
        None => Utc::now(),
    };

    let beliefs = serde_json::to_value(log.belief_at(at)).map_err(|e| e.to_string())?;
    Ok(json!({ "at": at.to_rfc3339(), "beliefs": beliefs }))
}

fn assumption_drift_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "baseline": {
                "type": "string",
                "description": "Path to the committed claims snapshot, relative to and inside the workspace root (default: state/assumptions/annotations.snapshot.json)"
            },
            "workspace": {
                "type": "string",
                "description": "Workspace dir to scan; must lie inside the workspace root (default: the root)"
            }
        }
    })
}

/// Compare the workspace's current `@ai:` claims against a committed baseline
/// snapshot. `clean` is false when any claim may now be lying — `span_drifted`
/// (the annotated code changed) or `broken_bindings` (a cited test vanished);
/// `moved` / `added` / `removed` / `verdict_changed` are informational. The
/// agent-facing counterpart of the `ix-assumption-graph-drift --check` CLI/CI
/// gate, so an agent can check its own edits before committing.
#[ix_skill(
    domain = "assumption",
    name = "assumption.drift",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::assumption_graph::assumption_drift_schema"
)]
pub fn assumption_drift(params: Value) -> Result<Value, String> {
    use ix_assumption_graph::drift;

    let root = workspace_root();
    let workspace = workspace_param(&root, &params)?;
    let baseline_path = params
        .get("baseline")
        .and_then(|v| v.as_str())
        .unwrap_or("state/assumptions/annotations.snapshot.json");

    let path = confine(&root, "baseline", baseline_path)?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {baseline_path}: {e}"))?;
    let baseline: drift::Snapshot =
        serde_json::from_str(&text).map_err(|e| parse_error("baseline", baseline_path, &e))?;

    let current = drift::snapshot(&workspace).map_err(|e| e.to_string())?;
    let mut report = drift::diff(&baseline, &current);
    drift::verify_bindings(&workspace, &current, &mut report);

    Ok(json!({
        "clean": report.is_clean(),
        "claims": current.claims.len(),
        "report": serde_json::to_value(&report).map_err(|e| e.to_string())?,
    }))
}

fn assumption_claims_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File or directory prefix (e.g. crates/ix-optick/src/lib.rs or crates/ix-fuzzy) whose @ai: claims to return"
            },
            "workspace": {
                "type": "string",
                "description": "Workspace dir to scan; must lie inside the workspace root (default: the root)"
            }
        },
        "required": ["path"]
    })
}

/// Return the `@ai:` claims anchored under a file or directory prefix — the
/// per-file drill-down (claim, verdict, kind, line, evidence) that
/// `assumption.query`'s crate-level facets don't expose. Lets an agent see the
/// invariants/assumptions living in the code it is about to edit.
#[ix_skill(
    domain = "assumption",
    name = "assumption.claims",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::assumption_graph::assumption_claims_schema"
)]
pub fn assumption_claims(params: Value) -> Result<Value, String> {
    use ix_assumption_graph::drift;

    let path = params
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or("`path` is required")?;
    let needle = path.replace('\\', "/");
    let prefix = format!("{}/", needle.trim_end_matches('/'));

    let workspace = workspace_param(&workspace_root(), &params)?;

    let snap = drift::snapshot(&workspace).map_err(|e| e.to_string())?;
    let claims: Vec<_> = snap
        .claims
        .into_iter()
        .filter(|c| c.path == needle || c.path.starts_with(&prefix))
        .collect();

    Ok(json!({ "path": needle, "count": claims.len(), "claims": claims }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confine_accepts_existing_paths_inside_the_root() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("state")).unwrap();
        std::fs::write(root.path().join("state/log.jsonl"), "").unwrap();

        assert!(confine(root.path(), "log", "state/log.jsonl").is_ok());
        assert!(confine(root.path(), "workspace", "state").is_ok());
        let absolute = root.path().join("state/log.jsonl");
        assert!(confine(root.path(), "log", absolute.to_str().unwrap()).is_ok());
    }

    #[test]
    fn confine_rejects_parent_dir_even_when_it_lands_inside() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("a")).unwrap();
        let err = confine(root.path(), "workspace", "a/../a").unwrap_err();
        assert!(err.contains("`..` is not allowed"), "{err}");
    }

    #[test]
    fn confine_gives_one_message_for_outside_and_missing() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let secret = elsewhere.path().join("id_ed25519");
        std::fs::write(&secret, "PRIVATE").unwrap();
        let missing = elsewhere.path().join("nope");

        let existing = secret.to_str().unwrap();
        let absent = missing.to_str().unwrap();
        let err_existing = confine(root.path(), "log", existing).unwrap_err();
        let err_absent = confine(root.path(), "log", absent).unwrap_err();
        assert_eq!(
            err_existing.replace(existing, "<p>"),
            err_absent.replace(absent, "<p>"),
            "the error must not reveal whether a file outside the root exists"
        );
        assert!(err_existing.contains("not an existing path inside the workspace root"));
    }

    #[test]
    fn confine_rejects_a_symlink_that_leads_out_of_the_root() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::fs::write(elsewhere.path().join("secret.json"), "{}").unwrap();
        let link = root.path().join("link");
        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(elsewhere.path(), &link);
        // Symlinks need Developer Mode or elevation on Windows; a directory
        // junction does not, and is the same escape.
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_dir(elsewhere.path(), &link).or_else(|_| {
            std::process::Command::new("cmd")
                .arg("/C")
                .arg("mklink")
                .arg("/J")
                .arg(&link)
                .arg(elsewhere.path())
                .output()
                .and_then(|o| {
                    if o.status.success() {
                        Ok(())
                    } else {
                        Err(std::io::Error::other("mklink /J failed"))
                    }
                })
        });
        // A symlink on Unix and a junction on Windows need no privileges, so a
        // failure here is a broken test environment, not a reason to skip.
        made.expect("create a symlink or junction");
        assert!(
            link.join("secret.json").exists(),
            "precondition: link resolves"
        );
        let err = confine(root.path(), "research", "link/secret.json").unwrap_err();
        assert!(
            err.contains("not an existing path inside the workspace root"),
            "{err}"
        );
    }

    #[test]
    fn parse_errors_do_not_echo_file_contents() {
        let e = serde_json::from_str::<Vec<ResearchClaim>>(r#"["SECRET-VALUE"]"#).unwrap_err();
        assert!(
            e.to_string().contains("SECRET-VALUE"),
            "precondition: serde echoes it"
        );
        let msg = parse_error("research", "claims.json", &e);
        assert!(!msg.contains("SECRET-VALUE"), "{msg}");
        assert!(msg.contains("line 1"), "{msg}");
    }
}
