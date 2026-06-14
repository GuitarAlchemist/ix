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
//! `governance.graph`.

use std::path::{Path, PathBuf};

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

fn assumption_query_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": "Workspace dir to scan for @ai: annotations (default: auto-detect)"
            },
            "research": {
                "type": "string",
                "description": "Optional path to a research-claims.json file to fold into the graph"
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
    let workspace = params
        .get("workspace")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(workspace_root);

    let research: Vec<ResearchClaim> = match params.get("research").and_then(|v| v.as_str()) {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("read {p}: {e}"))?;
            serde_json::from_str(&text).map_err(|e| format!("parse {p}: {e}"))?
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
                "description": "Path to belief-events.jsonl (default: state/assumptions/belief-events.jsonl)"
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

    let contents =
        std::fs::read_to_string(log_path).map_err(|e| format!("read {log_path}: {e}"))?;
    let log = BeliefLog::from_jsonl(&contents).map_err(|e| e.to_string())?;

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
                "description": "Path to the committed claims snapshot (default: state/assumptions/annotations.snapshot.json)"
            },
            "workspace": {
                "type": "string",
                "description": "Workspace dir to scan (default: auto-detect)"
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

    let workspace = params
        .get("workspace")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(workspace_root);
    let baseline_path = params
        .get("baseline")
        .and_then(|v| v.as_str())
        .unwrap_or("state/assumptions/annotations.snapshot.json");

    let text =
        std::fs::read_to_string(baseline_path).map_err(|e| format!("read {baseline_path}: {e}"))?;
    let baseline: drift::Snapshot =
        serde_json::from_str(&text).map_err(|e| format!("parse {baseline_path}: {e}"))?;

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
                "description": "Workspace dir to scan (default: auto-detect)"
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

    let workspace = params
        .get("workspace")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(workspace_root);

    let snap = drift::snapshot(&workspace).map_err(|e| e.to_string())?;
    let claims: Vec<_> = snap
        .claims
        .into_iter()
        .filter(|c| c.path == needle || c.path.starts_with(&prefix))
        .collect();

    Ok(json!({ "path": needle, "count": claims.len(), "claims": claims }))
}
