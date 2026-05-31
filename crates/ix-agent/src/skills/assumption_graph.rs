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
//!
//! Both are stateless reads (a full scan / log replay per call), like
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
            }
        }
    })
}

/// Build the unified assumption graph for a workspace and return its faceted
/// navigation view (counts by namespace / kind / domain + escalated claims).
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
    let view = graph.view().map_err(|e| e.to_string())?;
    serde_json::to_value(view).map_err(|e| e.to_string())
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
