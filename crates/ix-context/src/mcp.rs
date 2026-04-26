//! MCP tool wrapper for `ix_context_walk`.
//!
//! This module is a thin JSON envelope over [`crate::walk::Walker`] — it
//! parses a [`WalkRequest`] from `serde_json::Value`, dispatches through
//! the walker, and returns a [`WalkResponse`] whose `bundle` field is the
//! wire form of [`crate::model::ContextBundle`].
//!
//! # Why not register directly with ix-agent here
//!
//! The ix-agent MCP server has its own skill-registration plumbing
//! (see `ix_agent::registry_bridge`) that consumes `#[ix_skill]`-annotated
//! functions from `ix-registry`. Wiring the context walker into that
//! system is downstream work: for the MVP, `ix-context::mcp` exposes the
//! handler as a pure function so consumers can register it under whichever
//! dispatcher they prefer (ix-agent, a custom bench harness, or a direct
//! test fixture) without ix-context taking a hard dependency on ix-agent.
//!
//! The envelope schema is stable regardless of the eventual registration
//! path, so downstream wiring is a matter of picking the integration point,
//! not rewriting this file.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::index::ProjectIndex;
use crate::model::ContextBundle;
use crate::walk::{WalkBudget, WalkStrategy, Walker};

/// Request envelope for the `ix_context_walk` tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalkRequest {
    /// Fully-qualified free-function path to walk from, e.g.
    /// `"ix_math::eigen::jacobi"`.
    pub target: String,
    /// Strategy name: one of `"callers"`, `"callees"`, `"siblings"`, or
    /// `"cochange"`. The MCP JSON form uses short strings so the tool is
    /// ergonomic for Claude to call.
    pub strategy: String,
    /// Optional strategy-specific parameters — `max_depth` for
    /// callers/callees, `min_commits_shared` for cochange.
    #[serde(default)]
    pub strategy_params: StrategyParams,
    /// Optional budget — filled in with sensible defaults if omitted.
    #[serde(default)]
    pub budget: BudgetParams,
}

/// Strategy-specific parameters carried in [`WalkRequest::strategy_params`].
///
/// The fields that don't apply to the requested strategy are simply
/// ignored. This keeps the JSON surface flat instead of forcing Claude to
/// pick a variant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StrategyParams {
    /// BFS depth cap for callers/callees strategies. Default 3.
    #[serde(default = "default_max_depth")]
    pub max_depth: u8,
    /// Minimum co-change count threshold for the `cochange` strategy.
    /// Default 2.
    #[serde(default = "default_min_commits_shared")]
    pub min_commits_shared: u32,
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self {
            max_depth: default_max_depth(),
            min_commits_shared: default_min_commits_shared(),
        }
    }
}

fn default_max_depth() -> u8 {
    3
}
fn default_min_commits_shared() -> u32 {
    2
}

/// Budget envelope used by MCP consumers that don't want to specify a
/// `Duration` over the wire.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BudgetParams {
    #[serde(default = "default_max_nodes")]
    pub max_nodes: usize,
    #[serde(default = "default_max_edges")]
    pub max_edges: usize,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

impl Default for BudgetParams {
    fn default() -> Self {
        Self {
            max_nodes: default_max_nodes(),
            max_edges: default_max_edges(),
            timeout_ms: default_timeout_ms(),
        }
    }
}

// Align MCP defaults with WalkBudget::default_generous so library
// callers and MCP callers get identical behavior out of the box.
fn default_max_nodes() -> usize {
    1024
}
fn default_max_edges() -> usize {
    4096
}
fn default_timeout_ms() -> u64 {
    30_000
}

impl From<BudgetParams> for WalkBudget {
    fn from(p: BudgetParams) -> Self {
        WalkBudget {
            max_nodes: p.max_nodes,
            max_edges: p.max_edges,
            timeout: Duration::from_millis(p.timeout_ms),
        }
    }
}

/// Response envelope returned by `ix_context_walk`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalkResponse {
    pub bundle: ContextBundle,
}

/// Errors raised by the MCP wrapper. These map 1:1 to JSON-RPC error
/// objects when wired into a real MCP server.
#[derive(Debug, thiserror::Error, Serialize, Deserialize, PartialEq)]
pub enum WalkError {
    #[error("unknown strategy: {0} — expected one of callers, callees, siblings, cochange")]
    UnknownStrategy(String),
}

/// Pure handler for an `ix_context_walk` call. Reads a [`WalkRequest`],
/// dispatches through the walker, and returns a [`WalkResponse`].
///
/// Always returns a well-formed bundle. When the target is not found, the
/// walker emits an empty bundle with an explanatory `BudgetSkip` step — no
/// errors bubble up except for genuinely invalid input (e.g., an unknown
/// strategy name).
pub fn handle_walk_request(
    index: &ProjectIndex,
    req: WalkRequest,
) -> Result<WalkResponse, WalkError> {
    let strategy = match req.strategy.as_str() {
        "callers" | "callers_transitive" => WalkStrategy::CallersTransitive {
            max_depth: req.strategy_params.max_depth,
        },
        "callees" | "callees_transitive" => WalkStrategy::CalleesTransitive {
            max_depth: req.strategy_params.max_depth,
        },
        "siblings" | "module_siblings" => WalkStrategy::ModuleSiblings,
        "cochange" | "git_cochange" => WalkStrategy::GitCochange {
            min_commits_shared: req.strategy_params.min_commits_shared,
        },
        other => return Err(WalkError::UnknownStrategy(other.to_string())),
    };

    let walker = Walker::new(index);
    let bundle = walker.walk_from_free_fn(&req.target, strategy, req.budget.into());
    Ok(WalkResponse { bundle })
}

/// Convenience: dispatch a request encoded as `serde_json::Value`. Returns
/// the response as `serde_json::Value` so MCP consumers can drop it
/// straight into their result envelope.
pub fn handle_json_request(
    index: &ProjectIndex,
    req: serde_json::Value,
) -> Result<serde_json::Value, WalkError> {
    let parsed: WalkRequest = serde_json::from_value(req)
        .map_err(|e| WalkError::UnknownStrategy(format!("request parse failed: {e}")))?;
    let response = handle_walk_request(index, parsed)?;
    serde_json::to_value(response)
        .map_err(|e| WalkError::UnknownStrategy(format!("response serialize failed: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_fixture(dir: &std::path::Path, files: &[(&str, &str)]) {
        for (rel, contents) in files {
            let path = dir.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create_dir_all");
            }
            fs::write(&path, contents).expect("write fixture");
        }
    }

    fn build_index(files: &[(&str, &str)]) -> (tempfile::TempDir, ProjectIndex) {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_fixture(tmp.path(), files);
        let idx = ProjectIndex::build(tmp.path()).expect("build");
        (tmp, idx)
    }

    fn mini_fixture() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"
pub fn alpha() {}
pub fn beta() { alpha(); }
pub fn gamma() { beta(); }
"#,
            ),
        ]
    }

    // ── Envelope serde ─────────────────────────────────────────────────

    #[test]
    fn walk_request_serde_roundtrip_with_defaults() {
        let req = WalkRequest {
            target: "mini::alpha".to_string(),
            strategy: "callers".to_string(),
            strategy_params: StrategyParams::default(),
            budget: BudgetParams::default(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: WalkRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn walk_request_parses_with_omitted_optionals() {
        // Minimum viable request: just target + strategy.
        let json = r#"{"target":"mini::alpha","strategy":"callers"}"#;
        let req: WalkRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.target, "mini::alpha");
        assert_eq!(req.strategy, "callers");
        assert_eq!(req.strategy_params.max_depth, 3);
        assert_eq!(req.strategy_params.min_commits_shared, 2);
        // Defaults now match WalkBudget::default_generous.
        assert_eq!(req.budget.max_nodes, 1024);
        assert_eq!(req.budget.max_edges, 4096);
        assert_eq!(req.budget.timeout_ms, 30_000);
    }

    #[test]
    fn budget_params_convert_to_walk_budget() {
        let p = BudgetParams {
            max_nodes: 100,
            max_edges: 500,
            timeout_ms: 2_500,
        };
        let b: WalkBudget = p.into();
        assert_eq!(b.max_nodes, 100);
        assert_eq!(b.max_edges, 500);
        assert_eq!(b.timeout, Duration::from_millis(2_500));
    }

    // ── Strategy name parsing ──────────────────────────────────────────

    #[test]
    fn handle_walk_request_accepts_short_strategy_names() {
        let (_tmp, idx) = build_index(&mini_fixture());
        for short in ["callers", "callees", "siblings", "cochange"] {
            let req = WalkRequest {
                target: "mini::beta".to_string(),
                strategy: short.to_string(),
                strategy_params: StrategyParams::default(),
                budget: BudgetParams::default(),
            };
            let out = handle_walk_request(&idx, req);
            assert!(
                out.is_ok(),
                "strategy {short} should be accepted, got {out:?}"
            );
        }
    }

    #[test]
    fn handle_walk_request_accepts_long_strategy_names() {
        let (_tmp, idx) = build_index(&mini_fixture());
        for long in [
            "callers_transitive",
            "callees_transitive",
            "module_siblings",
            "git_cochange",
        ] {
            let req = WalkRequest {
                target: "mini::beta".to_string(),
                strategy: long.to_string(),
                strategy_params: StrategyParams::default(),
                budget: BudgetParams::default(),
            };
            let out = handle_walk_request(&idx, req);
            assert!(
                out.is_ok(),
                "strategy {long} should be accepted, got {out:?}"
            );
        }
    }

    #[test]
    fn handle_walk_request_rejects_unknown_strategy() {
        let (_tmp, idx) = build_index(&mini_fixture());
        let req = WalkRequest {
            target: "mini::beta".to_string(),
            strategy: "nonsense".to_string(),
            strategy_params: StrategyParams::default(),
            budget: BudgetParams::default(),
        };
        match handle_walk_request(&idx, req) {
            Err(WalkError::UnknownStrategy(name)) => assert_eq!(name, "nonsense"),
            other => panic!("expected UnknownStrategy error, got {:?}", other),
        }
    }

    // ── End-to-end dispatch ────────────────────────────────────────────

    #[test]
    fn handle_walk_request_produces_bundle_for_known_target() {
        let (_tmp, idx) = build_index(&mini_fixture());
        let req = WalkRequest {
            target: "mini::beta".to_string(),
            strategy: "callees".to_string(),
            strategy_params: StrategyParams {
                max_depth: 2,
                min_commits_shared: 1,
            },
            budget: BudgetParams::default(),
        };
        let response = handle_walk_request(&idx, req).expect("walk ok");
        assert_eq!(response.bundle.strategy, "callees_transitive");
        // beta -> alpha edge should be visible
        let has_alpha = response
            .bundle
            .edges
            .iter()
            .any(|e| matches!(&e.to, crate::model::ResolvedOrAmbiguous::Resolved { id } if id.contains("alpha")));
        assert!(
            has_alpha,
            "beta -> alpha edge missing: {:#?}",
            response.bundle.edges
        );
    }

    #[test]
    fn handle_json_request_roundtrips_through_value() {
        let (_tmp, idx) = build_index(&mini_fixture());
        let req_json = serde_json::json!({
            "target": "mini::gamma",
            "strategy": "callees",
            "strategy_params": { "max_depth": 2 }
        });
        let response_value = handle_json_request(&idx, req_json).expect("dispatch");
        // Response must deserialize back into a WalkResponse.
        let response: WalkResponse = serde_json::from_value(response_value).expect("deserialize");
        assert_eq!(response.bundle.strategy, "callees_transitive");
    }

    // ── Replay-identity smoke ──────────────────────────────────────────

    #[test]
    fn same_request_twice_produces_identical_walk_trace() {
        // The governance-instrument contract: two identical walks over
        // the same index produce bit-identical traces. This is the
        // replayability property that lets Demerzel audit an agent's
        // informational state after the fact.
        let (_tmp, idx) = build_index(&mini_fixture());
        let req1 = WalkRequest {
            target: "mini::gamma".to_string(),
            strategy: "callees".to_string(),
            strategy_params: StrategyParams::default(),
            budget: BudgetParams::default(),
        };
        let req2 = req1.clone();
        let bundle1 = handle_walk_request(&idx, req1).unwrap().bundle;
        let bundle2 = handle_walk_request(&idx, req2).unwrap().bundle;
        assert_eq!(
            bundle1.walk_trace, bundle2.walk_trace,
            "walk traces must be bit-identical across replay"
        );
        assert_eq!(bundle1.nodes, bundle2.nodes);
        assert_eq!(bundle1.edges, bundle2.edges);
    }
}
