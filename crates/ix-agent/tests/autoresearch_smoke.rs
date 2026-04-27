//! MCP smoke test for `ix_autoresearch_run`.
//!
//! Verifies the handler is reachable through `ToolRegistry::call`,
//! enforces the iteration cap, accepts the documented strategies, and
//! returns a JSON outcome with the expected fields.

use ix_agent::tools::ToolRegistry;
use serde_json::json;
use tempfile::TempDir;

#[test]
fn ix_autoresearch_run_grammar_greedy_returns_outcome_shape() {
    let reg = ToolRegistry::new();
    let dir = TempDir::new().unwrap();
    let result = reg
        .call(
            "ix_autoresearch_run",
            json!({
                "target": "grammar",
                "iterations": 5,
                "strategy": "greedy",
                "seed": 1,
                "state_dir": dir.path().to_string_lossy().to_string(),
            }),
        )
        .expect("autoresearch_run should succeed");

    assert_eq!(result["target"], json!("grammar"));
    assert_eq!(result["iterations"], json!(5));
    assert!(result["run_id"].is_string());
    assert!(result["log_path"].is_string());
    // best_reward is set unless every iteration errored — Greedy on
    // GrammarTarget::default_smoke never errors, so we expect a number.
    assert!(result["best_reward"].is_number());
    // Cost ledger fields
    assert!(result["cost"]["total_elapsed_ms"].is_number());
    assert_eq!(result["cost"]["eval_failure_count"], json!(0));
}

#[test]
fn ix_autoresearch_run_rejects_iterations_above_mcp_cap() {
    let reg = ToolRegistry::new();
    let err = reg
        .call(
            "ix_autoresearch_run",
            json!({
                "target": "grammar",
                "iterations": 100_000,  // way over MCP_ITERATION_CAP = 10_000
                "strategy": "greedy",
            }),
        )
        .expect_err("expected MCP cap error");
    assert!(
        err.contains("MCP cap"),
        "error should mention MCP cap; got: {err}"
    );
}

#[test]
fn ix_autoresearch_run_rejects_unknown_strategy() {
    let reg = ToolRegistry::new();
    let err = reg
        .call(
            "ix_autoresearch_run",
            json!({
                "iterations": 5,
                "strategy": "ucb_directions",  // v1.5+ only
            }),
        )
        .expect_err("expected unknown-strategy error");
    assert!(err.contains("unknown strategy"), "got: {err}");
}

#[test]
fn ix_autoresearch_run_rejects_unknown_target() {
    let reg = ToolRegistry::new();
    let err = reg
        .call(
            "ix_autoresearch_run",
            json!({
                "target": "optick",  // Phase 5 only
                "iterations": 5,
            }),
        )
        .expect_err("expected unknown-target error");
    assert!(err.contains("unknown target"), "got: {err}");
}

#[test]
fn ix_autoresearch_run_sa_with_calibration_runs_to_completion() {
    let reg = ToolRegistry::new();
    let dir = TempDir::new().unwrap();
    let result = reg
        .call(
            "ix_autoresearch_run",
            json!({
                "iterations": 10,
                "strategy": "sa",
                // initial_temperature omitted ⇒ Ben-Ameur calibration
                "cooling_rate": 0.95,
                "seed": 99,
                "state_dir": dir.path().to_string_lossy().to_string(),
            }),
        )
        .expect("SA with auto-calibration should succeed");
    assert_eq!(result["iterations"], json!(10));
    assert!(result["best_reward"].is_number());
}

#[test]
fn ix_autoresearch_run_rejects_zero_iterations() {
    let reg = ToolRegistry::new();
    let err = reg
        .call(
            "ix_autoresearch_run",
            json!({ "iterations": 0 }),
        )
        .expect_err("expected zero-iterations error");
    assert!(err.contains("≥ 1"), "got: {err}");
}
