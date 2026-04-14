//! Validator tests for `ToolRegistry::validate_pipeline_spec`.
//!
//! The full compile flow needs a live MCP client for sampling, so
//! those paths are exercised separately via the binary + a real
//! client. Here we pin down the validator's behaviour on hand-crafted
//! specs, which is what keeps bad LLM output from reaching
//! `ix_pipeline_run`.

use ix_agent::tools::ToolRegistry;
use serde_json::json;

#[test]
fn accepts_a_valid_single_step_spec() {
    let reg = ToolRegistry::new();
    let spec = json!({
        "steps": [
            {
                "id": "s01_stats",
                "tool": "ix_stats",
                "asset_name": "demo.stats",
                "arguments": { "data": [1.0, 2.0, 3.0] }
            }
        ]
    });
    let (errors, warnings) = reg.validate_pipeline_spec(&spec);
    assert!(errors.is_empty(), "expected no errors, got {errors:?}");
    assert!(warnings.is_empty(), "expected no warnings, got {warnings:?}");
}

#[test]
fn accepts_a_chained_two_step_spec() {
    let reg = ToolRegistry::new();
    let spec = json!({
        "steps": [
            {
                "id": "a",
                "tool": "ix_stats",
                "asset_name": "demo.a",
                "arguments": { "data": [1.0, 2.0] }
            },
            {
                "id": "b",
                "tool": "ix_fft",
                "asset_name": "demo.b",
                "depends_on": ["a"],
                "arguments": { "signal": [1.0, 2.0, 3.0, 4.0] }
            }
        ]
    });
    let (errors, _) = reg.validate_pipeline_spec(&spec);
    assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn rejects_unknown_tool() {
    let reg = ToolRegistry::new();
    let spec = json!({
        "steps": [
            {
                "id": "s01",
                "tool": "ix_does_not_exist",
                "arguments": {}
            }
        ]
    });
    let (errors, _) = reg.validate_pipeline_spec(&spec);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("unknown tool 'ix_does_not_exist'"));
}

#[test]
fn rejects_duplicate_ids() {
    let reg = ToolRegistry::new();
    let spec = json!({
        "steps": [
            { "id": "a", "tool": "ix_stats", "arguments": { "data": [1.0] } },
            { "id": "a", "tool": "ix_stats", "arguments": { "data": [2.0] } }
        ]
    });
    let (errors, _) = reg.validate_pipeline_spec(&spec);
    assert!(errors.iter().any(|e| e.contains("duplicate id 'a'")));
}

#[test]
fn rejects_depends_on_with_unknown_target() {
    let reg = ToolRegistry::new();
    let spec = json!({
        "steps": [
            {
                "id": "a",
                "tool": "ix_stats",
                "depends_on": ["ghost"],
                "arguments": { "data": [1.0] }
            }
        ]
    });
    let (errors, _) = reg.validate_pipeline_spec(&spec);
    assert!(
        errors.iter().any(|e| e.contains("unknown step 'ghost'")),
        "{errors:?}"
    );
}

#[test]
fn rejects_dependency_cycle() {
    let reg = ToolRegistry::new();
    let spec = json!({
        "steps": [
            {
                "id": "a",
                "tool": "ix_stats",
                "depends_on": ["b"],
                "arguments": { "data": [1.0] }
            },
            {
                "id": "b",
                "tool": "ix_stats",
                "depends_on": ["a"],
                "arguments": { "data": [2.0] }
            }
        ]
    });
    let (errors, _) = reg.validate_pipeline_spec(&spec);
    assert!(
        !errors.is_empty(),
        "cycle should produce at least one error"
    );
}

#[test]
fn warns_when_asset_name_missing() {
    let reg = ToolRegistry::new();
    let spec = json!({
        "steps": [
            {
                "id": "a",
                "tool": "ix_stats",
                "arguments": { "data": [1.0] }
            }
        ]
    });
    let (errors, warnings) = reg.validate_pipeline_spec(&spec);
    assert!(errors.is_empty());
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("no 'asset_name'"));
}

#[test]
fn rejects_empty_steps_array() {
    let reg = ToolRegistry::new();
    let spec = json!({ "steps": [] });
    let (errors, _) = reg.validate_pipeline_spec(&spec);
    assert!(errors.iter().any(|e| e.contains("empty")));
}

#[test]
fn rejects_missing_steps_field() {
    let reg = ToolRegistry::new();
    let spec = json!({ "note": "no steps field here" });
    let (errors, _) = reg.validate_pipeline_spec(&spec);
    assert!(errors.iter().any(|e| e.contains("missing 'steps'")));
}

#[test]
fn accepts_cost_anomaly_pipeline_shape() {
    // Smoke check against the same three-step DAG the first canonical
    // showcase uses. This is what the validator must not reject.
    let reg = ToolRegistry::new();
    let spec = json!({
        "steps": [
            {
                "id": "baseline",
                "tool": "ix_stats",
                "asset_name": "cost.baseline",
                "arguments": { "data": [1.0, 2.0, 3.0] }
            },
            {
                "id": "spectrum",
                "tool": "ix_fft",
                "asset_name": "cost.spectrum",
                "depends_on": ["baseline"],
                "arguments": { "signal": [1.0, 2.0, 3.0, 4.0] }
            },
            {
                "id": "anomalies",
                "tool": "ix_kmeans",
                "asset_name": "cost.anomalies",
                "depends_on": ["baseline", "spectrum"],
                "arguments": {
                    "data": [[1.0], [2.0], [3.0], [10.0]],
                    "k": 2,
                    "max_iter": 50
                }
            }
        ]
    });
    let (errors, _) = reg.validate_pipeline_spec(&spec);
    assert!(errors.is_empty(), "{errors:?}");
}
