//! `ix_node_catalog` + `ix_pipeline_validate` — the node catalog and the
//! offline validator for `ix_pipeline_run` specs, exercised through the same
//! `call_with_ctx` entry point `main.rs` uses for `tools/call`.

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};

fn call(name: &str, args: Value) -> Value {
    let (ctx, _rx) = ServerContext::new();
    ToolRegistry::new()
        .call_with_ctx(name, args, &ctx)
        .unwrap_or_else(|e| panic!("{name} failed: {e}"))
}

fn validate(spec: Value) -> Value {
    call("ix_pipeline_validate", spec)
}

fn error_codes(report: &Value) -> Vec<(String, Value)> {
    report["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .map(|e| (e["code"].as_str().unwrap().to_string(), e["step"].clone()))
        .collect()
}

#[test]
fn catalog_lists_every_tool_with_schema_and_tier() {
    let catalog = call("ix_node_catalog", json!({}));
    let nodes = catalog["nodes"].as_array().expect("nodes array");
    assert_eq!(catalog["count"], nodes.len());
    assert_eq!(nodes.len(), ToolRegistry::new().tool_names().count());

    let stats = nodes
        .iter()
        .find(|n| n["name"] == "ix_stats")
        .expect("ix_stats in catalog");
    assert_eq!(stats["dispatch"], "registry");
    assert_eq!(stats["input_schema"]["type"], "object");
    assert_eq!(stats["required_inputs"], json!(["data"]));
    assert_eq!(stats["approval"]["action_kind"], "read");
    assert_eq!(stats["approval"]["tier"], "tier_one");

    let cache = nodes.iter().find(|n| n["name"] == "ix_cache").unwrap();
    assert_eq!(cache["approval"]["tier"], "tier_two");

    let own = nodes
        .iter()
        .find(|n| n["name"] == "ix_node_catalog")
        .unwrap();
    assert_eq!(own["dispatch"], "manual");
    assert_eq!(own["output_schema"], Value::Null);
    assert_eq!(own["approval"]["tier"], "tier_one");
}

#[test]
fn valid_chained_pipeline_passes_with_order_and_tier() {
    let report = validate(json!({
        "steps": [
            { "id": "a", "tool": "ix_stats", "arguments": { "data": [1.0, 2.0, 3.0] } },
            {
                "id": "b",
                "tool": "ix_cache",
                "depends_on": ["a"],
                "arguments": { "operation": "set", "key": "k", "value": "$a.mean" }
            }
        ]
    }));
    assert_eq!(report["valid"], true, "{report:#}");
    assert!(
        report["warnings"].as_array().unwrap().is_empty(),
        "{report:#}"
    );
    assert_eq!(report["execution_order"], json!(["a", "b"]));
    assert_eq!(report["max_tier"], "tier_two");
    assert_eq!(report["requires_approval"], false);
    assert_eq!(report["steps"][0]["tier"], "tier_one");
}

#[test]
fn cycle_is_reported() {
    let report = validate(json!({
        "steps": [
            { "id": "a", "tool": "ix_stats", "arguments": { "data": [1.0] }, "depends_on": ["b"] },
            { "id": "b", "tool": "ix_stats", "arguments": { "data": [1.0] }, "depends_on": ["a"] }
        ]
    }));
    assert_eq!(report["valid"], false);
    assert!(
        error_codes(&report).iter().any(|(c, _)| c == "cycle"),
        "{report:#}"
    );
    assert_eq!(report["execution_order"], Value::Null);
}

#[test]
fn unknown_tool_is_reported_against_its_step() {
    let report = validate(json!({
        "steps": [{ "id": "x", "tool": "ix_does_not_exist", "arguments": {} }]
    }));
    assert_eq!(report["valid"], false);
    assert_eq!(
        error_codes(&report),
        vec![("unknown_tool".to_string(), json!("x"))]
    );
    // An unknown tool has no tier to contribute.
    assert_eq!(report["max_tier"], Value::Null);
}

#[test]
fn missing_required_input_is_reported() {
    let report = validate(json!({
        "steps": [{ "id": "s", "tool": "ix_stats", "arguments": {} }]
    }));
    assert_eq!(report["valid"], false);
    let errors = report["errors"].as_array().unwrap();
    assert_eq!(errors.len(), 1, "{report:#}");
    assert_eq!(errors[0]["code"], "missing_required_input");
    assert_eq!(errors[0]["step"], "s");
    assert!(errors[0]["message"].as_str().unwrap().contains("'data'"));
}

#[test]
fn references_to_undefined_steps_are_reported() {
    let report = validate(json!({
        "steps": [{
            "id": "s",
            "tool": "ix_stats",
            "depends_on": ["ghost"],
            "arguments": { "data": "$phantom.values" }
        }]
    }));
    let codes = error_codes(&report);
    assert_eq!(
        codes,
        vec![
            ("unknown_step_reference".to_string(), json!("s")),
            ("unknown_step_reference".to_string(), json!("s")),
        ],
        "{report:#}"
    );
}

#[test]
fn reference_to_a_step_not_upstream_is_a_warning() {
    let report = validate(json!({
        "steps": [
            { "id": "a", "tool": "ix_stats", "arguments": { "data": [1.0] } },
            { "id": "b", "tool": "ix_stats", "arguments": { "data": "$a.values" } }
        ]
    }));
    assert_eq!(report["valid"], true, "{report:#}");
    assert_eq!(report["warnings"][0]["code"], "undeclared_dependency");
    assert_eq!(report["warnings"][0]["step"], "b");
}

#[test]
fn missing_steps_is_a_structured_error_not_a_call_failure() {
    let report = validate(json!({}));
    assert_eq!(report["valid"], false);
    assert_eq!(report["errors"][0]["code"], "missing_steps");
}
