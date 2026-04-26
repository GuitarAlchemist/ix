//! Smoke tests for `ix_code_catalog` against the live catalog.
//!
//! These tests do NOT re-verify the catalog's correctness (the
//! underlying `ix_code::catalog` module's unit tests already pin down
//! shape, counts, and query behaviour). They verify the MCP handler
//! shape: that filters combine, errors are actionable, and the
//! returned JSON has the fields the manual documents.

use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};

fn call(args: Value) -> Result<Value, String> {
    ToolRegistry::new().call("ix_code_catalog", args)
}

#[test]
fn no_filters_returns_every_tool() {
    let result = call(json!({})).expect("no-filter call");
    let counts = result["counts"].as_object().expect("counts present");
    let total = counts["total"].as_u64().expect("total");
    assert!(total >= 30, "expected 30+ total tools, got {total}");
    let matched = result["matched"].as_u64().expect("matched");
    assert_eq!(matched, total, "no filter means every tool matches");
}

#[test]
fn language_filter_narrows_to_rust_suite() {
    let result = call(json!({ "language": "rust" })).expect("rust call");
    let tools = result["tools"].as_array().expect("tools array");
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    for required in ["Kani", "Verus", "Creusot", "Miri", "Loom", "rustdoc"] {
        assert!(
            names.contains(&required),
            "rust query missing required tool {required}; got {names:?}"
        );
    }
}

#[test]
fn category_filter_is_exclusive() {
    let result = call(json!({ "category": "formal_verification" })).expect("formal call");
    let tools = result["tools"].as_array().unwrap();
    for t in tools {
        assert_eq!(
            t["category"], "formal_verification",
            "category filter should be exclusive"
        );
    }
    // Kani, Verus, Coq, Lean, Z3 should all land in this bucket.
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    for required in ["Kani", "Verus", "Coq", "Lean", "Z3"] {
        assert!(
            names.contains(&required),
            "formal_verification missing {required}"
        );
    }
}

#[test]
fn combined_filter_language_and_category() {
    let result = call(json!({
        "language": "rust",
        "category": "formal_verification"
    }))
    .expect("combined call");
    let tools = result["tools"].as_array().unwrap();
    assert!(!tools.is_empty());
    for t in tools {
        assert_eq!(t["category"], "formal_verification");
        let langs: Vec<&str> = t["languages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            langs.contains(&"rust") || langs.contains(&"language-agnostic"),
            "expected rust or language-agnostic, got {langs:?}"
        );
    }
}

#[test]
fn technique_substring_search_works() {
    let result = call(json!({ "technique": "cyclomatic" })).expect("technique call");
    let tools = result["tools"].as_array().unwrap();
    assert!(!tools.is_empty());
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    // Radon, gocyclo, and ix_code_analyze all mention cyclomatic in
    // their technique field.
    assert!(names.contains(&"Radon") || names.contains(&"gocyclo"));
}

#[test]
fn invalid_category_returns_helpful_error() {
    let err = call(json!({ "category": "nonsense" })).expect_err("should reject");
    assert!(
        err.contains("unknown category") && err.contains("static_analysis"),
        "expected enum hint in error, got: {err}"
    );
}

#[test]
fn counts_per_category_sum_to_total() {
    let result = call(json!({})).expect("no-filter call");
    let c = result["counts"].as_object().unwrap();
    let total = c["total"].as_u64().unwrap();
    let sum: u64 = c["static_analysis"].as_u64().unwrap()
        + c["formal_verification"].as_u64().unwrap()
        + c["safety_memory"].as_u64().unwrap()
        + c["statistical_analysis"].as_u64().unwrap()
        + c["documentation"].as_u64().unwrap()
        + c["numeric_library"].as_u64().unwrap()
        + c["ml_framework"].as_u64().unwrap()
        + c["fuzzing"].as_u64().unwrap()
        + c["supply_chain"].as_u64().unwrap();
    assert_eq!(sum, total, "category counts must sum to total");
}

#[test]
fn fuzzing_and_supply_chain_categories_are_live() {
    let fuzzing = call(json!({ "category": "fuzzing" })).expect("fuzzing call");
    let names: Vec<String> = fuzzing["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str().map(String::from))
        .collect();
    assert!(names.iter().any(|n| n == "proptest"));
    assert!(names.iter().any(|n| n == "cargo-fuzz"));

    let sc = call(json!({ "category": "supply_chain" })).expect("supply_chain call");
    let sc_names: Vec<String> = sc["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str().map(String::from))
        .collect();
    assert!(sc_names.iter().any(|n| n == "cargo-audit"));
    assert!(sc_names.iter().any(|n| n == "Trivy"));
}

#[test]
fn ml_framework_category_surfaces_rust_ml_stack() {
    let result = call(json!({ "category": "ml_framework" })).expect("ml_framework call");
    let tools = result["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    for required in ["Burn", "Candle", "Linfa", "SmartCore", "Tract"] {
        assert!(
            names.contains(&required),
            "ml_framework missing {required}; got {names:?}"
        );
    }
}
