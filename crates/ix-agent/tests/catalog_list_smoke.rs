//! Smoke tests for `ix_catalog_list`.
//!
//! Asserts the meta-tool surfaces every registered catalog with
//! its expected name, scope, and non-zero entry count.

use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};

fn call(args: Value) -> Result<Value, String> {
    ToolRegistry::new().call("ix_catalog_list", args)
}

#[test]
fn lists_all_three_registered_catalogs() {
    let result = call(json!({})).expect("catalog_list call");
    let total = result["total_catalogs"].as_u64().expect("total_catalogs");
    assert_eq!(total, 3, "expected 3 registered catalogs");

    let catalogs = result["catalogs"].as_array().expect("catalogs array");
    let names: Vec<&str> = catalogs.iter().filter_map(|c| c["name"].as_str()).collect();
    for required in ["code_analysis", "grammar", "rfc"] {
        assert!(
            names.contains(&required),
            "missing catalog '{required}'; got {names:?}"
        );
    }
}

#[test]
fn every_catalog_has_non_empty_scope_and_non_zero_count() {
    let result = call(json!({})).expect("call");
    let catalogs = result["catalogs"].as_array().unwrap();
    for c in catalogs {
        let name = c["name"].as_str().unwrap();
        let scope = c["scope"].as_str().unwrap();
        let count = c["entry_count"].as_u64().unwrap();
        assert!(!scope.is_empty(), "{name} has empty scope");
        assert!(count > 0, "{name} has zero entries");
        // Counts are catalog-specific; sanity-check totals.
        assert!(c["counts"]["total"].as_u64().unwrap() > 0);
    }
}

#[test]
fn code_analysis_catalog_has_ml_framework_category_count() {
    let result = call(json!({})).expect("call");
    let code = result["catalogs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "code_analysis")
        .expect("code_analysis catalog");
    // Proves the counts are catalog-specific, not a uniform shape —
    // code_analysis should surface per-category breakdowns.
    assert!(code["counts"]["ml_framework"].as_u64().unwrap() > 0);
    assert!(code["counts"]["formal_verification"].as_u64().unwrap() > 0);
}

#[test]
fn rfc_catalog_counts_report_obsolescence_presence() {
    let result = call(json!({})).expect("call");
    let rfc = result["catalogs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "rfc")
        .expect("rfc catalog");
    assert!(
        rfc["counts"]["obsoleted"].as_u64().unwrap() > 0,
        "rfc catalog should have obsoleted entries in its counts"
    );
}
