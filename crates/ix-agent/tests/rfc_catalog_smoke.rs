//! Smoke tests for `ix_rfc_catalog`.
//!
//! Pins down the MCP handler wiring and the obsolescence-graph
//! round-trip — in particular that `current_standard_for("http")`
//! returns 9110 and not 2616, and that `obsolescence_chain=2616`
//! reaches 9110 via the 2014 split.

use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};

fn call(args: Value) -> Result<Value, String> {
    ToolRegistry::new().call("ix_rfc_catalog", args)
}

#[test]
fn no_filters_returns_every_entry() {
    let result = call(json!({})).expect("no-filter call");
    let total = result["counts"]["total"].as_u64().expect("total");
    let matched = result["matched"].as_u64().expect("matched");
    assert_eq!(matched, total);
    assert!(total >= 60, "expected 60+ RFC entries, got {total}");
}

#[test]
fn number_lookup_exact_match() {
    let result = call(json!({ "number": 9110 })).expect("number call");
    assert_eq!(result["matched"].as_u64(), Some(1));
    let entries = result["entries"].as_array().unwrap();
    assert_eq!(entries[0]["number"].as_u64(), Some(9110));
    assert_eq!(entries[0]["title"].as_str(), Some("HTTP Semantics"));
}

#[test]
fn topic_http_returns_both_current_and_obsoleted() {
    let result = call(json!({ "topic": "http" })).expect("topic call");
    let entries = result["entries"].as_array().unwrap();
    let numbers: Vec<u64> = entries
        .iter()
        .filter_map(|e| e["number"].as_u64())
        .collect();
    // Unfiltered http query should include both the current stack
    // (9110, 9111, 9112) and the historical chain (2616, 7230-7235).
    assert!(numbers.contains(&9110));
    assert!(numbers.contains(&2616));
}

#[test]
fn current_standard_filter_drops_obsoleted() {
    let result = call(json!({
        "topic": "http",
        "current_standard": true
    }))
    .expect("current_standard call");
    let entries = result["entries"].as_array().unwrap();
    let numbers: Vec<u64> = entries
        .iter()
        .filter_map(|e| e["number"].as_u64())
        .collect();
    assert!(
        numbers.contains(&9110),
        "current HTTP standard must include 9110; got {:?}",
        numbers
    );
    assert!(
        !numbers.contains(&2616),
        "current HTTP standard must NOT include obsoleted 2616; got {:?}",
        numbers
    );
    assert!(
        !numbers.contains(&7230),
        "current HTTP standard must NOT include obsoleted 7230"
    );
}

#[test]
fn obsolescence_chain_for_2616_reaches_9110() {
    let result = call(json!({ "obsolescence_chain": 2616 })).expect("chain call");
    assert_eq!(result["chain_for"].as_u64(), Some(2616));
    let entries = result["entries"].as_array().unwrap();
    let numbers: Vec<u64> = entries
        .iter()
        .filter_map(|e| e["number"].as_u64())
        .collect();
    // Must include the seed, the intermediates, and the current.
    assert!(numbers.contains(&2616), "chain must include seed 2616");
    assert!(
        numbers.contains(&9110),
        "chain must reach 9110 via 7230-7235"
    );
}

#[test]
fn rfc_2616_shows_obsoleted_by_fields() {
    let result = call(json!({ "number": 2616 })).expect("2616 lookup");
    let entry = &result["entries"][0];
    let obsoleted_by = entry["obsoleted_by"].as_array().unwrap();
    let numbers: Vec<u64> = obsoleted_by.iter().filter_map(|v| v.as_u64()).collect();
    // 2616 was replaced by the 2014 split.
    assert!(numbers.contains(&7230));
    assert!(numbers.contains(&7231));
}

#[test]
fn status_filter_is_exclusive() {
    let result = call(json!({ "status": "internet_standard" })).expect("status call");
    let entries = result["entries"].as_array().unwrap();
    for e in entries {
        assert_eq!(e["status"], "internet_standard");
    }
}

#[test]
fn unknown_status_returns_clear_error() {
    let err = call(json!({ "status": "nonsense" })).expect_err("should reject");
    assert!(
        err.contains("unknown status") && err.contains("internet_standard"),
        "expected enum hint in error, got: {err}"
    );
}
