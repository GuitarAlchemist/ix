//! Smoke tests for the three `ix_grothendieck_*` MCP tools.
//!
//! Verifies end-to-end JSON shapes through `ToolRegistry::call` so a regression
//! in either the handler or the schema would fail here, not just at the unit
//! level inside `ix-bracelet`.

use ix_agent::tools::ToolRegistry;
use serde_json::json;

#[test]
fn ix_grothendieck_delta_major_to_augmented_triad() {
    let reg = ToolRegistry::new();
    let result = reg
        .call(
            "ix_grothendieck_delta",
            json!({
                "source": [0, 4, 7], // C major triad
                "target": [0, 4, 8], // C augmented triad
            }),
        )
        .expect("delta tool should succeed");

    assert_eq!(
        result["delta"],
        json!([0, 0, -1, 2, -1, 0]),
        "expected canonical maj→aug delta"
    );
    assert_eq!(result["l1_norm"], json!(4));
    assert_eq!(result["is_zero"], json!(false));
    assert_eq!(result["source_icv"], json!([0, 0, 1, 1, 1, 0]));
    assert_eq!(result["target_icv"], json!([0, 0, 0, 3, 0, 0]));
}

#[test]
fn ix_grothendieck_delta_zero_for_same_orbit() {
    let reg = ToolRegistry::new();
    // C major triad and D major triad are in the same D₁₂ orbit → zero delta
    let result = reg
        .call(
            "ix_grothendieck_delta",
            json!({
                "source": [0, 4, 7],
                "target": [2, 6, 9],
            }),
        )
        .expect("delta tool should succeed");
    assert_eq!(result["is_zero"], json!(true));
    assert_eq!(result["l1_norm"], json!(0));
}

#[test]
fn ix_grothendieck_nearby_zero_budget_returns_source_orbit() {
    let reg = ToolRegistry::new();
    let result = reg
        .call(
            "ix_grothendieck_nearby",
            json!({
                "source": [0, 4, 7],
                "max_l1": 0,
            }),
        )
        .expect("nearby tool should succeed");

    let count = result["count"].as_u64().expect("count is integer");
    assert!(count >= 1, "should at least return the source itself");
    let results = result["results"].as_array().expect("results array");
    // Every result at cost 0 must have an all-zero delta
    for r in results {
        assert_eq!(r["cost"], json!(0));
        assert_eq!(r["delta"], json!([0, 0, 0, 0, 0, 0]));
    }
}

#[test]
fn ix_grothendieck_nearby_respects_limit() {
    let reg = ToolRegistry::new();
    let result = reg
        .call(
            "ix_grothendieck_nearby",
            json!({
                "source": [0, 4, 7],
                "max_l1": 6,
                "limit": 5,
            }),
        )
        .expect("nearby tool should succeed");
    assert!(result["count"].as_u64().unwrap() <= 5);
    assert!(result["results"].as_array().unwrap().len() <= 5);
}

#[test]
fn ix_grothendieck_path_finds_route_between_triads() {
    let reg = ToolRegistry::new();
    let result = reg
        .call(
            "ix_grothendieck_path",
            json!({
                "source": [0, 4, 7], // C major triad
                "target": [0, 3, 7], // C minor triad
                "max_steps": 5,
            }),
        )
        .expect("path tool should succeed");

    assert_eq!(result["found"], json!(true));
    let path = result["path"].as_array().expect("path array");
    assert!(!path.is_empty());
    // Endpoints must match the requested source/target
    assert_eq!(path.first().unwrap(), &json!([0, 4, 7]));
    assert_eq!(path.last().unwrap(), &json!([0, 3, 7]));
}

#[test]
fn ix_grothendieck_path_empty_when_cardinality_mismatch() {
    let reg = ToolRegistry::new();
    let result = reg
        .call(
            "ix_grothendieck_path",
            json!({
                "source": [0, 4, 7],
                "target": [0, 4, 7, 11],
            }),
        )
        .expect("path tool should succeed");
    assert_eq!(result["found"], json!(false));
    assert_eq!(result["path"].as_array().unwrap().len(), 0);
}

#[test]
fn ix_grothendieck_delta_rejects_non_array_source() {
    let reg = ToolRegistry::new();
    let err = reg
        .call(
            "ix_grothendieck_delta",
            json!({
                "source": "not-an-array",
                "target": [0, 4, 7],
            }),
        )
        .expect_err("expected an error for non-array source");
    assert!(err.contains("source"), "error should mention 'source': {err}");
}
