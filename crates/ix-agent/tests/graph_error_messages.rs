//! P0.2 regression — ix_graph error messages must name the field,
//! the offending value, and the reason. The previous "Invalid 'from'"
//! string was the single worst UX in the handler layer and cost 15
//! minutes during the adversarial refactor oracle build.
//!
//! Every test here was chosen to pin down one specific failure mode
//! the old implementation used to hide.

use ix_agent::tools::ToolRegistry;
use serde_json::json;

fn run_graph(args: serde_json::Value) -> Result<serde_json::Value, String> {
    ToolRegistry::new().call("ix_graph", args)
}

#[test]
fn accepts_integer_edges() {
    let result = run_graph(json!({
        "operation": "topological_sort",
        "n_nodes": 3,
        "edges": [[0, 1, 1.0], [1, 2, 1.0]]
    }));
    let r = result.expect("integer edges should be accepted");
    assert_eq!(r["is_dag"], true);
}

#[test]
fn accepts_integral_float_edges() {
    // Building edges from `[[f64; 3]; N]` const arrays is the common
    // way this used to fail. Integral floats (1.0, 2.0) must now
    // parse as node indices.
    let result = run_graph(json!({
        "operation": "topological_sort",
        "n_nodes": 3,
        "edges": [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0]]
    }));
    let r = result.expect("integral float edges should be accepted");
    assert_eq!(r["is_dag"], true);
}

#[test]
fn rejects_fractional_float_with_specific_message() {
    let result = run_graph(json!({
        "operation": "topological_sort",
        "n_nodes": 3,
        "edges": [[0.5, 1, 1.0]]
    }));
    let err = result.expect_err("fractional float should reject");
    assert!(
        err.contains("edge[0]") && err.contains("'from'") && err.contains("float 0.5"),
        "expected specific error naming the field + offending value, got: {err}"
    );
}

#[test]
fn rejects_negative_node_index_with_specific_message() {
    let result = run_graph(json!({
        "operation": "topological_sort",
        "n_nodes": 3,
        "edges": [[-1, 1, 1.0]]
    }));
    let err = result.expect_err("negative index should reject");
    assert!(
        err.contains("edge[0]")
            && err.contains("'from'")
            && err.contains("non-negative")
            && err.contains("-1"),
        "expected specific error naming the field + reason, got: {err}"
    );
}

#[test]
fn rejects_string_where_node_expected_with_kind_named() {
    let result = run_graph(json!({
        "operation": "topological_sort",
        "n_nodes": 3,
        "edges": [["a", 1, 1.0]]
    }));
    let err = result.expect_err("string node id should reject");
    assert!(
        err.contains("edge[0]") && err.contains("'from'") && err.contains("string"),
        "expected error naming the field + kind, got: {err}"
    );
}

#[test]
fn rejects_missing_from_slot() {
    let result = run_graph(json!({
        "operation": "topological_sort",
        "n_nodes": 3,
        "edges": [[]]
    }));
    let err = result.expect_err("empty edge should reject");
    assert!(
        err.contains("edge[0]") && err.contains("at least [from, to]"),
        "expected shape error naming the minimum, got: {err}"
    );
}

#[test]
fn rejects_out_of_range_node_index() {
    let result = run_graph(json!({
        "operation": "topological_sort",
        "n_nodes": 3,
        "edges": [[0, 99, 1.0]]
    }));
    let err = result.expect_err("out-of-range should reject");
    assert!(
        err.contains("edge[0]") && err.contains("99") && err.contains("n_nodes = 3"),
        "expected bounds error naming the value + n_nodes, got: {err}"
    );
}

#[test]
fn rejects_edge_that_is_not_an_array() {
    let result = run_graph(json!({
        "operation": "topological_sort",
        "n_nodes": 3,
        "edges": ["not an edge"]
    }));
    let err = result.expect_err("non-array edge should reject");
    assert!(
        err.contains("edge[0]") && err.contains("[from, to, weight]"),
        "expected shape error, got: {err}"
    );
}

#[test]
fn rejects_missing_n_nodes_with_helpful_message() {
    let result = run_graph(json!({
        "operation": "topological_sort",
        "edges": []
    }));
    let err = result.expect_err("missing n_nodes should reject");
    assert!(
        err.contains("n_nodes") && err.contains("positive integer"),
        "expected helpful missing-field error, got: {err}"
    );
}

#[test]
fn rejects_unknown_operation_with_enum_hint() {
    let result = run_graph(json!({
        "operation": "nonsense",
        "n_nodes": 3,
        "edges": []
    }));
    let err = result.expect_err("unknown op should reject");
    assert!(
        err.contains("unknown operation") && err.contains("pagerank"),
        "expected enum hint in error, got: {err}"
    );
}
