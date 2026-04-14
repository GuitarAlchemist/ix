//! Smoke tests for `ix_cargo_deps` against the ix workspace itself.
//!
//! Runs the tool on the live repo state, so we can't assert absolute
//! crate counts or exact dep lists (they drift). Instead we pin
//! down invariants: ix-math is a leaf, ix-agent is the heaviest
//! crate, edges reference valid node ids, etc.

use ix_agent::tools::ToolRegistry;
use serde_json::json;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p
}

fn run_cargo_deps() -> serde_json::Value {
    let args = json!({ "workspace_root": workspace_root().display().to_string() });
    ToolRegistry::new()
        .call("ix_cargo_deps", args)
        .expect("cargo_deps failed")
}

#[test]
fn returns_well_formed_node_edge_structure() {
    let result = run_cargo_deps();

    let n_nodes = result["n_nodes"].as_u64().expect("n_nodes") as usize;
    let nodes = result["nodes"].as_array().expect("nodes");
    let edges = result["edges"].as_array().expect("edges");

    assert!(n_nodes >= 20, "ix workspace should have 20+ crates, got {n_nodes}");
    assert_eq!(nodes.len(), n_nodes);

    // Every node has the expected fields. We don't assert on
    // the name prefix — the workspace contains a few non-`ix-*`
    // crates (e.g. `memristive-markov`) that legitimately belong
    // under crates/.
    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(node["id"].as_u64().unwrap() as usize, i);
        let name = node["name"].as_str().unwrap();
        assert!(!name.is_empty(), "crate name must be non-empty");
        assert!(node["sloc"].as_u64().is_some());
        assert!(node["file_count"].as_u64().is_some());
        assert!(node["dep_count"].as_u64().is_some());
    }

    // Every edge references valid node ids.
    for (i, edge) in edges.iter().enumerate() {
        let arr = edge.as_array().expect("edge is array");
        assert_eq!(arr.len(), 3, "edge[{i}] must be [from, to, weight]");
        let from = arr[0].as_u64().unwrap() as usize;
        let to = arr[1].as_u64().unwrap() as usize;
        let w = arr[2].as_f64().unwrap();
        assert!(from < n_nodes, "edge[{i}].from out of range");
        assert!(to < n_nodes, "edge[{i}].to out of range");
        assert_ne!(from, to, "edge[{i}] is a self-loop");
        assert_eq!(w, 1.0, "edge[{i}].weight should be 1.0");
    }
}

#[test]
fn ix_math_is_a_leaf_with_sloc() {
    let result = run_cargo_deps();
    let nodes = result["nodes"].as_array().unwrap();
    let math = nodes
        .iter()
        .find(|n| n["name"] == "ix-math")
        .expect("ix-math must be present");
    // ix-math depends on nothing else in the workspace.
    let dep_count = math["dep_count"].as_u64().unwrap();
    assert_eq!(
        dep_count, 0,
        "ix-math should be a leaf (dep_count = 0), got {dep_count}"
    );
    // And has a nontrivial SLOC (> 1000 lines).
    assert!(math["sloc"].as_u64().unwrap() > 1000);
}

#[test]
fn ix_agent_is_the_heaviest_node() {
    let result = run_cargo_deps();
    let nodes = result["nodes"].as_array().unwrap();
    let heaviest = nodes
        .iter()
        .max_by_key(|n| n["sloc"].as_u64().unwrap_or(0))
        .unwrap();
    assert_eq!(
        heaviest["name"].as_str(),
        Some("ix-agent"),
        "ix-agent should be the largest crate"
    );
}

#[test]
fn output_is_directly_consumable_by_ix_graph() {
    let result = run_cargo_deps();
    // Use the emitted n_nodes + edges as the input to ix_graph's
    // topological_sort. This validates that the format is shape-
    // compatible without needing a manual conversion.
    let reg = ToolRegistry::new();
    let graph_result = reg
        .call(
            "ix_graph",
            json!({
                "operation": "topological_sort",
                "n_nodes": result["n_nodes"],
                "edges": result["edges"],
            }),
        )
        .expect("ix_graph call");
    assert_eq!(
        graph_result["is_dag"], true,
        "the ix workspace dep graph must be a DAG"
    );
}

#[test]
fn output_feeds_ix_graph_pagerank() {
    let result = run_cargo_deps();
    let reg = ToolRegistry::new();
    let pr = reg
        .call(
            "ix_graph",
            json!({
                "operation": "pagerank",
                "n_nodes": result["n_nodes"],
                "edges": result["edges"],
                "damping": 0.85,
                "iterations": 100,
            }),
        )
        .expect("pagerank");
    let pagerank = pr["pagerank"].as_object().expect("pagerank map");
    let n_nodes = result["n_nodes"].as_u64().unwrap() as usize;
    assert_eq!(pagerank.len(), n_nodes);
}
