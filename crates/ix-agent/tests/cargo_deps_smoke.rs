//! Smoke tests for `ix_cargo_deps` against the ix workspace itself.
//!
//! Runs the tool on the live repo state, so we can't assert absolute
//! crate counts or exact dep lists (they drift). Instead we pin
//! down invariants: ix-math is a leaf, ix-agent is the heaviest
//! crate, edges reference valid node ids, etc.

use ix_agent::tools::ToolRegistry;
use serde_json::json;
use std::collections::BTreeSet;
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

    assert!(
        n_nodes >= 20,
        "ix workspace should have 20+ crates, got {n_nodes}"
    );
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
fn denormalized_projections_align_with_nodes() {
    let result = run_cargo_deps();
    let n_nodes = result["n_nodes"].as_u64().unwrap() as usize;

    let sloc = result["sloc"].as_array().expect("sloc vector");
    let file_counts = result["file_counts"]
        .as_array()
        .expect("file_counts vector");
    let dep_counts = result["dep_counts"].as_array().expect("dep_counts vector");
    let names = result["names"].as_array().expect("names vector");
    let features = result["features"].as_array().expect("features matrix");

    assert_eq!(sloc.len(), n_nodes);
    assert_eq!(file_counts.len(), n_nodes);
    assert_eq!(dep_counts.len(), n_nodes);
    assert_eq!(names.len(), n_nodes);
    assert_eq!(features.len(), n_nodes);

    // Each feature row must be a 3-column matrix (sloc, file_count, dep_count).
    for row in features {
        let r = row.as_array().unwrap();
        assert_eq!(r.len(), 3);
    }

    // Projections must align with the nodes array element-wise.
    let nodes = result["nodes"].as_array().unwrap();
    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(
            sloc[i].as_f64().unwrap(),
            node["sloc"].as_u64().unwrap() as f64
        );
        assert_eq!(
            file_counts[i].as_f64().unwrap(),
            node["file_count"].as_u64().unwrap() as f64
        );
        assert_eq!(
            dep_counts[i].as_f64().unwrap(),
            node["dep_count"].as_u64().unwrap() as f64
        );
        assert_eq!(names[i].as_str().unwrap(), node["name"].as_str().unwrap());
    }
}

#[test]
fn sloc_vector_feeds_ix_stats_directly() {
    // End-to-end shape test: cargo_deps output must flow into ix_stats
    // without manual extraction. This is the path the live oracle uses.
    let cargo = run_cargo_deps();
    let sloc = cargo["sloc"].clone();
    let reg = ToolRegistry::new();
    let stats = reg
        .call("ix_stats", json!({ "data": sloc }))
        .expect("stats");
    assert!(stats["mean"].as_f64().unwrap() > 0.0);
    assert!(stats["max"].as_f64().unwrap() >= stats["mean"].as_f64().unwrap());
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

/// Node names and edges from `result`, as sorted sets.
fn names_and_edges(result: &serde_json::Value) -> (BTreeSet<String>, BTreeSet<(String, String)>) {
    let names: Vec<String> = result["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap().to_string())
        .collect();
    let edges = result["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let from = e[0].as_u64().unwrap() as usize;
            let to = e[1].as_u64().unwrap() as usize;
            (names[from].clone(), names[to].clone())
        })
        .collect();
    (names.into_iter().collect(), edges)
}

#[test]
fn graph_matches_cargo_metadata() {
    // Oracle: cargo's own view of the workspace. Regression for the tool
    // counting directories the workspace excludes (ix-duck-ext) or never
    // lists (ix-router-spike), and missing `[dependencies.<name>]` tables.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let out = std::process::Command::new(cargo)
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--offline",
        ])
        .current_dir(workspace_root())
        .output()
        .expect("run cargo metadata");
    assert!(out.status.success(), "cargo metadata failed");
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let packages = meta["packages"].as_array().unwrap();
    let expected_names: BTreeSet<String> = packages
        .iter()
        .map(|p| p["name"].as_str().unwrap().to_string())
        .collect();
    let expected_edges: BTreeSet<(String, String)> = packages
        .iter()
        .flat_map(|p| {
            let from = p["name"].as_str().unwrap().to_string();
            let names = &expected_names;
            p["dependencies"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d["name"].as_str().unwrap().to_string())
                .filter(move |d| names.contains(d))
                .map(move |d| (from.clone(), d))
                .collect::<Vec<_>>()
        })
        .filter(|(from, to)| from != to)
        .collect();

    let result = run_cargo_deps();
    let (names, edges) = names_and_edges(&result);
    assert_eq!(
        names, expected_names,
        "nodes must be exactly the workspace members"
    );
    assert_eq!(
        edges, expected_edges,
        "edges must match cargo's dependency list"
    );
}

#[test]
fn skips_directories_the_workspace_does_not_list() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let write = |rel: &str, body: &str| {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    };
    write(
        "Cargo.toml",
        "[workspace]\nexclude = [\"crates/c\"]\nmembers = [\n    \"crates/a\", # leaf\n    \"crates/b\",\n]\n",
    );
    write("crates/a/Cargo.toml", "[package]\nname = \"a\"\n");
    write(
        "crates/b/Cargo.toml",
        "[package]\nname = \"b\"\n\n[dependencies.a]\npath = \"../a\"\n\n[target.'cfg(unix)'.dependencies]\nc = { path = \"../c\" }\n",
    );
    write("crates/c/Cargo.toml", "[package]\nname = \"c\"\n");
    write("crates/d/Cargo.toml", "[package]\nname = \"d\"\n");

    let result = ToolRegistry::new()
        .call(
            "ix_cargo_deps",
            json!({ "workspace_root": root.display().to_string() }),
        )
        .expect("cargo_deps failed");
    let (names, edges) = names_and_edges(&result);
    assert_eq!(names, BTreeSet::from(["a".to_string(), "b".to_string()]));
    assert_eq!(edges, BTreeSet::from([("b".to_string(), "a".to_string())]));
    assert_eq!(result["non_members"], json!(["c", "d"]));
}

#[test]
fn glob_members_include_every_crate_directory() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/*\"]\n",
    )
    .unwrap();
    for name in ["x", "y"] {
        let crate_dir = root.join("crates").join(name);
        std::fs::create_dir_all(&crate_dir).unwrap();
        std::fs::write(
            crate_dir.join("Cargo.toml"),
            format!("[package]\nname = \"{name}\"\n"),
        )
        .unwrap();
    }
    let result = ToolRegistry::new()
        .call(
            "ix_cargo_deps",
            json!({ "workspace_root": root.display().to_string() }),
        )
        .expect("cargo_deps failed");
    assert_eq!(result["names"], json!(["x", "y"]));
    assert_eq!(result["non_members"], json!([]));
}
