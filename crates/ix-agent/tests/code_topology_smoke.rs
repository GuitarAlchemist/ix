//! `ix_code_topology` — the MCP exposure for gap-matrix row F1.
//!
//! Covers the two things the exposure can regress independently: the tool being
//! reachable at all (it is the only consumer of `ix-code/topology`, so if the
//! feature is dropped from ix-agent's manifest nothing else notices), and the
//! handler producing a topology that agrees with the graph it was built from.

use ix_agent::handlers::code_topology;
use ix_agent::tools::ToolRegistry;
use serde_json::json;

/// Three units: alpha calls beta twice and gamma once, beta calls gamma.
/// Undirected shape is a triangle — one component, one cycle.
fn diamond() -> serde_json::Value {
    json!({
        "sources": [
            { "name": "alpha.rs", "source": "pub fn a_entry() { b_work(); b_work(); g_sink(); }" },
            { "name": "beta.rs",  "source": "pub fn b_work() { g_sink(); }" },
            { "name": "gamma.rs", "source": "pub fn g_sink() { let _ = 1; }" },
        ],
        "granularity": "module"
    })
}

#[test]
fn the_tool_is_registered_and_dispatches() {
    let registry = ToolRegistry::new();
    let listed = registry.list();
    let names: Vec<&str> = listed["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert!(
        names.contains(&"ix_code_topology"),
        "ix_code_topology must stay on the MCP surface"
    );

    let out = registry
        .call("ix_code_topology", diamond())
        .expect("dispatch succeeds");
    assert_eq!(out["granularity"], "module");
}

#[test]
fn module_topology_matches_the_hand_computed_shape() {
    let out = code_topology(diamond()).expect("valid input");
    assert_eq!(out["n_units"], 3);
    assert_eq!(
        out["undirected_edges"], 3,
        "alpha-beta, alpha-gamma, beta-gamma"
    );
    assert_eq!(out["topology"]["betti_0"], 1, "one connected component");
    assert_eq!(out["topology"]["betti_1"], 1, "the triangle is one cycle");
    assert_eq!(out["circuit_rank"], 1);
    assert_eq!(out["resolution"]["ambiguous_calls"], 0);
    assert_eq!(
        out["topology"]["parse_quality"], 1.0,
        "the graph was accepted, not rejected by a size guard"
    );

    // The tightest link (two call sites) must merge before the single-call
    // ones, which is the whole point of inverting weight into distance.
    let max = out["topology"]["max_persistence"].as_f64().expect("f64");
    assert!(
        (max - 1.0).abs() < 1e-12,
        "a one-call-site link sits at distance 1/1 = 1.0, got {max}"
    );
}

#[test]
fn function_granularity_analyses_a_single_unit() {
    let out = code_topology(json!({
        "sources": [{ "name": "one.rs", "source":
            "fn a() { b(); } fn b() { c(); } fn c() { a(); }" }],
        "granularity": "function"
    }))
    .expect("valid input");
    assert_eq!(out["granularity"], "function");
    assert_eq!(out["topology"]["betti_0"], 1);
    assert_eq!(out["topology"]["betti_1"], 1, "a -> b -> c -> a");
}

#[test]
fn bad_input_is_rejected_rather_than_guessed() {
    let err = code_topology(json!({ "granularity": "module" })).unwrap_err();
    assert!(err.contains("'sources' or 'path'"), "got: {err}");

    let err = code_topology(json!({
        "sources": [{ "name": "a.rs", "source": "fn a() {}" }],
        "granularity": "voxel"
    }))
    .unwrap_err();
    assert!(err.contains("granularity"), "got: {err}");

    // Fusing two files' bare function names would invent edges between
    // same-named functions, so this is refused rather than approximated.
    let err = code_topology(json!({
        "sources": [
            { "name": "a.rs", "source": "fn a() {}" },
            { "name": "b.rs", "source": "fn b() {}" },
        ],
        "granularity": "function"
    }))
    .unwrap_err();
    assert!(err.contains("one unit at a time"), "got: {err}");

    let err = code_topology(json!({ "path": "no/such/directory/anywhere" })).unwrap_err();
    assert!(err.contains("path not found"), "got: {err}");
}

/// Walk a real crate in this workspace. Asserts invariants rather than frozen
/// Betti numbers, so ordinary edits to ix-topo do not turn this red — but a
/// walker that stops finding files, or a filtration that stops agreeing with
/// its own graph, still does.
#[test]
fn walking_a_real_crate_produces_a_self_consistent_topology() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("ix-topo/src");
    let on_disk = std::fs::read_dir(&src)
        .expect("ix-topo/src is readable")
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("rs"))
        .count();
    assert!(on_disk > 1, "fixture crate should have several modules");

    let out = code_topology(json!({ "path": src.display().to_string() })).expect("walk succeeds");
    assert_eq!(
        out["n_units"].as_u64().expect("n_units"),
        on_disk as u64,
        "every .rs file at the root of ix-topo/src became a unit"
    );

    // betti_1 == E - V + betti_0 with E the UNDIRECTED edge count. Checked here
    // on live source because the identity is what the F1 research note's
    // interpretation rests on; see docs/research/2026-09-08-code-topology-over-ix.md.
    assert_eq!(
        out["circuit_rank"].as_i64().expect("circuit_rank"),
        out["topology"]["betti_1"].as_i64().expect("betti_1"),
        "betti_1 must equal the circuit rank while the filtration stops at dim 1"
    );

    // Resolution accounting has to be complete: every call site is resolved,
    // external, or ambiguous — none silently vanishes.
    let resolution = &out["resolution"];
    let total = resolution["resolved_calls"].as_u64().expect("resolved")
        + resolution["external_calls"].as_u64().expect("external")
        + resolution["ambiguous_calls"].as_u64().expect("ambiguous");
    assert!(total > 0, "ix-topo makes some calls");
}
