//! 48-tool parity test — protects the MCP surface during the manual→registry
//! migration.
//!
//! Every tool name in `EXPECTED` must remain reachable through
//! `ToolRegistry::list()` regardless of whether it's sourced manually or via
//! the capability registry. The test fails if any historical tool vanishes
//! during migration.

use ix_agent::tools::ToolRegistry;
use std::collections::HashSet;

/// The 48 MCP tools exposed by ix-agent (46 registry + ix_demo and
/// ix_explain_algorithm, both manual).
const EXPECTED: &[&str] = &[
    "ix_adversarial_fgsm",
    "ix_bandit",
    "ix_bloom_filter",
    "ix_cache",
    "ix_category",
    "ix_chaos_lyapunov",
    "ix_code_analyze",
    "ix_context_walk",
    "ix_demo",
    "ix_distance",
    "ix_evolution",
    "ix_explain_algorithm",
    "ix_federation_discover",
    "ix_fft",
    "ix_fractal",
    "ix_ga_bridge",
    "ix_game_nash",
    "ix_governance_belief",
    "ix_governance_check",
    "ix_governance_graph",
    "ix_governance_graph_rescan",
    "ix_governance_persona",
    "ix_governance_policy",
    "ix_gradient_boosting",
    "ix_grammar_evolve",
    "ix_grammar_search",
    "ix_grammar_weights",
    "ix_graph",
    "ix_hyperloglog",
    "ix_kmeans",
    "ix_linear_regression",
    "ix_markov",
    "ix_ml_pipeline",
    "ix_ml_predict",
    "ix_nn_forward",
    "ix_number_theory",
    "ix_optimize",
    "ix_pipeline",
    "ix_random_forest",
    "ix_rotation",
    "ix_search",
    "ix_sedenion",
    "ix_stats",
    "ix_supervised",
    "ix_tars_bridge",
    "ix_topo",
    "ix_trace_ingest",
    "ix_viterbi",
];

fn exposed_names() -> HashSet<String> {
    let reg = ToolRegistry::new();
    let list = reg.list();
    list["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| {
            t["name"]
                .as_str()
                .expect("tool.name is string")
                .to_string()
        })
        .collect()
}

#[test]
fn parity_all_44_tools_reachable() {
    let exposed = exposed_names();
    let missing: Vec<&&str> = EXPECTED.iter().filter(|n| !exposed.contains(**n)).collect();
    assert!(
        missing.is_empty(),
        "Tools vanished after migration: {:?}",
        missing
    );
    // Stricter: the tool list should have EXACTLY the 43 expected names — not
    // more (no duplicates from naming mismatches) and not fewer. If this drifts
    // past 43, a migration is producing a name that doesn't match the original.
    assert_eq!(
        exposed.len(),
        EXPECTED.len(),
        "expected exactly {} tools, got {} — names diverged from originals: extras={:?}",
        EXPECTED.len(),
        exposed.len(),
        exposed
            .iter()
            .filter(|n| !EXPECTED.contains(&n.as_str()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn parity_expected_count() {
    // Sanity: 46 registry tools + ix_demo + ix_explain_algorithm = 48.
    // Both ix_demo and ix_explain_algorithm are manually registered.
    // If this drifts, update both EXPECTED and this assertion in the same commit.
    assert_eq!(EXPECTED.len(), 48);
}

#[test]
fn loop_detector_trips_on_repeated_dispatch() {
    // Reach into the shared detector, clear it, then spam a known-good
    // registry-backed tool 11 times through the dispatcher. The 11th call
    // must return a circuit-breaker error containing "circuit breaker
    // tripped" rather than an actual skill result.
    //
    // Uses `ix_stats` (batch1) with a tiny data array so each legitimate
    // call is cheap. The detector's default is 10 calls / 5 min — the
    // 11th call exceeds threshold.
    use ix_agent::registry_bridge;
    let detector = registry_bridge::shared_loop_detector();
    detector.clear_key("ix_stats");

    let params = serde_json::json!({ "data": [1.0, 2.0, 3.0] });
    for _ in 0..10 {
        let result = registry_bridge::dispatch("ix_stats", params.clone());
        assert!(result.is_ok(), "first 10 calls should succeed: {result:?}");
    }
    // 11th call trips the circuit breaker.
    let tripped = registry_bridge::dispatch("ix_stats", params.clone());
    match tripped {
        Err(msg) => {
            assert!(
                msg.contains("circuit breaker tripped"),
                "expected circuit-breaker error, got: {msg}"
            );
            assert!(msg.contains("ix_stats"));
            assert!(msg.contains("threshold"));
        }
        Ok(value) => panic!("11th call should have been blocked, got: {value}"),
    }

    // Cleanup so other tests in the same binary are not affected.
    detector.clear_key("ix_stats");
}

#[test]
fn parity_batch1_tools_are_registry_backed() {
    // Sanity: the 6 tools migrated in Week 2 batch 1 should now be sourced
    // from the capability registry, not from the manual handler list.
    let batch1_skills = [
        "stats",
        "distance",
        "fft",
        "kmeans",
        "linear_regression",
        "governance.belief",
    ];
    for skill in batch1_skills {
        assert!(
            ix_registry::by_name(skill).is_some(),
            "batch1 skill missing from registry: {skill}"
        );
    }
}

#[test]
fn parity_batch2_tools_are_registry_backed() {
    // All 28 batch2 skill names should be discoverable via ix-registry.
    let batch2_skills = [
        "optimize",
        "markov",
        "viterbi",
        "search",
        "game.nash",
        "chaos.lyapunov",
        "adversarial.fgsm",
        "bloom_filter",
        "grammar.weights",
        "grammar.evolve",
        "grammar.search",
        "rotation",
        "number_theory",
        "fractal",
        "sedenion",
        "topo",
        "category",
        "nn.forward",
        "bandit",
        "evolution",
        "random_forest",
        "gradient_boosting",
        "supervised",
        "graph",
        "hyperloglog",
        "governance.check",
        "governance.persona",
        "governance.policy",
    ];
    for skill in batch2_skills {
        assert!(
            ix_registry::by_name(skill).is_some(),
            "batch2 skill missing from registry: {skill}"
        );
    }
}

#[test]
fn parity_all_43_registry_backed() {
    // After batch1 (6) + batch2 (28) + batch3 (9+1 context.walk) + prime_radiant (2)
    // migration, all 46 algorithm tools are registry-backed. ix_demo is manual.
    let registry_count = ix_registry::count();
    assert_eq!(
        registry_count, 46,
        "expected 46 registry skills (6 + 28 + 10 + 2), got {registry_count}"
    );
}

#[test]
fn parity_batch3_tools_are_registry_backed() {
    let batch3_skills = [
        "pipeline",
        "cache",
        "federation.discover",
        "trace.ingest",
        "ml_pipeline",
        "ml_predict",
        "code_analyze",
        "tars_bridge",
        "ga_bridge",
    ];
    for skill in batch3_skills {
        assert!(
            ix_registry::by_name(skill).is_some(),
            "batch3 skill missing from registry: {skill}"
        );
    }
}

#[test]
fn registry_backed_calls_dispatch_correctly() {
    // End-to-end: call a registry-backed tool via ToolRegistry and confirm
    // the registry dispatch path works.
    let reg = ToolRegistry::new();

    // batch1: ix_stats → stats
    let params = serde_json::json!({ "data": [1.0, 2.0, 3.0, 4.0, 5.0] });
    let result = reg.call("ix_stats", params).expect("ix_stats via registry");
    let mean = result["mean"].as_f64().expect("mean field");
    assert!((mean - 3.0).abs() < 1e-9);

    // batch2: ix_number_theory → number_theory
    let params = serde_json::json!({ "operation": "gcd", "a": 48, "b": 18 });
    let result = reg
        .call("ix_number_theory", params)
        .expect("ix_number_theory via registry");
    assert_eq!(result["gcd"].as_u64(), Some(6));
}
