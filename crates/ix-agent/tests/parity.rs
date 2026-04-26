//! 64-tool parity test — protects the MCP surface during the manual→registry
//! migration and any subsequent additions.
//!
//! Every tool name in `EXPECTED` must remain reachable through
//! `ToolRegistry::list()` regardless of whether it's sourced manually or via
//! the capability registry. The test fails if any historical tool vanishes
//! or if a new tool is added without updating this allowlist — an
//! intentional rate-limiter so every surface change is reviewed.

use ix_agent::tools::ToolRegistry;
use std::collections::HashSet;

/// The 56 MCP tools exposed by ix-agent. The first 48 are registry-backed,
/// plus ix_demo, ix_explain_algorithm, and ix_triage_session (the manual
/// ServerContext-routed surface), plus the 4 pipeline tools added during
/// the R1/R2/R7-Week-2/NL-compiler work: ix_pipeline_run, ix_pipeline_list,
/// ix_autograd_run, ix_pipeline_compile, plus the P1.1/P1.2 source
/// adapters ix_git_log + ix_cargo_deps.
const EXPECTED: &[&str] = &[
    "ix_adversarial_fgsm",
    "ix_ast_query",
    "ix_autograd_run",
    "ix_bandit",
    "ix_bloom_filter",
    "ix_cache",
    "ix_cargo_deps",
    "ix_catalog_list",
    "ix_category",
    "ix_chaos_lyapunov",
    "ix_code_analyze",
    "ix_code_catalog",
    "ix_code_smells",
    "ix_context_walk",
    "ix_demo",
    "ix_distance",
    "ix_evolution",
    "ix_explain_algorithm",
    "ix_federation_discover",
    "ix_fft",
    "ix_fractal",
    "ix_fuzzy_eval",
    "ix_ga_bridge",
    "ix_game_nash",
    "ix_git_log",
    "ix_governance_belief",
    "ix_governance_check",
    "ix_governance_graph",
    "ix_governance_graph_rescan",
    "ix_governance_persona",
    "ix_governance_policy",
    "ix_gradient_boosting",
    "ix_grammar_catalog",
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
    "ix_optick_search",
    "ix_pipeline",
    "ix_pipeline_compile",
    "ix_pipeline_list",
    "ix_pipeline_run",
    "ix_random_forest",
    "ix_rfc_catalog",
    "ix_rotation",
    "ix_search",
    "ix_sedenion",
    "ix_session_flywheel_export",
    "ix_stats",
    "ix_supervised",
    "ix_tars_bridge",
    "ix_topo",
    "ix_trace_ingest",
    "ix_triage_session",
    "ix_viterbi",
];

fn exposed_names() -> HashSet<String> {
    let reg = ToolRegistry::new();
    let list = reg.list();
    list["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().expect("tool.name is string").to_string())
        .collect()
}

#[test]
fn parity_all_64_tools_reachable() {
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
    // Sanity: 48 registry tools + ix_demo + ix_explain_algorithm +
    // ix_triage_session + ix_pipeline_run + ix_pipeline_list +
    // ix_autograd_run + ix_pipeline_compile + ix_git_log +
    // ix_cargo_deps + ix_code_catalog + ix_catalog_list +
    // ix_grammar_catalog + ix_rfc_catalog + ix_ast_query +
    // ix_optick_search + ix_code_smells = 64.
    // If this drifts, update both EXPECTED and this assertion in the
    // same commit.
    assert_eq!(EXPECTED.len(), 64);
}

#[test]
fn dispatch_action_runs_approval_middleware_on_known_tool() {
    // Path E integration test: dispatch_action runs the full
    // middleware chain (currently just ApprovalMiddleware with
    // defaults) before invoking the terminal RegistryLookupHandler.
    //
    // Uses `ix_stats` which is classified as ActionKind::Read by
    // ix-approval — should auto-approve and reach the handler.
    use ix_agent::registry_bridge;
    use ix_agent_core::{AgentAction, ReadContext};

    let cx = ReadContext::synthetic_for_legacy();
    let action = AgentAction::InvokeTool {
        tool_name: "ix_stats".to_string(),
        params: serde_json::json!({ "data": [1.0, 2.0, 3.0, 4.0, 5.0] }),
        ordinal: 0,
        target_hint: None,
    };

    let outcome = registry_bridge::dispatch_action(&cx, action)
        .expect("ix_stats should be Tier 1 Read and auto-approve");

    // The handler's value should be the stats summary — assert the
    // JSON is an object (exact structure is ix_stats' business).
    assert!(
        outcome.value.is_object(),
        "ix_stats should return a JSON object, got {:?}",
        outcome.value
    );
}

#[test]
fn dispatch_action_blocks_unknown_tool_via_approval() {
    // Unknown tools classify as ActionKind::Unknown and get promoted
    // to Tier 3 by ApprovalMiddleware. dispatch_action should return
    // an ActionError::Blocked with BlockCode::ApprovalRequired —
    // proving the full middleware chain fires on the new path.
    use ix_agent::registry_bridge;
    use ix_agent_core::event::BlockCode;
    use ix_agent_core::{ActionError, AgentAction, ReadContext};

    let cx = ReadContext::synthetic_for_legacy();
    let action = AgentAction::InvokeTool {
        tool_name: "totally_fake_tool_xyz".to_string(),
        params: serde_json::json!({}),
        ordinal: 0,
        target_hint: None,
    };

    match registry_bridge::dispatch_action(&cx, action) {
        Err(ActionError::Blocked { code, blocker, .. }) => {
            assert_eq!(code, BlockCode::ApprovalRequired);
            assert_eq!(blocker, "ix_approval");
        }
        other => {
            panic!("expected Blocked(ApprovalRequired) from approval middleware, got {other:?}")
        }
    }
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
    // After batch1 (6) + batch2 (28) + batch3 (10+1 context.walk +
    // session.flywheel_export) + prime_radiant (2) migration, all 47
    // algorithm tools are registry-backed. ix_demo is manual.
    let registry_count = ix_registry::count();
    assert_eq!(
        registry_count, 48,
        "expected 48 registry skills, got {registry_count}"
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
