//! 73-tool parity test — protects the MCP surface during the manual→registry
//! migration and any subsequent additions.
//!
//! Every tool name in `EXPECTED` must remain reachable through
//! `ToolRegistry::list()` regardless of whether it's sourced manually or via
//! the capability registry. The test fails if any historical tool vanishes
//! or if a new tool is added without updating this allowlist — an
//! intentional rate-limiter so every surface change is reviewed.
//!
//! `EXPECTED` is cross-checked against `state/registry/skills.snapshot.json`,
//! the generated inventory oracle (ix#185). The hand-typed counts that used to
//! live here are gone: drift now reports *which* capability moved, and
//! `ix doctor --write` regenerates the snapshot.

use ix_agent::tools::ToolRegistry;
use std::collections::{BTreeSet, HashSet};

/// The 73 MCP tools exposed by ix-agent. The first 48 are registry-backed,
/// plus ix_demo, ix_explain_algorithm, and ix_triage_session (the manual
/// ServerContext-routed surface), plus the 4 pipeline tools added during
/// the R1/R2/R7-Week-2/NL-compiler work: ix_pipeline_run, ix_pipeline_list,
/// ix_autograd_run, ix_pipeline_compile, plus the P1.1/P1.2/P1.3 source
/// adapters ix_git_log + ix_cargo_deps + ix_git_churn, plus the 3
/// ix_grothendieck_* PC-set algebra tools backed by ix-bracelet, plus
/// ix_petri_analyze (Petri-net deadlock/boundedness/liveness, ix-petri).
const EXPECTED: &[&str] = &[
    "ix_adversarial_fgsm",
    "ix_annotations_scan",
    "ix_assumption_belief_at",
    "ix_assumption_claims",
    "ix_assumption_drift",
    "ix_assumption_query",
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
    "ix_git_churn",
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
    "ix_autoresearch_run",
    "ix_grammar_search",
    "ix_grammar_weights",
    "ix_graph",
    "ix_grothendieck_delta",
    "ix_grothendieck_nearby",
    "ix_grothendieck_path",
    "ix_hyperloglog",
    "ix_dbscan",
    "ix_eigen",
    "ix_feature_importances",
    "ix_svd",
    "ix_gmm",
    "ix_wavelet_denoise",
    "ix_fir_filter",
    "ix_spectrogram",
    "ix_autocorrelation",
    "ix_analyze_reference",
    "ix_spectral_distance",
    "ix_kmeans",
    "ix_linear_regression",
    "ix_pca",
    "ix_silhouette",
    "ix_markov",
    "ix_mesh_correlate",
    "ix_ml_pipeline",
    "ix_ml_predict",
    "ix_nl_to_pipeline",
    "ix_nn_forward",
    "ix_number_theory",
    "ix_optimize",
    "ix_optick_search",
    "ix_petri_analyze",
    "ix_pipeline",
    "ix_pipeline_compile",
    "ix_pipeline_list",
    "ix_pipeline_run",
    "ix_quality_gate_history",
    "ix_random_forest",
    "ix_rfc_catalog",
    "ix_rotation",
    "ix_search",
    "ix_sedenion",
    "ix_sentrux_annotate",
    "ix_session_flywheel_export",
    "ix_stats",
    "ix_supervised",
    "ix_tars_bridge",
    "ix_thinker_hits",
    "ix_topo",
    "ix_trace_ingest",
    "ix_tsne",
    "ix_triage_session",
    "ix_viterbi",
    "ix_voicings_payload",
];

/// `EXPECTED` plus any feature-gated tools present in this build. The `maintain-gate`
/// feature adds `ix_maintain_gate` (pulls bundled DuckDB; off in the default/CI build), so
/// both the default surface (93) and the `--features maintain-gate` surface (94) stay green.
fn expected() -> Vec<&'static str> {
    // `mut` is only used when the feature adds a tool; allow the default-build no-op.
    #[allow(unused_mut)]
    let mut v = EXPECTED.to_vec();
    #[cfg(feature = "maintain-gate")]
    v.push("ix_maintain_gate");
    v
}

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
    let exp = expected();
    let missing: Vec<&&str> = exp.iter().filter(|n| !exposed.contains(**n)).collect();
    assert!(
        missing.is_empty(),
        "Tools vanished after migration: {:?}",
        missing
    );
    // Stricter: the tool list should have EXACTLY the expected names — not more (no
    // duplicates from naming mismatches) and not fewer. `expected()` includes any
    // feature-gated tools present in this build, so both surfaces stay exact.
    assert_eq!(
        exposed.len(),
        exp.len(),
        "expected exactly {} tools, got {} — names diverged from originals: extras={:?}",
        exp.len(),
        exposed.len(),
        exposed
            .iter()
            .filter(|n| !exp.contains(&n.as_str()))
            .collect::<Vec<_>>()
    );
}

/// Load the generated inventory snapshot — the oracle that replaced this
/// file's hand-typed counts (ix#185).
///
/// `assert_eq!(EXPECTED.len(), 94)` used to live here. It carried a 40-line
/// running tally in a comment, and it is the oracle that drifted and turned
/// `main` red when `mesh_correlate` landed. The snapshot is regenerated with
/// `cargo run -p ix-skill --bin ix -- doctor --write`, so adding a tool
/// produces a reviewable diff of *names* instead of a number nobody can
/// sanity-check.
fn snapshot_names(section: &str) -> BTreeSet<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../state/registry/skills.snapshot.json");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "reading {}: {e}\n  \
             regenerate it with `cargo run -p ix-skill --bin ix -- doctor --write`",
            path.display()
        )
    });
    let doc: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
    doc[section]["names"]
        .as_array()
        .unwrap_or_else(|| panic!("{}: {section}.names is not an array", path.display()))
        .iter()
        .map(|v| {
            v.as_str()
                .expect("snapshot names are strings")
                .to_string()
        })
        .collect()
}

/// Assert two name sets match, reporting the difference in both directions.
fn assert_same_names(live: &BTreeSet<String>, snapshot: &BTreeSet<String>, what: &str) {
    let only_live: Vec<&String> = live.difference(snapshot).collect();
    let only_snapshot: Vec<&String> = snapshot.difference(live).collect();
    assert!(
        only_live.is_empty() && only_snapshot.is_empty(),
        "{what} drifted from state/registry/skills.snapshot.json\n  \
         in this build but not the snapshot: {only_live:?}\n  \
         in the snapshot but not this build: {only_snapshot:?}\n  \
         If the change is intended, run \
         `cargo run -p ix-skill --bin ix -- doctor --write` and commit the diff."
    );
}

/// The MCP tool allowlist in this file must agree with the generated snapshot.
///
/// `EXPECTED` stays hand-maintained on purpose — it is the rate-limiter that
/// forces every surface change through review. What is gone is the *count*:
/// the two lists are now compared by name, so neither can drift silently.
#[test]
fn parity_expected_matches_generated_snapshot() {
    let expected: BTreeSet<String> = EXPECTED.iter().map(|s| (*s).to_string()).collect();
    assert_same_names(&expected, &snapshot_names("mcp_tools"), "EXPECTED (MCP tools)");
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
        "mesh_correlate",
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

/// Every `#[ix_skill]` registration must appear in the generated snapshot.
///
/// This replaces `assert_eq!(ix_registry::count(), 66)`, which asserted a
/// number rather than an inventory: a rename that kept the count constant
/// used to slip through.
#[test]
fn parity_registry_matches_generated_snapshot() {
    let live: BTreeSet<String> = ix_registry::all().map(|d| d.name.to_string()).collect();
    assert_same_names(&live, &snapshot_names("skills"), "capability registry skills");
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

    // batch1: ix_eigen → eigen (exercises the MCP-name dispatch path for the new
    // skill, not just the wrapper). [[2,1],[1,2]] has top eigenvalue 3.
    let params = serde_json::json!({ "matrix": [[2.0, 1.0], [1.0, 2.0]] });
    let result = reg.call("ix_eigen", params).expect("ix_eigen via registry");
    let top = result["eigenvalues"][0].as_f64().expect("eigenvalues[0]");
    assert!((top - 3.0).abs() < 1e-9, "top eigenvalue ~3, got {top}");
}

/// End-to-end: the feature-gated `ix_maintain_gate` tool dispatches through the manual
/// `handler` path and returns a verdict. Only the two hard lenses are exercised (no
/// loops/embeddings dirs) so the gate is deterministic and avoids touching live ga state.
#[cfg(feature = "maintain-gate")]
#[test]
fn maintain_gate_tool_returns_a_verdict() {
    let here = env!("CARGO_MANIFEST_DIR");
    let fx = format!("{here}/../ix-duck/tests/fixtures/maintain");
    let reg = ToolRegistry::new();
    let args = serde_json::json!({
        "hits_path": format!("{fx}/hits_up.jsonl"),
        "corpus_dir": format!("{fx}/corpus-pass"),
        "run_at": "2026-06-20T00:00:00Z"
    });
    let v = reg
        .call("ix_maintain_gate", args)
        .expect("ix_maintain_gate should dispatch and run");
    assert_eq!(
        v["status"], "T",
        "metric up + guardrail held → T; got {v:?}"
    );
    assert_eq!(v["decision"], "accept");
    // A missing required arg is a clean error, not a panic.
    assert!(reg.call("ix_maintain_gate", serde_json::json!({})).is_err());
}
