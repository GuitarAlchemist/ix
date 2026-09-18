//! MCP-surface parity test — protects the tool surface during the manual→registry
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

/// The MCP tools exposed by ix-agent. Deliberately **not** prefixed with a
/// hand-typed count: the number here read "72" while the list below held 96
/// entries, because every tool-adding PR bumped the list and left the prose
/// alone. `state/registry/skills.snapshot.json` is the count oracle (ix#185);
/// this list is the *name* oracle. The first 48 are registry-backed,
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
    "ix_code_topology",
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
    // ix#193 — proposed surface, see crates/ix-agent/src/skills/batch1.rs
    "ix_kalman",
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

/// Every registry-backed tool must be named in `ix-approval`'s classification table.
///
/// `ToolRegistry::call` routes registry-backed tools through `dispatch_action`, whose
/// `ApprovalMiddleware` sends an unclassified name to `ActionKind::Unknown` → Tier 3 →
/// blocked. So adding an `#[ix_skill]` without a matching `classify_action_kind` entry
/// ships a tool that is listed, unit-tested below the gate, and refused on every MCP call
/// (`ix_petri_analyze`, `ix_mesh_correlate` and the four `ix_assumption_*` tools were).
/// Manual tools go through the same gate (ix#350); see
/// [`every_exposed_tool_including_manual_has_an_explicit_approval_classification`].
///
/// A tool that *should* be gated goes in one of the explicit gated tables
/// (`SHELL_COMMAND_TOOLS`, `WEB_FETCH_TOOLS`, `EDIT_OUT_OF_PROJECT_TOOLS`), not the
/// silent `Unknown` default.
#[test]
fn every_registry_backed_tool_has_an_explicit_approval_classification() {
    use ix_agent::registry_bridge::mcp_name;
    use ix_approval::{classify_action_kind, ActionKind};

    let unclassified: Vec<String> = ix_registry::all()
        .map(|d| mcp_name(d.name))
        .filter(|name| classify_action_kind(name) == ActionKind::Unknown)
        .collect();
    assert!(
        unclassified.is_empty(),
        "registry-backed tools with no ix-approval classification — every MCP call to \
         them is blocked at Tier 3: {unclassified:?}\n  \
         Add each to crates/ix-approval/src/classify.rs in the table its effects warrant: \
         READ_TOOLS (no side effects), EDIT_IN_PROJECT_TOOLS (writes workspace or \
         in-process state), or a gated table (SHELL_COMMAND_TOOLS, WEB_FETCH_TOOLS, \
         EDIT_OUT_OF_PROJECT_TOOLS) for Tier 3."
    );
}

/// Every tool on the MCP surface — registry-backed *and* hand-registered in
/// `ToolRegistry` — must be named in `ix-approval`'s classification table.
///
/// Manual tools used to be invoked directly and skipped the approval gate entirely
/// (ix#350). They now run through `registry_bridge::dispatch_manual`, so an unclassified
/// manual tool is refused at Tier 3 on every call. Enumerating the live surface (not
/// `EXPECTED`) means a newly registered manual tool fails here until it is classified.
#[test]
fn every_exposed_tool_including_manual_has_an_explicit_approval_classification() {
    use ix_approval::{classify_action_kind, ActionKind};

    let mut unclassified: Vec<String> = exposed_names()
        .into_iter()
        .filter(|name| classify_action_kind(name) == ActionKind::Unknown)
        .collect();
    unclassified.sort();
    assert!(
        unclassified.is_empty(),
        "MCP tools with no ix-approval classification — every call to them is \
         blocked at Tier 3: {unclassified:?}\n  \
         Add each to crates/ix-approval/src/classify.rs in the table its effects warrant: \
         READ_TOOLS (no side effects), EDIT_IN_PROJECT_TOOLS (writes workspace or \
         in-process state, or spawns a fixed read-only subprocess), or a gated table \
         (SHELL_COMMAND_TOOLS, WEB_FETCH_TOOLS, EDIT_OUT_OF_PROJECT_TOOLS) for Tier 3."
    );
}

/// A manual tool classified Tier 3 is refused through `ToolRegistry::call` before its
/// handler runs — proof the manual path reaches the gate. Unrefused, `ix_nl_to_pipeline`
/// would spawn the `ix` binary and `ix_sentrux_annotate` the named executable.
#[test]
fn manual_tier_three_tools_are_refused_by_the_approval_gate() {
    let marker = in_root_tempdir();
    let out = marker.path().join("annotations.jsonl");
    for (tool, args) in [
        (
            "ix_nl_to_pipeline",
            serde_json::json!({ "sentence": "compute stats of 1 2 3" }),
        ),
        (
            "ix_sentrux_annotate",
            serde_json::json!({
                "workspace": "crates/ix-approval",
                "mode": "sidecar",
                "out": out.to_str().unwrap(),
            }),
        ),
    ] {
        let err = ToolRegistry::new()
            .call(tool, args)
            .expect_err("a Tier-3 manual tool must be refused");
        assert!(
            err.starts_with("ix_approval: action blocked (ApprovalRequired)"),
            "{tool}: expected an approval refusal, got: {err}"
        );
    }
    assert!(!out.exists(), "a refused tool must not write");
}

/// The auto-approved (Tier 2) manual tools that take a caller path confine it to the
/// workspace: `repo_root` for the git tools (it also becomes `safe.directory`) and
/// `state_dir` for `ix_autoresearch_run`, which writes there.
#[test]
fn tier_two_manual_tools_refuse_paths_outside_the_workspace() {
    use ix_agent::registry_bridge::shared_loop_detector;
    use serde_json::json;

    let registry = ToolRegistry::new();
    let call = |tool: &str, args: serde_json::Value| {
        shared_loop_detector().clear_key(tool);
        registry.call(tool, args)
    };
    let outside = tempfile::tempdir().unwrap();
    let outside_dir = outside.path().to_str().unwrap().to_string();
    let new_state = outside.path().join("state");

    let cases = [
        (
            "ix_git_log",
            json!({ "path": "crates", "repo_root": outside_dir }),
        ),
        ("ix_git_churn", json!({ "repo_root": outside_dir })),
        (
            "ix_autoresearch_run",
            json!({ "iterations": 1, "state_dir": new_state.to_str().unwrap() }),
        ),
    ];
    for (tool, args) in cases {
        let err =
            call(tool, args.clone()).expect_err("a path outside the workspace must be refused");
        assert!(err.contains("inside an allowed"), "{tool} {args}: {err}");
    }
    assert!(
        !new_state.exists(),
        "a refused state_dir must not be created"
    );

    for (tool, args) in [
        (
            "ix_git_log",
            json!({ "path": "crates", "repo_root": "../escape" }),
        ),
        ("ix_git_churn", json!({ "repo_root": "../escape" })),
        (
            "ix_autoresearch_run",
            json!({ "iterations": 1, "state_dir": "../escape" }),
        ),
    ] {
        let err = call(tool, args.clone()).expect_err("`..` must be refused");
        // `..` is resolved lexically and the root check decides, so an escaping
        // one is refused as an outside path (ix#350).
        assert!(err.contains("inside an allowed"), "{tool} {args}: {err}");
    }

    // An in-root repo root still works.
    let out = call(
        "ix_git_log",
        json!({ "path": "crates/ix-approval", "since_days": 30, "repo_root": "." }),
    )
    .expect("the workspace root must be accepted as repo_root");
    assert!(out["commits"].is_number(), "{out}");

    // `ix_maintain_gate` is auto-approved too, and every path it takes is the
    // caller's. Only compiled with the feature that pulls bundled DuckDB.
    #[cfg(feature = "maintain-gate")]
    {
        let err = call(
            "ix_maintain_gate",
            json!({ "hits_path": outside_dir, "corpus_dir": "state" }),
        )
        .expect_err("a hits_path outside the workspace must be refused");
        assert!(err.contains("inside an allowed"), "{err}");
        let err = call(
            "ix_maintain_gate",
            json!({ "hits_path": "Cargo.toml", "corpus_dir": "state", "repo_dir": outside_dir }),
        )
        .expect_err("a repo_dir outside the workspace must be refused");
        assert!(err.contains("inside an allowed"), "{err}");
    }
}

/// The Tier-1 manual tools read whatever path the caller names, and Tier 1
/// auto-approves, so each path is confined the same way (ix#350 review). Checked
/// through `ToolRegistry::call`, with a file whose contents must never appear in
/// an error message.
#[test]
fn tier_one_manual_tools_refuse_paths_outside_the_workspace() {
    use ix_agent::registry_bridge::shared_loop_detector;
    use serde_json::json;

    let registry = ToolRegistry::new();
    let call = |tool: &str, args: serde_json::Value| {
        shared_loop_detector().clear_key(tool);
        registry.call(tool, args)
    };
    let outside = tempfile::tempdir().unwrap();
    let secret = outside.path().join("secret.rs");
    std::fs::write(&secret, "fn secret_value() { /* SECRET-VALUE */ }").unwrap();
    let secret_file = secret.to_str().unwrap().to_string();
    let outside_dir = outside.path().to_str().unwrap().to_string();

    // (tool, argument builder, the path is a file rather than a directory)
    type Build = fn(&str) -> serde_json::Value;
    let tools: [(&str, Build, bool); 9] = [
        ("ix_cargo_deps", |p| json!({ "workspace_root": p }), false),
        ("ix_pipeline_list", |p| json!({ "root": p }), false),
        (
            "ix_quality_gate_history",
            |p| json!({ "ledger_path": p }),
            true,
        ),
        ("ix_code_topology", |p| json!({ "path": p }), false),
        (
            "ix_ast_query",
            |p| json!({ "query": "(function_item) @f", "path": p }),
            true,
        ),
        ("ix_code_smells", |p| json!({ "dir": p }), false),
        ("ix_code_smells", |p| json!({ "path": p }), true),
        ("ix_annotations_scan", |p| json!({ "workspace": p }), false),
        (
            "ix_optick_search",
            |p| json!({ "query": [0.0], "index_path": p }),
            true,
        ),
    ];

    for (tool, build, takes_file) in tools {
        let raw = if takes_file {
            secret_file.clone()
        } else {
            outside_dir.clone()
        };
        let args = build(&raw);
        let err = match call(tool, args.clone()) {
            Ok(v) => panic!("{tool} {args}: a path outside the workspace was accepted: {v}"),
            Err(e) => e,
        };
        assert!(err.contains("inside an allowed"), "{tool} {args}: {err}");
        assert!(!err.contains("SECRET"), "{tool} leaked contents: {err}");

        let args = build("../escape");
        let err = call(tool, args.clone()).expect_err("`..` must be refused");
        assert!(err.contains("inside an allowed"), "{tool} {args}: {err}");
    }

    // `test_files` entries are read by the reconciler, which resolves them with
    // `workspace.join(entry)` — and `join` drops the base for an absolute entry,
    // so each entry is confined on its own (ix#350 review).
    for entry in [secret_file.as_str(), "../escape.rs"] {
        let args = json!({ "workspace": "crates/ix-approval", "test_files": [entry] });
        let err = call("ix_annotations_scan", args.clone())
            .expect_err("a test_files entry outside the workspace must be refused");
        assert!(err.contains("inside an allowed"), "{args}: {err}");
        assert!(
            err.contains("test_files"),
            "the error must name the parameter: {err}"
        );
        assert!(!err.contains("SECRET"), "leaked contents: {err}");
    }
    let out = call(
        "ix_annotations_scan",
        json!({
            "workspace": "crates/ix-approval",
            "test_files": ["src/classify.rs"]
        }),
    )
    .expect("an in-root test_files entry must be accepted");
    assert!(out.is_object(), "{out}");

    // In-root paths keep working.
    let out = call("ix_cargo_deps", json!({ "workspace_root": "." }))
        .expect("the workspace itself must be walked");
    assert!(out["n_nodes"].as_u64().unwrap_or(0) > 0, "{out}");
    let out = call(
        "ix_code_smells",
        json!({ "path": "crates/ix-approval/src/classify.rs" }),
    )
    .expect("an in-root file must be scanned");
    assert_eq!(out["path"], "crates/ix-approval/src/classify.rs", "{out}");
}

/// `ix_pipeline_run` is gated and its handler dispatches each step through the gate
/// again. Before the chain was held behind an `Arc`, the nested dispatch waited on the
/// chain mutex its own caller held, so this call never returned. The nested calls still
/// count against the loop detector.
#[test]
fn gated_pipeline_run_dispatches_gated_steps_without_deadlock() {
    use ix_agent::registry_bridge::shared_loop_detector;
    use ix_agent::server_context::ServerContext;

    // No other test in this binary calls this tool, so its count is ours alone.
    const STEP_TOOL: &str = "ix_grothendieck_delta";
    shared_loop_detector().clear_key(STEP_TOOL);
    shared_loop_detector().clear_key("ix_pipeline_run");

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (ctx, _rx) = ServerContext::new();
        let step = |id: &str| {
            serde_json::json!({
                "id": id,
                "tool": STEP_TOOL,
                "arguments": { "source": [0, 4, 7], "target": [0, 4, 8] }
            })
        };
        let out = ToolRegistry::new().call_with_ctx(
            "ix_pipeline_run",
            serde_json::json!({ "steps": [step("a"), step("b")] }),
            &ctx,
        );
        let _ = tx.send(out);
    });
    let out = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("ix_pipeline_run deadlocked on a nested gated dispatch")
        .expect("pipeline with gated steps must succeed");
    assert_eq!(out["results"]["a"]["is_zero"], false, "{out}");
    assert_eq!(shared_loop_detector().count(STEP_TOOL), 2);
    assert!(shared_loop_detector().count("ix_pipeline_run") >= 1);
}

/// `ix_petri_analyze` through the exact entry point `main.rs` uses for `tools/call`.
/// Before its classification it returned
/// `ix_approval: action blocked (ApprovalRequired)` here.
#[test]
fn petri_analyze_is_reachable_through_mcp_dispatch() {
    use ix_agent::server_context::ServerContext;

    let (ctx, _rx) = ServerContext::new();
    let out = ToolRegistry::new()
        .call_with_ctx(
            "ix_petri_analyze",
            serde_json::json!({
                "places": [{ "id": "lock", "tokens": 1 }, "working"],
                "transitions": ["acquire"],
                "arcs": [
                    { "source": "lock", "target": "acquire" },
                    { "source": "acquire", "target": "working" }
                ]
            }),
            &ctx,
        )
        .expect("ix_petri_analyze must not be refused by the approval gate");
    assert_eq!(out["deadlock_free"]["verdict"], "fails");
    assert_eq!(out["deadlock_free"]["detail"][0]["witness"][0], "acquire");
}

/// `ix_pipeline_run` is a manual tool, gated itself (ix#350), and each step goes back
/// through `ToolRegistry::call` — so a step naming an unclassified registry tool failed
/// the whole pipeline with the same approval refusal. The nested dispatch also checks
/// the middleware chain is not held across the outer handler (it would deadlock).
#[test]
fn pipeline_run_step_reaches_a_newly_classified_tool() {
    use ix_agent::server_context::ServerContext;

    let (ctx, _rx) = ServerContext::new();
    let out = ToolRegistry::new()
        .call_with_ctx(
            "ix_pipeline_run",
            serde_json::json!({
                "steps": [{
                    "id": "mesh",
                    "tool": "ix_mesh_correlate",
                    "arguments": { "series": [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]] }
                }]
            }),
            &ctx,
        )
        .expect("a pipeline step must not be refused by the approval gate");
    let mesh = &out["results"]["mesh"];
    assert_eq!(mesh["n_streams"], 2);
    // Values only the real handler computes: the two series are perfectly
    // correlated, so both nodes get component id 0 (`components[node]`).
    let r = mesh["correlation"][0][1]
        .as_f64()
        .expect("correlation matrix entry");
    assert!((r - 1.0).abs() < 1e-9, "expected r = 1.0, got {r}");
    assert_eq!(mesh["components"], serde_json::json!([0, 0]));
}

/// The four `ix_assumption_*` tools run auto-approved and take caller paths, so the
/// paths are confined to the workspace root. Checked through the MCP entry point.
#[test]
fn assumption_tools_refuse_paths_outside_the_workspace() {
    use ix_agent::server_context::ServerContext;

    let (ctx, _rx) = ServerContext::new();
    let registry = ToolRegistry::new();
    let outside = tempfile::tempdir().unwrap();
    let secret = outside.path().join("secret.json");
    std::fs::write(&secret, r#"["SECRET-VALUE"]"#).unwrap();
    let secret = secret.to_str().unwrap();
    let outside_dir = outside.path().to_str().unwrap();

    let cases = [
        (
            "ix_assumption_query",
            serde_json::json!({ "research": secret }),
        ),
        (
            "ix_assumption_query",
            serde_json::json!({ "workspace": outside_dir }),
        ),
        (
            "ix_assumption_belief_at",
            serde_json::json!({ "log": secret }),
        ),
        (
            "ix_assumption_drift",
            serde_json::json!({ "baseline": secret }),
        ),
        (
            "ix_assumption_claims",
            serde_json::json!({ "path": "crates", "workspace": outside_dir }),
        ),
    ];
    for (tool, args) in cases {
        let err = registry
            .call_with_ctx(tool, args.clone(), &ctx)
            .expect_err("a path outside the workspace must be refused");
        assert!(
            err.contains("not an existing path inside an allowed root"),
            "{tool} {args}: {err}"
        );
        assert!(
            !err.contains("SECRET-VALUE"),
            "{tool} leaked contents: {err}"
        );
    }

    let err = registry
        .call_with_ctx(
            "ix_assumption_belief_at",
            serde_json::json!({ "log": "../escape.jsonl" }),
            &ctx,
        )
        .expect_err("`..` must be refused");
    // Since ix#350 a `..` is resolved lexically and the root check decides, so
    // this one is refused for leaving the root rather than for its shape.
    assert!(err.contains("inside an allowed"), "{err}");

    // A relative workspace inside the root still works.
    let out = registry
        .call_with_ctx(
            "ix_assumption_claims",
            serde_json::json!({ "path": "src", "workspace": "crates/ix-approval" }),
            &ctx,
        )
        .expect("a workspace inside the root must be accepted");
    assert_eq!(out["path"], "src");
}

/// The repo root, as `path_confine::workspace_root` resolves it under `cargo test`.
/// Without the Windows verbatim `\\?\` prefix, which the tools refuse as input.
fn confine_test_root() -> std::path::PathBuf {
    let canonical = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root");
    match canonical.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(plain) => std::path::PathBuf::from(plain),
        None => canonical,
    }
}

/// A scratch directory inside the workspace root, removed on drop. It lives in
/// the gitignored `target/confine-test/`, so a killed run leaves nothing in the
/// tree and tests that walk the workspace sources do not meet its links.
fn in_root_tempdir() -> tempfile::TempDir {
    let parent = confine_test_root().join("target").join("confine-test");
    std::fs::create_dir_all(&parent).expect("create target/confine-test");
    tempfile::Builder::new()
        .tempdir_in(parent)
        .expect("tempdir inside the workspace root")
}

#[cfg(unix)]
fn link_dir(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

/// Symlinks need Developer Mode or elevation on Windows; a directory junction
/// does not, and is the same escape.
#[cfg(windows)]
fn link_dir(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(target, link).or_else(|_| {
        std::process::Command::new("cmd")
            .arg("/C")
            .arg("mklink")
            .arg("/J")
            .arg(link)
            .arg(target)
            .output()
            .and_then(|o| {
                if o.status.success() {
                    Ok(())
                } else {
                    Err(std::io::Error::other("mklink /J failed"))
                }
            })
    })
}

/// Every auto-approved (Tier 1 / Tier 2) registry tool that takes a caller path
/// confines it: a file or directory outside the workspace, `..`, and a link inside
/// the workspace that leads out are all refused through the MCP entry point, with
/// one message that echoes no file contents, while an in-root path still works.
#[test]
fn auto_approved_tools_refuse_paths_outside_the_workspace() {
    use ix_agent::registry_bridge::shared_loop_detector;
    use ix_agent::server_context::ServerContext;
    use serde_json::json;

    let (ctx, _rx) = ServerContext::new();
    let registry = ToolRegistry::new();
    let call = |tool: &str, args: serde_json::Value| {
        shared_loop_detector().clear_key(tool);
        registry.call_with_ctx(tool, args, &ctx)
    };

    let outside = tempfile::tempdir().unwrap();
    let secret = outside.path().join("secret.csv");
    std::fs::write(&secret, "SECRET-VALUE\nSECRET-VALUE\n").unwrap();
    std::fs::write(outside.path().join("trace.json"), r#"{"SECRET":"VALUE"}"#).unwrap();
    std::fs::write(outside.path().join("lib.rs"), "fn secret_value() {}").unwrap();
    let secret_file = secret.to_str().unwrap().to_string();
    let outside_dir = outside.path().to_str().unwrap().to_string();

    // A link inside the workspace that points at the outside directory.
    let scratch = in_root_tempdir();
    let link = scratch.path().join("escape");
    link_dir(outside.path(), &link).expect("create a symlink or junction");
    let rel = |p: &std::path::Path| {
        p.strip_prefix(confine_test_root())
            .unwrap()
            .to_str()
            .unwrap()
            .replace('\\', "/")
    };
    // Absolute, so the export destination (whose relative paths resolve against
    // ~/.ga/traces) meets the same in-root link as every other tool.
    let link_file = link.join("secret.csv").to_str().unwrap().to_string();
    let link_dir_arg = link.to_str().unwrap().to_string();

    // (tool, argument builder) — the builder places the path where the tool reads it.
    type Build = fn(&str) -> serde_json::Value;
    let tools: [(&str, Build, bool); 9] = [
        ("ix_code_analyze", |p| json!({ "path": p }), true),
        (
            "ix_context_walk",
            |p| json!({ "target": "x::y", "strategy": "callers", "workspace_root": p }),
            false,
        ),
        ("ix_governance_graph", |p| json!({ "root": p }), false),
        (
            "ix_governance_graph_rescan",
            |p| json!({ "root": p, "last_scan_epoch": 0 }),
            false,
        ),
        (
            "ix_ml_pipeline",
            |p| json!({ "source": { "type": "csv", "path": p } }),
            true,
        ),
        (
            "ix_tars_bridge",
            |p| json!({ "action": "prepare_traces", "trace_dir": p }),
            false,
        ),
        ("ix_trace_ingest", |p| json!({ "dir": p }), false),
        (
            "ix_session_flywheel_export",
            |p| json!({ "session_log": p }),
            true,
        ),
        (
            "ix_session_flywheel_export",
            |p| json!({ "session_log": "Cargo.toml", "trace_dir": p }),
            false,
        ),
    ];

    for (tool, build, takes_file) in tools {
        let (target, via_link) = if takes_file {
            (secret_file.clone(), link_file.clone())
        } else {
            (outside_dir.clone(), link_dir_arg.clone())
        };
        for raw in [target, via_link] {
            let args = build(&raw);
            let err = match call(tool, args.clone()) {
                Ok(v) => panic!("{tool} {args}: a path outside the workspace was accepted: {v}"),
                Err(e) => e,
            };
            // Reads say "not an existing path inside an allowed root", the export
            // destination "not inside an allowed destination root".
            assert!(err.contains("inside an allowed"), "{tool} {args}: {err}");
            assert!(!err.contains("SECRET"), "{tool} leaked contents: {err}");
        }
        let args = build("../escape");
        let err = call(tool, args.clone()).expect_err("`..` must be refused");
        // Since ix#350 confinement resolves `..` and refuses the result for
        // leaving the root. `ix_ml_pipeline` keeps its own earlier `..` check.
        assert!(
            err.contains("inside an allowed") || err.contains("must not contain '..'"),
            "{tool} {args}: {err}"
        );
    }
    // Refusing a write destination must not create it.
    assert!(!outside.path().join("traces").exists());

    // On Windows, UNC, device and verbatim paths are refused by their shape,
    // before anything is resolved. Only local shapes are used here: a named
    // pipe that does not exist, and a verbatim path to a real in-root file,
    // which the root check alone would have accepted.
    #[cfg(windows)]
    {
        let verbatim = format!(r"\\?\{}", confine_test_root().join("Cargo.toml").display());
        for raw in [r"\\.\pipe\ix-confine-test-absent", verbatim.as_str()] {
            for (tool, args) in [
                ("ix_code_analyze", json!({ "path": raw })),
                ("ix_session_flywheel_export", json!({ "session_log": raw })),
                ("ix_trace_ingest", json!({ "dir": raw })),
                (
                    "ix_session_flywheel_export",
                    json!({ "session_log": "Cargo.toml", "trace_dir": raw }),
                ),
            ] {
                let err = call(tool, args.clone()).expect_err("a non-local path shape must be refused");
                assert!(
                    err.contains("absolute path on a local drive"),
                    "{tool} {args}: {err}"
                );
            }
        }
    }

    // The export destination admits only the operator's trace locations: the
    // workspace itself, including the harness config under it, is refused.
    let in_root_dest = scratch.path().join("out/new");
    for dest in [
        in_root_dest.to_str().unwrap().to_string(),
        confine_test_root().join(".claude").to_str().unwrap().to_string(),
    ] {
        let err = call(
            "ix_session_flywheel_export",
            json!({ "session_log": "Cargo.toml", "trace_dir": dest, "trace_id": "settings" }),
        )
        .expect_err("a workspace destination must be refused");
        assert!(err.contains("not inside an allowed destination root"), "{dest}: {err}");
    }
    assert!(!scratch.path().join("out").exists());

    // A trace id is a file name, not a path: refused before anything is written.
    let absolute_id = outside.path().join("settings");
    for id in [absolute_id.to_str().unwrap(), "../../.claude/settings"] {
        let err = call(
            "ix_session_flywheel_export",
            json!({ "session_log": "Cargo.toml", "trace_id": id }),
        )
        .expect_err("a path-shaped trace_id must be refused");
        assert!(err.contains("is not a plain file name"), "{id}: {err}");
    }
    assert!(!outside.path().join("settings.json").exists());

    // A persona name is a file stem under the personas directory, not a path.
    for name in ["../../secret", "a/b", "C:\\x", "a\u{0}b"] {
        let err = call("ix_governance_persona", json!({ "persona": name }))
            .expect_err("a path-shaped persona name must be refused");
        assert!(err.contains("is not a persona name"), "{name}: {err}");
    }

    // In-root calls keep working.
    let out = call(
        "ix_code_analyze",
        json!({ "path": "crates/ix-approval/src/classify.rs" }),
    )
    .expect("an in-root file must be analyzed");
    assert!(out.is_object(), "{out}");

    call(
        "ix_context_walk",
        json!({
            "target": "ix_approval::classify::classify_action_kind",
            "strategy": "callers",
            "workspace_root": "crates/ix-approval"
        }),
    )
    .expect("an in-root workspace must be indexed");

    let graph = call(
        "ix_governance_graph",
        json!({ "root": "crates/ix-approval" }),
    )
    .expect("an in-root governance root must be scanned");
    assert_eq!(graph["total_nodes"], 0, "{graph}");
    let rescan = call(
        "ix_governance_graph_rescan",
        json!({ "root": "crates/ix-approval", "last_scan_epoch": 0 }),
    )
    .expect("an in-root rescan must work");
    assert_eq!(rescan["changed"], true, "{rescan}");

    let data = scratch.path().join("data.csv");
    let mut csv = String::from("a,b,label\n");
    for i in 0..20 {
        csv.push_str(&format!("{i},{},{}\n", i * 2, i % 2));
    }
    std::fs::write(&data, csv).unwrap();
    let trained = call(
        "ix_ml_pipeline",
        json!({ "source": { "type": "csv", "path": rel(&data), "target_column": "label" } }),
    )
    .expect("an in-root CSV must load");
    assert!(trained.is_object(), "{trained}");

    // Unparseable contents inside the root (ragged rows): the call fails, and
    // the error names the file, not its text.
    let junk = scratch.path().join("junk.csv");
    std::fs::write(&junk, "SECRET-VALUE\nSECRET-VALUE,SECRET-VALUE\n").unwrap();
    let e = call(
        "ix_ml_pipeline",
        json!({ "source": { "type": "csv", "path": rel(&junk) } }),
    )
    .expect_err("a CSV with ragged rows must be refused");
    assert!(e.contains("CSV load error"), "{e}");
    assert!(!e.contains("SECRET"), "parse error leaked contents: {e}");

    let traces = scratch.path().join("traces");
    std::fs::create_dir_all(&traces).unwrap();
    let ingest = call("ix_trace_ingest", json!({ "dir": rel(&traces) }))
        .expect("an in-root trace dir must be ingested");
    assert_eq!(ingest["total_traces"], 0, "{ingest}");
    let prepared = call(
        "ix_tars_bridge",
        json!({ "action": "prepare_traces", "trace_dir": rel(&traces) }),
    )
    .expect("an in-root trace dir must be prepared");
    assert_eq!(prepared["stats"]["total_traces"], 0, "{prepared}");

    // An in-root session log is admitted (the destination check comes next and
    // refuses the in-root dir). Exports that succeed write under a trace root
    // and are covered in `session_log_wiring.rs` and `path_confine_env.rs`,
    // which control those locations.
    let log_path = scratch.path().join("session.jsonl");
    drop(ix_session::SessionLog::open(&log_path).unwrap());
    let err = call(
        "ix_session_flywheel_export",
        json!({ "session_log": rel(&log_path), "trace_dir": in_root_dest.to_str().unwrap() }),
    )
    .expect_err("the in-root destination is refused");
    assert!(err.contains("`trace_dir`"), "the session log must pass first: {err}");
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
    assert_same_names(
        &live,
        &snapshot_names("skills"),
        "capability registry skills",
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
    // Built from the workspace root rather than `<crate>/..`: the tool confines
    // its paths, and a `..` segment resolved into an existing in-root path is
    // accepted (ix#350) — but spelling the fixture without one keeps the
    // fixture independent of that rule.
    let fx = confine_test_root()
        .join("crates/ix-duck/tests/fixtures/maintain")
        .display()
        .to_string()
        .replace('\\', "/");
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
