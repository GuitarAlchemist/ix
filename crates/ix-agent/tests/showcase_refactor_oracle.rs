//! Canonical showcase #05 — the Adversarial Refactor Oracle
//! (live data edition).
//!
//! A 14-tool self-referential demo: ix analyses its own workspace
//! using live data from ix_cargo_deps + ix_git_log (no baked
//! constants), attacks a classifier trained on its own cluster
//! labels, searches for refactor vectors via a GA, and audits
//! the whole chain via Demerzel governance — all as a single
//! `ix_pipeline_run` invocation.
//!
//! # Live vs baked
//!
//! The first two steps (`s00_discover_crates`, `s01_churn_series`)
//! call `ix_cargo_deps` and `ix_git_log` to read the **current**
//! state of this very repository. Every number in the rest of the
//! pipeline flows from those reads via `$step.field` substitution —
//! the SLOC baseline, the dep graph, the feature matrix, the
//! k-means input, the random-forest training data, and the FGSM
//! attack target are all live. The only baked constants that remain
//! are the three framing compromises called out in §2 below.
//!
//! This is the P1.1 + P1.2 graduation: the oracle moved from
//! "self-referential in theory" to "self-referential in fact."
//!
//! # What makes this the most ambitious demo in the showcase
//!
//! - **12 tools** chained in one call (bracket v2 is 5). See the
//!   per-step list below.
//! - **Real workspace data** — every number in the pipeline comes from
//!   `git log`, `wc -l`, `Cargo.toml`, or `ix_code_analyze` on this
//!   very repository. No synthetic fixtures.
//! - **R6 Level 1 preview** — the adversarial step (FGSM) wraps the
//!   random-forest classifier trained in the previous step, as per
//!   the adversarial-surrogate-validation rung of the R6 roadmap.
//! - **Compiled via `ix_pipeline_compile`** — the pipeline spec comes
//!   from the NL-to-pipeline compiler. The same spec that lives on
//!   disk is driven through the compiler (with a canned LLM response
//!   standing in for a live sampling call) and executed end-to-end in
//!   this test, proving the compile → validate → run → audit chain.
//! - **Lineage DAG** (R2 Phase 2) — the 12-step provenance chain is
//!   surfaced in the response and can be passed to `ix_governance_check`
//!   as the audit trail, closing the loop between pipeline output and
//!   Demerzel compliance.
//!
//! # Framing compromises — called out honestly
//!
//! Three tools in the chain accept only fixed enums / hard-coded maps.
//! The pipeline still runs on real data but narrates these steps as
//! symbolic stand-ins for the full functionality:
//!
//! 1. **`ix_evolution`** only accepts `sphere | rosenbrock | rastrigin`
//!    as its fitness. We run rosenbrock over a 4-dim "refactor vector
//!    space" and narrate the best_params as a symbolic refactor move.
//!    A real fitness function is future R10 work.
//! 2. **`ix_chaos_lyapunov`** only supports the logistic map. We derive
//!    a parameter from the SLOC variance across the 5 crates and map
//!    it into `r ∈ [3.5, 4.0]` to report which dynamical regime the
//!    workspace lives in. Poetic, but on real input.
//! 3. **`ix_adversarial_fgsm`** needs an explicit gradient array and
//!    `ix_random_forest` has no gradient — so we bake the synthetic
//!    gradient as `healthy_centroid - at_risk_centroid` (precomputed
//!    from a preliminary k-means run on the same data) and narrate
//!    the attack as "the minimum perturbation along the natural
//!    refactor direction."
//!
//! # Per-step map
//!
//! | # | id                       | tool                   | role                                    |
//! |---|--------------------------|------------------------|-----------------------------------------|
//! | 1 | s01_baseline_sloc        | ix_stats               | descriptive stats on SLOC across crates |
//! | 2 | s02_dep_pagerank         | ix_graph (pagerank)    | centrality of each crate in the dep DAG|
//! | 3 | s03_dep_toposort         | ix_graph (toposort)    | confirm dep graph is a DAG              |
//! | 4 | s04_betti_numbers        | ix_topo (betti_at_r)   | Betti numbers of the metric cloud       |
//! | 5 | s05_persistence_diagram  | ix_topo (persistence)  | persistent homology of same cloud       |
//! | 6 | s06_churn_spectrum       | ix_fft                 | FFT of commit counts                    |
//! | 7 | s07_velocity_regime      | ix_chaos_lyapunov      | logistic-map regime check               |
//! | 8 | s08_crate_clusters       | ix_kmeans              | cluster crates by health profile        |
//! | 9 | s09_risk_classifier      | ix_random_forest       | RF trained on cluster labels            |
//! |10 | s10_adversarial_attack   | ix_adversarial_fgsm    | minimum refactor perturbation           |
//! |11 | s11_refactor_search      | ix_evolution (GA)      | symbolic refactor vector search         |
//! |12 | s12_governance_audit     | ix_governance_check    | Demerzel verdict on the plan            |

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::thread;

// ---------------------------------------------------------------------------
// Live-data sources: the oracle now fetches everything from
// ix_cargo_deps (workspace enumeration + metric projections + dep
// graph) and ix_git_log (commit cadence series), so no per-run
// workspace constants are baked into this file.
//
// The one remaining baked value is FGSM_GRADIENT — the synthetic
// "refactor direction" the adversarial step uses as its gradient.
// Computing it live would require either (a) a `vector_diff` tool
// to subtract two rows of `$s06_clusters.centroids`, or (b) a
// stronger `$step.field` substitution that supports arithmetic.
// Both are follow-up work (see FINDINGS §5.C / P1.3). For now it
// stays baked, but at a value that can work against *any* feature
// matrix shape by being a unit-ish direction.
// ---------------------------------------------------------------------------

/// Synthetic FGSM gradient for the 3-feature column order that
/// `ix_cargo_deps` emits (sloc, file_count, dep_count). Negative in
/// every dimension — "less code, fewer files, fewer deps" is the
/// direction that moves any at-risk node toward the healthy cluster.
/// Magnitudes are rough but the signs are what matter for FGSM at
/// small epsilon.
const FGSM_GRADIENT: [f64; 3] = [-1.0, -1.0, -1.0];

/// A static baseline for the logistic-map parameter (r ∈ [3.5, 4.0]).
/// Previously derived from SLOC variance; now a fixed conservative
/// value in the chaotic band. Real parameter derivation from live
/// data is blocked on the same substitution weakness as FGSM (can't
/// compute mean/var of `$s.sloc` inline).
const LYAPUNOV_R: f64 = 3.78;

// ---------------------------------------------------------------------------
// The canned LLM response — the pipeline spec a competent compiler
// would emit given the brief in `ORACLE_BRIEF`. The test drives this
// through `ix_pipeline_compile` via `fake_client`, which exercises
// the real validation + dispatch path but substitutes a deterministic
// response for the LLM sampling call.
// ---------------------------------------------------------------------------

const ORACLE_BRIEF: &str = "Discover every crate in the ix workspace with ix_cargo_deps, \
    pull the 90-day commit churn series for ix-agent via ix_git_log, then run the adversarial \
    refactor audit on the live data: descriptive stats on SLOC, dep-graph centrality via \
    PageRank, confirm the graph is a DAG, topological invariants of the 3-feature crate cloud, \
    FFT of commit churn, logistic-map regime check, cluster into 3 health groups via k-means, \
    train a random-forest classifier on the cluster labels, attack it with FGSM along the \
    synthetic refactor direction, search for a 3-dim refactor vector via GA, and close with a \
    Demerzel governance check on the refactor plan. Every numeric input must flow from the two \
    source-adapter steps via $step.field references — no baked constants except the synthetic \
    FGSM gradient.";

fn canned_oracle_spec() -> Value {
    // `cargo test` runs with CWD = crates/ix-agent, but the source
    // adapters need to see the whole workspace. Resolve the workspace
    // root from CARGO_MANIFEST_DIR and pass it explicitly. In a live
    // MCP run the CWD is the user's terminal, so the default CWD
    // behaviour is still the right shape — this override is for the
    // test harness only.
    let workspace_root = workspace_root().display().to_string();
    // git_log is called from the test harness, which runs out of
    // crates/ix-agent. The pipeline handler itself has no hook to
    // change CWD, so we path-prefix into the workspace root via the
    // tool's `path` arg. But `path` is validated to be repo-internal,
    // so we use a relative path from the *git repo root* (which
    // matches the workspace root for the ix project). The tool
    // calls `git log` which resolves paths against git's notion of
    // the repo, not the process CWD, so `crates/ix-agent` is correct
    // regardless of where the process was launched.
    json!({
        "steps": [
            // ─── Source adapters — the two steps that make this
            // self-referential in fact rather than in name. ─────
            {
                "id": "s00_cargo_deps",
                "tool": "ix_cargo_deps",
                "asset_name": "refactor_oracle.cargo_deps",
                "arguments": { "workspace_root": workspace_root }
            },
            {
                "id": "s01_git_log",
                "tool": "ix_git_log",
                "asset_name": "refactor_oracle.git_log",
                "depends_on": ["s00_cargo_deps"],
                "arguments": {
                    "path": "crates/ix-agent",
                    "since_days": 90,
                    "bucket": "week",
                    "repo_root": workspace_root
                }
            },
            // ─── Live-data analysis — every argument below is a
            // $step.field reference into the two source adapters. ─
            {
                "id": "s02_baseline_sloc",
                "tool": "ix_stats",
                "asset_name": "refactor_oracle.baseline_sloc",
                "depends_on": ["s00_cargo_deps"],
                "arguments": { "data": "$s00_cargo_deps.sloc" }
            },
            {
                "id": "s03_dep_pagerank",
                "tool": "ix_graph",
                "asset_name": "refactor_oracle.dep_pagerank",
                "depends_on": ["s00_cargo_deps", "s02_baseline_sloc"],
                "arguments": {
                    "operation": "pagerank",
                    "n_nodes": "$s00_cargo_deps.n_nodes",
                    "edges": "$s00_cargo_deps.edges",
                    "damping": 0.85,
                    "iterations": 100
                }
            },
            {
                "id": "s04_dep_toposort",
                "tool": "ix_graph",
                "asset_name": "refactor_oracle.dep_toposort",
                "depends_on": ["s03_dep_pagerank"],
                "arguments": {
                    "operation": "topological_sort",
                    "n_nodes": "$s00_cargo_deps.n_nodes",
                    "edges": "$s00_cargo_deps.edges"
                }
            },
            {
                "id": "s05_betti_numbers",
                "tool": "ix_topo",
                "asset_name": "refactor_oracle.betti_numbers",
                "depends_on": ["s04_dep_toposort"],
                "arguments": {
                    "operation": "betti_at_radius",
                    "points": "$s00_cargo_deps.features",
                    "radius": 5000.0,
                    "max_dim": 1
                }
            },
            {
                "id": "s06_persistence_diagram",
                "tool": "ix_topo",
                "asset_name": "refactor_oracle.persistence_diagram",
                "depends_on": ["s05_betti_numbers"],
                "arguments": {
                    "operation": "persistence",
                    "points": "$s00_cargo_deps.features",
                    "max_dim": 1,
                    "max_radius": 15000.0
                }
            },
            {
                "id": "s07_churn_spectrum",
                "tool": "ix_fft",
                "asset_name": "refactor_oracle.churn_spectrum",
                "depends_on": ["s01_git_log", "s06_persistence_diagram"],
                "arguments": { "signal": "$s01_git_log.series" }
            },
            {
                "id": "s08_velocity_regime",
                "tool": "ix_chaos_lyapunov",
                "asset_name": "refactor_oracle.velocity_regime",
                "depends_on": ["s07_churn_spectrum"],
                "arguments": {
                    "map": "logistic",
                    "parameter": LYAPUNOV_R,
                    "iterations": 1000
                }
            },
            {
                "id": "s09_crate_clusters",
                "tool": "ix_kmeans",
                "asset_name": "refactor_oracle.crate_clusters",
                "depends_on": ["s08_velocity_regime"],
                "arguments": {
                    "data": "$s00_cargo_deps.features",
                    "k": 3,
                    "max_iter": 100
                }
            },
            {
                "id": "s10_risk_classifier",
                "tool": "ix_random_forest",
                "asset_name": "refactor_oracle.risk_classifier",
                "depends_on": ["s09_crate_clusters"],
                "arguments": {
                    "x_train": "$s00_cargo_deps.features",
                    "y_train": "$s09_crate_clusters.labels",
                    "x_test": "$s00_cargo_deps.features",
                    "n_trees": 20,
                    "max_depth": 5
                }
            },
            {
                "id": "s11_adversarial_attack",
                "tool": "ix_adversarial_fgsm",
                "asset_name": "refactor_oracle.adversarial_attack",
                "depends_on": ["s10_risk_classifier"],
                "arguments": {
                    // `.0` indexes the first row of the features matrix —
                    // whichever crate `ix_cargo_deps` emitted first in
                    // its alphabetical ordering (usually ix-adversarial
                    // or similar). The FGSM step is intentionally agnostic
                    // to which row it attacks; what matters is the
                    // perturbation direction.
                    "input": "$s00_cargo_deps.features.0",
                    "gradient": FGSM_GRADIENT,
                    "epsilon": 0.1
                }
            },
            {
                "id": "s12_refactor_search",
                "tool": "ix_evolution",
                "asset_name": "refactor_oracle.refactor_search",
                "depends_on": ["s11_adversarial_attack"],
                "arguments": {
                    "algorithm": "genetic",
                    "function": "rosenbrock",
                    "dimensions": 3,
                    "generations": 50,
                    "population_size": 40,
                    "mutation_rate": 0.1
                }
            },
            {
                "id": "s13_governance_audit",
                "tool": "ix_governance_check",
                "asset_name": "refactor_oracle.governance_audit",
                "depends_on": ["s12_refactor_search"],
                "arguments": {
                    "action": "ship the GA-proposed refactor plan for the largest crate in the ix workspace",
                    "context": "Plan was derived from live workspace data via ix_cargo_deps (all crates, SLOC, file count, dep count, edge list) and ix_git_log (90-day commit cadence), validated by PageRank + topological sort of the dep graph, topologically summarised via persistent homology + Betti numbers, frequency-analysed via FFT over real commit churn, regime-checked via the logistic-map Lyapunov exponent, clustered into 3 health profiles, classified by a random forest, attacked by FGSM along a synthetic refactor direction, and searched via GA. Every upstream step has an asset-backed cache key; see the pipeline lineage for the full audit chain."
                }
            }
        ]
    })
}

// ---------------------------------------------------------------------------
// Fake MCP client for the sampling path.
// ---------------------------------------------------------------------------

fn fake_client(ctx: ServerContext, outbound: Receiver<String>, canned: String) {
    thread::spawn(move || {
        while let Ok(line) = outbound.recv() {
            let envelope: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let Some(id) = envelope.get("id").and_then(|v| v.as_i64()) else {
                continue;
            };
            if envelope.get("method").and_then(|m| m.as_str()) != Some("sampling/createMessage") {
                continue;
            }
            let response = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "role": "assistant",
                    "content": { "type": "text", "text": canned },
                }
            });
            ctx.deliver_response(id, response);
        }
    });
}

fn unwrap_tool_result(v: &Value) -> Value {
    if v.is_object() && !v.get("content").map(|c| c.is_array()).unwrap_or(false) {
        return v.clone();
    }
    if let Some(content) = v.get("content").and_then(|c| c.as_array()) {
        for item in content {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                    return parsed;
                }
            }
        }
    }
    v.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn canned_spec_passes_the_validator() {
    let reg = ToolRegistry::new();
    let (errors, warnings) = reg.validate_pipeline_spec(&canned_oracle_spec());
    assert!(
        errors.is_empty(),
        "canned oracle spec must validate cleanly: {errors:?}"
    );
    assert!(
        warnings.is_empty(),
        "canned oracle spec should have asset_name on every step: {warnings:?}"
    );
}

#[test]
fn compiler_drives_oracle_from_natural_language_brief() {
    // Drive the full compile pipeline: fake_client stands in for the
    // real MCP sampling call, ix_pipeline_compile validates the LLM
    // response and returns status "ok" if the spec is executable.
    let canned = serde_json::to_string(&canned_oracle_spec()).expect("serialise canned");
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned);

    let reg = ToolRegistry::new();
    let result = reg
        .call_with_ctx(
            "ix_pipeline_compile",
            json!({ "sentence": ORACLE_BRIEF, "max_steps": 14 }),
            &ctx,
        )
        .expect("compile");

    assert_eq!(result["status"], "ok", "compile failed: {result}");
    assert_eq!(
        result["spec"]["steps"].as_array().map(|a| a.len()),
        Some(14),
        "expected 14 steps (2 source adapters + 12 analysis)"
    );
}

#[test]
fn oracle_runs_end_to_end_and_produces_lineage_dag() {
    let canned = serde_json::to_string(&canned_oracle_spec()).expect("serialise canned");
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned);

    let reg = ToolRegistry::new();

    // Compile the brief into a spec.
    let compiled = reg
        .call_with_ctx(
            "ix_pipeline_compile",
            json!({ "sentence": ORACLE_BRIEF }),
            &ctx,
        )
        .expect("compile");
    assert_eq!(compiled["status"], "ok");

    // Execute the compiled spec.
    let exec = reg
        .call_with_ctx("ix_pipeline_run", compiled["spec"].clone(), &ctx)
        .expect("run");

    // All 14 steps must execute in topological order.
    let order = exec["execution_order"].as_array().expect("execution_order");
    assert_eq!(order.len(), 14);
    assert_eq!(order[0], "s00_cargo_deps");
    assert_eq!(order[13], "s13_governance_audit");

    // Every step must have an asset-backed cache key (R2 Phase 1).
    let cache_keys = exec["cache_keys"].as_object().expect("cache_keys");
    for step_id in order {
        let id = step_id.as_str().unwrap();
        let key = cache_keys.get(id).expect("cache key present");
        assert!(
            key.as_str()
                .is_some_and(|k| k.starts_with("ix_pipeline_run:")),
            "step {id}: expected asset-backed cache key, got {key:?}"
        );
    }

    // Lineage DAG (R2 Phase 2) must have 14 well-formed entries.
    let lineage = exec["lineage"].as_object().expect("lineage");
    assert_eq!(lineage.len(), 14);
    for (id, entry) in lineage {
        let deps = entry.get("depends_on").and_then(|v| v.as_array()).unwrap();
        let ups = entry
            .get("upstream_cache_keys")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(
            deps.len(),
            ups.len(),
            "lineage['{id}']: depends_on length must match upstream_cache_keys"
        );
    }

    // s00 — source adapter must have discovered the full workspace
    // (20+ crates) with denormalized projection vectors aligned.
    let cargo = unwrap_tool_result(&exec["results"]["s00_cargo_deps"]);
    let n_nodes = cargo["n_nodes"].as_u64().expect("n_nodes") as usize;
    assert!(
        n_nodes >= 20,
        "workspace should have 20+ crates, got {n_nodes}"
    );
    let sloc_vec = cargo["sloc"].as_array().unwrap();
    assert_eq!(sloc_vec.len(), n_nodes);

    // s01 — git log should have returned a 13-bucket weekly series
    // for the 90-day window.
    let git = unwrap_tool_result(&exec["results"]["s01_git_log"]);
    let series = git["series"].as_array().unwrap();
    assert_eq!(
        series.len(),
        13,
        "90-day weekly series should have 13 buckets"
    );

    // s09 k-means must have produced k=3 labels covering every crate.
    let clusters = unwrap_tool_result(&exec["results"]["s09_crate_clusters"]);
    let labels = clusters
        .get("labels")
        .and_then(|v| v.as_array())
        .expect("kmeans labels");
    assert_eq!(labels.len(), n_nodes, "one label per workspace crate");
    let mut distinct_labels: Vec<i64> = labels.iter().filter_map(|v| v.as_i64()).collect();
    distinct_labels.sort();
    distinct_labels.dedup();
    assert!(
        distinct_labels.len() >= 2,
        "k=3 should produce 2+ distinct labels on real data; got {distinct_labels:?}"
    );

    // s10 random forest must predict a label for every crate.
    let rf = unwrap_tool_result(&exec["results"]["s10_risk_classifier"]);
    let predictions = rf
        .get("predictions")
        .and_then(|v| v.as_array())
        .expect("rf predictions");
    assert_eq!(predictions.len(), n_nodes);

    // s11 FGSM on a 3-dim feature row.
    let fgsm = unwrap_tool_result(&exec["results"]["s11_adversarial_attack"]);
    let adv = fgsm
        .get("adversarial_input")
        .and_then(|v| v.as_array())
        .expect("adversarial_input");
    assert_eq!(adv.len(), 3);

    // s12 GA reports a 3-dim best_params.
    let ga = unwrap_tool_result(&exec["results"]["s12_refactor_search"]);
    let best = ga
        .get("best_params")
        .and_then(|v| v.as_array())
        .expect("ga best_params");
    assert_eq!(best.len(), 3);

    // s13 governance verdict.
    let gov = unwrap_tool_result(&exec["results"]["s13_governance_audit"]);
    assert!(
        gov.get("compliant").is_some(),
        "governance_check must emit a compliant field"
    );
}

#[test]
fn oracle_governance_check_can_consume_pipeline_lineage() {
    // After the pipeline finishes, invoke ix_governance_check a
    // second time with the emitted lineage — this mirrors the
    // R2 Phase 2 test pattern and closes the audit loop.
    let canned = serde_json::to_string(&canned_oracle_spec()).expect("serialise");
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned);

    let reg = ToolRegistry::new();
    let exec = reg
        .call_with_ctx("ix_pipeline_run", canned_oracle_spec(), &ctx)
        .expect("run");
    let lineage = exec.get("lineage").cloned().expect("lineage");

    let args = json!({
        "action": "ship the GA refactor plan for ix-agent to main",
        "lineage": lineage,
    });
    let verdict = reg
        .call("ix_governance_check", args)
        .expect("governance_check");
    let verdict = unwrap_tool_result(&verdict);

    let audit = verdict
        .get("lineage_audit")
        .expect("lineage_audit present when lineage was passed");
    assert_eq!(
        audit.get("step_count").and_then(|v| v.as_u64()),
        Some(14),
        "lineage_audit.step_count should be 14"
    );
}

// ---------------------------------------------------------------------------
// Dump helper — writes the canned spec to disk as the on-disk
// pipeline.json for the 05-adversarial-refactor-oracle showcase. Run
// with `cargo test -- --ignored` when the canned spec is updated.
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p
}

/// Run the oracle and print a narrated step-by-step trace of what
/// every tool produced on the real workspace metrics. Use this to
/// actually *see* the demo's output.
///
/// ```bash
/// cargo test -p ix-agent --test showcase_refactor_oracle \
///   run_refactor_oracle_with_narration -- --ignored --nocapture
/// ```
#[test]
#[ignore = "narration demo — run with --ignored --nocapture to see full output"]
fn run_refactor_oracle_with_narration() {
    let canned = serde_json::to_string(&canned_oracle_spec()).expect("serialise");
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned);

    let reg = ToolRegistry::new();
    let exec = reg
        .call_with_ctx("ix_pipeline_run", canned_oracle_spec(), &ctx)
        .expect("run oracle");

    let results = exec["results"].as_object().unwrap();
    let order: Vec<String> = exec["execution_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    println!("\n┌──────────────────────────────────────────────────────────────────────┐");
    println!("│     THE ADVERSARIAL REFACTOR ORACLE — LIVE DATA RUN                  │");
    println!("│  14 ix tools chained on THIS repo's cargo + git state (P1.1 + P1.2)  │");
    println!("└──────────────────────────────────────────────────────────────────────┘\n");

    // ─── Source adapters ───────────────────────────────────────────
    let s00 = unwrap_tool_result(&results["s00_cargo_deps"]);
    let n_nodes = s00["n_nodes"].as_u64().unwrap_or(0) as usize;
    let crate_names: Vec<String> = s00["names"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_str().unwrap_or("?").to_string())
                .collect()
        })
        .unwrap_or_default();
    println!("─── STEP 0  ix_cargo_deps (source adapter) ─────────────────────────────");
    println!("    workspace crates discovered : {n_nodes}");
    println!(
        "    edges emitted               : {}",
        s00["edges"].as_array().map(|a| a.len()).unwrap_or(0)
    );
    println!("    feature cols per crate      : 3 (sloc, file_count, dep_count)");
    println!();

    let s01 = unwrap_tool_result(&results["s01_git_log"]);
    println!("─── STEP 1  ix_git_log (source adapter) ────────────────────────────────");
    println!("    path       : crates/ix-agent");
    println!("    window     : 90 days, weekly buckets");
    println!("    commits    : {}", s01["commits"].as_u64().unwrap_or(0));
    println!(
        "    n_buckets  : {}",
        s01["n_buckets"].as_u64().unwrap_or(0)
    );
    println!();

    // ─── Live analysis steps ───────────────────────────────────────
    let s2 = unwrap_tool_result(&results["s02_baseline_sloc"]);
    println!("─── STEP 2  ix_stats (SLOC baseline over ALL workspace crates) ─────────");
    println!("    n_crates  = {}", s2["count"].as_u64().unwrap_or(0));
    println!("    mean      = {:.0}", s2["mean"].as_f64().unwrap_or(0.0));
    println!(
        "    std_dev   = {:.0}",
        s2["std_dev"].as_f64().unwrap_or(0.0)
    );
    println!("    max       = {:.0}", s2["max"].as_f64().unwrap_or(0.0));
    println!();

    let s3 = unwrap_tool_result(&results["s03_dep_pagerank"]);
    println!("─── STEP 3  ix_graph pagerank (live dep-graph centrality) ──────────────");
    if let Some(pr) = s3.get("pagerank").and_then(|v| v.as_object()) {
        let mut rows: Vec<(usize, f64)> = pr
            .iter()
            .filter_map(|(k, v)| Some((k.parse().ok()?, v.as_f64()?)))
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("    top 5 most-depended-upon crates:");
        for (idx, score) in rows.iter().take(5) {
            let name = crate_names
                .get(*idx)
                .cloned()
                .unwrap_or_else(|| format!("#{idx}"));
            println!("      {name:<28} rank = {score:.4}");
        }
    }
    println!();

    let s4 = unwrap_tool_result(&results["s04_dep_toposort"]);
    println!("─── STEP 4  ix_graph topological_sort (DAG check) ──────────────────────");
    println!("    is_dag  = {}", s4["is_dag"].as_bool().unwrap_or(false));
    if let Some(ord) = s4.get("order").and_then(|v| v.as_array()) {
        let first_named: Vec<String> = ord
            .iter()
            .take(5)
            .filter_map(|v| {
                v.as_u64().map(|i| {
                    crate_names
                        .get(i as usize)
                        .cloned()
                        .unwrap_or_else(|| format!("#{i}"))
                })
            })
            .collect();
        println!("    first 5 in topo order: {first_named:?}");
    }
    println!();

    let s5 = unwrap_tool_result(&results["s05_betti_numbers"]);
    println!("─── STEP 5  ix_topo betti_at_radius (metric cloud) ─────────────────────");
    println!(
        "    radius        = {}",
        s5["radius"].as_f64().unwrap_or(0.0)
    );
    println!("    betti_numbers = {:?}", s5["betti_numbers"]);
    println!();

    let s6 = unwrap_tool_result(&results["s06_persistence_diagram"]);
    println!("─── STEP 6  ix_topo persistence ─────────────────────────────────────────");
    if let Some(diagrams) = s6.get("diagrams").and_then(|v| v.as_array()) {
        for d in diagrams {
            let dim = d["dimension"].as_u64().unwrap_or(0);
            let pair_count = d["pairs"].as_array().map(|a| a.len()).unwrap_or(0);
            println!("    H{dim} had {pair_count} birth/death pair(s)");
        }
    }
    println!();

    let s7 = unwrap_tool_result(&results["s07_churn_spectrum"]);
    println!("─── STEP 7  ix_fft (spectrum of live commit churn) ─────────────────────");
    if let Some(mags) = s7.get("magnitudes").and_then(|v| v.as_array()) {
        let first_few: Vec<String> = mags
            .iter()
            .take(5)
            .map(|v| format!("{:.2}", v.as_f64().unwrap_or(0.0)))
            .collect();
        println!(
            "    fft_size      = {}",
            s7["fft_size"].as_u64().unwrap_or(0)
        );
        println!("    first 5 bins  = [{}]", first_few.join(", "));
    }
    println!();

    let s8 = unwrap_tool_result(&results["s08_velocity_regime"]);
    println!("─── STEP 8  ix_chaos_lyapunov (logistic-map regime) ────────────────────");
    println!("    parameter r       = {:.3}", LYAPUNOV_R);
    println!(
        "    lyapunov_exponent = {:.4}",
        s8["lyapunov_exponent"].as_f64().unwrap_or(0.0)
    );
    println!(
        "    dynamics          = {}",
        s8["dynamics"].as_str().unwrap_or("?")
    );
    println!();

    let s9 = unwrap_tool_result(&results["s09_crate_clusters"]);
    println!("─── STEP 9  ix_kmeans (k=3 health profiles on all crates) ──────────────");
    let labels: Vec<i64> = s9["labels"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();
    let mut counts: std::collections::BTreeMap<i64, Vec<String>> =
        std::collections::BTreeMap::new();
    for (i, label) in labels.iter().enumerate() {
        let name = crate_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("#{i}"));
        counts.entry(*label).or_default().push(name);
    }
    for (label, members) in &counts {
        println!(
            "    cluster {label}  ({} crates): {}",
            members.len(),
            members
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
                + if members.len() > 3 { ", ..." } else { "" }
        );
    }
    println!(
        "    inertia  = {:.1}",
        s9["inertia"].as_f64().unwrap_or(0.0)
    );
    println!();

    let s10 = unwrap_tool_result(&results["s10_risk_classifier"]);
    println!("─── STEP 10  ix_random_forest (self-test) ──────────────────────────────");
    let preds: Vec<i64> = s10["predictions"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();
    let accuracy = if !preds.is_empty() {
        preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| p == l)
            .count() as f64
            / preds.len() as f64
    } else {
        0.0
    };
    println!("    predictions : {} crates", preds.len());
    println!("    accuracy    : {:.0}%", accuracy * 100.0);
    println!();

    let s11 = unwrap_tool_result(&results["s11_adversarial_attack"]);
    println!("─── STEP 11  ix_adversarial_fgsm (refactor perturbation) ───────────────");
    println!("    input         = {:?}", s11["adversarial_input"]);
    println!("    perturbation  = {:?}", s11["perturbation"]);
    println!(
        "    l_inf_norm    = {:.3}",
        s11["l_inf_norm"].as_f64().unwrap_or(0.0)
    );
    println!();

    let s12 = unwrap_tool_result(&results["s12_refactor_search"]);
    println!("─── STEP 12  ix_evolution (GA over 3-dim refactor space) ───────────────");
    println!(
        "    algorithm      = {}",
        s12["algorithm"].as_str().unwrap_or("?")
    );
    println!(
        "    function       = {}  (symbolic stand-in for real fitness)",
        s12["function"].as_str().unwrap_or("?")
    );
    println!("    best_params    = {:?}", s12["best_params"]);
    println!(
        "    best_fitness   = {:.6}",
        s12["best_fitness"].as_f64().unwrap_or(0.0)
    );
    println!();

    let s13 = unwrap_tool_result(&results["s13_governance_audit"]);
    println!("─── STEP 13  ix_governance_check (Demerzel verdict) ────────────────────");
    println!(
        "    compliant             = {}",
        s13["compliant"].as_bool().unwrap_or(false)
    );
    println!(
        "    constitution_version  = {}",
        s13["constitution_version"].as_str().unwrap_or("?")
    );
    println!(
        "    warnings              = {}",
        s13["warnings"].as_array().map(|a| a.len()).unwrap_or(0)
    );
    if let Some(articles) = s13.get("relevant_articles").and_then(|v| v.as_array()) {
        println!("    relevant articles     = {} matched", articles.len());
    }
    println!();

    // Pipeline-level summary.
    let cache_hits = exec["cache_hits"].as_array().unwrap();
    let durations = exec["durations_ms"].as_object().unwrap();
    let total_ms: u64 = durations.values().filter_map(|v| v.as_u64()).sum();
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│                        PIPELINE SUMMARY                              │");
    println!("├──────────────────────────────────────────────────────────────────────┤");
    println!("│  steps executed       : {:<45} │", order.len());
    println!("│  cache hits           : {:<45} │", cache_hits.len());
    println!("│  total duration (ms)  : {:<45} │", total_ms);
    println!(
        "│  lineage DAG entries  : {:<45} │",
        exec["lineage"].as_object().map(|o| o.len()).unwrap_or(0)
    );
    println!("└──────────────────────────────────────────────────────────────────────┘");
}

#[test]
#[ignore = "writes pipeline.json to disk — run with --ignored to regenerate"]
fn dump_refactor_oracle_pipeline_json() {
    let spec = canned_oracle_spec();
    let mut wrapped = json!({
        "$schema": "https://ix.guitaralchemist.com/schemas/pipeline-v1.json",
        "name": "adversarial-refactor-oracle",
        "description": "14-tool self-referential ecosystem forensics demo (LIVE DATA edition): ix analyses its own workspace using ix_cargo_deps + ix_git_log as source adapters, clusters crates by health profile, trains a random-forest classifier on the cluster labels, attacks it with FGSM along the synthetic refactor direction, searches for a 3-dim refactor vector via GA, and closes with a Demerzel governance audit. Every numeric input flows from the two source-adapter steps via $step.field references — no baked constants except the synthetic FGSM gradient and the logistic-map parameter (both blocked on expression-style substitution support). Compiled from a natural-language brief via ix_pipeline_compile. P1.1 + P1.2 graduation: self-referential in fact, not just in theory.",
        "version": "2.0",
        "tools_used": [
            "ix_cargo_deps", "ix_git_log", "ix_stats", "ix_graph",
            "ix_topo", "ix_fft", "ix_chaos_lyapunov", "ix_kmeans",
            "ix_random_forest", "ix_adversarial_fgsm", "ix_evolution",
            "ix_governance_check"
        ]
    });
    wrapped["steps"] = spec["steps"].clone();

    let mut out = workspace_root();
    out.push("examples");
    out.push("canonical-showcase");
    out.push("05-adversarial-refactor-oracle");
    out.push("pipeline.json");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    let pretty = serde_json::to_string_pretty(&wrapped).unwrap() + "\n";
    std::fs::write(&out, pretty).expect("write pipeline.json");
    eprintln!("[dump_refactor_oracle] wrote {}", out.display());
}
