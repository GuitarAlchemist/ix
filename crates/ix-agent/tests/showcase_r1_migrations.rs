//! R1 migration regression tests for the three remaining canonical
//! showcase demos: chaos-detective, governance-gauntlet, sprint-oracle.
//!
//! Each demo's `pipeline.json` is generated from the hand-chained
//! `DemoScenario` via `dump_showcase_pipelines.rs`, so bit-identity to
//! the originals is already guaranteed by construction for the argument
//! payloads. What these tests verify is that the pipeline specs
//! successfully execute end-to-end through `ix_pipeline_run`, preserve
//! the expected structural outputs (step count, asset-backed cache
//! keys), and — for deterministic demos — capture the original "aha"
//! signal (e.g. positive Lyapunov exponent for chaos-detective).

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::Value;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p
}

fn load_spec(folder: &str) -> Value {
    let mut path = workspace_root();
    path.push("examples");
    path.push("canonical-showcase");
    path.push(folder);
    path.push("pipeline.json");
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut spec: Value = serde_json::from_str(&raw).expect("valid JSON");
    let steps = spec
        .get_mut("steps")
        .map(|v| v.take())
        .expect("spec has 'steps'");
    serde_json::json!({ "steps": steps })
}

fn make_ctx() -> ServerContext {
    let (ctx, _rx) = ServerContext::new();
    ctx
}

fn run_pipeline(folder: &str) -> Value {
    let reg = ToolRegistry::new();
    let ctx = make_ctx();
    reg.call_with_ctx("ix_pipeline_run", load_spec(folder), &ctx)
        .unwrap_or_else(|e| panic!("pipeline_run failed for {folder}: {e}"))
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

fn assert_lineage_well_formed(result: &Value, expected_step_count: usize) {
    let lineage = result
        .get("lineage")
        .and_then(|v| v.as_object())
        .expect("lineage present");
    assert_eq!(lineage.len(), expected_step_count, "lineage entry per step");
    for (id, entry) in lineage {
        let tool = entry.get("tool").and_then(|v| v.as_str());
        assert!(tool.is_some(), "lineage['{id}'] missing tool");
        assert!(
            entry.get("depends_on").is_some(),
            "lineage['{id}'] missing depends_on"
        );
        assert!(
            entry.get("upstream_cache_keys").is_some(),
            "lineage['{id}'] missing upstream_cache_keys"
        );
        let deps = entry.get("depends_on").and_then(|v| v.as_array()).unwrap();
        let ups = entry
            .get("upstream_cache_keys")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(
            deps.len(),
            ups.len(),
            "lineage['{id}']: depends_on and upstream_cache_keys must have equal length"
        );
    }
}

fn assert_all_cache_keys_set(result: &Value, expected_step_count: usize) {
    let order = result
        .get("execution_order")
        .and_then(|v| v.as_array())
        .expect("execution_order present");
    assert_eq!(
        order.len(),
        expected_step_count,
        "execution_order length mismatch"
    );

    let cache_keys = result
        .get("cache_keys")
        .and_then(|v| v.as_object())
        .expect("cache_keys present");
    for step in order {
        let id = step.as_str().expect("step id is string");
        let key = cache_keys
            .get(id)
            .unwrap_or_else(|| panic!("cache_keys missing '{id}'"));
        assert!(
            key.as_str()
                .is_some_and(|k| k.starts_with("ix_pipeline_run:")),
            "step '{id}' expected asset-backed cache key, got {key:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Chaos detective — deterministic, checks positive Lyapunov exponent.
// ---------------------------------------------------------------------------

#[test]
fn chaos_detective_replays_via_pipeline_run() {
    let result = run_pipeline("02-chaos-detective");
    assert_all_cache_keys_set(&result, 4);
    assert_lineage_well_formed(&result, 4);

    let order: Vec<String> = result
        .get("execution_order")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    // Lyapunov step is the third one.
    let lyapunov_id = order
        .iter()
        .find(|id| id.contains("lyapunov"))
        .expect("lyapunov step id");
    let lyapunov = result
        .get("results")
        .and_then(|r| r.get(lyapunov_id.as_str()))
        .expect("lyapunov result");
    let lyapunov = unwrap_tool_result(lyapunov);
    let exponent = lyapunov
        .get("lyapunov_exponent")
        .and_then(|v| v.as_f64())
        .expect("lyapunov_exponent field");
    assert!(
        exponent > 0.0,
        "chaos-detective should find positive Lyapunov exponent (deterministic chaos), got {exponent}"
    );
}

#[test]
fn chaos_detective_second_run_hits_cache() {
    let _first = run_pipeline("02-chaos-detective");
    let second = run_pipeline("02-chaos-detective");
    let hits = second
        .get("cache_hits")
        .and_then(|v| v.as_array())
        .expect("cache_hits");
    assert_eq!(hits.len(), 4, "all 4 steps should be cache hits on replay");
}

// ---------------------------------------------------------------------------
// Governance gauntlet — smoke-check, structural only.
// ---------------------------------------------------------------------------

#[test]
fn governance_gauntlet_replays_via_pipeline_run() {
    let result = run_pipeline("03-governance-gauntlet");
    assert_all_cache_keys_set(&result, 5);
    assert_lineage_well_formed(&result, 5);

    // Every step result must exist and be non-error.
    let results = result
        .get("results")
        .and_then(|v| v.as_object())
        .expect("results map");
    assert_eq!(results.len(), 5);
    for (id, r) in results {
        let unwrapped = unwrap_tool_result(r);
        assert!(
            unwrapped.get("error").is_none(),
            "step '{id}' returned error: {unwrapped}"
        );
    }
}

#[test]
fn governance_gauntlet_second_run_hits_cache() {
    let _first = run_pipeline("03-governance-gauntlet");
    let second = run_pipeline("03-governance-gauntlet");
    let hits = second
        .get("cache_hits")
        .and_then(|v| v.as_array())
        .expect("cache_hits");
    assert_eq!(hits.len(), 5);
}

// ---------------------------------------------------------------------------
// Sprint oracle — smoke-check; regression-on-trend for the linreg step.
// ---------------------------------------------------------------------------

#[test]
fn sprint_oracle_replays_via_pipeline_run() {
    let result = run_pipeline("04-sprint-oracle");
    assert_all_cache_keys_set(&result, 4);
    assert_lineage_well_formed(&result, 4);

    let order: Vec<String> = result
        .get("execution_order")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    // Linear regression step should produce a positive slope
    // (sprint velocity is trending up over 12 sprints).
    let linreg_id = order
        .iter()
        .find(|id| id.contains("trend_line"))
        .cloned()
        .expect("linreg step id");
    let linreg = result
        .get("results")
        .and_then(|r| r.get(linreg_id.as_str()))
        .expect("linreg result");
    let linreg = unwrap_tool_result(linreg);
    let slope = linreg
        .get("weights")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_f64())
        .expect("linreg weights");
    assert!(
        slope > 0.0,
        "sprint velocity should trend up, got slope={slope}"
    );
}

// ---------------------------------------------------------------------------
// ix_pipeline_list — discovers all 4 showcase pipelines.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// R2 Phase 2 — lineage DAG has correct upstream_cache_keys on chained steps.
// ---------------------------------------------------------------------------

#[test]
fn governance_check_consumes_pipeline_lineage() {
    // Run a pipeline to produce a lineage map, then feed that map into
    // ix_governance_check and assert the audit trail round-trips into
    // the response as `lineage_audit`.
    let reg = ToolRegistry::new();
    let ctx = make_ctx();
    let pipeline_out = reg
        .call_with_ctx("ix_pipeline_run", load_spec("04-sprint-oracle"), &ctx)
        .expect("pipeline_run");
    let lineage = pipeline_out
        .get("lineage")
        .cloned()
        .expect("lineage present in pipeline output");

    let args = serde_json::json!({
        "action": "commit the sprint forecast to the backlog dashboard",
        "lineage": lineage,
    });
    let result = reg
        .call("ix_governance_check", args)
        .expect("governance_check call");
    let result = unwrap_tool_result(&result);

    let audit = result
        .get("lineage_audit")
        .expect("lineage_audit present when lineage was passed");
    assert_eq!(
        audit.get("step_count").and_then(|v| v.as_u64()),
        Some(4),
        "lineage_audit.step_count should reflect 4 pipeline steps"
    );
    let steps = audit
        .get("steps")
        .and_then(|v| v.as_array())
        .expect("steps array");
    assert_eq!(steps.len(), 4);
    for entry in steps {
        assert!(entry.get("step_id").is_some());
        assert!(entry.get("tool").is_some());
        assert!(entry.get("upstream_cache_keys").is_some());
    }
}

#[test]
fn sprint_oracle_lineage_walks_upstream_cache_keys() {
    let result = run_pipeline("04-sprint-oracle");
    assert_lineage_well_formed(&result, 4);

    let order: Vec<String> = result
        .get("execution_order")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    let lineage = result.get("lineage").unwrap();
    let cache_keys = result.get("cache_keys").unwrap();

    // First step has no upstream; every subsequent step must have
    // exactly one upstream cache_key equal to its parent's cache_key.
    for (i, id) in order.iter().enumerate() {
        let entry = lineage.get(id).unwrap();
        let deps = entry.get("depends_on").and_then(|v| v.as_array()).unwrap();
        if i == 0 {
            assert!(deps.is_empty(), "first step has no dependencies");
        } else {
            assert_eq!(
                deps.len(),
                1,
                "step {id} should depend on exactly one parent"
            );
            let parent = deps[0].as_str().unwrap();
            let parent_key = cache_keys.get(parent).unwrap();
            let upstream = entry
                .get("upstream_cache_keys")
                .and_then(|v| v.as_array())
                .unwrap();
            assert_eq!(upstream.len(), 1);
            assert_eq!(
                &upstream[0], parent_key,
                "step {id}: upstream_cache_keys must equal parent '{parent}' cache_key"
            );
        }
    }
}

#[test]
fn pipeline_list_discovers_canonical_showcase() {
    let reg = ToolRegistry::new();
    let mut root = workspace_root();
    root.push("examples");
    root.push("canonical-showcase");

    let args = serde_json::json!({ "root": root.display().to_string() });
    let result = reg
        .call("ix_pipeline_list", args)
        .expect("pipeline_list call");
    let result = unwrap_tool_result(&result);

    let pipelines = result
        .get("pipelines")
        .and_then(|v| v.as_array())
        .expect("pipelines array");
    let names: Vec<&str> = pipelines
        .iter()
        .filter_map(|p| p.get("name").and_then(|n| n.as_str()))
        .collect();
    for expected in [
        "cost-anomaly-hunter",
        "chaos-detective",
        "governance-gauntlet",
        "sprint-oracle",
    ] {
        assert!(
            names.contains(&expected),
            "expected pipeline '{expected}' in {names:?}"
        );
    }

    // Every entry must have a positive step_count and at least one tool.
    for p in pipelines {
        let step_count = p.get("step_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let tools = p.get("tools").and_then(|v| v.as_array()).unwrap();
        let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        assert!(step_count > 0, "pipeline '{name}' has zero steps");
        assert!(!tools.is_empty(), "pipeline '{name}' lists no tools");
    }
}

#[test]
fn sprint_oracle_second_run_hits_cache() {
    let _first = run_pipeline("04-sprint-oracle");
    let second = run_pipeline("04-sprint-oracle");
    let hits = second
        .get("cache_hits")
        .and_then(|v| v.as_array())
        .expect("cache_hits");
    assert_eq!(hits.len(), 4);
}
