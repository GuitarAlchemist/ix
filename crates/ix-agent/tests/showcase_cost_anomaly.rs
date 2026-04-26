//! Canonical showcase validation test — cost anomaly hunter demo.
//!
//! This test is the proof-point for R1 (`ix_pipeline_run`) + R2 Phase 1
//! (content-addressed caching): it loads the demo spec from
//! `examples/canonical-showcase/01-cost-anomaly-hunter/pipeline.json`,
//! runs it through `ToolRegistry::call_with_ctx`, and asserts that the
//! same 3 anomaly days are flagged as the hand-chained version in the
//! original HTML dashboard (days 23, 52, 71 — the injected incidents).
//!
//! A second run of the same spec must hit the cache on every step,
//! validating that R2 Phase 1 delivers the "replay is fast" promise.

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::Value;
use std::path::PathBuf;

fn spec_path() -> PathBuf {
    // Tests run with CWD = crate root (ix-agent), so walk up to the
    // workspace root and into the canonical-showcase folder.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.push("examples");
    p.push("canonical-showcase");
    p.push("01-cost-anomaly-hunter");
    p.push("pipeline.json");
    p
}

fn load_spec() -> Value {
    let path = spec_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut spec: Value = serde_json::from_str(&raw).expect("valid JSON");
    // Strip out the metadata fields the pipeline runner does not consume,
    // leaving only { steps: [...] }.
    let steps = spec
        .get_mut("steps")
        .map(|v| v.take())
        .expect("spec has steps");
    serde_json::json!({ "steps": steps })
}

fn make_ctx() -> ServerContext {
    let (ctx, _rx) = ServerContext::new();
    ctx
}

/// Run the cost-anomaly-hunter pipeline end-to-end and assert the
/// 3 expected anomaly days are clustered into the small cluster.
#[test]
fn cost_anomaly_hunter_replays_via_pipeline_run() {
    let reg = ToolRegistry::new();
    let ctx = make_ctx();

    let spec = load_spec();
    let result = reg
        .call_with_ctx("ix_pipeline_run", spec, &ctx)
        .expect("pipeline_run failed");

    // --- Execution order ---
    let order = result
        .get("execution_order")
        .and_then(|v| v.as_array())
        .expect("execution_order present");
    assert_eq!(order.len(), 3);
    assert_eq!(order[0], "baseline");
    assert_eq!(order[1], "spectrum");
    assert_eq!(order[2], "anomalies");

    // --- Baseline stats ---
    let baseline = result
        .get("results")
        .and_then(|r| r.get("baseline"))
        .expect("baseline result");
    let mean = extract_scalar(baseline, "mean").expect("mean");
    assert!(
        (mean - 993.87).abs() < 1.0,
        "baseline mean expected ~993.87, got {mean}"
    );

    // --- FFT spectrum exists and has the right size ---
    let spectrum = result
        .get("results")
        .and_then(|r| r.get("spectrum"))
        .expect("spectrum result");
    let mags = extract_array(spectrum, "magnitudes").expect("magnitudes");
    assert_eq!(
        mags.len(),
        128,
        "FFT magnitudes should be 128 (zero-padded)"
    );

    // --- k-means clusters ---
    // The 90 daily costs cluster into 2 groups: normal (~87 days) and
    // anomaly (3 days). Verify labels match the expected [23, 52, 71].
    let kmeans = result
        .get("results")
        .and_then(|r| r.get("anomalies"))
        .expect("anomalies result");
    let labels = extract_array(kmeans, "labels").expect("kmeans labels");
    let centroids = extract_matrix(kmeans, "centroids").expect("centroids");

    // Identify which cluster index is the "high-cost" anomaly cluster
    // by finding the centroid with the higher mean value.
    assert_eq!(centroids.len(), 2, "k=2 clusters");
    let anomaly_cluster_idx = if centroids[0][0] > centroids[1][0] {
        0
    } else {
        1
    };

    let anomaly_days: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            v.as_u64().and_then(|l| {
                if l as usize == anomaly_cluster_idx {
                    Some(i)
                } else {
                    None
                }
            })
        })
        .collect();

    assert_eq!(
        anomaly_days,
        vec![23, 52, 71],
        "anomaly days should match the injected incidents"
    );

    // --- Cache keys populated (R2 Phase 1) ---
    let cache_keys = result
        .get("cache_keys")
        .and_then(|v| v.as_object())
        .expect("cache_keys present");
    for id in ["baseline", "spectrum", "anomalies"] {
        let k = cache_keys
            .get(id)
            .unwrap_or_else(|| panic!("cache_keys missing '{id}'"));
        assert!(
            k.is_string() && k.as_str().unwrap().starts_with("ix_pipeline_run:"),
            "step '{id}' should have a content-addressed cache key"
        );
    }

    // --- Durations recorded ---
    let durations = result
        .get("durations_ms")
        .and_then(|v| v.as_object())
        .expect("durations_ms present");
    assert_eq!(durations.len(), 3);
}

/// Second run must hit the cache on every step. `cache_hits` should
/// contain all 3 step IDs.
#[test]
fn cost_anomaly_hunter_second_run_hits_cache() {
    let reg = ToolRegistry::new();
    let ctx = make_ctx();

    // Prime the cache.
    let _first = reg
        .call_with_ctx("ix_pipeline_run", load_spec(), &ctx)
        .expect("first run");

    // Second run — every step should be a cache hit.
    let second = reg
        .call_with_ctx("ix_pipeline_run", load_spec(), &ctx)
        .expect("second run");

    let hits = second
        .get("cache_hits")
        .and_then(|v| v.as_array())
        .expect("cache_hits");

    // All 3 steps should be cache hits on the replay.
    let hit_ids: Vec<&str> = hits.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(hit_ids.len(), 3, "all 3 steps should hit cache on replay");
    assert!(hit_ids.contains(&"baseline"));
    assert!(hit_ids.contains(&"spectrum"));
    assert!(hit_ids.contains(&"anomalies"));

    // Durations should be 0 for cached steps.
    let durations = second
        .get("durations_ms")
        .and_then(|v| v.as_object())
        .expect("durations_ms");
    for id in ["baseline", "spectrum", "anomalies"] {
        let d = durations.get(id).and_then(|v| v.as_u64()).unwrap();
        assert_eq!(d, 0, "cached step '{id}' should report 0ms duration");
    }
}

// ---------------------------------------------------------------------------
// Helpers: robustly extract fields from tool results, handling both the
// bare JSON object form and the MCP content envelope form.
// ---------------------------------------------------------------------------

fn unwrap_tool_result(v: &Value) -> Value {
    // Bare object (what the direct handlers return)
    if v.is_object() && !v.get("content").map(|c| c.is_array()).unwrap_or(false) {
        return v.clone();
    }
    // MCP content envelope: { content: [{ type: "text", text: "{...json...}" }] }
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

fn extract_scalar(v: &Value, key: &str) -> Option<f64> {
    unwrap_tool_result(v).get(key).and_then(|x| x.as_f64())
}

fn extract_array(v: &Value, key: &str) -> Option<Vec<Value>> {
    unwrap_tool_result(v)
        .get(key)
        .and_then(|x| x.as_array().cloned())
}

fn extract_matrix(v: &Value, key: &str) -> Option<Vec<Vec<f64>>> {
    let raw = unwrap_tool_result(v)
        .get(key)
        .and_then(|x| x.as_array().cloned())?;
    let mut out = Vec::with_capacity(raw.len());
    for row in raw {
        let row_vec: Vec<f64> = row.as_array()?.iter().filter_map(|v| v.as_f64()).collect();
        out.push(row_vec);
    }
    Some(out)
}
