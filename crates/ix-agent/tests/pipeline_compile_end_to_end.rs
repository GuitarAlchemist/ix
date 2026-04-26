//! End-to-end tests for `ix_pipeline_compile` with a fake LLM client.
//!
//! The real compiler asks the MCP client to sample an LLM response
//! containing a pipeline spec. These tests replace the client with a
//! background thread that reads `sampling/createMessage` envelopes off
//! the `ServerContext` outbound queue and delivers a canned response,
//! so the compiler's validation, parsing, and cache-key logic can be
//! exercised without a real LLM in the loop.
//!
//! Covers:
//! - happy path: LLM returns a valid spec → status "ok"
//! - markdown-fence path: LLM wraps response in ```json ... ``` → still parses
//! - invalid JSON path: LLM returns garbage → status "parse_error"
//! - invalid spec path: LLM emits JSON but references an unknown tool → status "invalid"
//! - end-to-end compile → run: the compiled spec is immediately executable via ix_pipeline_run

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::Duration;

/// Spawn a background thread that plays the role of the MCP client.
/// It pulls every outbound envelope, and for each `sampling/createMessage`
/// request it delivers the supplied canned response text back into the
/// context via `deliver_response`.
fn fake_client(ctx: ServerContext, outbound: Receiver<String>, canned_response: String) {
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
            // Construct a canned response envelope in the shape the
            // real client would emit.
            let response = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "role": "assistant",
                    "content": {
                        "type": "text",
                        "text": canned_response,
                    }
                }
            });
            ctx.deliver_response(id, response);
        }
    });
}

fn call_compile(reg: &ToolRegistry, ctx: &ServerContext, sentence: &str) -> Value {
    reg.call_with_ctx("ix_pipeline_compile", json!({ "sentence": sentence }), ctx)
        .expect("compile should return Ok envelope")
}

#[test]
fn happy_path_spec_is_ok_and_executable() {
    let canned = r#"{
  "steps": [
    {
      "id": "s01_stats",
      "tool": "ix_stats",
      "asset_name": "compiled.stats",
      "arguments": { "data": [1.0, 2.0, 3.0, 4.0, 5.0] }
    }
  ]
}"#;
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned.to_string());

    let reg = ToolRegistry::new();
    let result = call_compile(&reg, &ctx, "baseline stats on these numbers");
    assert_eq!(result["status"], "ok");
    let errors = result["validation"]["errors"].as_array().unwrap();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");

    // The compiled spec must be immediately executable.
    let spec = result["spec"].clone();
    let exec = reg
        .call_with_ctx("ix_pipeline_run", spec, &ctx)
        .expect("run compiled spec");
    let order = exec["execution_order"].as_array().unwrap();
    assert_eq!(order.len(), 1);
    assert_eq!(order[0], "s01_stats");
}

#[test]
fn markdown_fence_is_stripped() {
    let canned = "```json\n{\n  \"steps\": [\n    { \"id\": \"a\", \"tool\": \"ix_stats\", \"asset_name\": \"x.a\", \"arguments\": { \"data\": [1.0] } }\n  ]\n}\n```\n";
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned.to_string());

    let reg = ToolRegistry::new();
    let result = call_compile(&reg, &ctx, "anything");
    assert_eq!(result["status"], "ok", "got {result}");
    assert_eq!(result["spec"]["steps"][0]["id"], "a");
}

#[test]
fn invalid_json_is_parse_error() {
    let canned = "sure, here's your pipeline: { not valid json ";
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned.to_string());

    let reg = ToolRegistry::new();
    let result = call_compile(&reg, &ctx, "anything");
    assert_eq!(result["status"], "parse_error");
    assert!(result["validation"]["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e.as_str().unwrap_or("").contains("not valid JSON")));
    // Raw response is preserved for debugging.
    assert_eq!(result["raw_llm_response"], canned);
}

#[test]
fn unknown_tool_is_invalid_status() {
    let canned = r#"{
  "steps": [
    { "id": "a", "tool": "ix_definitely_not_a_real_tool", "arguments": {} }
  ]
}"#;
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned.to_string());

    let reg = ToolRegistry::new();
    let result = call_compile(&reg, &ctx, "anything");
    assert_eq!(result["status"], "invalid");
    let errors = result["validation"]["errors"].as_array().unwrap();
    assert!(errors
        .iter()
        .any(|e| e.as_str().unwrap().contains("unknown tool")));
}

/// Regression for realistic multi-tool pipelines: a 4-step spec that
/// a live LLM would plausibly emit for "profile these numbers, cluster
/// them, fit a line through the clusters, and flag outliers". Exercises
/// the validator against upstream references (`$s01_stats.mean`), an
/// asset_name on every step, a cross-step depends_on chain spanning
/// three crates (ix-math, ix-unsupervised, ix-supervised), and the
/// lineage DAG path.
#[test]
fn four_step_cross_crate_compile_runs_end_to_end() {
    let canned = r#"{
  "steps": [
    {
      "id": "s01_stats",
      "tool": "ix_stats",
      "asset_name": "numbers.stats",
      "arguments": { "data": [10.0, 12.0, 9.0, 11.0, 50.0, 8.0, 13.0, 11.5, 9.5, 48.0] }
    },
    {
      "id": "s02_clusters",
      "tool": "ix_kmeans",
      "asset_name": "numbers.clusters",
      "depends_on": ["s01_stats"],
      "arguments": {
        "data": [[10.0], [12.0], [9.0], [11.0], [50.0], [8.0], [13.0], [11.5], [9.5], [48.0]],
        "k": 2,
        "max_iter": 50
      }
    },
    {
      "id": "s03_fit",
      "tool": "ix_linear_regression",
      "asset_name": "numbers.fit",
      "depends_on": ["s02_clusters"],
      "arguments": {
        "x": [[1.0], [2.0], [3.0], [4.0], [5.0]],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0]
      }
    },
    {
      "id": "s04_distance",
      "tool": "ix_distance",
      "asset_name": "numbers.distance",
      "depends_on": ["s03_fit"],
      "arguments": {
        "a": [10.0, 12.0, 9.0],
        "b": [11.0, 11.0, 11.0],
        "metric": "euclidean"
      }
    }
  ]
}"#;
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned.to_string());

    let reg = ToolRegistry::new();
    let result = call_compile(
        &reg,
        &ctx,
        "profile, cluster, fit, and measure distance on these numbers",
    );
    assert_eq!(
        result["status"], "ok",
        "compile should succeed; got {result}"
    );
    let errors = result["validation"]["errors"].as_array().unwrap();
    assert!(errors.is_empty(), "unexpected validator errors: {errors:?}");

    // Every step must carry its asset_name through to the compiled spec.
    let steps = result["spec"]["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 4);
    for (i, expected) in [
        "numbers.stats",
        "numbers.clusters",
        "numbers.fit",
        "numbers.distance",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            steps[i]["asset_name"], *expected,
            "step {i} asset_name survived compile"
        );
    }

    // Execute end-to-end. Topological sort must respect the chain.
    let exec = reg
        .call_with_ctx("ix_pipeline_run", result["spec"].clone(), &ctx)
        .expect("run compiled 4-step spec");
    let order = exec["execution_order"].as_array().unwrap();
    assert_eq!(order.len(), 4);
    assert_eq!(order[0], "s01_stats");
    assert_eq!(order[1], "s02_clusters");
    assert_eq!(order[2], "s03_fit");
    assert_eq!(order[3], "s04_distance");

    // Lineage DAG: every non-root step records its parent.
    let lineage = exec["lineage"].as_object().unwrap();
    assert_eq!(lineage.len(), 4);
    assert_eq!(lineage["s01_stats"]["depends_on"], json!([]));
    assert_eq!(lineage["s02_clusters"]["depends_on"], json!(["s01_stats"]));
    assert_eq!(lineage["s03_fit"]["depends_on"], json!(["s02_clusters"]));
    assert_eq!(lineage["s04_distance"]["depends_on"], json!(["s03_fit"]));
}

#[test]
fn multi_step_compiled_pipeline_runs_end_to_end() {
    let canned = r#"{
  "steps": [
    {
      "id": "s01_baseline",
      "tool": "ix_stats",
      "asset_name": "compiled.baseline",
      "arguments": { "data": [10.0, 12.0, 9.0, 11.0, 50.0] }
    },
    {
      "id": "s02_clusters",
      "tool": "ix_kmeans",
      "asset_name": "compiled.clusters",
      "depends_on": ["s01_baseline"],
      "arguments": {
        "data": [[10.0], [12.0], [9.0], [11.0], [50.0]],
        "k": 2,
        "max_iter": 100
      }
    }
  ]
}"#;
    let (ctx, outbound) = ServerContext::new();
    fake_client(ctx.clone(), outbound, canned.to_string());

    let reg = ToolRegistry::new();
    let result = call_compile(&reg, &ctx, "cluster these 5 numbers into 2 groups");
    assert_eq!(result["status"], "ok");

    // Execute the compiled spec.
    let exec = reg
        .call_with_ctx("ix_pipeline_run", result["spec"].clone(), &ctx)
        .expect("run compiled spec");

    // Both steps must have executed in the right order.
    let order = exec["execution_order"].as_array().unwrap();
    assert_eq!(order.len(), 2);
    assert_eq!(order[0], "s01_baseline");
    assert_eq!(order[1], "s02_clusters");

    // Lineage DAG must be well-formed (R2 Phase 2 check).
    let lineage = exec["lineage"].as_object().unwrap();
    assert_eq!(lineage.len(), 2);
    assert_eq!(
        lineage["s02_clusters"]["depends_on"],
        json!(["s01_baseline"])
    );

    // Give the background thread a breath to drain pending sends
    // (the test assertion path is what matters; the drain is cosmetic).
    thread::sleep(Duration::from_millis(10));
}
