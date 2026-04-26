//! R7 Week 2 kickoff — integration test for `ix_autograd_run`.
//!
//! Proves that the autograd crate is reachable from the MCP layer
//! end-to-end: the caller submits a single JSON request naming a
//! differentiable tool and its inputs, and receives both the
//! forward outputs and per-input gradients in one response.
//!
//! This is the path to pipeline-level gradient descent: an external
//! caller (Python client, notebook, another IX pipeline) can poll
//! ix_autograd_run in a loop, maintain Adam state outside the MCP
//! boundary, and train a model without re-marshaling the tape.

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};

fn make_ctx() -> ServerContext {
    let (ctx, _rx) = ServerContext::new();
    ctx
}

fn extract_scalar(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| {
        v.as_array()
            .and_then(|a| a.first().and_then(|x| x.as_f64()))
    })
}

#[test]
fn autograd_run_linear_regression_forward_and_backward() {
    let reg = ToolRegistry::new();
    let ctx = make_ctx();

    // x: [5, 3], w: [3, 1], b: [1, 1], y: [5, 1]
    let args = json!({
        "tool": "linear_regression",
        "inputs": {
            "x": [
                [0.1, 0.2, 0.3],
                [0.4, -0.1, 0.5],
                [-0.2, 0.6, 0.1],
                [0.3, 0.2, -0.4],
                [0.5, -0.3, 0.2]
            ],
            "w": [[0.7], [-0.5], [0.3]],
            "b": [[0.1]],
            "y": [[0.2], [0.1], [-0.1], [0.3], [0.0]]
        }
    });

    let result = reg
        .call_with_ctx("ix_autograd_run", args, &ctx)
        .expect("autograd_run failed");

    // --- tool name echoed ---
    assert_eq!(
        result.get("tool").and_then(|v| v.as_str()),
        Some("linear_regression")
    );

    // --- forward outputs present ---
    let forward = result
        .get("forward")
        .and_then(|v| v.as_object())
        .expect("forward object");
    assert!(forward.contains_key("y_hat"));
    assert!(forward.contains_key("loss"));

    // Loss is a scalar (rank-0 tensor serialized as f64)
    let loss = forward
        .get("loss")
        .and_then(extract_scalar)
        .expect("loss scalar");
    assert!(loss.is_finite());
    assert!(loss >= 0.0, "MSE loss should be non-negative");

    // --- gradients present for all three trainable parameters, not y ---
    let grads = result
        .get("gradients")
        .and_then(|v| v.as_object())
        .expect("gradients object");
    assert!(grads.contains_key("x"));
    assert!(grads.contains_key("w"));
    assert!(grads.contains_key("b"));
    assert!(
        !grads.contains_key("y"),
        "y is a target, should not receive a gradient"
    );

    // w gradient should be a 3x1 matrix (same shape as w input)
    let w_grad = grads.get("w").and_then(|v| v.as_array()).expect("w array");
    assert_eq!(w_grad.len(), 3);
    let first_row = w_grad[0].as_array().expect("w[0] array");
    assert_eq!(first_row.len(), 1);
}

#[test]
fn autograd_run_stats_variance_forward_and_backward() {
    let reg = ToolRegistry::new();
    let ctx = make_ctx();

    // Variance of 6 numbers; output is scalar, gradient shape == input shape.
    let args = json!({
        "tool": "stats_variance",
        "inputs": {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        }
    });

    let result = reg
        .call_with_ctx("ix_autograd_run", args, &ctx)
        .expect("autograd_run failed");

    let forward = result
        .get("forward")
        .and_then(|v| v.as_object())
        .expect("forward object");
    let variance = forward
        .get("variance")
        .and_then(extract_scalar)
        .expect("variance scalar");
    // Population variance of 1..=6 = mean((v - 3.5)^2) = 17.5/6 ≈ 2.916667
    assert!((variance - 2.916666_f64).abs() < 1e-5);

    let grads = result
        .get("gradients")
        .and_then(|v| v.as_object())
        .expect("gradients object");
    let x_grad = grads.get("x").and_then(|v| v.as_array()).expect("x grad");
    assert_eq!(x_grad.len(), 6);
    // Each element of the gradient should be finite.
    for (i, v) in x_grad.iter().enumerate() {
        let f = v.as_f64().unwrap_or_else(|| panic!("x_grad[{i}] non-f64"));
        assert!(f.is_finite(), "x_grad[{i}] = {f}");
    }
}

#[test]
fn autograd_run_rejects_unknown_tool() {
    let reg = ToolRegistry::new();
    let ctx = make_ctx();
    let args = json!({
        "tool": "bogus_tool",
        "inputs": { "x": [1.0, 2.0] }
    });
    let result = reg.call_with_ctx("ix_autograd_run", args, &ctx);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("unknown tool"));
}

#[test]
fn autograd_run_listed_in_tools() {
    let reg = ToolRegistry::new();
    let list = reg.list();
    let tools = list.get("tools").and_then(|v| v.as_array()).expect("tools");
    let found = tools
        .iter()
        .any(|t| t.get("name").and_then(|n| n.as_str()) == Some("ix_autograd_run"));
    assert!(found, "ix_autograd_run should appear in tools/list");
}
