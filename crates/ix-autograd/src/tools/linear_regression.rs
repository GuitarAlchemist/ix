//! Linear regression as a `DifferentiableTool`.
//!
//! Day 2 of Week 1 per examples/canonical-showcase/ix-roadmap-plan-v1.md §5.
//!
//! # Design note
//!
//! A production MSE loss would be `sum((y_hat - y) ^ 2) / n`. That needs
//! subtraction and scalar division, neither of which are in Day 2's
//! primitive set (add, mul, sum, matmul only). To stay strict to the
//! Day 2 scope and still exercise the full chain (matmul → add →
//! elementwise mul → sum), this tool uses:
//!
//! ```text
//!     y_hat = x @ w + b          // matmul + broadcast add
//!     loss  = sum(y_hat * y_hat) // mul + sum
//! ```
//!
//! This is a differentiable quadratic in the parameters and is
//! sufficient to verify that the autograd machinery produces correct
//! gradients for a real chained pipeline. Day 3 will upgrade to a full
//! MSE loss once `sub` and scalar `div` land.

use crate::ops;
use crate::tape::{DiffContext, TensorHandle};
use crate::tensor::Tensor;
use crate::tool::{DifferentiableTool, ValueMap};
use crate::{AutogradError, Result};

pub struct LinearRegressionTool;

impl LinearRegressionTool {
    /// Build the forward graph on the tape and return handles for the
    /// three inputs and the scalar loss. Useful for tests that want to
    /// run `backward` directly without going through the `ValueMap`
    /// boundary.
    pub fn build_graph(
        ctx: &mut DiffContext,
        x: Tensor,
        w: Tensor,
        b: Tensor,
    ) -> Result<LinregHandles> {
        let x_h = ops::input(ctx, x);
        let w_h = ops::input(ctx, w);
        let b_h = ops::input(ctx, b);

        // y_hat = x @ w + b   (b is [1, 1], broadcasts to [n, 1])
        let xw = ops::matmul(ctx, x_h, w_h)?;
        let y_hat = ops::add(ctx, xw, b_h)?;

        // loss = sum(y_hat * y_hat)
        let sq = ops::mul(ctx, y_hat, y_hat)?;
        let loss = ops::sum(ctx, sq)?;

        Ok(LinregHandles {
            x: x_h,
            w: w_h,
            b: b_h,
            y_hat,
            loss,
        })
    }
}

/// Handles returned by `build_graph` for downstream use.
#[derive(Debug, Clone, Copy)]
pub struct LinregHandles {
    pub x: TensorHandle,
    pub w: TensorHandle,
    pub b: TensorHandle,
    pub y_hat: TensorHandle,
    pub loss: TensorHandle,
}

impl DifferentiableTool for LinearRegressionTool {
    fn name(&self) -> &'static str {
        "ix_linear_regression"
    }

    fn forward(&self, ctx: &mut DiffContext, inputs: &ValueMap) -> Result<ValueMap> {
        let x = inputs
            .get("x")
            .cloned()
            .ok_or_else(|| AutogradError::MissingInput("x".into()))?;
        let w = inputs
            .get("w")
            .cloned()
            .ok_or_else(|| AutogradError::MissingInput("w".into()))?;
        let b = inputs
            .get("b")
            .cloned()
            .ok_or_else(|| AutogradError::MissingInput("b".into()))?;

        let handles = Self::build_graph(ctx, x, w, b)?;

        // Stash the handles as tool state so `backward` can find them.
        let state = serde_json::json!({
            "x": handles.x.0,
            "w": handles.w.0,
            "b": handles.b.0,
            "y_hat": handles.y_hat.0,
            "loss": handles.loss.0,
        });
        ctx.tool_state
            .insert("ix_linear_regression.last".into(), state);

        let y_hat_value = ctx
            .tape
            .get(handles.y_hat)
            .ok_or(AutogradError::InvalidHandle(handles.y_hat))?
            .value
            .clone();
        let loss_value = ctx
            .tape
            .get(handles.loss)
            .ok_or(AutogradError::InvalidHandle(handles.loss))?
            .value
            .clone();

        let mut out = ValueMap::new();
        out.insert("y_hat".into(), y_hat_value);
        out.insert("loss".into(), loss_value);
        Ok(out)
    }

    fn backward(&self, ctx: &mut DiffContext, _out_grads: &ValueMap) -> Result<ValueMap> {
        // Pull the handles recorded during `forward`.
        let state = ctx
            .tool_state
            .get("ix_linear_regression.last")
            .ok_or_else(|| AutogradError::MissingSaved("ix_linear_regression.last".into()))?
            .clone();

        let loss_idx = state
            .get("loss")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AutogradError::MissingSaved("loss handle".into()))?
            as usize;
        let x_idx = state
            .get("x")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AutogradError::MissingSaved("x handle".into()))?
            as usize;
        let w_idx = state
            .get("w")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AutogradError::MissingSaved("w handle".into()))?
            as usize;
        let b_idx = state
            .get("b")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AutogradError::MissingSaved("b handle".into()))?
            as usize;

        let loss_h = TensorHandle(loss_idx);
        let seed = ndarray::Array::from_elem(ndarray::IxDyn(&[]), 1.0_f64);
        let grads = ctx.backward(loss_h, seed)?;

        let mut out = ValueMap::new();
        if let Some(g) = grads.get(&TensorHandle(x_idx)) {
            out.insert("x".into(), Tensor::from_array(g.clone()));
        }
        if let Some(g) = grads.get(&TensorHandle(w_idx)) {
            out.insert("w".into(), Tensor::from_array(g.clone()));
        }
        if let Some(g) = grads.get(&TensorHandle(b_idx)) {
            out.insert("b".into(), Tensor::from_array(g.clone()));
        }
        Ok(out)
    }
}
