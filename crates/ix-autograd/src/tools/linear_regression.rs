//! Linear regression as a `DifferentiableTool`.
//!
//! Day 2 introduced this tool with a stand-in loss (`sum(y_hat * y_hat)`)
//! because subtraction and scalar division were not yet available.
//! Day 3 upgrades it to a proper mean-squared-error loss using the new
//! `sub`, `mean`, and `div_scalar` primitives:
//!
//! ```text
//!     y_hat    = x @ w + b              // matmul + broadcast add
//!     residual = y_hat - y              // sub
//!     sq       = residual * residual    // mul
//!     loss     = mean(sq)               // sum + div_scalar
//! ```
//!
//! Day 3 also replaces the fragile `serde_json::Value` tool state from
//! Day 2 with a typed `LinregState` struct stored via
//! `DiffContext::set_tool_state::<LinregState>`, following the Day 2
//! review finding §3.2.
//!
//! Only `x`, `w`, and `b` receive gradients. `y` is a target (observed
//! data) and is not trainable.

use crate::ops;
use crate::tape::{DiffContext, TensorHandle};
use crate::tensor::Tensor;
use crate::tool::{DifferentiableTool, ValueMap};
use crate::{AutogradError, Result};

/// Linear regression with mean-squared-error loss as a
/// `DifferentiableTool`. See the module docs for the exact forward
/// graph.
pub struct LinearRegressionTool;

/// Typed tool state stored in `DiffContext` between forward and backward.
/// Replaces the Day 2 `serde_json::Value` bag.
#[derive(Debug, Clone, Copy)]
pub struct LinregState {
    /// Leaf handle for the design matrix `x`.
    pub x: TensorHandle,
    /// Leaf handle for the weight vector `w`.
    pub w: TensorHandle,
    /// Leaf handle for the scalar bias `b`.
    pub b: TensorHandle,
    /// Leaf handle for the observed target `y` (not a trainable
    /// parameter).
    pub y: TensorHandle,
    /// Node handle for the prediction `y_hat = x @ w + b`.
    pub y_hat: TensorHandle,
    /// Node handle for the scalar MSE loss.
    pub loss: TensorHandle,
}

impl LinearRegressionTool {
    /// Build the forward graph on the tape and return handles for the
    /// inputs, the prediction, and the scalar MSE loss.
    pub fn build_graph(
        ctx: &mut DiffContext,
        x: Tensor,
        w: Tensor,
        b: Tensor,
        y: Tensor,
    ) -> Result<LinregState> {
        let x_h = ops::input(ctx, x);
        let w_h = ops::input(ctx, w);
        let b_h = ops::input(ctx, b);
        let y_h = ops::input(ctx, y);

        // y_hat = x @ w + b    (b is [1, 1], broadcasts to [n, 1])
        let xw = ops::matmul(ctx, x_h, w_h)?;
        let y_hat = ops::add(ctx, xw, b_h)?;

        // residual = y_hat - y    (same shape [n, 1])
        let residual = ops::sub(ctx, y_hat, y_h)?;

        // sq = residual * residual
        let sq = ops::mul(ctx, residual, residual)?;

        // loss = mean(sq)    (scalar — Day 3 mean is sum/n)
        let loss = ops::mean(ctx, sq)?;

        Ok(LinregState {
            x: x_h,
            w: w_h,
            b: b_h,
            y: y_h,
            y_hat,
            loss,
        })
    }
}

const STATE_KEY: &str = "ix_linear_regression.last";

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
        let y = inputs
            .get("y")
            .cloned()
            .ok_or_else(|| AutogradError::MissingInput("y".into()))?;

        let state = Self::build_graph(ctx, x, w, b, y)?;
        ctx.set_tool_state(STATE_KEY, state);

        let y_hat_value = ctx
            .tape
            .get(state.y_hat)
            .ok_or(AutogradError::InvalidHandle(state.y_hat))?
            .value
            .clone();
        let loss_value = ctx
            .tape
            .get(state.loss)
            .ok_or(AutogradError::InvalidHandle(state.loss))?
            .value
            .clone();

        let mut out = ValueMap::new();
        out.insert("y_hat".into(), y_hat_value);
        out.insert("loss".into(), loss_value);
        Ok(out)
    }

    fn backward(&self, ctx: &mut DiffContext, _out_grads: &ValueMap) -> Result<ValueMap> {
        let state = *ctx
            .get_tool_state::<LinregState>(STATE_KEY)
            .ok_or_else(|| AutogradError::MissingSaved(STATE_KEY.into()))?;

        let seed = ndarray::Array::from_elem(ndarray::IxDyn(&[]), 1.0_f64);
        let grads = ctx.backward(state.loss, seed)?;

        let mut out = ValueMap::new();
        if let Some(g) = grads.get(&state.x) {
            out.insert("x".into(), Tensor::from_array(g.clone()));
        }
        if let Some(g) = grads.get(&state.w) {
            out.insert("w".into(), Tensor::from_array(g.clone()));
        }
        if let Some(g) = grads.get(&state.b) {
            out.insert("b".into(), Tensor::from_array(g.clone()));
        }
        // Note: y is a target, not a parameter — we intentionally do not
        // return its gradient even though the walker computes one.
        Ok(out)
    }
}
