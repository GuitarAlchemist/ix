//! `StatsVarianceTool` — variance reduction wrapped as a
//! `DifferentiableTool`.
//!
//! Day 4 of R7 Autograd-IX Week 1. This is the second tool wrapped
//! after `LinearRegressionTool` and serves as the proof-point that
//! the trait + tool-state pattern from Day 3 scales to a second tool
//! without surprises.
//!
//! The forward graph is a direct call to [`crate::ops::variance`],
//! which itself decomposes into `mean → sub → mul → mean` on the tape.
//! Backward delegates to the reverse walker — no custom backward
//! function is needed.

use crate::ops;
use crate::tape::{DiffContext, TensorHandle};
use crate::tensor::Tensor;
use crate::tool::{DifferentiableTool, ValueMap};
use crate::{AutogradError, Result};

/// Variance reduction over a single tensor input.
///
/// Inputs:  `x` — any-rank tensor.
/// Outputs: `variance` — scalar tensor (rank 0).
pub struct StatsVarianceTool;

/// Typed tool state for `StatsVarianceTool`.
#[derive(Debug, Clone, Copy)]
pub struct StatsVarianceState {
    /// Leaf handle for the input tensor `x`.
    pub x: TensorHandle,
    /// Node handle for the scalar variance output.
    pub variance: TensorHandle,
}

const STATE_KEY: &str = "ix_stats_variance.last";

impl DifferentiableTool for StatsVarianceTool {
    fn name(&self) -> &'static str {
        "ix_stats_variance"
    }

    fn forward(&self, ctx: &mut DiffContext, inputs: &ValueMap) -> Result<ValueMap> {
        let x = inputs
            .get("x")
            .cloned()
            .ok_or_else(|| AutogradError::MissingInput("x".into()))?;
        let x_h = ops::input(ctx, x);
        let variance = ops::variance(ctx, x_h)?;

        ctx.set_tool_state(STATE_KEY, StatsVarianceState { x: x_h, variance });

        let variance_value = ctx
            .tape
            .get(variance)
            .ok_or(AutogradError::InvalidHandle(variance))?
            .value
            .clone();

        let mut out = ValueMap::new();
        out.insert("variance".into(), variance_value);
        Ok(out)
    }

    fn backward(&self, ctx: &mut DiffContext, _out_grads: &ValueMap) -> Result<ValueMap> {
        let state = *ctx
            .get_tool_state::<StatsVarianceState>(STATE_KEY)
            .ok_or_else(|| AutogradError::MissingSaved(STATE_KEY.into()))?;

        let seed = ndarray::Array::from_elem(ndarray::IxDyn(&[]), 1.0_f64);
        let grads = ctx.backward(state.variance, seed)?;

        let mut out = ValueMap::new();
        if let Some(g) = grads.get(&state.x) {
            out.insert("x".into(), Tensor::from_array(g.clone()));
        }
        Ok(out)
    }
}
