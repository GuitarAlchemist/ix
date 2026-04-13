//! The `DifferentiableTool` trait — the IX-autograd equivalent of
//! `ix-agent::Tool` but with explicit forward + backward methods.
//!
//! Per the Codex code review, every differentiable tool writes both
//! methods explicitly. There is no automatic-backward path (we don't
//! depend on candle). If a tool is purely a composition of the
//! primitive ops in `crate::ops`, its `backward` can just walk the
//! sub-tape those ops built.

use crate::tape::DiffContext;
use crate::tensor::Tensor;
use crate::Result;
use std::collections::HashMap;

/// Named bag of tensors flowing between tools.
pub type ValueMap = HashMap<String, Tensor>;

/// The core interface every differentiable IX MCP tool implements.
///
/// A tool is a pair of functions (`forward`, `backward`) plus a
/// `name`. `forward` runs during the normal pipeline execution and
/// (when the execution mode requires a tape) records nodes on
/// `ctx.tape`. `backward` runs during a gradient-requiring reverse
/// walk and returns gradients keyed by input name.
pub trait DifferentiableTool: Send + Sync {
    /// The name under which this tool is registered in `ix-agent`.
    fn name(&self) -> &'static str;

    /// Whether this tool can run in `ExecutionMode::Train` or `Mixed`.
    /// Default true — we only implement this trait for differentiable
    /// tools, but individual instances may opt out at runtime.
    fn supports_grad(&self) -> bool {
        true
    }

    /// Forward pass. Records nodes onto `ctx.tape` when in a
    /// tape-requiring mode; runs pure numeric in `Eager`.
    fn forward(&self, ctx: &mut DiffContext, inputs: &ValueMap) -> Result<ValueMap>;

    /// Backward pass. Given upstream gradients keyed by output name,
    /// compute and return gradients keyed by input name. Called by the
    /// pipeline executor during the reverse walk.
    fn backward(&self, ctx: &mut DiffContext, out_grads: &ValueMap) -> Result<ValueMap>;
}
