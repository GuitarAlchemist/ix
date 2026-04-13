//! Execution modes for a differentiable pipeline.
//!
//! The four modes were merged from the Codex + Claude code review:
//!
//! - `Eager` — current ix-pipeline behaviour, plain values, no tape.
//! - `Train` — all tools must be differentiable, tape is built, gradients
//!   flow end-to-end. Used for pipeline-level gradient descent.
//! - `Mixed` — differentiable subgraph runs on the tape; non-differentiable
//!   upstream tools run in `Eager` and their outputs are frozen as constants
//!   at the boundary. No gradient flows into non-differentiable tools.
//!   Claude's addition.
//! - `VerifyFiniteDiff` — run the finite-difference verifier on all tools
//!   that report `supports_grad() == true`. Used by CI and integration
//!   tests to catch backward bugs early.

use serde::{Deserialize, Serialize};

/// How the pipeline executor should run a DAG.
///
/// The mode threads through every [`crate::tool::DifferentiableTool`]
/// call and determines whether the tool builds a tape, runs plain
/// numeric forward, or runs the finite-diff verifier.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Current `ix-pipeline` behaviour: plain values, no tape, no
    /// gradient tracking. The default and the cheapest mode.
    #[default]
    Eager,
    /// All tools on the DAG must be differentiable. A tape is built,
    /// and gradients flow end-to-end on `backward`.
    Train,
    /// Differentiable subgraph runs on the tape; non-differentiable
    /// upstream tools run in `Eager` and their outputs are frozen as
    /// constants at the boundary. No gradient flows into
    /// non-differentiable tools. Claude's addition from the Day 2
    /// code review, for gradual migration of the IX toolbox.
    Mixed,
    /// Run the finite-difference verifier on all tools that report
    /// `supports_grad() == true`. Used by CI and integration tests
    /// to catch backward bugs early.
    VerifyFiniteDiff,
}

impl ExecutionMode {
    /// Whether this mode requires building a tape.
    pub fn requires_tape(self) -> bool {
        matches!(self, Self::Train | Self::Mixed | Self::VerifyFiniteDiff)
    }

    /// Whether this mode allows non-differentiable tools to participate.
    pub fn allows_non_diff(self) -> bool {
        matches!(self, Self::Eager | Self::Mixed)
    }
}
