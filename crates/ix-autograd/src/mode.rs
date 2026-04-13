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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionMode {
    #[default]
    Eager,
    Train,
    Mixed,
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
