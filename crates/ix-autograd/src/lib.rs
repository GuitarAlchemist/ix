#![warn(missing_docs)]
//! Reverse-mode automatic differentiation for IX pipelines.
//!
//! This crate adds differentiable programming to the `ix-pipeline`
//! DAG executor. It provides a Wengert-style tape, a set of primitive
//! ops (`add`, `sub`, `mul`, `sum`, `matmul`, `div_scalar`, `mean`,
//! `variance`) with hand-written backward functions, and a
//! [`prelude::DifferentiableTool`] trait that lets existing IX MCP
//! tools participate in gradient-based pipeline optimization.
//!
//! Delivered incrementally per Week 1 of R7 in
//! `examples/canonical-showcase/ix-roadmap-plan-v1.md`:
//!
//! - **Day 1** — crate scaffold, [`prelude::ExecutionMode`], tape skeleton
//! - **Day 2** — primitive ops (`add`, `mul`, `sum`, `matmul`) + finite-diff verifier
//! - **Day 3** — `sub`, `div_scalar`, `mean`, `variance`, full MSE linear regression,
//!   typed tool state, Adam training demo
//! - **Day 4** — hardening and broadcasting polish (planned)
//! - **Day 5** — FFT backward behind the `fft-autograd` feature flag (planned)
//!
//! See the `examples/canonical-showcase/r7-day2-review.md` for the
//! multi-provider code review that informed the Day 3 task ordering.
//!
//! # Design decisions locked in by the Codex + Claude code review
//! (`examples/canonical-showcase/r7-autograd-codex-review.md`):
//!
//! - Build **scratch over `ndarray`**, not on top of candle-core or burn.
//! - **Reverse-mode Wengert tape**, dynamic (not type-level generic).
//! - Explicit `forward` + `backward` methods on `DifferentiableTool`.
//! - `ExecutionMode::{Eager, Train, Mixed, VerifyFiniteDiff}` threaded
//!   through the pipeline executor (Mixed is Claude's addition; the
//!   others are Codex's).
//! - FFT backward is gated behind the `fft-autograd` feature flag
//!   (Day 5 of Week 1) and must handle Hermitian mirroring with DC
//!   and Nyquist edge cases.

/// Execution mode enum that tells the pipeline executor whether to
/// build a tape, run plain numeric forward, or run the finite-diff
/// verifier.
pub mod mode;

/// Primitive differentiable operations: `add`, `sub`, `mul`, `sum`,
/// `matmul`, `div_scalar`, `mean`, `variance`, plus the `input` leaf
/// helper and the `DiffContext::backward` reverse walker.
pub mod ops;

/// Wengert-style reverse-mode tape: [`tape::TapeNode`],
/// [`tape::TensorHandle`], and the [`tape::DiffContext`] runtime
/// context that carries the tape between tools.
pub mod tape;

/// Tape-aware tensor type wrapping [`ndarray::ArrayD`]`<f64>`.
pub mod tensor;

/// The [`tool::DifferentiableTool`] trait — the interface every
/// differentiable IX MCP tool must implement.
pub mod tool;

/// Concrete `DifferentiableTool` implementations for specific MCP
/// tools (currently just linear regression).
pub mod tools;

/// Convenience re-exports: the minimum types a downstream crate needs
/// to build a forward pass, run `backward`, and extract gradients.
pub mod prelude {
    pub use crate::mode::ExecutionMode;
    pub use crate::tape::{DiffContext, Tape, TensorHandle};
    pub use crate::tensor::{Tensor, TensorData};
    pub use crate::tool::{DifferentiableTool, ValueMap};
}

/// Error type returned by every fallible operation in this crate.
#[derive(Debug, thiserror::Error)]
pub enum AutogradError {
    /// Two tensors had incompatible shapes for an operation that
    /// requires matching or broadcast-compatible shapes.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Shape the op expected.
        expected: Vec<usize>,
        /// Shape the op actually received.
        actual: Vec<usize>,
    },
    /// A required input key was missing from a `ValueMap` handed to
    /// `DifferentiableTool::forward`.
    #[error("missing input: {0}")]
    MissingInput(String),
    /// A required saved-state key was missing from the tool state bag
    /// or from a tape node's `saved` field.
    #[error("missing saved state: {0}")]
    MissingSaved(String),
    /// A `TensorHandle` referred to an index past the end of the tape.
    /// Usually a bug in a custom op or a tool reusing a handle from
    /// a stale `DiffContext`.
    #[error("tensor handle {0:?} not found on tape")]
    InvalidHandle(tape::TensorHandle),
    /// A tool was invoked in an execution mode it does not support —
    /// for example, a non-differentiable tool asked to run in `Train`.
    #[error("tool `{tool}` does not support autograd in mode {mode:?}")]
    UnsupportedMode {
        /// Name of the tool that refused the mode.
        tool: String,
        /// The mode the tool could not satisfy.
        mode: mode::ExecutionMode,
    },
    /// An op was called with a tensor rank outside its supported set
    /// (e.g. a 1-D vector passed to a 2-D-only `matmul`). Day 3
    /// replacement for the earlier `ShapeMismatch` hack on matmul.
    #[error("{op}: unsupported rank — supported {supported:?}, got {actual}")]
    UnsupportedRank {
        /// Op name, e.g. `"matmul"`.
        op: &'static str,
        /// Set of ranks the op does support.
        supported: Vec<usize>,
        /// Rank the op was actually handed.
        actual: usize,
    },
    /// A catch-all for numerical issues that do not fit any other
    /// variant — division by zero, empty reductions, unknown op names
    /// during the reverse walk, etc.
    #[error("numerical: {0}")]
    Numerical(String),
}

/// Shorthand for `std::result::Result<T, AutogradError>`.
pub type Result<T> = std::result::Result<T, AutogradError>;
