//! Reverse-mode automatic differentiation for IX pipelines.
//!
//! Week 1 of R7 per `examples/canonical-showcase/ix-roadmap-plan-v1.md`.
//! This file is a scaffold; Days 2-5 fill in the real implementations.
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

pub mod mode;
pub mod ops;
pub mod tape;
pub mod tensor;
pub mod tool;
pub mod tools;

pub mod prelude {
    pub use crate::mode::ExecutionMode;
    pub use crate::tape::{DiffContext, Tape, TensorHandle};
    pub use crate::tensor::{Tensor, TensorData};
    pub use crate::tool::{DifferentiableTool, ValueMap};
}

#[derive(Debug, thiserror::Error)]
pub enum AutogradError {
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("missing input: {0}")]
    MissingInput(String),
    #[error("missing saved state: {0}")]
    MissingSaved(String),
    #[error("tensor handle {0:?} not found on tape")]
    InvalidHandle(tape::TensorHandle),
    #[error("tool `{tool}` does not support autograd in mode {mode:?}")]
    UnsupportedMode {
        tool: String,
        mode: mode::ExecutionMode,
    },
    #[error("numerical: {0}")]
    Numerical(String),
}

pub type Result<T> = std::result::Result<T, AutogradError>;
