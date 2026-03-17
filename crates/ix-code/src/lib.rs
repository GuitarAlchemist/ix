//! # ix-code
//!
//! Code analysis: extract complexity metrics from source code using
//! `rust-code-analysis` (Mozilla). Computes cyclomatic complexity,
//! cognitive complexity, Halstead metrics, SLOC, and maintainability index.
//!
//! Results can be converted to feature vectors for ML pipelines.

pub mod analyze;
pub mod metrics;
