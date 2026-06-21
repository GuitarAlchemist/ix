//! Convergence criteria for optimization loops.
//!
//! The criterion itself now lives in [`ix_math::convergence`] — the one home
//! shared by the optimizers here and the iterative learners in `ix-unsupervised`.
//! This alias preserves the historical `ix_optimize::convergence::ConvergenceCriteria`
//! path (struct-literal construction with `max_iterations` / `tolerance` fields
//! still works, since they are the same public fields).
pub use ix_math::convergence::Convergence as ConvergenceCriteria;
