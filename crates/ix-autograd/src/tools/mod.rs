//! Differentiable wrappers for IX MCP tools.
//!
//! Day 2 of Week 1 per examples/canonical-showcase/ix-roadmap-plan-v1.md §5.
//! Day 2 ships `linear_regression` as the first real `DifferentiableTool`.
//! Day 3 will add variance, mean, then wrap `ix_stats` and start the
//! CATIA bracket chain.

pub mod linear_regression;
pub mod stats_variance;
