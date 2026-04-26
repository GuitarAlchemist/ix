//! # ix-optimize
//!
//! Optimization algorithms: gradient descent variants, simulated annealing,
//! particle swarm optimization, and convergence utilities.

pub mod annealing;
pub mod convergence;
pub mod gradient;
pub mod pso;
pub mod traits;

pub use traits::{ObjectiveFunction, Optimizer};
