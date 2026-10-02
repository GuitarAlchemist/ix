//! # ix-optimize
//!
//! Optimization algorithms: gradient descent variants, simulated annealing,
//! particle swarm optimization, ant colony optimization for the travelling
//! salesman problem, and convergence utilities.

pub mod aco;
pub mod annealing;
pub mod convergence;
pub mod gradient;
pub mod pso;
pub mod traits;

pub use traits::{ObjectiveFunction, Optimizer};
