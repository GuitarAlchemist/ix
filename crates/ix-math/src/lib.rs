//! # ix-math
//!
//! Core math primitives for the machin ML toolkit.
//! Linear algebra, statistics, distances, activation functions, and numerical calculus.

pub mod activation;
pub mod bsp;
pub mod calculus;
pub mod distance;
pub mod dual_quaternion;
pub mod eigen;
pub mod error;
pub mod geometric_space;
pub mod hyperbolic;
pub mod linalg;
pub mod plucker;
pub mod poincare_hierarchy;
pub mod preprocessing;
pub mod primes;
pub mod quaternion;
pub mod random;
pub mod sedenion;
pub mod stats;
pub mod svd;

pub use ndarray;
