//! # ix-math
//!
//! Core math primitives for the machin ML toolkit.
//! Linear algebra, statistics, distances, activation functions, and numerical calculus.

pub mod linalg;
pub mod stats;
pub mod distance;
pub mod activation;
pub mod calculus;
pub mod random;
pub mod hyperbolic;
pub mod quaternion;
pub mod dual_quaternion;
pub mod plucker;
pub mod primes;
pub mod sedenion;
pub mod bsp;
pub mod error;
pub mod preprocessing;

pub use ndarray;
