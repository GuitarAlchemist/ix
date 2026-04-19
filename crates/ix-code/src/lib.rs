//! # ix-code
//!
//! Code analysis: extract complexity metrics from source code using
//! lightweight keyword-based analysis plus optional tree-sitter AST,
//! git history trajectories, persistent homology, governance verdicts,
//! and advanced math features.
//!
//! Results can be converted to feature vectors for ML pipelines.
//!
//! ## Feature flags
//!
//! - `semantic` -- tree-sitter AST + call-graph extraction (Layer 2)
//! - `trajectory` -- git2 metric trajectory analysis (Layer 3)
//! - `topology` -- persistent homology on call graphs (Layer 4)
//! - `gates` -- hexavalent governance verdicts (Layer 5)
//! - `advanced` -- hyperbolic embeddings, K-theory, spectral, BSP (Layer 6)
//! - `physics` -- chaos, Kalman, wavelets, Markov, Laplacian (Layer 7)
//! - `full` -- all layers enabled

pub mod analyze;
pub mod catalog;
pub mod metrics;
pub mod smells;

#[cfg(feature = "semantic")]
pub mod semantic;

#[cfg(feature = "trajectory")]
pub mod trajectory;

#[cfg(feature = "topology")]
pub mod topology;

#[cfg(feature = "gates")]
pub mod gates;

#[cfg(feature = "gates")]
pub mod aggregate;

#[cfg(feature = "advanced")]
pub mod advanced;

#[cfg(feature = "physics")]
pub mod physics;
