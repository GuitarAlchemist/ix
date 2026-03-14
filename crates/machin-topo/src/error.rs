//! Error types for the topology crate.

use thiserror::Error;

/// Errors that can occur in topological operations.
#[derive(Debug, Error)]
pub enum TopoError {
    /// A parameter was invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A computation failed.
    #[error("computation error: {0}")]
    ComputationError(String),
}
