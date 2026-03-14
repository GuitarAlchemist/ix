//! Error types for the dynamics crate.

use thiserror::Error;

/// Errors that can occur in dynamics operations.
#[derive(Debug, Error)]
pub enum DynamicsError {
    /// A parameter was invalid (e.g., non-rotation matrix).
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A numerical operation failed (e.g., singular matrix).
    #[error("numerical error: {0}")]
    NumericalError(String),
}
