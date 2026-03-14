//! Error types for the K-theory crate.

use thiserror::Error;

/// Errors that can occur in K-theory operations.
#[derive(Debug, Error)]
pub enum KTheoryError {
    /// A parameter was invalid (e.g., non-square matrix).
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A numerical operation failed.
    #[error("numerical error: {0}")]
    NumericalError(String),
}
