use thiserror::Error;

/// Errors that can occur when working with governance artifacts.
#[derive(Debug, Error)]
pub enum GovernanceError {
    /// An I/O error occurred while reading a governance artifact.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Failed to parse a governance artifact.
    #[error("parse error: {0}")]
    ParseError(String),

    /// A governance artifact failed validation.
    #[error("validation error: {0}")]
    ValidationError(String),

    /// An action violates a constitutional article.
    #[error("constitutional violation: Article {article_number} — {description}")]
    ConstitutionViolation {
        article_number: u8,
        description: String,
    },

    /// The requested persona was not found.
    #[error("persona not found: {0}")]
    PersonaNotFound(String),

    /// The requested policy was not found.
    #[error("policy not found: {0}")]
    PolicyNotFound(String),
}

pub type Result<T> = std::result::Result<T, GovernanceError>;
