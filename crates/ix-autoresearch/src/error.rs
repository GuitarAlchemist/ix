//! Closed enum of error categories the kernel may emit.
//!
//! `EvalCategory` is the subset that target adapters return from
//! `Experiment::evaluate`. Per the security review, target adapters must
//! never let raw subprocess stderr flow into log entries — `Iteration.error`
//! is rendered via `Display` over this typed enum, not `format!("{stderr}")`.

use std::time::Duration;
use thiserror::Error;

/// Top-level kernel error type.
#[derive(Debug, Error)]
pub enum AutoresearchError {
    #[error("evaluation failed: {0}")]
    EvalFailed(EvalCategory),

    #[error("evaluation timed out after {0:?}")]
    TimedOut(Duration),

    #[error("subprocess hard-killed by watchdog: {detail}")]
    HardKilled { detail: String },

    #[error("aborting run: {threshold} consecutive hard-kills exceeded threshold")]
    HardKillCascade { threshold: usize },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("config hash collision detected (different configs produced same hash)")]
    HashCollision,

    #[error("invalid run id: {0}")]
    InvalidRunId(String),

    #[error("invalid milestone slug: {0}")]
    InvalidSlug(String),

    #[error("milestone slug already exists: {slug} (use --force to overwrite)")]
    MilestoneSlugCollision { slug: String },

    #[error("promote sanitization failed: {detail}")]
    PromoteSanitizationFailed { detail: String },
}

/// Closed categorization of evaluation failures. Each adapter maps its
/// concrete failures into one of these. Crucially, the `String` payloads
/// must be sanitized (no raw stderr, no API keys, no absolute paths) per
/// the security review. Use `Display` for user-facing text.
#[derive(Debug, Clone)]
pub enum EvalCategory {
    /// Subprocess returned a non-zero exit code.
    SubprocessFailedExitCode { code: i32 },
    /// Expected output file from a subprocess was missing.
    /// `path` is run-dir-relative, never absolute.
    MissingExpectedFile { path: String },
    /// Expected JSON could not be parsed.
    JsonParseFailed { reason: String },
    /// In-process evaluator returned an error other than the categories above.
    InternalError { reason: String },
    /// Adapter declined to evaluate the requested config (e.g. simplex
    /// constraint violated, cardinality mismatch).
    InvalidConfig { reason: String },
}

impl std::fmt::Display for EvalCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SubprocessFailedExitCode { code } => write!(f, "subprocess exit code {code}"),
            Self::MissingExpectedFile { path } => write!(f, "missing expected file {path}"),
            Self::JsonParseFailed { reason } => write!(f, "JSON parse failed: {reason}"),
            Self::InternalError { reason } => write!(f, "internal: {reason}"),
            Self::InvalidConfig { reason } => write!(f, "invalid config: {reason}"),
        }
    }
}
