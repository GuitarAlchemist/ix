//! In-source AI annotation extractor.
//!
//! Walks a workspace, finds comments matching the `@ai:` marker syntax defined in
//! `docs/contracts/2026-05-24-ai-annotation.contract.md`, and emits a stream of
//! [`Annotation`] structs (canonical JSON shape).
//!
//! ```ignore
//! // @ai:invariant arr is sorted ascending [T:test conf:0.95 src:test_search.rs:42]
//! ```
//!
//! The extractor is intentionally language-agnostic — it matches against the
//! `@ai:` token inside any single-line comment (`//`, `#`, `--`) or a single-line
//! block comment (`/* ... */`, `<!-- ... -->`). Multi-line block comments are
//! NOT supported in v1.

pub mod parser;
pub mod reconciler;
pub mod types;
pub mod walker;

pub use parser::{parse_line, ParsedMarker};
pub use reconciler::{reconcile, ReconcilerConfig, ReconciliationReport};
pub use types::{
    Annotation, AnnotationKind, Certainty, Location, Reconciliation, Source, TruthValue,
    SCHEMA_VERSION,
};
pub use walker::extract;

use std::path::Path;

/// High-level convenience: walk `workspace`, return all annotations.
///
/// Equivalent to [`walker::extract`].
pub fn extract_workspace(workspace: &Path) -> Result<Vec<Annotation>, Error> {
    walker::extract(workspace)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}
