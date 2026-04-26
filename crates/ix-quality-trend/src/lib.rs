//! # ix-quality-trend
//!
//! Aggregates timestamped quality snapshots from the GuitarAlchemist ecosystem
//! (embedding diagnostics, voicing analysis audits, chatbot QA) and emits an
//! executive-readable markdown trend report.
//!
//! The tool is **schema-tolerant**: all fields on the typed snapshot structs
//! are `Option<T>`, so older snapshots that predate a given metric still load
//! cleanly. Missing categories produce a "no data" section rather than an error.
//!
//! ## Layout
//!
//! ```text
//! <snapshots-dir>/
//!   embeddings/YYYY-MM-DD.json
//!   voicing-analysis/YYYY-MM-DD.json
//!   chatbot-qa/YYYY-MM-DD.json
//! ```
//!
//! Dates are parsed from the filename stem; non-date filenames are skipped.

pub mod report;
pub mod snapshot;
pub mod trend;

pub use report::{
    build_health_artifact, is_key_metric_name, QualityAlert, QualityHealthArtifact,
    QualityHealthStatus, QualityTrendSummary,
};
pub use snapshot::{
    ChatbotQaSnapshot, DatedSnapshot, EmbeddingsSnapshot, SnapshotCategory, SnapshotSet,
    VoicingAnalysisSnapshot,
};
pub use trend::{DriftFlag, MetricSeries, MetricTrend, RegressionFlag, TrendDirection};
