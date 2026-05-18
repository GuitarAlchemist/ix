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
//! ## Loader behaviour
//!
//! Filenames are first matched against a `YYYY-MM-DD` prefix (so
//! `2026-05-15.json` and `2026-05-15-soak.json` both load with date
//! `2026-05-15`). If that fails, the loader falls back to (a) a `timestamp`
//! field inside the JSON, then (b) the file's modification time. Only when
//! all three sources fail is the file skipped — and that skip is now a
//! **warning on stderr plus a manifest entry**, not a silent drop.
//!
//! See [`load_with`] and [`LoadOptions`] for the strict / quiet / manifest
//! switches.

pub mod report;
pub mod snapshot;
pub mod trend;

pub use report::{
    build_health_artifact, is_key_metric_name, QualityAlert, QualityHealthArtifact,
    QualityHealthStatus, QualityTrendSummary,
};
pub use snapshot::{
    load_all, load_with, ChatbotQaSnapshot, DatedSnapshot, EmbeddingsSnapshot, LoadOptions,
    LoaderManifest, LoaderStatus, ManifestEntry, SnapshotCategory, SnapshotSet,
    VoicingAnalysisSnapshot,
};
pub use trend::{DriftFlag, MetricSeries, MetricTrend, RegressionFlag, TrendDirection};
