//! Freshness gate: compare a fresh ingest against the committed catalog. Any
//! divergence means the catalog is stale (someone added/changed a learning
//! without regenerating) — the green-but-dead guard.
//!
//! The gate itself lives in [`ix_registrar`]; it is re-exported here so callers
//! keep using `ix_streeling::check::{drift, DriftReport}`. The drift comparison
//! is keyed by [`LearningRecord`]'s [`ix_registrar::Record`] impl (see `model`).

pub use ix_registrar::{drift, DriftReport};
