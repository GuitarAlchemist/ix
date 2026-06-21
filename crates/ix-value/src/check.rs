//! Freshness gate: compare a fresh ingest against the committed catalog. Any
//! divergence means the catalog is stale (someone edited a manifest without
//! regenerating) — the green-but-dead guard.
//!
//! The gate itself lives in [`ix_registrar`]; it is re-exported here so callers
//! keep using `ix_value::check::{drift, DriftReport}`. Because item ids are bare
//! (not repo-prefixed), [`ValueRecord`]'s [`ix_registrar::Record`] impl uses a
//! repo-scoped composite dedup key (see `model`).

pub use ix_registrar::{drift, DriftReport};
