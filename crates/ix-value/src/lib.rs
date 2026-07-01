//! `ix-value` — the Business-Value Scorecard generator: federate each repo's
//! hand-authored `state/value/manifest.json` into a scored catalog
//! (`state/value/catalog.jsonl`), RICE→stars.
//!
//! Plain Rust — **no DuckDB dependency**. DuckDB only *reads* the emitted catalog.
//! Source-of-truth stays in the per-repo manifests; this is a derived registrar
//! layer, a second payload over the same federation shape as `ix-streeling`. See
//! `docs/plans/2026-06-14-003-feat-business-value-scorecard-plan.md`.

pub mod check;
pub mod ingest;
pub mod model;

// Shared federation surface, provided by the registrar deep module. Re-exported
// at the historical paths so callers keep using `ix_value::{default_roots,
// to_jsonl, from_jsonl}` unchanged.
pub use ix_registrar::{default_roots, from_jsonl, to_jsonl};
