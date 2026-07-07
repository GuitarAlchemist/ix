//! `ix-streeling` — Streeling University's generator: turn the ecosystem's
//! learning files into a federated catalog (`state/streeling/catalog.jsonl`) and
//! a campus index (`docs/streeling/README.md`).
//!
//! This crate is one **adapter** over the [`ix_registrar`] deep module: it owns
//! the *ingest* (walk repos, parse Markdown frontmatter into [`LearningRecord`]s)
//! and re-exports the shared federation surface (roots, JSONL, the drift gate).
//!
//! Plain Rust — **no DuckDB dependency**. DuckDB (via `ix-duck` / duckdb-skills)
//! only *reads* the emitted catalog. Source-of-truth stays in each repo's files;
//! this is a derived index/registrar layer. See
//! `docs/plans/2026-06-14-002-feat-streeling-university-plan.md`.

pub mod campus;
pub mod check;
pub mod ingest;
pub mod model;
pub mod search;

// Shared federation surface, provided by the registrar deep module. Re-exported
// at the historical paths so callers keep using `ix_streeling::{default_roots,
// to_jsonl, from_jsonl}` unchanged.
pub use ix_registrar::{default_roots, from_jsonl, to_jsonl};
