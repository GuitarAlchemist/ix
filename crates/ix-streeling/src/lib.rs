//! `ix-streeling` — Streeling University's generator: turn the ecosystem's
//! learning files into a federated catalog (`state/streeling/catalog.jsonl`) and
//! a campus index (`docs/streeling/README.md`).
//!
//! Plain Rust — **no DuckDB dependency**. DuckDB (via `ix-duck` / duckdb-skills)
//! only *reads* the emitted catalog. Source-of-truth stays in each repo's files;
//! this is a derived index/registrar layer. See
//! `docs/plans/2026-06-14-002-feat-streeling-university-plan.md`.

pub mod campus;
pub mod check;
pub mod ingest;
pub mod model;

use ingest::SourceRoot;
use model::LearningRecord;
use std::path::Path;

/// Default federation roots: this repo (`ix`) + sibling `ga`. tars/Demerzel are
/// fast-follow (need adapters), so they're intentionally absent here.
// @ai:assumption sibling clones live beside the ix repo (../ga); a missing sibling is skipped, not fatal [P:assumed conf:0.7 src:ix-streeling::tests::ingest_tolerates_absent_sibling]
pub fn default_roots(ix_root: &Path) -> Vec<SourceRoot> {
    // Canonicalize so `--repo-root .` resolves a real parent (".".parent() is "").
    let ix_abs = ix_root.canonicalize().unwrap_or_else(|_| ix_root.to_path_buf());
    let mut roots = vec![SourceRoot::new("ix", ix_abs.clone())];
    if let Some(parent) = ix_abs.parent() {
        roots.push(SourceRoot::new("ga", parent.join("ga")));
    }
    roots
}

/// Serialize records as JSONL (one object per line, trailing newline).
pub fn to_jsonl(records: &[LearningRecord]) -> String {
    let mut s: String = records
        .iter()
        .filter_map(|r| serde_json::to_string(r).ok())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

/// Parse a JSONL catalog, ignoring blank lines.
pub fn from_jsonl(s: &str) -> Vec<LearningRecord> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}
