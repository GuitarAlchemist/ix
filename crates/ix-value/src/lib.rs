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

use ingest::SourceRoot;
use model::ValueRecord;
use std::path::Path;

/// Default federation roots: this repo (`ix`) + sibling `ga`. tars/Demerzel are
/// fast-follow; absent siblings roll up to "unset" (graceful).
// @ai:assumption sibling clones live beside the ix repo (../ga); a missing manifest is reported, not fatal [P:assumed conf:0.7 src:ix-value::tests::absent_sibling_is_reported_not_fatal]
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
pub fn to_jsonl(records: &[ValueRecord]) -> String {
    let mut s: String = records
        .iter()
        .filter_map(|r| serde_json::to_string(r).ok())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

/// Parse a JSONL catalog, ignoring blank lines.
pub fn from_jsonl(s: &str) -> Vec<ValueRecord> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}
