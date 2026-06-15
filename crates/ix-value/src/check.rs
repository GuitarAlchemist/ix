//! Freshness gate: compare a fresh ingest against the committed catalog. Any
//! divergence means the catalog is stale (someone edited a manifest without
//! regenerating) — the green-but-dead guard. Mirrors `ix-streeling::check`.

use crate::model::ValueRecord;
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct DriftReport {
    /// In manifests but not catalogued.
    pub missing: Vec<String>,
    /// In the catalog but no longer in any scanned manifest.
    pub extra: Vec<String>,
    /// Same id, different content.
    pub changed: Vec<String>,
}

impl DriftReport {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.extra.is_empty() && self.changed.is_empty()
    }
}

/// Composite key — item ids are bare (not repo-prefixed), so scope the key by repo
/// to avoid cross-repo id collisions.
fn key(r: &ValueRecord) -> String {
    format!("{}\u{1}{}", r.repo, r.id)
}

fn records_equal(a: &ValueRecord, b: &ValueRecord) -> bool {
    serde_json::to_string(a).ok() == serde_json::to_string(b).ok()
}

/// Compute drift between a fresh ingest and the committed catalog, scoped to the
/// repos actually scanned (`seen_repos`) — a committed record from an absent
/// sibling is not flagged stale, so the gate works in a single-repo CI checkout.
pub fn drift(fresh: &[ValueRecord], committed: &[ValueRecord], seen_repos: &[String]) -> DriftReport {
    let fmap: BTreeMap<String, &ValueRecord> = fresh.iter().map(|r| (key(r), r)).collect();
    let cmap: BTreeMap<String, &ValueRecord> = committed.iter().map(|r| (key(r), r)).collect();
    let seen = |repo: &str| seen_repos.iter().any(|r| r == repo);
    let mut report = DriftReport::default();
    for (k, fr) in &fmap {
        match cmap.get(k) {
            None => report.missing.push(fr.id.clone()),
            Some(cr) if !records_equal(fr, cr) => report.changed.push(fr.id.clone()),
            Some(_) => {}
        }
    }
    for (k, cr) in &cmap {
        if !fmap.contains_key(k) && seen(&cr.repo) {
            report.extra.push(cr.id.clone());
        }
    }
    report.missing.sort();
    report.extra.sort();
    report.changed.sort();
    report
}
