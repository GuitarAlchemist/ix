//! Freshness gate: compare a fresh ingest against the committed catalog. Any
//! divergence means the catalog is stale (someone added/changed a learning
//! without regenerating) — the green-but-dead guard.

use crate::model::LearningRecord;
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct DriftReport {
    /// In sources but not catalogued (uncatalogued learnings).
    pub missing: Vec<String>,
    /// In the catalog but no longer in sources (stale/removed).
    pub extra: Vec<String>,
    /// Same id, different content.
    pub changed: Vec<String>,
}

impl DriftReport {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.extra.is_empty() && self.changed.is_empty()
    }
}

fn records_equal(a: &LearningRecord, b: &LearningRecord) -> bool {
    serde_json::to_string(a).ok() == serde_json::to_string(b).ok()
}

/// Compute drift between a fresh ingest and the committed catalog, scoped to the
/// repos actually scanned (`seen_repos`). A committed record from a repo that
/// wasn't scanned (e.g. a sibling clone absent in CI) is **not** flagged stale —
/// its freshness is that repo's responsibility. This makes the gate viable in a
/// single-repo CI checkout.
pub fn drift(
    fresh: &[LearningRecord],
    committed: &[LearningRecord],
    seen_repos: &[String],
) -> DriftReport {
    let fmap: BTreeMap<&str, &LearningRecord> = fresh.iter().map(|r| (r.id.as_str(), r)).collect();
    let cmap: BTreeMap<&str, &LearningRecord> =
        committed.iter().map(|r| (r.id.as_str(), r)).collect();
    let seen = |repo: &str| seen_repos.iter().any(|r| r == repo);
    let mut report = DriftReport::default();
    for (id, fr) in &fmap {
        match cmap.get(id) {
            None => report.missing.push((*id).to_string()),
            Some(cr) if !records_equal(fr, cr) => report.changed.push((*id).to_string()),
            Some(_) => {}
        }
    }
    for (id, cr) in &cmap {
        if !fmap.contains_key(id) && seen(&cr.repo) {
            report.extra.push((*id).to_string());
        }
    }
    report.missing.sort();
    report.extra.sort();
    report.changed.sort();
    report
}
