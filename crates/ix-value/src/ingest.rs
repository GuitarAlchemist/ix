//! Read each repo's `state/value/manifest.json` and normalize it into scored
//! [`ValueRecord`]s. Tolerant by design: a missing manifest reports the repo as
//! missing (not fatal); a malformed manifest or an out-of-range item is
//! skipped-and-counted, never fatal (cf. `ix-streeling`).

use crate::model::{
    axes_valid, score01, stars, Item, Kind, Manifest, RepoScore, ValueRecord, SCHEMA_VERSION,
};

// A repo to scan for a value manifest — the registrar's shared root type,
// re-exported so callers keep using `ix_value::ingest::SourceRoot`.
pub use ix_registrar::SourceRoot;

/// Outcome of an ingest pass.
#[derive(Debug, Default)]
pub struct IngestReport {
    pub records: Vec<ValueRecord>,
    /// Items skipped (out-of-range axes or otherwise unusable).
    pub skipped: usize,
    /// Repos whose manifest parsed.
    pub roots_seen: Vec<String>,
    /// Repos with no (or unreadable) manifest — graceful, not fatal.
    pub roots_missing: Vec<String>,
}

/// Build one item record. Precondition: `axes_valid(item.*)` already passed, so
/// the `as u8` narrowing is lossless (axes are in `1..=5`).
fn item_record(repo: &str, item: &Item) -> ValueRecord {
    let (r, i, c) = (item.reach as u8, item.impact as u8, item.confidence as u8);
    ValueRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        id: item.id.clone(),
        repo: repo.to_string(),
        kind: item.kind,
        title: item.title.clone(),
        reach: r,
        impact: i,
        confidence: c,
        stars: stars(r, i, c),
        score01: score01(r, i, c),
        rationale: item.rationale.clone(),
    }
}

/// Build the repo rollup row: use an explicit `repo_score` if present, else the
/// mean of the items' `score01` (plain mean v1; reach-weighting is a revisit
/// trigger). Returns `None` when there is nothing to roll up.
fn rollup_record(repo: &str, rs: Option<&RepoScore>, items: &[ValueRecord]) -> Option<ValueRecord> {
    let (reach, impact, confidence, stars_v, score, rationale) = match rs {
        // Precondition: axes_valid passed, so `as u8` is lossless (1..=5).
        Some(rs) if axes_valid(rs.reach, rs.impact, rs.confidence) => {
            let (r, i, c) = (rs.reach as u8, rs.impact as u8, rs.confidence as u8);
            (
                r,
                i,
                c,
                stars(r, i, c),
                score01(r, i, c),
                rs.rationale.clone(),
            )
        }
        _ => {
            if items.is_empty() {
                return None;
            }
            let n = items.len() as f64;
            let mean = |f: fn(&ValueRecord) -> f64| items.iter().map(f).sum::<f64>() / n;
            let score = mean(|r| r.score01);
            let round_axis = |f: fn(&ValueRecord) -> f64| (mean(f).round() as u8).clamp(1, 5);
            (
                round_axis(|r| r.reach as f64),
                round_axis(|r| r.impact as f64),
                round_axis(|r| r.confidence as f64),
                (score * 5.0).round().clamp(1.0, 5.0) as u8,
                score,
                Some(format!(
                    "rolled up from {} item(s) (mean score)",
                    items.len()
                )),
            )
        }
    };
    Some(ValueRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        id: repo.to_string(),
        repo: repo.to_string(),
        kind: Kind::Repo,
        title: repo.to_string(),
        reach,
        impact,
        confidence,
        stars: stars_v,
        score01: score,
        rationale,
    })
}

/// Ingest every configured root. Records are returned sorted by `id` for stable output.
// @ai:invariant ingest emits one record per valid item plus a repo rollup per scanned manifest; absent manifests are reported, not fatal [T:test conf:0.9 src:ix-value::tests::ingest_produces_items_and_explicit_rollup]
pub fn ingest(roots: &[SourceRoot]) -> IngestReport {
    let mut report = IngestReport::default();
    for sr in roots {
        let path = sr.root.join("state/value/manifest.json");
        let Ok(raw) = std::fs::read_to_string(&path) else {
            report.roots_missing.push(sr.repo.clone());
            continue;
        };
        let manifest: Manifest = match serde_json::from_str(&raw) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "ix-value: skipping {} — malformed manifest: {e}",
                    path.display()
                );
                report.roots_missing.push(sr.repo.clone());
                continue;
            }
        };
        report.roots_seen.push(sr.repo.clone());

        let mut repo_items: Vec<ValueRecord> = Vec::new();
        for item in &manifest.items {
            // `repo` is a generated-only kind (the rollup row). A hand-authored
            // item declaring it would surface as a phantom repo in the DuckDB
            // `WHERE kind='repo'` leaderboard — skip it.
            if item.kind == Kind::Repo {
                eprintln!(
                    "ix-value: skipping {}:{} — items may not declare kind 'repo' (generated only)",
                    manifest.repo, item.id
                );
                report.skipped += 1;
                continue;
            }
            if !axes_valid(item.reach, item.impact, item.confidence) {
                eprintln!(
                    "ix-value: skipping {}:{} — axes out of range 1..=5 (r={},i={},c={})",
                    manifest.repo, item.id, item.reach, item.impact, item.confidence
                );
                report.skipped += 1;
                continue;
            }
            repo_items.push(item_record(&manifest.repo, item));
        }
        if let Some(roll) = rollup_record(&manifest.repo, manifest.repo_score.as_ref(), &repo_items)
        {
            report.records.push(roll);
        }
        report.records.append(&mut repo_items);
    }
    report.records.sort_by(|a, b| a.id.cmp(&b.id));
    report
}
