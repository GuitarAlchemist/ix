//! `ix-registrar` — the **Federation / registrar** deep module (CONTEXT.md).
//!
//! A registrar turns each repo's per-repo source into a federated
//! `state/<x>/catalog.jsonl` and gates that catalog for drift. This crate owns
//! the parts that are *the same* whatever the payload is — root discovery, JSONL
//! serialization, and the repo-scoped freshness gate — behind one small
//! interface. The parts that genuinely vary (how a source file becomes a record)
//! stay in each adapter crate.
//!
//! Two adapters justify the seam: [`ix-streeling`](../ix_streeling) federates
//! learnings (one record per Markdown frontmatter), [`ix-value`](../ix_value)
//! federates the business-value scorecard (RICE rows from a JSON manifest). Each
//! supplies a [`Record`] impl; everything below is shared.
//!
//! Plain Rust — **no DuckDB dependency**. DuckDB only ever *reads* the emitted
//! catalog.

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A catalog row. The registrar is generic over this: it needs only enough of a
/// record to *identify* it (for drift dedup), *name* it (for drift reporting),
/// and *attribute* it to a repo (for repo-scoping the gate). Everything else —
/// the record's actual fields — is the adapter's business and is reached only
/// through `Serialize`.
pub trait Record: Serialize {
    /// Identity key for drift comparison; must be unique within one catalog.
    ///
    /// When the record's `id` is already globally unique (e.g. `"{repo}:{path}"`)
    /// this is just that id. When ids are bare and could collide across repos,
    /// scope it (e.g. `"{repo}\u{1}{id}"`).
    fn dedup_key(&self) -> String;

    /// Human-facing id surfaced in [`DriftReport`] entries.
    fn report_id(&self) -> String;

    /// Originating repo — the registrar uses it to scope the drift gate to the
    /// repos actually scanned.
    fn repo(&self) -> &str;
}

/// A repo to scan for a registrar's source. The adapter decides what file(s)
/// under `root` constitute the source; the registrar only needs the label and
/// the path.
pub struct SourceRoot {
    pub repo: String,
    pub root: PathBuf,
}

impl SourceRoot {
    pub fn new(repo: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self {
            repo: repo.into(),
            root: root.into(),
        }
    }
}

/// Default federation roots: this repo (`ix`) + sibling `ga`. tars/Demerzel are
/// fast-follow (they need their own source adapters), so they're intentionally
/// absent here.
// @ai:assumption sibling clones live beside the ix repo (../ga) [T:test conf:0.85 src:ix-registrar::tests::default_roots_includes_ix_and_ga_sibling]
pub fn default_roots(ix_root: &Path) -> Vec<SourceRoot> {
    // Canonicalize so `--repo-root .` resolves a real parent (".".parent() is "").
    let ix_abs = ix_root
        .canonicalize()
        .unwrap_or_else(|_| ix_root.to_path_buf());
    let mut roots = vec![SourceRoot::new("ix", ix_abs.clone())];
    if let Some(parent) = ix_abs.parent() {
        roots.push(SourceRoot::new("ga", parent.join("ga")));
    }
    roots
}

/// Serialize records as JSONL (one object per line, trailing newline).
pub fn to_jsonl<R: Serialize>(records: &[R]) -> String {
    let mut s: String = records
        .iter()
        .filter_map(|r| serde_json::to_string(r).ok())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

/// Parse a JSONL catalog, ignoring blank lines.
pub fn from_jsonl<R: DeserializeOwned>(s: &str) -> Vec<R> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// The freshness gate's verdict.
#[derive(Debug, Default)]
pub struct DriftReport {
    /// In sources but not catalogued (uncatalogued / needs regenerate).
    pub missing: Vec<String>,
    /// In the catalog but no longer in any scanned source (stale/removed).
    pub extra: Vec<String>,
    /// Same key, different content.
    pub changed: Vec<String>,
}

impl DriftReport {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.extra.is_empty() && self.changed.is_empty()
    }
}

fn records_equal<R: Serialize>(a: &R, b: &R) -> bool {
    serde_json::to_string(a).ok() == serde_json::to_string(b).ok()
}

/// Compute drift between a fresh ingest and the committed catalog, scoped to the
/// repos actually scanned (`seen_repos`). A committed record from a repo that
/// wasn't scanned (e.g. a sibling clone absent in CI) is **not** flagged stale —
/// its freshness is that repo's responsibility. This makes the gate viable in a
/// single-repo CI checkout.
// @ai:invariant drift is repo-scoped: a committed record whose repo is not in seen_repos is never reported as extra/stale [T:test conf:0.9 src:ix-registrar::tests::drift_is_repo_scoped]
pub fn drift<R: Record>(fresh: &[R], committed: &[R], seen_repos: &[String]) -> DriftReport {
    let fmap: BTreeMap<String, &R> = fresh.iter().map(|r| (r.dedup_key(), r)).collect();
    let cmap: BTreeMap<String, &R> = committed.iter().map(|r| (r.dedup_key(), r)).collect();
    let seen = |repo: &str| seen_repos.iter().any(|r| r == repo);
    let mut report = DriftReport::default();
    for (k, fr) in &fmap {
        match cmap.get(k) {
            None => report.missing.push(fr.report_id()),
            Some(cr) if !records_equal(fr, cr) => report.changed.push(fr.report_id()),
            Some(_) => {}
        }
    }
    for (k, cr) in &cmap {
        if !fmap.contains_key(k) && seen(cr.repo()) {
            report.extra.push(cr.report_id());
        }
    }
    report.missing.sort();
    report.extra.sort();
    report.changed.sort();
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct Row {
        id: String,
        repo: String,
        body: String,
    }

    impl Record for Row {
        // Bare ids → composite key so two repos can share an id without colliding.
        fn dedup_key(&self) -> String {
            format!("{}\u{1}{}", self.repo, self.id)
        }
        fn report_id(&self) -> String {
            self.id.clone()
        }
        fn repo(&self) -> &str {
            &self.repo
        }
    }

    fn row(id: &str, repo: &str, body: &str) -> Row {
        Row {
            id: id.into(),
            repo: repo.into(),
            body: body.into(),
        }
    }

    #[test]
    fn jsonl_roundtrips() {
        let recs = vec![row("a", "ix", "1"), row("b", "ga", "2")];
        let text = to_jsonl(&recs);
        assert!(text.ends_with('\n'));
        assert_eq!(text.lines().count(), 2);
        let back: Vec<Row> = from_jsonl(&text);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].id, "a");
        // Blank lines are ignored.
        let back2: Vec<Row> = from_jsonl(&format!("{text}\n\n"));
        assert_eq!(back2.len(), 2);
    }

    #[test]
    fn drift_detects_missing_extra_changed() {
        let fresh = vec![row("a", "ix", "1"), row("b", "ix", "2")];
        let seen = vec!["ix".to_string()];

        // Identical → clean.
        assert!(drift(&fresh, &fresh, &seen).is_clean());

        // Committed missing "b" → b is uncatalogued (missing).
        let committed = vec![row("a", "ix", "1")];
        let d = drift(&fresh, &committed, &seen);
        assert_eq!(d.missing, vec!["b".to_string()]);
        assert!(d.extra.is_empty() && d.changed.is_empty());

        // Committed has an extra "c" from a scanned repo → stale (extra).
        let committed = vec![row("a", "ix", "1"), row("b", "ix", "2"), row("c", "ix", "3")];
        let d = drift(&fresh, &committed, &seen);
        assert_eq!(d.extra, vec!["c".to_string()]);

        // Same key, different body → changed.
        let committed = vec![row("a", "ix", "CHANGED"), row("b", "ix", "2")];
        let d = drift(&fresh, &committed, &seen);
        assert_eq!(d.changed, vec!["a".to_string()]);
    }

    #[test]
    fn drift_is_repo_scoped() {
        let fresh = vec![row("a", "ix", "1")];
        let seen = vec!["ix".to_string()];
        // A committed record from an UNSCANNED repo (tars) must not be flagged stale.
        let committed = vec![row("a", "ix", "1"), row("z", "tars", "9")];
        let d = drift(&fresh, &committed, &seen);
        assert!(
            d.is_clean(),
            "tars record ignored when tars wasn't scanned: {d:?}"
        );
    }

    #[test]
    fn default_roots_includes_ix_and_ga_sibling() {
        let tmp = std::env::temp_dir();
        let roots = default_roots(&tmp);
        assert_eq!(roots[0].repo, "ix");
        assert!(roots.iter().any(|r| r.repo == "ga"));
        // ga is discovered as a sibling of ix (shares the same parent).
        let ga = roots.iter().find(|r| r.repo == "ga").unwrap();
        assert!(ga.root.ends_with("ga"));
    }
}
