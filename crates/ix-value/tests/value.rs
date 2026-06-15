//! Integration tests for the ix-value generator: ingest, rollup, tolerant skips,
//! and the repo-scoped drift gate. Uses temp manifest dirs (real on-disk shape).

use ix_value::check::drift;
use ix_value::ingest::{ingest, IngestReport, SourceRoot};
use ix_value::model::Kind;
use std::path::Path;

fn write_manifest(root: &Path, body: &str) {
    let dir = root.join("state/value");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("manifest.json"), body).unwrap();
}

fn ingest_one(repo: &str, root: &Path) -> IngestReport {
    ingest(&[SourceRoot::new(repo, root)])
}

#[test]
fn ingest_produces_items_and_explicit_rollup() {
    let dir = tempfile::tempdir().unwrap();
    write_manifest(
        dir.path(),
        r#"{
          "schema_version": "0.1.0", "repo": "ix",
          "items": [
            {"id": "crate/ix-optick", "kind": "demo", "title": "OPTIC-K", "reach": 4, "impact": 5, "confidence": 4},
            {"id": "crate/ix-governance", "kind": "demo", "title": "Governance", "reach": 3, "impact": 5, "confidence": 3}
          ],
          "repo_score": {"reach": 4, "impact": 5, "confidence": 4, "rationale": "core"}
        }"#,
    );
    let rep = ingest_one("ix", dir.path());
    assert_eq!(rep.roots_seen, vec!["ix".to_string()]);
    assert_eq!(rep.records.len(), 3, "2 items + 1 repo rollup");
    let repo_row = rep.records.iter().find(|r| r.kind == Kind::Repo).unwrap();
    assert_eq!(repo_row.id, "ix");
    assert_eq!(repo_row.stars, 4); // geomean(4,5,4)=4.31 → 4
    // sorted by id
    assert!(rep.records.windows(2).all(|w| w[0].id <= w[1].id));
}

#[test]
fn rollup_falls_back_to_item_mean_when_no_repo_score() {
    let dir = tempfile::tempdir().unwrap();
    write_manifest(
        dir.path(),
        r#"{"repo": "ga", "items": [
            {"id": "demo/a", "title": "A", "reach": 5, "impact": 5, "confidence": 5},
            {"id": "demo/b", "title": "B", "reach": 1, "impact": 1, "confidence": 1}
        ]}"#,
    );
    let rep = ingest_one("ga", dir.path());
    let repo_row = rep.records.iter().find(|r| r.kind == Kind::Repo).unwrap();
    // mean of score01(5,5,5)=1.0 and score01(1,1,1)=0.2 → 0.6 → stars round(3.0)=3
    assert_eq!(repo_row.stars, 3);
    assert!(repo_row.rationale.as_deref().unwrap().contains("rolled up"));
}

#[test]
fn out_of_range_item_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    write_manifest(
        dir.path(),
        r#"{"repo": "ix", "items": [
            {"id": "ok", "title": "OK", "reach": 3, "impact": 3, "confidence": 3},
            {"id": "bad", "title": "Bad", "reach": 9, "impact": 3, "confidence": 3}
        ]}"#,
    );
    let rep = ingest_one("ix", dir.path());
    assert_eq!(rep.skipped, 1);
    assert!(rep.records.iter().all(|r| r.id != "bad"));
}

#[test]
fn out_of_u8_axis_skips_only_that_item_not_the_repo() {
    // A `999`/`-1` typo must NOT abort decoding the whole manifest (which would
    // erase the valid sibling item). Axes deserialize as i64 → per-item skip.
    let dir = tempfile::tempdir().unwrap();
    write_manifest(
        dir.path(),
        r#"{"repo": "ix", "items": [
            {"id": "ok", "title": "OK", "reach": 3, "impact": 3, "confidence": 3},
            {"id": "huge", "title": "Huge", "reach": 999, "impact": 3, "confidence": 3},
            {"id": "neg", "title": "Neg", "reach": -1, "impact": 3, "confidence": 3}
        ]}"#,
    );
    let rep = ingest_one("ix", dir.path());
    assert_eq!(rep.roots_seen, vec!["ix".to_string()], "manifest still parsed");
    assert_eq!(rep.skipped, 2, "both out-of-range items skipped");
    assert!(rep.records.iter().any(|r| r.id == "ok"), "valid item survives");
    assert!(rep.records.iter().all(|r| r.id != "huge" && r.id != "neg"));
}

#[test]
fn item_declaring_repo_kind_is_skipped() {
    // `kind: "repo"` on a hand-authored item would surface as a phantom repo in
    // the DuckDB `WHERE kind='repo'` leaderboard — must be skipped.
    let dir = tempfile::tempdir().unwrap();
    write_manifest(
        dir.path(),
        r#"{"repo": "ix", "items": [
            {"id": "ok", "kind": "demo", "title": "OK", "reach": 3, "impact": 3, "confidence": 3},
            {"id": "sneaky", "kind": "repo", "title": "Sneaky", "reach": 5, "impact": 5, "confidence": 5}
        ]}"#,
    );
    let rep = ingest_one("ix", dir.path());
    assert_eq!(rep.skipped, 1, "the kind:repo item is skipped");
    // Exactly one repo row — the generated rollup, id == repo name.
    let repo_rows: Vec<_> = rep.records.iter().filter(|r| r.kind == Kind::Repo).collect();
    assert_eq!(repo_rows.len(), 1);
    assert_eq!(repo_rows[0].id, "ix");
    assert!(rep.records.iter().all(|r| r.id != "sneaky"));
}

#[test]
fn malformed_manifest_is_skipped_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    write_manifest(dir.path(), "{ this is not json");
    let rep = ingest_one("ix", dir.path());
    assert!(rep.records.is_empty());
    assert_eq!(rep.roots_missing, vec!["ix".to_string()]);
}

#[test]
fn absent_sibling_is_reported_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    // ix has a manifest; ga sibling dir has none.
    write_manifest(
        dir.path(),
        r#"{"repo": "ix", "items": [{"id": "x", "title": "X", "reach": 2, "impact": 2, "confidence": 2}]}"#,
    );
    let absent = dir.path().join("no-ga-here");
    let rep = ingest(&[SourceRoot::new("ix", dir.path()), SourceRoot::new("ga", &absent)]);
    assert!(rep.roots_seen.contains(&"ix".to_string()));
    assert!(rep.roots_missing.contains(&"ga".to_string()));
    assert!(!rep.records.is_empty());
}

#[test]
fn drift_detects_uncatalogued_and_is_repo_scoped() {
    let dir = tempfile::tempdir().unwrap();
    write_manifest(
        dir.path(),
        r#"{"repo": "ix", "items": [{"id": "x", "title": "X", "reach": 2, "impact": 2, "confidence": 2}]}"#,
    );
    let rep = ingest_one("ix", dir.path());

    // Fresh vs empty committed → the ix records are "missing" (uncatalogued).
    let d = drift(&rep.records, &[], &rep.roots_seen);
    assert!(!d.is_clean());
    assert!(!d.missing.is_empty());

    // Fresh vs itself → clean.
    let d2 = drift(&rep.records, &rep.records, &rep.roots_seen);
    assert!(d2.is_clean());

    // A committed record from an UNSCANNED repo is NOT flagged stale (repo-scoped).
    let mut foreign = rep.records.clone();
    for r in &mut foreign {
        r.repo = "tars".to_string();
    }
    let d3 = drift(&rep.records, &foreign, &rep.roots_seen);
    assert!(d3.extra.is_empty(), "tars records ignored when tars wasn't scanned");
}
