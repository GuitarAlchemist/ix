//! Integration tests for the Streeling generator against a fixture repo layout.

use ix_streeling::{campus, check, default_roots, from_jsonl, ingest::ingest, to_jsonl};
use std::fs;
use std::path::Path;

fn write(path: &Path, body: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

/// Build a parent dir containing an `ix` repo and (optionally) a `ga` sibling.
fn fixture(with_ga: bool) -> (tempfile::TempDir, std::path::PathBuf) {
    let parent = tempfile::tempdir().unwrap();
    let ix = parent.path().join("ix");

    write(
        &ix.join("docs/solutions/build-errors/wdac.md"),
        "---\ntitle: \"WDAC blocks test binaries\"\ncategory: build-errors\ndate: 2026-04-10\ntags: [windows, cargo]\nsymptom: \"os error 4551\"\nroot_cause: \"App Control policy\"\n---\nbody\n",
    );
    write(
        &ix.join("docs/plans/2026-01-01-001-feat-thing-plan.md"),
        "---\ntitle: \"A plan\"\ntype: feat\nstatus: active\ndate: 2026-01-01\n---\nplan body\n",
    );
    write(
        &ix.join("docs/brainstorms/2026-01-02-topic-brainstorm.md"),
        "---\ndate: 2026-01-02\ntopic: some-topic\n---\nbrainstorm body\n",
    );
    // Malformed: no frontmatter at all -> skipped.
    write(&ix.join("docs/solutions/README.md"), "# index, no frontmatter\n");

    if with_ga {
        let ga = parent.path().join("ga");
        write(
            &ga.join("docs/solutions/integration-issues/cors.md"),
            "---\ntitle: \"CORS on dev server\"\ncategory: integration-issues\ndate: 2026-03-01\ntags: [cors]\n---\nbody\n",
        );
    }
    (parent, ix)
}

#[test]
// @ai:invariant ingest parses YAML frontmatter from docs/solutions/plans/brainstorms across ix+ga into LearningRecords and skips files without valid frontmatter [T:test conf:0.9 src:ix-streeling::tests::ingest_parses_and_skips]
fn ingest_parses_and_skips() {
    let (_parent, ix) = fixture(true);
    let rep = ingest(&default_roots(&ix));

    // ix: 1 solution + 1 plan + 1 brainstorm; ga: 1 solution. README skipped.
    assert_eq!(rep.records.len(), 4, "records: {:#?}", rep.records);
    assert_eq!(rep.skipped, 1, "the no-frontmatter README should be skipped");
    assert!(rep.roots_seen.contains(&"ix".to_string()));
    assert!(rep.roots_seen.contains(&"ga".to_string()));

    let wdac = rep.records.iter().find(|r| r.title.contains("WDAC")).unwrap();
    assert_eq!(wdac.repo, "ix");
    assert_eq!(wdac.category, "build-errors");
    assert_eq!(wdac.symptom.as_deref(), Some("os error 4551"));
    assert_eq!(wdac.id, format!("ix:{}", wdac.path));
    assert!(rep.records.iter().any(|r| r.repo == "ga" && r.category == "integration-issues"));
    assert!(rep.records.iter().any(|r| r.title == "some-topic"), "brainstorm topic -> title");
}

#[test]
// @ai:assumption a missing sibling clone is skipped (reported in roots_missing), not fatal [T:test conf:0.9 src:ix-streeling::tests::ingest_tolerates_absent_sibling]
fn ingest_tolerates_absent_sibling() {
    let (_parent, ix) = fixture(false); // no ga
    let rep = ingest(&default_roots(&ix));
    assert!(rep.roots_missing.contains(&"ga".to_string()));
    assert!(rep.records.iter().all(|r| r.repo == "ix"));
    assert!(!rep.records.is_empty());
}

#[test]
fn campus_render_is_idempotent_and_preserves_intro() {
    let (_parent, ix) = fixture(true);
    let records = ingest(&default_roots(&ix)).records;

    let first = campus::render(&records, None);
    // Inject a custom intro and re-render: intro preserved, body deterministic.
    let custom = first.replace(
        "Streeling University is the front door",
        "MY CUSTOM INTRO — Streeling University is the front door",
    );
    let second = campus::render(&records, Some(&custom));
    assert!(second.contains("MY CUSTOM INTRO"), "intro must be preserved");

    // Idempotent: rendering again from the same inputs yields the same output.
    let third = campus::render(&records, Some(&second));
    assert_eq!(second, third, "campus render must be idempotent");
}

#[test]
// @ai:invariant streeling check fails (drift not clean) when a learning is uncatalogued, and passes when the catalog matches the sources [T:test conf:0.9 src:ix-streeling::tests::check_detects_drift]
fn check_detects_drift() {
    let (_parent, ix) = fixture(true);
    let fresh = ingest(&default_roots(&ix)).records;

    let seen = vec!["ix".to_string(), "ga".to_string()];
    // A catalog that matches -> clean.
    let committed = from_jsonl(&to_jsonl(&fresh));
    assert!(check::drift(&fresh, &committed, &seen).is_clean());

    // Drop one record from the catalog -> drift (uncatalogued/missing).
    let stale: Vec<_> = committed.iter().skip(1).cloned().collect();
    let d = check::drift(&fresh, &stale, &seen);
    assert!(!d.is_clean());
    assert_eq!(d.missing.len(), 1);

    // A committed record from an UNSCANNED repo is not flagged stale.
    let mut with_foreign = committed.clone();
    if let Some(mut r) = committed.first().cloned() {
        r.repo = "tars".to_string();
        r.id = ix_streeling::model::LearningRecord::make_id("tars", &r.path);
        with_foreign.push(r);
    }
    assert!(
        check::drift(&fresh, &with_foreign, &seen).is_clean(),
        "records from a repo not in seen_repos must be ignored"
    );
}
