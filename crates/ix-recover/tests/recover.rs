//! In-process behaviour of `ix-recover`. The real process kills are in
//! `kill.rs`; here a kill is never needed, and the corruptions are written by
//! the test itself (simulated evidence damage, labelled as such).

use ix_recover::{canonical_json, Attempt, Begin, Error, Point, Record, Store};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

const SUMMARY: &[u8] = b"{\"mean\":2}\n";
const LOCK: &[u8] = b"schema: ix-lock/v2\n";

fn store() -> (tempfile::TempDir, Store) {
    let t = tempfile::tempdir().unwrap();
    let s = Store::open(t.path().join("runs")).unwrap();
    (t, s)
}

fn bindings() -> Value {
    json!({ "spec_sha256": "sha256:aa", "ix_exe_sha256": "sha256:bb" })
}

fn fresh(s: &Store, id: &str) -> Attempt {
    match s.begin(id, &bindings()).unwrap() {
        Begin::Fresh(a) => a,
        other => panic!("expected Fresh, got {other:?}"),
    }
}

fn published(s: &Store, id: &str) -> Record {
    fresh(s, id)
        .publish(&[("summary.json", SUMMARY), ("ix.lock", LOCK)])
        .unwrap()
}

fn run_dir(s: &Store, id: &str) -> PathBuf {
    s.root().join(id)
}

fn attempt_dir(s: &Store, id: &str, rec: &Record) -> PathBuf {
    run_dir(s, id).join("attempts").join(rec.attempt_id())
}

fn expect_corrupt(s: &Store, id: &str, needle: &str) {
    match s.begin(id, &bindings()) {
        Err(Error::Corrupt(m)) => {
            assert!(m.contains(needle), "corrupt message {m:?} lacks {needle:?}")
        }
        other => panic!("expected Corrupt mentioning {needle:?}, got {other:?}"),
    }
}

#[test]
fn test_publish_then_begin_replays_the_same_files() {
    let (_t, s) = store();
    let rec = published(&s, "r1");
    assert_eq!(rec.file("summary.json"), Some(SUMMARY));
    match s.begin("r1", &bindings()).unwrap() {
        Begin::Done(again) => assert_eq!(again, rec),
        other => panic!("expected Done, got {other:?}"),
    }
}

#[test]
fn test_different_bindings_conflict_and_name_the_binding() {
    let (_t, s) = store();
    published(&s, "r1");
    let other = json!({ "spec_sha256": "sha256:cc", "ix_exe_sha256": "sha256:bb" });
    match s.begin("r1", &other) {
        Err(Error::Conflict(m)) => {
            assert!(m.contains("spec_sha256"), "{m}");
            assert!(!m.contains("ix_exe_sha256"), "{m}");
        }
        other => panic!("expected Conflict, got {other:?}"),
    }
}

#[test]
fn test_bindings_are_recorded_before_any_publish() {
    let (_t, s) = store();
    drop(fresh(&s, "r1")); // begun, never published
    let other = json!({ "spec_sha256": "sha256:cc", "ix_exe_sha256": "sha256:bb" });
    assert!(matches!(s.begin("r1", &other), Err(Error::Conflict(_))));
    assert!(matches!(s.begin("r1", &bindings()), Ok(Begin::Fresh(_))));
}

#[test]
fn test_a_held_run_is_unknown_until_released() {
    let (_t, s) = store();
    let held = fresh(&s, "r1");
    assert!(matches!(s.begin("r1", &bindings()), Err(Error::Unknown(_))));
    drop(held);
    assert!(matches!(s.begin("r1", &bindings()), Ok(Begin::Fresh(_))));
}

#[test]
fn test_a_missing_receipt_is_repaired_once() {
    let (_t, s) = store();
    let rec = published(&s, "r1");
    fs::remove_file(run_dir(&s, "r1").join("receipt.json")).unwrap();
    match s.begin("r1", &bindings()).unwrap() {
        Begin::Repaired(again) => assert_eq!(again, rec),
        other => panic!("expected Repaired, got {other:?}"),
    }
    assert!(matches!(s.begin("r1", &bindings()), Ok(Begin::Done(_))));
}

#[test]
fn test_foreign_commit_during_attempt_is_refused() {
    let (_t, s) = store();
    let dir = run_dir(&s, "r1");
    let mut planted = false;
    let result = fresh(&s, "r1").publish_observed(&[("summary.json", SUMMARY)], &mut |p| {
        if p == Point::Staged {
            fs::write(dir.join("committed.json"), b"{}").unwrap();
            planted = true;
        }
    });
    assert!(planted);
    match result {
        Err(Error::Corrupt(m)) => assert!(m.contains("appeared"), "{m}"),
        other => panic!("expected Corrupt, got {other:?}"),
    }
    assert!(!dir.join("receipt.json").exists());
    expect_corrupt(&s, "r1", "committed.json");
}

#[test]
fn test_a_tampered_file_is_corrupt() {
    let (_t, s) = store();
    let rec = published(&s, "r1");
    fs::write(
        attempt_dir(&s, "r1", &rec).join("summary.json"),
        b"{\"mean\":3}\n",
    )
    .unwrap();
    expect_corrupt(&s, "r1", "does not match its digest");
}

#[test]
fn test_an_extra_file_in_the_attempt_is_corrupt() {
    let (_t, s) = store();
    let rec = published(&s, "r1");
    fs::write(attempt_dir(&s, "r1", &rec).join("extra"), b"x").unwrap();
    expect_corrupt(&s, "r1", "holds");
}

#[test]
fn test_a_missing_file_in_the_attempt_is_corrupt() {
    let (_t, s) = store();
    let rec = published(&s, "r1");
    fs::remove_file(attempt_dir(&s, "r1", &rec).join("ix.lock")).unwrap();
    expect_corrupt(&s, "r1", "holds");
}

fn rewrite_receipt(s: &Store, id: &str, edit: impl FnOnce(&str) -> String) {
    let path = run_dir(s, id).join("receipt.json");
    let text = fs::read_to_string(&path).unwrap();
    fs::write(&path, edit(&text)).unwrap();
}

#[test]
fn test_an_unknown_receipt_field_is_corrupt() {
    let (_t, s) = store();
    published(&s, "r1");
    rewrite_receipt(&s, "r1", |t| t.replacen('{', "{\n  \"extra\": 1,", 1));
    expect_corrupt(&s, "r1", "unknown field");
}

#[test]
fn test_a_duplicated_receipt_field_is_corrupt() {
    let (_t, s) = store();
    published(&s, "r1");
    rewrite_receipt(&s, "r1", |t| {
        t.replacen('{', "{\n  \"attempt_id\": \"att-x\",", 1)
    });
    expect_corrupt(&s, "r1", "duplicate field");
}

#[test]
fn test_a_receipt_for_another_manifest_is_corrupt() {
    let (_t, s) = store();
    published(&s, "r1");
    let path = run_dir(&s, "r1").join("receipt.json");
    let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    v["manifest_sha256"] = json!("sha256:00");
    fs::write(&path, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    expect_corrupt(&s, "r1", "manifest_sha256");
}

#[test]
fn test_an_intent_that_does_not_match_its_digest_is_corrupt() {
    let (_t, s) = store();
    published(&s, "r1");
    let path = run_dir(&s, "r1").join("intent.json");
    let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    v["bindings"]["spec_sha256"] = json!("sha256:zz");
    fs::write(&path, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    expect_corrupt(&s, "r1", "bindings_sha256");
}

#[test]
fn test_a_commit_without_an_intent_is_corrupt() {
    let (_t, s) = store();
    published(&s, "r1");
    fs::remove_file(run_dir(&s, "r1").join("intent.json")).unwrap();
    expect_corrupt(&s, "r1", "without intent.json");
}

#[test]
fn test_unsafe_run_ids_are_refused() {
    let (_t, s) = store();
    let long = "x".repeat(129);
    for id in [
        "",
        ".",
        "..",
        "a/b",
        "a\\b",
        "a b",
        "trailing.",
        "con",
        "NUL.txt",
        "lpt1",
        "Com9",
        &long,
    ] {
        assert!(
            matches!(s.begin(id, &bindings()), Err(Error::Usage(_))),
            "run id {id:?} was accepted"
        );
    }
    assert!(
        fs::read_dir(s.root()).unwrap().next().is_none(),
        "a refused id left a directory"
    );
}

#[test]
fn test_bad_bindings_are_refused() {
    let (_t, s) = store();
    assert!(matches!(s.begin("r1", &json!(["a"])), Err(Error::Usage(_))));
    assert!(matches!(
        s.begin("r1", &json!({ "k": 0.5 })),
        Err(Error::Usage(_))
    ));
}

#[test]
fn test_duplicate_or_reserved_file_names_are_refused() {
    let (_t, s) = store();
    for files in [
        vec![("a", SUMMARY), ("A", SUMMARY)],
        vec![("manifest.json", SUMMARY)],
        vec![("../x", SUMMARY)],
    ] {
        assert!(matches!(
            fresh(&s, "r1").publish(&files),
            Err(Error::Usage(_))
        ));
    }
    assert!(matches!(s.begin("r1", &bindings()), Ok(Begin::Fresh(_))));
}

#[cfg(windows)]
#[test]
fn test_run_ids_differing_only_in_case_conflict_on_windows() {
    let (_t, s) = store();
    published(&s, "Run1");
    match s.begin("run1", &bindings()) {
        Err(Error::Conflict(m)) => assert!(m.contains("case-insensitive"), "{m}"),
        other => panic!("expected Conflict, got {other:?}"),
    }
}

#[test]
fn test_canonical_json_sorts_keys_at_every_depth() {
    let v = json!({ "b": [1, { "d": 2, "c": "x" }], "a": true });
    assert_eq!(canonical_json(&v), r#"{"a":true,"b":[1,{"c":"x","d":2}]}"#);
}
