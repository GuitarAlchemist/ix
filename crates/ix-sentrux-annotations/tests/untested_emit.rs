//! Integration test for the `--emit-untested` mode.
//!
//! Spec from the task brief:
//!
//! > with 2 business-value annotations + 5 untested files in fixture,
//! > emits exactly 2 untested smells (the intersection)
//!
//! We construct a temp workspace with two source files carrying
//! `@ai:business-value` markers, load the Pro-tier fixture (5 untested
//! files, two of which are the business-value files), run the
//! intersection, and assert the resulting JSONL contains exactly the two
//! intersection annotations with the contract-mandated metadata.

use ix_sentrux_annotations::{
    emit_sidecar,
    test_gaps::parse_test_gaps_response,
    untested::{business_value_paths_from_workspace, untested_high_value_annotations},
};
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn intersection_with_2_bv_and_5_untested_emits_exactly_2() {
    // Step 1: build the workspace with two business-value files. The
    // remaining three "untested" entries in the fixture have NO source
    // file in this workspace — that's fine; the intersection filter is
    // purely path-based and doesn't need the files to exist.
    let dir = tempdir().unwrap();
    let ws = dir.path();
    fs::create_dir_all(ws.join("crates/foo/src")).unwrap();
    fs::create_dir_all(ws.join("crates/bar/src")).unwrap();

    // Business-value markers — these are the source-of-truth product-owner
    // tags that the bridge filters by.
    fs::write(
        ws.join("crates/foo/src/core.rs"),
        "// @ai:business-value chord recognizer core [T:manually-reviewed conf:0.95 src:po@2026-05-24]\nfn core() {}\n",
    )
    .unwrap();
    fs::write(
        ws.join("crates/bar/src/main.rs"),
        "// @ai:business-value voicing search hot path [T:manually-reviewed conf:0.9 src:po@2026-05-24]\nfn main() {}\n",
    )
    .unwrap();

    // Step 2: load the Pro-tier test_gaps fixture from the canonical path.
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/test_gaps_pro.json");
    let raw = fs::read_to_string(&fixture_path).expect("fixture present");
    let bare: serde_json::Value = serde_json::from_str(&raw).unwrap();
    // Parse the fixture as a bare TestGapsReport (no MCP envelope).
    let report: ix_sentrux_annotations::test_gaps::TestGapsReport =
        serde_json::from_value(bare).unwrap();
    assert_eq!(report.untested_files.len(), 5, "fixture has 5 untested");

    // Step 3: scan the workspace for business-value annotations.
    let bv_paths = business_value_paths_from_workspace(ws);
    assert_eq!(bv_paths.len(), 2, "workspace has 2 business-value files");

    // Step 4: compute the intersection.
    let now = "2026-05-24T18:00:00Z";
    let annos = untested_high_value_annotations(&report.untested_files, &bv_paths, now);

    // The contract: exactly 2 untested-smell annotations (the intersection).
    assert_eq!(
        annos.len(),
        2,
        "expected 2 intersection annotations, got {}",
        annos.len()
    );
    let paths: Vec<&str> = annos.iter().map(|a| a.location.path.as_str()).collect();
    assert!(paths.contains(&"crates/foo/src/core.rs"));
    assert!(paths.contains(&"crates/bar/src/main.rs"));

    // Step 5: emit the sidecar JSONL and verify on-disk shape.
    let outcome = emit_sidecar(
        ws,
        &annos,
        Some(&ws.join("state/quality/ai-annotations-sentrux-untested.jsonl")),
    )
    .unwrap();
    assert_eq!(outcome.written, 2);
    let body = fs::read_to_string(ws.join("state/quality/ai-annotations-sentrux-untested.jsonl"))
        .unwrap();
    assert_eq!(body.lines().count(), 2);
    for line in body.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["kind"], "smell");
        assert_eq!(v["truth_value"], "F");
        assert_eq!(v["certainty"], "detected-by-sentrux");
        assert_eq!(v["confidence"], 1.0);
        assert_eq!(v["source"]["author"], "sentrux");
        assert!(v["source"]["evidence"]
            .as_str()
            .unwrap()
            .starts_with("sentrux-test-gaps@"));
        assert_eq!(v["claim"], "no test coverage detected by sentrux");
        assert_eq!(v["location"]["line_start"], 1);
        assert_eq!(v["location"]["line_end"], 1);
    }
}

#[test]
fn intersection_is_empty_when_workspace_has_no_business_value() {
    // Safety gate: even if sentrux reports 1000 untested files, an
    // operator who hasn't tagged any business-value files yet sees ZERO
    // emitted annotations. Prevents the bridge from spamming a fresh
    // codebase.
    let dir = tempdir().unwrap();
    let ws = dir.path();
    fs::create_dir_all(ws.join("src")).unwrap();
    fs::write(ws.join("src/lib.rs"), "// no markers here\nfn foo() {}\n").unwrap();

    let untested: Vec<String> = (0..1000).map(|i| format!("src/file{i}.rs")).collect();
    let bv = business_value_paths_from_workspace(ws);
    assert!(bv.is_empty());
    let annos = untested_high_value_annotations(&untested, &bv, "now");
    assert!(annos.is_empty());
}

#[test]
fn fixture_envelope_round_trips_through_parse_test_gaps_response() {
    // Cross-check that the MCP envelope form parses the same way as the
    // bare TestGapsReport form, so callers can use either fixture shape
    // interchangeably.
    let bare_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/test_gaps_pro.json");
    let bare_raw = fs::read_to_string(&bare_path).unwrap();

    // Wrap the bare fixture in an MCP envelope and parse via the public
    // unwrap helper — both paths should reach the same TestGapsReport.
    let envelope = serde_json::json!({
        "result": {
            "content": [{
                "type": "text",
                "text": bare_raw
            }]
        }
    });
    let via_envelope = parse_test_gaps_response(&envelope).expect("parses");
    let via_bare: ix_sentrux_annotations::test_gaps::TestGapsReport =
        serde_json::from_str(&bare_raw).unwrap();

    assert_eq!(via_envelope.untested_files, via_bare.untested_files);
    assert_eq!(via_envelope.untested, via_bare.untested);
    assert!(via_envelope.has_per_file());
}
