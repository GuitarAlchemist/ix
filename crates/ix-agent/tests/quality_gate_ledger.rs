//! `ix_quality_gate_history` end-to-end, through the same entry point
//! `main.rs` uses for `tools/call`.
//!
//! The tool read `state/quality/gate-ledger.jsonl` from the day it shipped,
//! but nothing in ix wrote that file, so every call returned `count: 0` from
//! an absent ledger — a shape an agent reads as "no gate failures". The
//! producer now exists (`ix doctor`), and these tests hold both halves: real
//! rows come back, and an absent ledger says so out loud instead of
//! answering with a reassuring zero.

use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use ix_quality_trend::{append_entry, GateDecision, GateLedgerEntry, GateMetric};
use serde_json::{json, Value};
use std::path::Path;
use tempfile::TempDir;

/// A temp dir inside the workspace root: since ix#350 an explicit
/// `ledger_path` is confined to the allowed roots, and the system temp dir is
/// not one of them.
fn in_root_tempdir() -> TempDir {
    let parent = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/confine-test");
    std::fs::create_dir_all(&parent).expect("create target/confine-test");
    // Canonical, without the Windows verbatim prefix the tools refuse as input.
    let canonical = parent.canonicalize().expect("canonical confine-test dir");
    let parent = match canonical.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(plain) => std::path::PathBuf::from(plain),
        None => canonical,
    };
    tempfile::Builder::new()
        .tempdir_in(parent)
        .expect("tempdir inside the workspace root")
}

/// Call the tool the way the MCP server does.
fn call(params: Value) -> Value {
    let (ctx, _rx) = ServerContext::new();
    ToolRegistry::new()
        .call_with_ctx("ix_quality_gate_history", params, &ctx)
        .expect("ix_quality_gate_history should not error")
}

fn entry(source: &str, domain: &str, decision: GateDecision, value: f64) -> GateLedgerEntry {
    GateLedgerEntry::new(
        source,
        domain,
        decision,
        GateMetric {
            name: "doctor_checks_failing".to_string(),
            value,
            threshold: Some(0.0),
            trend: None,
        },
    )
}

/// Pin `run_at`. Two entries minted in the same instant would otherwise sort
/// by whatever `Utc::now()` resolution the host happens to give us, and the
/// newest-first assertions below would be a coin flip.
fn at(mut e: GateLedgerEntry, rfc3339: &str) -> GateLedgerEntry {
    e.run_at = rfc3339.parse().expect("valid RFC3339");
    e
}

fn seed(path: &Path, entries: &[GateLedgerEntry]) {
    for e in entries {
        append_entry(path, e).expect("seed ledger");
    }
}

#[test]
fn reads_back_rows_a_producer_appended() {
    let dir = in_root_tempdir();
    let path = dir.path().join("state/quality/gate-ledger.jsonl");
    seed(
        &path,
        &[
            at(
                entry("ix-doctor", "harness", GateDecision::Pass, 0.0),
                "2026-09-01T10:00:00Z",
            ),
            at(
                entry("ix-doctor", "harness", GateDecision::Fail, 2.0),
                "2026-09-02T10:00:00Z",
            ),
        ],
    );

    let out = call(json!({ "ledger_path": path.display().to_string() }));

    assert_eq!(out["ledger_status"], "present");
    assert_eq!(out["count"], 2);
    assert_eq!(out["note"], Value::Null, "nothing to warn about");
    let rows = out["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row["schema"], "quality-gate-ledger-v1");
        assert_eq!(row["source"], "ix-doctor");
        assert_eq!(row["metric"]["name"], "doctor_checks_failing");
    }
    // Newest first, and the failing run is the later one.
    assert_eq!(rows[0]["decision"], "fail");
}

/// The honesty test. An absent ledger must not look like a clean run.
#[test]
fn absent_ledger_is_reported_as_absent_not_as_zero_failures() {
    let dir = in_root_tempdir();
    let path = dir.path().join("state/quality/gate-ledger.jsonl");
    assert!(!path.exists());

    let out = call(json!({ "ledger_path": path.display().to_string() }));

    assert_eq!(out["ledger_status"], "absent");
    assert_eq!(out["count"], 0);

    let note = out["note"]
        .as_str()
        .expect("absent ledger must carry a note");
    assert!(
        note.contains("no quality gate has recorded a run"),
        "note must say no gate ran, got: {note}"
    );
    assert!(
        note.contains("NOT evidence that gates passed"),
        "note must refuse the pass reading, got: {note}"
    );
    assert!(
        note.contains("ix doctor") || note.contains("doctor"),
        "note must name the producer that fixes it, got: {note}"
    );
}

#[test]
fn empty_ledger_is_distinguished_from_an_absent_one() {
    let dir = in_root_tempdir();
    let path = dir.path().join("gate-ledger.jsonl");
    std::fs::write(&path, "\n").expect("write empty ledger");

    let out = call(json!({ "ledger_path": path.display().to_string() }));

    assert_eq!(out["ledger_status"], "empty");
    assert_eq!(out["count"], 0);
    assert!(out["note"]
        .as_str()
        .expect("note")
        .contains("not as a pass"));
}

/// "Gates ran, none matched your filter" and "no gate ever ran" both give
/// `count: 0`. They must not give the same `ledger_status`.
#[test]
fn filtered_to_nothing_still_reports_the_ledger_as_present() {
    let dir = in_root_tempdir();
    let path = dir.path().join("gate-ledger.jsonl");
    seed(
        &path,
        &[entry("ix-doctor", "harness", GateDecision::Pass, 0.0)],
    );

    let out = call(json!({
        "ledger_path": path.display().to_string(),
        "source": "a-producer-that-never-ran",
    }));

    assert_eq!(out["count"], 0);
    assert_eq!(out["ledger_status"], "present");
    assert!(out["note"]
        .as_str()
        .expect("note")
        .contains("none match these filters"));
}

#[test]
fn filters_select_by_source_domain_and_decision() {
    let dir = in_root_tempdir();
    let path = dir.path().join("gate-ledger.jsonl");
    seed(
        &path,
        &[
            entry("ix-doctor", "harness", GateDecision::Pass, 0.0),
            entry("sentrux", "structural", GateDecision::Fail, 1.0),
            entry("ix-doctor", "harness", GateDecision::Fail, 3.0),
        ],
    );
    let p = path.display().to_string();

    let by_source = call(json!({ "ledger_path": p, "source": "sentrux" }));
    assert_eq!(by_source["count"], 1);
    assert_eq!(by_source["rows"][0]["domain"], "structural");

    let by_decision = call(json!({ "ledger_path": p, "decision": "fail" }));
    assert_eq!(by_decision["count"], 2);

    let both = call(json!({
        "ledger_path": p,
        "domain": "harness",
        "decision": "fail",
    }));
    assert_eq!(both["count"], 1);
    assert_eq!(both["rows"][0]["metric"]["value"], 3.0);

    let capped = call(json!({ "ledger_path": p, "limit": 1 }));
    assert_eq!(capped["count"], 1);
}

/// Legacy v0 rows (ga's chatbot-PR shape) share the file. They are excluded
/// from `rows` by design, but they still count as history — the ledger is
/// `present`, not `empty`.
#[test]
fn legacy_v0_rows_count_as_present_history() {
    let dir = in_root_tempdir();
    let path = dir.path().join("gate-ledger.jsonl");
    std::fs::write(
        &path,
        "{\"pr\":155,\"branch\":\"chatbot/x\",\"decision\":\"merged-clean\"}\n",
    )
    .expect("write legacy row");

    let out = call(json!({ "ledger_path": path.display().to_string() }));

    assert_eq!(out["ledger_status"], "present");
    assert_eq!(out["count"], 0, "v0 rows are not returned as v1 rows");
    assert!(out["note"].as_str().expect("note").contains("none match"));
}

#[test]
fn a_bad_decision_filter_is_an_error_not_a_silent_empty_result() {
    let (ctx, _rx) = ServerContext::new();
    let err = ToolRegistry::new()
        .call_with_ctx(
            "ix_quality_gate_history",
            json!({ "decision": "probably" }),
            &ctx,
        )
        .expect_err("unknown decision must be rejected");
    assert!(err.contains("probably"), "got: {err}");
}
