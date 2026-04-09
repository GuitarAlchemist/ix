//! Integration tests exercising the `ix` binary end-to-end.
//!
//! Uses `assert_cmd` to spawn the CLI and inspect its stdout + exit code.
//! Keeps to batch1/batch2 skills whose handlers are deterministic and fast.

use assert_cmd::Command;
use predicates::prelude::*;

fn ix() -> Command {
    let mut cmd = Command::cargo_bin("ix").expect("ix binary built");
    // Point at the governance submodule two dirs up from the crate root.
    cmd.env(
        "IX_GOVERNANCE_DIR",
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../governance/demerzel"),
    );
    cmd
}

#[test]
fn list_skills_returns_all_entries() {
    let out = ix()
        .args(["--format", "json", "list", "skills"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
    let count = value["count"].as_u64().expect("count is number");
    // 45 = batch1 (6) + batch2 (28) + batch3 (9) + prime_radiant (2).
    // If this drifts, update the assertion alongside the batch changes.
    assert_eq!(count, 45, "expected 45 registry skills, got {count}");
}

#[test]
fn list_domains_includes_governance() {
    let out = ix()
        .args(["--format", "json", "list", "domains"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("\"governance\""));
    assert!(stdout.contains("\"math\""));
}

#[test]
fn describe_skill_shows_schema() {
    let out = ix()
        .args(["--format", "json", "describe", "skill", "stats"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
    assert_eq!(value["name"], "stats");
    assert_eq!(value["domain"], "math");
    assert!(value["schema"]["properties"]["data"].is_object());
}

#[test]
fn describe_skill_unknown_fails() {
    ix()
        .args(["describe", "skill", "does.not.exist"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("skill not found"));
}

#[test]
fn run_stats_via_stdin() {
    let out = ix()
        .args(["--format", "json", "run", "stats"])
        .write_stdin("{\"data\":[1.0,2.0,3.0,4.0,5.0]}")
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
    assert_eq!(value["mean"].as_f64(), Some(3.0));
    assert_eq!(value["count"].as_u64(), Some(5));
}

#[test]
fn run_number_theory_gcd_via_input_flag() {
    let out = ix()
        .args([
            "--format",
            "json",
            "run",
            "number_theory",
            "--input",
            "{\"operation\":\"gcd\",\"a\":48,\"b\":18}",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
    assert_eq!(value["gcd"].as_u64(), Some(6));
}

#[test]
fn run_unknown_skill_fails() {
    ix()
        .args(["run", "does.not.exist"])
        .write_stdin("{}")
        .assert()
        .failure()
        .stderr(predicate::str::contains("skill not found"));
}

#[test]
fn check_doctor_exits_with_hexavalent_code() {
    // Exit code is hexavalent: 0=T, 1=P, 2=U, 3=D, 4=F, 5=C.
    // When the governance submodule is present, verdict should be T (0).
    let assert = ix().args(["--format", "json", "check", "doctor"]).assert();
    // Accept 0 (T) or 1 (P) — warn if any optional check is missing.
    let code = assert.get_output().status.code().unwrap_or(-1);
    assert!(
        code == 0 || code == 1,
        "expected hexavalent T (0) or P (1), got {code}"
    );
}

#[test]
fn check_action_detects_dangerous_keywords() {
    let assert = ix()
        .args([
            "--format",
            "json",
            "check",
            "action",
            "delete the production database",
        ])
        .assert();
    // "delete" and "drop table" trigger D (doubtful).
    let code = assert.get_output().status.code().unwrap_or(-1);
    assert_eq!(code, 3, "dangerous keyword should yield D/doubtful (3)");
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
    assert_eq!(value["verdict"], "D");
    assert_eq!(value["dangerous_keywords_matched"], true);
}

#[test]
fn check_action_benign_is_true() {
    let assert = ix()
        .args(["--format", "json", "check", "action", "log the response"])
        .assert();
    let code = assert.get_output().status.code().unwrap_or(-1);
    assert!(
        code == 0 || code == 1,
        "benign action should be T (0) or P (1), got {code}"
    );
}

#[test]
fn list_skills_with_domain_filter() {
    let out = ix()
        .args([
            "--format", "json", "list", "skills", "--domain", "governance",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
    assert_eq!(value["count"].as_u64(), Some(6)); // 4 governance + 2 governance.graph skills
}

#[test]
fn serve_verb_reports_stub() {
    ix()
        .args(["serve", "repl"])
        .assert()
        .code(2) // UNKNOWN exit code for stubs
        .stderr(predicate::str::contains("coming"));
}
