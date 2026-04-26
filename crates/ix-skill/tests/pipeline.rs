//! Integration tests for `ix pipeline {new,validate,dag,run}`.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

fn ix_in(dir: &std::path::Path) -> Command {
    let mut cmd = Command::cargo_bin("ix").expect("ix binary built");
    cmd.current_dir(dir).env(
        "IX_GOVERNANCE_DIR",
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../governance/demerzel"),
    );
    cmd
}

fn tempdir(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("ix-pipeline-test-{}-{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn new_scaffolds_valid_yaml() {
    let dir = tempdir("new");
    let out = ix_in(&dir)
        .args(["--format", "json", "pipeline", "new", "demo"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["action"], "created");
    assert_eq!(value["stages"], 1);
    assert!(dir.join("ix.yaml").is_file());
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn new_refuses_to_overwrite() {
    let dir = tempdir("overwrite");
    fs::write(dir.join("ix.yaml"), "version: \"1\"\nstages: {}\n").unwrap();
    ix_in(&dir)
        .args(["pipeline", "new", "demo"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("already exists"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn validate_catches_unknown_skill() {
    let dir = tempdir("unknown_skill");
    fs::write(
        dir.join("ix.yaml"),
        r#"version: "1"
stages:
  bad:
    skill: no.such.skill
    args: {}
"#,
    )
    .unwrap();
    ix_in(&dir)
        .args(["pipeline", "validate"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown skill"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn validate_catches_unknown_dep() {
    let dir = tempdir("unknown_dep");
    fs::write(
        dir.join("ix.yaml"),
        r#"version: "1"
stages:
  a:
    skill: stats
    deps: [missing]
"#,
    )
    .unwrap();
    ix_in(&dir)
        .args(["pipeline", "validate"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown upstream stage"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn run_executes_single_stage() {
    let dir = tempdir("single");
    fs::write(
        dir.join("ix.yaml"),
        r#"version: "1"
stages:
  load:
    skill: stats
    args:
      data: [1.0, 2.0, 3.0, 4.0, 5.0]
"#,
    )
    .unwrap();
    let out = ix_in(&dir)
        .args(["--format", "json", "pipeline", "run"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["stages"]["load"]["output"]["mean"], 3.0);
    assert_eq!(value["stages"]["load"]["output"]["count"], 5);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn run_executes_two_stages_with_dep() {
    let dir = tempdir("two_stages");
    // Two-stage: compute stats, then check governance with the action text.
    fs::write(
        dir.join("ix.yaml"),
        r#"version: "1"
stages:
  load:
    skill: stats
    args:
      data: [10.0, 20.0, 30.0]
  audit:
    skill: governance.check
    args:
      action: "read statistics"
    deps: [load]
"#,
    )
    .unwrap();
    let out = ix_in(&dir)
        .args(["--format", "json", "pipeline", "run"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["stages"]["load"]["output"]["mean"], 20.0);
    assert!(value["stages"]["audit"]["output"].is_object());
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn dag_reports_parallel_depth() {
    let dir = tempdir("dag_depth");
    fs::write(
        dir.join("ix.yaml"),
        r#"version: "1"
stages:
  a:
    skill: stats
    args: { data: [1.0, 2.0] }
  b:
    skill: stats
    args: { data: [3.0, 4.0] }
  c:
    skill: governance.check
    args: { action: "combine" }
    deps: [a, b]
"#,
    )
    .unwrap();
    let out = ix_in(&dir)
        .args(["--format", "json", "pipeline", "dag"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    // a and b are independent → level 0. c depends on both → level 1.
    assert_eq!(value["parallel_depth"], 2);
    assert_eq!(value["levels"][0].as_array().unwrap().len(), 2);
    assert_eq!(value["levels"][1].as_array().unwrap().len(), 1);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn run_ndjson_streams_events() {
    let dir = tempdir("ndjson");
    fs::write(
        dir.join("ix.yaml"),
        r#"version: "1"
stages:
  load:
    skill: stats
    args: { data: [1.0, 2.0, 3.0] }
"#,
    )
    .unwrap();
    let out = ix_in(&dir)
        .args(["pipeline", "run", "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
        lines.len() >= 3,
        "expected start + stage + done, got: {stdout}"
    );
    // Every line must be valid JSON with an `event` key.
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line).expect("valid ndjson line");
        assert!(v["event"].is_string());
    }
    // First event is `start`, last is `done`.
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(lines[0]).unwrap()["event"],
        "start"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(lines.last().unwrap()).unwrap()["event"],
        "done"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn run_with_from_reference_flows_data() {
    let dir = tempdir("from_ref");
    // Use `{"from": "load"}` to pass load's output (full object) as `action`
    // to governance.check. governance.check expects string action, so this
    // should trigger a type error OR produce a plain result — either way the
    // `from` reference machinery should resolve without erroring early.
    fs::write(
        dir.join("ix.yaml"),
        r#"version: "1"
stages:
  load:
    skill: stats
    args: { data: [1.0, 2.0, 3.0] }
  audit:
    skill: governance.check
    args:
      action: "processed upstream stats"
      context: { from: load.mean }
"#,
    )
    .unwrap();
    let out = ix_in(&dir)
        .args(["--format", "json", "pipeline", "run"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    // Audit ran successfully; load's mean was 2.0.
    assert_eq!(value["stages"]["load"]["output"]["mean"], 2.0);
    assert!(value["stages"]["audit"]["output"].is_object());
    fs::remove_dir_all(&dir).ok();
}
