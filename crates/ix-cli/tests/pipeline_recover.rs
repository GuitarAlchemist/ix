//! `ix pipeline run --run-id` (provisional, `ix-recover`): one execution per
//! run id, replayed byte for byte afterwards. A removed receipt stands in for a
//! kill between the commit and the receipt (simulated here; real kills of the
//! ix process are in `pipeline_kill.rs`, of a store owner in
//! `crates/ix-recover/tests/kill.rs`).

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};

const SPEC: &str = r#"version: "1"
params:
  nums: null
stages:
  s:
    skill: stats
    args:
      data: { param: nums }
"#;

const PROVENANCE: &str = "state/thinking-machine/provenance.jsonl";

fn ix_in(dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("ix").expect("ix binary built");
    cmd.current_dir(dir).env(
        "IX_GOVERNANCE_DIR",
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../governance/demerzel"),
    );
    cmd
}

fn tempdir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("ix-recover-test-{}-{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("ix.yaml"), SPEC).unwrap();
    path
}

/// Run `ix --format json pipeline run --run-id ID --param nums=NUMS`;
/// return (success, stdout, stderr).
fn run(dir: &Path, id: &str, nums: &str) -> (bool, Vec<u8>, String) {
    let out = ix_in(dir)
        .args(["--format", "json", "pipeline", "run", "--run-id", id])
        .args(["--runs-dir", "runs", "--param"])
        .arg(format!("nums={nums}"))
        .output()
        .unwrap();
    (
        out.status.success(),
        out.stdout,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn attempts(dir: &Path, id: &str) -> usize {
    fs::read_dir(dir.join("runs").join(id).join("attempts"))
        .unwrap()
        .count()
}

/// A recoverable run writes nothing outside its runs directory.
fn assert_nothing_outside_the_store(dir: &Path) {
    assert!(
        !dir.join("ix.lock").exists(),
        "a recoverable run wrote ix.lock beside the spec"
    );
    assert!(
        !dir.join(PROVENANCE).exists(),
        "a recoverable run appended provenance"
    );
}

#[test]
fn test_run_id_replays_the_recorded_output_without_executing() {
    let dir = tempdir("replay");
    let (ok, first, err) = run(&dir, "r1", "[1.0, 2.0, 3.0]");
    assert!(ok, "first run failed: {err}");
    assert!(err.contains("published attempt"), "{err}");
    let summary: serde_json::Value = serde_json::from_slice(&first).unwrap();
    assert!(summary["stages"]["s"]["output"].is_object(), "{summary}");
    let (ok, again, err) = run(&dir, "r1", "[1.0, 2.0, 3.0]");
    assert!(ok, "replay failed: {err}");
    assert!(
        err.contains("replayed") && err.contains("nothing executed"),
        "{err}"
    );
    assert_eq!(again, first, "a replay must print the recorded bytes");
    assert_eq!(attempts(&dir, "r1"), 1, "a replay executed the pipeline");
    assert_nothing_outside_the_store(&dir);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_run_id_keeps_ix_lock_and_provenance_out_of_the_working_tree() {
    let dir = tempdir("outside");
    assert!(run(&dir, "r1", "[1.0, 2.0, 3.0]").0);
    assert_nothing_outside_the_store(&dir);
    let stored = dir.join("runs").join("r1").join("attempts");
    let attempt = fs::read_dir(&stored).unwrap().next().unwrap().unwrap();
    let lock = fs::read_to_string(attempt.path().join("ix.lock")).unwrap();
    assert!(lock.contains("ix-lock/v2"), "{lock}");
    // The same spec without --run-id writes both, so the checks above can fail.
    let out = ix_in(&dir)
        .args(["--format", "json", "pipeline", "run", "--param"])
        .arg("nums=[1.0, 2.0, 3.0]")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(dir.join("ix.lock").is_file() && dir.join(PROVENANCE).is_file());
    fs::remove_dir_all(&dir).ok();
}

/// The registered tools the review cites: each reads a file or the workspace,
/// runs a subprocess, or writes outside the store. Some carry the
/// `deterministic` tag, which is not the audit.
const UNAUDITED: [(&str, &str); 4] = [
    ("code_analyze", "path: \"ix.yaml\""),
    ("federation.discover", "domain: \"math\""),
    ("duckdb.query", "sql: \"select 1\""),
    ("session.flywheel_export", "session_path: \"session.json\""),
];

#[test]
fn test_run_id_refuses_unaudited_stages_before_the_store() {
    for (skill, args) in UNAUDITED {
        let dir = tempdir("unaudited");
        fs::write(
            dir.join("ix.yaml"),
            format!(
                "version: \"1\"\nstages:\n  s:\n    skill: stats\n    args:\n      data: [1.0, 2.0]\n  q:\n    skill: {skill}\n    args:\n      {args}\n"
            ),
        )
        .unwrap();
        let out = ix_in(&dir)
            .args(["pipeline", "run", "--run-id", "r1", "--runs-dir", "runs"])
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "{skill}: an unaudited stage must be refused"
        );
        assert!(
            err.contains("audited")
                && err.contains(&format!("q ({skill})"))
                && !err.contains("s (stats)"),
            "{skill}: {err}"
        );
        assert!(out.stdout.is_empty(), "{skill}");
        assert!(
            !dir.join("runs").exists(),
            "{skill}: a refused run touched the store"
        );
        assert_nothing_outside_the_store(&dir);
        fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn test_run_id_with_other_params_is_a_conflict() {
    let dir = tempdir("conflict");
    assert!(run(&dir, "r1", "[1.0, 2.0, 3.0]").0);
    let (ok, out, err) = run(&dir, "r1", "[1.0, 2.0, 4.0]");
    assert!(!ok, "a changed param under the same run id must fail");
    assert!(
        err.contains("conflict") && err.contains("spec_sha256"),
        "{err}"
    );
    assert!(out.is_empty());
    assert_eq!(attempts(&dir, "r1"), 1, "a conflict executed the pipeline");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_run_id_repairs_a_missing_receipt_and_replays() {
    let dir = tempdir("repair");
    let (ok, first, _) = run(&dir, "r1", "[1.0, 2.0, 3.0]");
    assert!(ok);
    fs::remove_file(dir.join("runs").join("r1").join("receipt.json")).unwrap();
    let (ok, again, err) = run(&dir, "r1", "[1.0, 2.0, 3.0]");
    assert!(ok, "repair failed: {err}");
    assert!(err.contains("receipt repaired"), "{err}");
    assert_eq!(again, first);
    assert!(dir.join("runs").join("r1").join("receipt.json").is_file());
    assert_eq!(attempts(&dir, "r1"), 1);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_run_id_refuses_unsafe_ids_and_json_streaming() {
    let dir = tempdir("refuse");
    let (ok, _, err) = run(&dir, "..", "[1.0]");
    assert!(!ok && err.contains("usage"), "{err}");
    let out = ix_in(&dir)
        .args(["pipeline", "run", "--json", "--run-id", "r1"])
        .args(["--param", "nums=[1.0]"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--json"));
    assert!(!dir.join("ix.lock").exists());
    fs::remove_dir_all(&dir).ok();
}
