//! Integration tests for Week 5 governance + belief verbs.

use assert_cmd::Command;
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
    path.push(format!("ix-gov-test-{}-{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

// ---- describe persona ----------------------------------------------------

#[test]
fn describe_persona_default() {
    let dir = tempdir("persona_default");
    let out = ix_in(&dir)
        .args(["--format", "json", "describe", "persona", "default"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["name"], "default");
    assert!(v["capabilities"].is_array());
    assert!(v["constraints"].is_array());
    assert!(v["voice"]["tone"].is_string());
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn describe_persona_unknown_fails() {
    let dir = tempdir("persona_unknown");
    ix_in(&dir)
        .args(["describe", "persona", "does-not-exist"])
        .assert()
        .failure();
    fs::remove_dir_all(&dir).ok();
}

// ---- describe policy -----------------------------------------------------

#[test]
fn describe_policy_alignment() {
    let dir = tempdir("policy_alignment");
    let out = ix_in(&dir)
        .args(["--format", "json", "describe", "policy", "alignment"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    // AlignmentPolicy is loaded as a generic Policy so `name` should be present;
    // the confidence thresholds live inside `extra`.
    assert!(v["description"].is_string());
    assert!(v["extra"]["confidence_thresholds"]["proceed_autonomously"].is_number());
    fs::remove_dir_all(&dir).ok();
}

// ---- beliefs set + show + get -------------------------------------------

#[test]
fn beliefs_set_creates_file_with_hexavalent_value() {
    let dir = tempdir("beliefs_set");
    let out = ix_in(&dir)
        .args([
            "--format",
            "json",
            "beliefs",
            "set",
            "test.prop",
            "This API is stable",
            "--truth",
            "P",
            "--confidence",
            "0.8",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["state"]["truth_value"], "P");
    assert_eq!(v["state"]["confidence"], 0.8);
    // The file itself
    let contents = fs::read_to_string(dir.join("state/beliefs/test.prop.belief.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
    assert_eq!(parsed["truth_value"], "P");
    assert_eq!(parsed["proposition"], "This API is stable");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn beliefs_set_rejects_invalid_truth() {
    let dir = tempdir("beliefs_bad_truth");
    ix_in(&dir)
        .args(["beliefs", "set", "foo", "proposition", "--truth", "X"])
        .assert()
        .failure();
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn beliefs_set_clamps_confidence() {
    let dir = tempdir("beliefs_clamp");
    let out = ix_in(&dir)
        .args([
            "--format",
            "json",
            "beliefs",
            "set",
            "over",
            "out of bounds",
            "--truth",
            "T",
            "--confidence",
            "1.7",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["state"]["confidence"], 1.0);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn beliefs_show_lists_set_beliefs() {
    let dir = tempdir("beliefs_show");
    // Seed two beliefs.
    ix_in(&dir)
        .args(["beliefs", "set", "a", "prop-a", "--truth", "T"])
        .assert()
        .success();
    ix_in(&dir)
        .args(["beliefs", "set", "b", "prop-b", "--truth", "D"])
        .assert()
        .success();
    let out = ix_in(&dir)
        .args(["--format", "json", "beliefs", "show"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["count"], 2);
    fs::remove_dir_all(&dir).ok();
}

// ---- beliefs snapshot ----------------------------------------------------

#[test]
fn beliefs_snapshot_captures_all_beliefs() {
    let dir = tempdir("snapshot");
    ix_in(&dir)
        .args(["beliefs", "set", "one", "first", "--truth", "T"])
        .assert()
        .success();
    ix_in(&dir)
        .args(["beliefs", "set", "two", "second", "--truth", "P"])
        .assert()
        .success();
    let out = ix_in(&dir)
        .args([
            "--format",
            "json",
            "beliefs",
            "snapshot",
            "pre-deploy check",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["action"], "snapshot");
    assert_eq!(v["captured_beliefs"], 2);
    // File was written to state/snapshots/ with kebab-case slug
    let snap_path = v["path"].as_str().unwrap();
    assert!(snap_path.contains("pre-deploy-check"));
    assert!(dir.join(snap_path).is_file());
    let snap_content = fs::read_to_string(dir.join(snap_path)).unwrap();
    let snap: serde_json::Value = serde_json::from_str(&snap_content).unwrap();
    assert_eq!(snap["count"], 2);
    assert_eq!(snap["trigger"], "manual");
    fs::remove_dir_all(&dir).ok();
}

// ---- ix.lock written by pipeline run -----------------------------------

#[test]
fn pipeline_run_writes_ix_lock() {
    let dir = tempdir("lock");
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
    ix_in(&dir)
        .args(["--format", "json", "pipeline", "run"])
        .assert()
        .success();
    let lock_path = dir.join("ix.lock");
    assert!(lock_path.is_file(), "ix.lock was not written");
    let lock_text = fs::read_to_string(&lock_path).unwrap();
    assert!(lock_text.contains("schema: ix-lock/v1"));
    assert!(lock_text.contains("skill: stats"));
    assert!(lock_text.contains("args_hash: fnv1a64:"));
    fs::remove_dir_all(&dir).ok();
}
