//! `scripts/risk_report_binding.py` holds the risk report to the changeset it gates (ix#316).
//!
//! On PR #313 the `risk-report` job wrote a `risk-report.md` for the PR's 25 files and a
//! `risk-report.json` for twelve files that never existed in ix, with a `critical` finding on
//! two of them. `enforce` reads the JSON, people read the Markdown, and nothing compared the
//! two. The script runs before `enforce` and fails closed unless the JSON names exactly the
//! PR's files, carries the digests of the evidence the job collected, and the Markdown shows
//! the same digests.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

const SCRIPT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/risk_report_binding.py"
);
const WORKFLOW: &str = include_str!("../../../.github/workflows/agent-blackbox.yml");

const CHANGED: &str = "Cargo.toml\ncrates/ix-loop/src/lib.rs\n";
const DIFF: &str = "diff --git a/Cargo.toml b/Cargo.toml\n+members = []\n";

/// The interpreter the job's `python` resolves to: `IX_PYTHON`, else `python3`, else
/// `python`. None found is a failure, not a skip: the gate this tests runs on Python.
fn python() -> String {
    let candidates = std::env::var("IX_PYTHON")
        .map(|p| vec![p])
        .unwrap_or_else(|_| vec!["python3".into(), "python".into()]);
    for c in &candidates {
        if let Ok(out) = Command::new(c).arg("--version").output() {
            let v = String::from_utf8_lossy(&out.stdout).to_string()
                + &String::from_utf8_lossy(&out.stderr);
            if out.status.success() && v.starts_with("Python 3") {
                return c.clone();
            }
        }
    }
    panic!("no Python 3 among {candidates:?}; set IX_PYTHON to one")
}

fn sha(s: &str) -> String {
    format!("{:x}", Sha256::digest(s.as_bytes()))
}

fn artifact(kind: &str, content: &str) -> Value {
    json!({ "kind": kind, "status": "present", "path": format!("dist/{kind}"),
            "sha256": sha(content), "bytes": content.len() })
}

/// What `analyze` writes for the job's own evidence.
fn report(files: &[&str], changed: &str, diff: &str) -> Value {
    json!({
        "verdict": "warn",
        "risk_score": 0.4,
        "changed_files": files,
        "evidence_integrity": { "algorithm": "sha256", "artifacts": [
            artifact("changed_files", changed),
            artifact("diff", diff),
        ]},
    })
}

/// The evidence table `analyze` renders into the Markdown.
fn markdown(report: &Value) -> String {
    let mut md = String::from("# Risk report\n\n## Evidence Integrity\n\n");
    md += "| Artifact | Status | SHA-256 | Bytes |\n|---|---|---|---|\n";
    for a in report["evidence_integrity"]["artifacts"]
        .as_array()
        .unwrap()
    {
        md += &format!(
            "| {} | `{}` | `{}` | `{}` |\n",
            a["kind"].as_str().unwrap(),
            a["status"].as_str().unwrap(),
            a["sha256"].as_str().unwrap(),
            a["bytes"]
        );
    }
    md
}

struct Dist(tempfile::TempDir);

impl Dist {
    fn new(report: Option<&Value>, md: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("changed-files.txt"), CHANGED).unwrap();
        std::fs::write(p.join("diff.patch"), DIFF).unwrap();
        if let Some(r) = report {
            std::fs::write(p.join("risk-report.json"), r.to_string()).unwrap();
        }
        std::fs::write(p.join("risk-report.md"), md).unwrap();
        Dist(dir)
    }

    fn path(&self) -> &Path {
        self.0.path()
    }

    /// Exit code and everything printed.
    fn bind(&self) -> (i32, String) {
        let out = Command::new(python())
            .arg(PathBuf::from(SCRIPT))
            .arg("--dist")
            .arg(self.path())
            .output()
            .expect("run the binding script");
        let text = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        (out.status.code().unwrap_or(-1), text)
    }
}

fn good() -> Value {
    report(&["Cargo.toml", "crates/ix-loop/src/lib.rs"], CHANGED, DIFF)
}

#[test]
fn a_report_of_this_changeset_is_bound() {
    let r = good();
    let (code, out) = Dist::new(Some(&r), &markdown(&r)).bind();
    assert_eq!(code, 0, "{out}");
}

#[test]
fn a_report_naming_other_files_fails_and_names_them() {
    // The #313 shape: the JSON describes files the PR does not change.
    let r = report(
        &["Cargo.toml", ".github/workflows/jules-auto-delegate.yml"],
        CHANGED,
        DIFF,
    );
    let (code, out) = Dist::new(Some(&r), &markdown(&r)).bind();
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("jules-auto-delegate.yml"), "{out}");
    assert!(out.contains("crates/ix-loop/src/lib.rs"), "{out}");
}

#[test]
fn a_report_of_other_evidence_fails() {
    let r = report(
        &["Cargo.toml", "crates/ix-loop/src/lib.rs"],
        CHANGED,
        "another diff\n",
    );
    let (code, out) = Dist::new(Some(&r), &markdown(&r)).bind();
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("diff"), "{out}");
}

#[test]
fn a_markdown_of_another_report_fails() {
    let r = good();
    let other = report(&["Cargo.toml"], "Cargo.toml\n", DIFF);
    let (code, out) = Dist::new(Some(&r), &markdown(&other)).bind();
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("risk-report.md"), "{out}");
}

#[test]
fn a_missing_or_unreadable_report_fails() {
    let (code, out) = Dist::new(None, "").bind();
    assert_eq!(code, 1, "{out}");
    let d = Dist::new(None, "");
    std::fs::write(d.path().join("risk-report.json"), "{ not json").unwrap();
    let (code, out) = d.bind();
    assert_eq!(code, 1, "{out}");
}

#[test]
fn changed_files_the_report_discovered_itself_fail() {
    // `analyze` falls back to `git diff` when no list is given: that is not this job's list.
    let mut r = good();
    r["evidence_integrity"]["artifacts"][0]["status"] = json!("discovered");
    let (code, out) = Dist::new(Some(&r), &markdown(&r)).bind();
    assert_eq!(code, 1, "{out}");
}

/// The step runs before `Enforce verdict`, on every event, label or not.
#[test]
fn the_workflow_binds_the_report_before_enforcing_it() {
    let wf: serde_yaml::Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    let steps = wf["jobs"]["risk-report"]["steps"]
        .as_sequence()
        .expect("risk-report steps");
    let at = |needle: &str| {
        steps
            .iter()
            .position(|s| s["run"].as_str().is_some_and(|r| r.contains(needle)))
    };
    let bind = at("scripts/risk_report_binding.py --dist dist").expect("a binding step");
    let enforce = at("agent_blackbox enforce").expect("the enforce step");
    let analyze = at("--out-dir dist)").expect("the analyze step");
    assert!(
        analyze < bind && bind < enforce,
        "{analyze} {bind} {enforce}"
    );
    assert!(
        steps[bind]["if"].is_null(),
        "the binding step runs whatever the labels say"
    );
}
