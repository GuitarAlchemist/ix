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

/// The step runs before the PR comment and `Enforce verdict`, on every event, label or not,
/// and the evidence is uploaded even when it fails.
#[test]
fn the_workflow_binds_the_report_before_showing_or_enforcing_it() {
    let wf: serde_yaml::Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    let steps = wf["jobs"]["risk-report"]["steps"]
        .as_sequence()
        .expect("risk-report steps");
    let at = |needle: &str| {
        steps
            .iter()
            .position(|s| s["run"].as_str().is_some_and(|r| r.contains(needle)))
    };
    let named = |name: &str| {
        steps
            .iter()
            .position(|s| s["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("a step named {name}"))
    };
    let bind = at("scripts/risk_report_binding.py --dist dist").expect("a binding step");
    let enforce = at("agent_blackbox enforce").expect("the enforce step");
    let analyze = at("--out-dir dist)").expect("the analyze step");
    let comment = named("Comment risk report");
    let upload = named("Upload risk artifacts");
    assert!(
        analyze < bind && bind < comment && bind < enforce,
        "analyze {analyze}, bind {bind}, comment {comment}, enforce {enforce}"
    );
    assert!(
        steps[bind]["if"].is_null(),
        "the binding step runs whatever the labels say"
    );
    assert_eq!(
        steps[upload]["if"].as_str(),
        Some("always()"),
        "the evidence of an unbound report stays downloadable"
    );
}

fn enforce_step() -> serde_yaml::Value {
    let wf: serde_yaml::Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    wf["jobs"]["risk-report"]["steps"]
        .as_sequence()
        .expect("risk-report steps")
        .iter()
        .find(|s| s["name"].as_str() == Some("Enforce verdict"))
        .expect("the enforce step")
        .clone()
}

/// The PR's draft flag and labels are read from GitHub when the step runs, never from the triggering event: a
/// re-run replays its run's event, so a draft's old run re-run after the PR is ready would still see a draft
/// (Codex round 57). Marking a PR ready runs the gate again.
#[test]
fn the_enforce_step_takes_nothing_from_the_event_but_the_pr_number() {
    let wf: serde_yaml::Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    let types = wf["on"]["pull_request"]["types"]
        .as_sequence()
        .expect("pull_request types");
    assert!(
        types.iter().any(|t| t.as_str() == Some("ready_for_review")),
        "marking a PR ready must run the gate again"
    );
    let step = enforce_step();
    let env = step["env"].as_mapping().expect("the enforce step's env");
    assert_eq!(
        step["env"]["PR_NUMBER"].as_str(),
        Some("${{ github.event.pull_request.number }}")
    );
    for (k, v) in env {
        let v = v.as_str().unwrap_or_default();
        assert!(
            !v.contains("github.event") || k.as_str() == Some("PR_NUMBER"),
            "{k:?} takes {v} from the event, which a re-run replays"
        );
    }
    let run = step["run"].as_str().expect("enforce script");
    assert!(
        !run.contains("${{"),
        "the script takes its inputs from env only:\n{run}"
    );
}

/// Git's bash on Windows (the `bash` a GitHub Windows runner uses; `System32\bash.exe` is WSL's), `bash`
/// elsewhere; `IX_BASH` overrides. None found is a failure, not a skip.
fn bash() -> PathBuf {
    if let Ok(b) = std::env::var("IX_BASH") {
        return b.into();
    }
    if cfg!(windows) {
        let out = Command::new("git")
            .arg("--exec-path")
            .output()
            .expect("git, to find Git's bash");
        // <git>/mingw64/libexec/git-core → <git>/bin/bash.exe
        let exec = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        let b = exec
            .ancestors()
            .nth(3)
            .expect("Git's install dir")
            .join("bin")
            .join("bash.exe");
        assert!(b.is_file(), "no bash at {}; set IX_BASH", b.display());
        return b;
    }
    "bash".into()
}

/// `gh api repos/<repo>/pulls/<n> --jq ...` answers the live PR from `STUB_*`; nothing else is expected.
const GH: &str = r#"#!/bin/bash
[ "$1 $2" = "api repos/GuitarAlchemist/ix/pulls/429" ] || { echo "unexpected: gh $*" >&2; exit 2; }
[ -z "$STUB_GH_DOWN" ] || exit 1
case "$*" in
  *labels*) echo "$STUB_LABELED" ;;
  *.draft*) echo "$STUB_DRAFT" ;;
  *) echo "unexpected: gh $*" >&2; exit 2 ;;
esac
"#;

/// `enforce` says `STUB_SAYS` and exits `STUB_EXIT`.
const PYTHON: &str = r#"#!/bin/bash
case "$*" in
  *"agent_blackbox enforce --report dist/risk-report.json"*) [ -z "$STUB_SAYS" ] || echo "$STUB_SAYS"; exit "$STUB_EXIT" ;;
  *) echo "unexpected: python $*" >&2; exit 2 ;;
esac
"#;

/// The step's script, run as GitHub runs a `shell: bash` step, with `gh` and `python` stubbed.
fn run_enforce(env: &[(&str, &str)]) -> (i32, String) {
    let dir = tempfile::tempdir().unwrap();
    let stubs = dir.path().join("stubs");
    std::fs::create_dir(&stubs).unwrap();
    for (name, body) in [("gh", GH), ("python", PYTHON)] {
        let p = stubs.join(name);
        std::fs::write(&p, body).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    let run = enforce_step()["run"]
        .as_str()
        .expect("enforce script")
        .replace("${{ github.event.pull_request.number }}", "${PR_NUMBER}");
    std::fs::write(
        dir.path().join("step.sh"),
        format!("PATH=\"$(cd \"$STUBS\" && pwd):$PATH\"\n{run}"),
    )
    .unwrap();
    let out = Command::new(bash())
        .args(["--noprofile", "--norc", "-eo", "pipefail", "step.sh"])
        .current_dir(dir.path())
        .env("STUBS", &stubs)
        .env("GITHUB_REPOSITORY", "GuitarAlchemist/ix")
        .env("PR_NUMBER", "429")
        .envs(env.iter().copied())
        .output()
        .expect("run the enforce step");
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    (out.status.code().unwrap_or(-1), text)
}

/// A draft's failing verdict, and only that, is a warning; everything else follows the PR as it is now. The
/// `AGENT_BLACKBOX_*` values are what an old event would still carry on a re-run. `enforce` at the pinned
/// AGENT_BLACKBOX_REF prints `Agent Blackbox verdict: fail` and exits 1 on that verdict, 2 on a missing one, and a
/// crash exits 1 with a traceback on stderr.
#[test]
fn the_verdict_is_enforced_on_the_pr_as_it_is_now() {
    /// What happens, the step's env, the exit it must end with, whether it shows a warning.
    type Case<'a> = (&'a str, &'a [(&'a str, &'a str)], i32, bool);
    let fail = "Agent Blackbox verdict: fail";
    #[rustfmt::skip]
    let cases: &[Case] = &[
        ("a ready PR whose verdict fails, re-run from an event that saw a labeled draft",
         &[("STUB_DRAFT", "false"), ("STUB_LABELED", "false"), ("STUB_SAYS", fail), ("STUB_EXIT", "1"),
           ("AGENT_BLACKBOX_DRAFT", "true"), ("AGENT_BLACKBOX_REVIEWED", "true")], 1, false),
        ("a draft whose verdict fails, re-run from an event that saw it ready",
         &[("STUB_DRAFT", "true"), ("STUB_LABELED", "false"), ("STUB_SAYS", fail), ("STUB_EXIT", "1"),
           ("AGENT_BLACKBOX_DRAFT", "false")], 0, true),
        ("a draft whose enforce crashed",
         &[("STUB_DRAFT", "true"), ("STUB_LABELED", "false"), ("STUB_SAYS", ""), ("STUB_EXIT", "1")], 1, false),
        ("a draft whose report has no verdict",
         &[("STUB_DRAFT", "true"), ("STUB_LABELED", "false"),
           ("STUB_SAYS", "Agent Blackbox verdict is invalid or missing: None"), ("STUB_EXIT", "2")], 1, false),
        ("a labeled PR whose verdict fails",
         &[("STUB_DRAFT", "false"), ("STUB_LABELED", "true"), ("STUB_SAYS", fail), ("STUB_EXIT", "1")], 0, false),
        ("a verdict that passes",
         &[("STUB_DRAFT", "false"), ("STUB_LABELED", "false"), ("STUB_SAYS", "Agent Blackbox verdict: pass"),
           ("STUB_EXIT", "0")], 0, false),
        ("GitHub not answering, for a failing verdict",
         &[("STUB_GH_DOWN", "1"), ("STUB_SAYS", fail), ("STUB_EXIT", "1")], 1, false),
    ];
    let mut wrong = Vec::new();
    for (what, env, want, warned) in cases {
        let (code, out) = run_enforce(env);
        let warning = out.contains("::warning");
        if code != *want || warning != *warned {
            wrong.push(format!(
                "{what}: exit {code} (want {want}), warning {warning} (want {warned}):\n{out}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n\n"));
}
