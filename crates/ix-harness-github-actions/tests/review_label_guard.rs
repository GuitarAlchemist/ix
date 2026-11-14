//! A push that changes a protected path takes the `agent-blackbox-reviewed` label off.
//!
//! The label says a person reviewed the PR's changes to the paths `agent-blackbox.policy.json`
//! blocks or marks one-way. It stayed on through later pushes, so a workflow edit pushed after
//! the review merged under it (retro of 2026-10-22). `scripts/review_label_guard.py` compares
//! the protected content of the PR before and after a push; the `risk-report` job removes the
//! label when it changed, or when the two cannot be compared.
use serde_yaml::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

const SCRIPT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/review_label_guard.py"
);
const WORKFLOW: &str = include_str!("../../../.github/workflows/agent-blackbox.yml");
const POLICY: &str = include_str!("../../../agent-blackbox.policy.json");
const STEP: &str = "Withdraw the review label on a push that changes protected paths";

/// The interpreter the job's `python` resolves to: `IX_PYTHON`, else `python3`, else
/// `python`. None found is a failure, not a skip: the guard this tests runs on Python.
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

struct Repo(tempfile::TempDir);

impl Repo {
    /// `main` with the real policy, a workflow, a lock file and a source file.
    fn new() -> Self {
        let r = Repo(tempfile::tempdir().unwrap());
        r.git(&["init", "-q", "-b", "main"]);
        r.commit(
            &[
                ("agent-blackbox.policy.json", Some(POLICY)),
                (".github/workflows/ci.yml", Some("name: ci\n")),
                ("Cargo.lock", Some("# lock v1\n")),
                ("crates/a/src/lib.rs", Some("// a v1\n")),
                ("README.md", Some("readme v1\n")),
            ],
            "base",
        );
        r
    }

    fn path(&self) -> &Path {
        self.0.path()
    }

    fn git(&self, args: &[&str]) -> String {
        let out = Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.autocrlf=false",
            ])
            .args(args)
            .current_dir(self.path())
            .output()
            .expect("git");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Writes (or, for `None`, deletes) each file, commits, and returns the commit.
    fn commit(&self, files: &[(&str, Option<&str>)], msg: &str) -> String {
        for (path, content) in files {
            let p = self.path().join(path);
            match content {
                Some(c) => {
                    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                    std::fs::write(&p, c).unwrap();
                }
                None => std::fs::remove_file(&p).unwrap(),
            }
        }
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "-m", msg]);
        self.git(&["rev-parse", "HEAD"])
    }

    /// Exit code and everything printed.
    fn guard(&self, base: &str, before: &str, after: &str) -> (i32, String) {
        let out = Command::new(python())
            .arg(PathBuf::from(SCRIPT))
            .args(["--base", base, "--before", before, "--after", after])
            .current_dir(self.path())
            .output()
            .expect("run the guard");
        let text = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        (out.status.code().unwrap_or(-1), text)
    }
}

/// `main`, and a PR branch whose first push edits the workflow and the source.
fn reviewed_pr() -> (Repo, String, String) {
    let r = Repo::new();
    let base = r.git(&["rev-parse", "HEAD"]);
    r.git(&["checkout", "-q", "-b", "pr"]);
    let before = r.commit(
        &[
            (".github/workflows/ci.yml", Some("name: ci\non: push\n")),
            ("crates/a/src/lib.rs", Some("// a v2\n")),
        ],
        "first push",
    );
    (r, base, before)
}

#[test]
fn a_push_that_leaves_protected_paths_alone_keeps_the_label() {
    let (r, base, before) = reviewed_pr();
    let after = r.commit(&[("crates/a/src/lib.rs", Some("// a v3\n"))], "source only");
    let (code, out) = r.guard(&base, &before, &after);
    assert_eq!(code, 0, "{out}");
}

#[test]
fn a_push_that_changes_a_protected_path_withdraws_it_and_names_it() {
    let (r, base, before) = reviewed_pr();
    let after = r.commit(
        &[(".github/workflows/ci.yml", Some("name: ci\non: [push]\n"))],
        "workflow again",
    );
    let (code, out) = r.guard(&base, &before, &after);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains(".github/workflows/ci.yml"), "{out}");
}

#[test]
fn a_push_that_first_touches_a_one_way_path_withdraws_it() {
    let (r, base, before) = reviewed_pr();
    let after = r.commit(&[("Cargo.lock", Some("# lock v2\n"))], "lock");
    let (code, out) = r.guard(&base, &before, &after);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("Cargo.lock"), "{out}");
}

#[test]
fn a_push_that_reverts_a_reviewed_protected_change_withdraws_it() {
    let (r, base, before) = reviewed_pr();
    let after = r.commit(
        &[(".github/workflows/ci.yml", Some("name: ci\n"))],
        "revert",
    );
    let (code, out) = r.guard(&base, &before, &after);
    assert_eq!(code, 1, "{out}");
}

/// Merging `main` brings `main`'s protected changes into the branch, not into the PR's diff.
#[test]
fn merging_main_that_changed_a_protected_path_the_pr_leaves_alone_keeps_it() {
    let (r, _, before) = reviewed_pr();
    r.git(&["checkout", "-q", "main"]);
    let main = r.commit(&[("Cargo.lock", Some("# lock v2\n"))], "main moves");
    r.git(&["checkout", "-q", "pr"]);
    r.git(&["merge", "-q", "--no-edit", "main"]);
    let after = r.git(&["rev-parse", "HEAD"]);
    let (code, out) = r.guard(&main, &before, &after);
    assert_eq!(code, 0, "{out}");
}

#[test]
fn a_rebase_that_keeps_the_protected_content_keeps_it() {
    let (r, _, before) = reviewed_pr();
    r.git(&["checkout", "-q", "main"]);
    let main = r.commit(&[("README.md", Some("readme v2\n"))], "main moves");
    r.git(&["checkout", "-q", "pr"]);
    r.git(&["rebase", "-q", "main"]);
    let after = r.git(&["rev-parse", "HEAD"]);
    assert_ne!(after, before);
    let (code, out) = r.guard(&main, &before, &after);
    assert_eq!(code, 0, "{out}");
}

/// A force-push can leave the old head where the job cannot fetch it.
#[test]
fn a_push_that_cannot_be_compared_withdraws_it() {
    let (r, base, _) = reviewed_pr();
    let after = r.git(&["rev-parse", "HEAD"]);
    let gone = "0123456789abcdef0123456789abcdef01234567";
    let (code, out) = r.guard(&base, gone, &after);
    // 3, not Python's own 2 (a missing script, a bad argument).
    assert_eq!(code, 3, "{out}");
    assert!(out.contains(gone), "{out}");
}

/// The PR cannot exempt its own paths: the base's policy decides what is protected.
#[test]
fn the_base_policy_decides_not_the_pushed_one() {
    let r = Repo::new();
    let base = r.git(&["rev-parse", "HEAD"]);
    r.git(&["checkout", "-q", "-b", "pr"]);
    let loose = POLICY.replace("\"Cargo.lock\"", "\"Cargo.lock.unused\"");
    assert_ne!(
        loose, POLICY,
        "the fixture drops Cargo.lock from the policy"
    );
    let before = r.commit(&[("agent-blackbox.policy.json", Some(&loose))], "loosen");
    let after = r.commit(&[("Cargo.lock", Some("# lock v2\n"))], "lock");
    let (code, out) = r.guard(&base, &before, &after);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("Cargo.lock"), "{out}");
}

fn steps() -> Vec<Value> {
    let wf: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    wf["jobs"]["risk-report"]["steps"]
        .as_sequence()
        .expect("risk-report steps")
        .clone()
}

fn guard_step() -> Value {
    steps()
        .into_iter()
        .find(|s| s["name"].as_str() == Some(STEP))
        .expect("the guard step")
}

/// The guard acts once per push, right after the checkout: a later step failing must not
/// skip it, and a re-run, which replays an old push, must not take off a label applied since.
#[test]
fn the_guard_runs_once_per_push_before_anything_can_fail() {
    let steps = steps();
    let at = |name: &str| {
        steps
            .iter()
            .position(|s| s["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("a step named {name}"))
    };
    assert_eq!(at(STEP), at("Checkout target repo") + 1);
    assert_eq!(
        guard_step()["if"].as_str(),
        Some("github.event.action == 'synchronize' && github.run_attempt == 1")
    );
    let run = guard_step()["run"]
        .as_str()
        .expect("guard script")
        .to_string();
    assert!(!run.contains("${{"), "inputs come from env only:\n{run}");
    assert!(
        run.contains("scripts/review_label_guard.py"),
        "the step runs the guard:\n{run}"
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

/// The labels of PR 440 are `STUB_LABELED`; a DELETE of the label is logged to `deleted`.
const GH: &str = r#"#!/bin/bash
[ -z "$STUB_GH_DOWN" ] || exit 1
case "$*" in
  "api repos/GuitarAlchemist/ix/issues/440/labels --jq "*) echo "$STUB_LABELED" ;;
  "api -X DELETE repos/GuitarAlchemist/ix/issues/440/labels/agent-blackbox-reviewed")
    echo deleted >> "$STUBS/deleted"; [ -z "$STUB_DELETE_FAILS" ] || exit 1 ;;
  *) echo "unexpected: gh $*" >&2; exit 2 ;;
esac
"#;

/// The guard answers `STUB_GUARD`.
const PYTHON: &str = r#"#!/bin/bash
case "$*" in
  "scripts/review_label_guard.py --base b0 --before b1 --after b2") exit "$STUB_GUARD" ;;
  *) echo "unexpected: python $*" >&2; exit 2 ;;
esac
"#;

/// The step's script, run as GitHub runs a `shell: bash` step, with `gh` and `python` stubbed.
/// Exit code, output, and whether the label was deleted.
fn run_step(env: &[(&str, &str)]) -> (i32, String, bool) {
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
    let run = guard_step()["run"]
        .as_str()
        .expect("guard script")
        .to_string();
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
        .env("PR_NUMBER", "440")
        .env("BASE_SHA", "b0")
        .env("BEFORE_SHA", "b1")
        .env("AFTER_SHA", "b2")
        .envs(env.iter().copied())
        .output()
        .expect("run the guard step");
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    (
        out.status.code().unwrap_or(-1),
        text,
        stubs.join("deleted").exists(),
    )
}

#[test]
fn the_step_takes_the_label_off_when_the_guard_says_so() {
    /// What happens, the step's env, the exit it must end with, whether the label goes.
    type Case<'a> = (&'a str, &'a [(&'a str, &'a str)], i32, bool);
    #[rustfmt::skip]
    let cases: &[Case] = &[
        ("protected paths unchanged", &[("STUB_GUARD", "0"), ("STUB_LABELED", "true")], 0, false),
        ("protected paths changed, labeled", &[("STUB_GUARD", "1"), ("STUB_LABELED", "true")], 0, true),
        ("cannot compare, labeled", &[("STUB_GUARD", "3"), ("STUB_LABELED", "true")], 0, true),
        ("protected paths changed, not labeled", &[("STUB_GUARD", "1"), ("STUB_LABELED", "false")], 0, false),
        ("the guard crashed, labeled", &[("STUB_GUARD", "2"), ("STUB_LABELED", "true")], 0, true),
        // The labels cannot be read: the DELETE is tried, and a DELETE that fails fails the step,
        // so a label that may still be on shows as a red job.
        ("GitHub not answering", &[("STUB_GUARD", "1"), ("STUB_GH_DOWN", "1")], 1, false),
        ("the DELETE fails", &[("STUB_GUARD", "1"), ("STUB_LABELED", "true"), ("STUB_DELETE_FAILS", "1")], 1, true),
    ];
    let mut wrong = Vec::new();
    for (what, env, want, deleted) in cases {
        let (code, out, gone) = run_step(env);
        if code != *want || gone != *deleted {
            wrong.push(format!(
                "{what}: exit {code} (want {want}), deleted {gone} (want {deleted}):\n{out}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n\n"));
}
