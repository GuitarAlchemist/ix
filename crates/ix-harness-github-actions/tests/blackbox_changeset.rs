//! Agent Blackbox reports what the PR changes, not what `main` took since the PR was pushed.
//!
//! On #507 (2026-11-11) the job diffed `pull_request.base.sha`, a commit of 2026-11-02, against
//! the test merge: 22 files, `.github/workflows/ci.yml` among them, for a PR that changes 8
//! files under `go/ixctl`. The verdict was `fail` on a blocked path the PR never touched.
//!
//! These tests run the workflow's own step, as the runner does (`bash -eo pipefail`), in
//! synthetic repositories.
use serde_yaml::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const WORKFLOW: &str = include_str!("../../../.github/workflows/agent-blackbox.yml");
const STEP: &str = "Establish the changeset of this PR";

fn steps() -> Vec<Value> {
    let wf: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    wf["jobs"]["risk-report"]["steps"]
        .as_sequence()
        .expect("risk-report has steps")
        .clone()
}

fn script() -> String {
    let step = steps()
        .into_iter()
        .find(|s| s["name"].as_str() == Some(STEP))
        .unwrap_or_else(|| panic!("no step named {STEP:?}"));
    assert_eq!(step["shell"].as_str(), Some("bash"));
    assert_eq!(
        step["env"]["MERGE_SHA"].as_str(),
        Some("${{ github.sha }}"),
        "the merge the job checked out"
    );
    assert_eq!(
        step["env"]["HEAD_SHA"].as_str(),
        Some("${{ github.event.pull_request.head.sha }}")
    );
    step["run"].as_str().expect("a run script").to_string()
}

/// Git's bash on Windows (`System32\bash.exe` is WSL's), `bash` elsewhere; `IX_BASH` overrides.
fn bash() -> PathBuf {
    if let Ok(b) = std::env::var("IX_BASH") {
        return b.into();
    }
    if cfg!(windows) {
        let out = Command::new("git")
            .arg("--exec-path")
            .output()
            .expect("git, to find Git's bash");
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

struct Repo(tempfile::TempDir);

impl Repo {
    /// `main` with a workflow, a source file and a note.
    fn new() -> Self {
        let r = Repo(tempfile::tempdir().unwrap());
        r.git(&["init", "-q", "-b", "main"]);
        r.write(".github/workflows/ci.yml", "name: ci\n");
        r.write("go/ixctl/need.go", "package main // v1\n");
        r.write("docs/note.md", "note v1\n");
        r.commit("base");
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
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn write(&self, rel: &str, content: &str) {
        let p = self.path().join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }

    fn commit(&self, msg: &str) -> String {
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "-m", msg]);
        self.git(&["rev-parse", "HEAD"])
    }

    /// The test merge GitHub builds: `main` as it is now, with `head` merged in.
    fn test_merge(&self, head: &str) -> String {
        self.git(&["checkout", "-q", "--detach", "main"]);
        self.git(&["merge", "-q", "--no-ff", "-m", "test merge", head]);
        self.git(&["rev-parse", "HEAD"])
    }

    /// Runs the step; returns whether it succeeded, the listed files, and what it printed.
    fn establish(&self, merge: &str, head: &str) -> (bool, Vec<String>, String) {
        let _ = fs::remove_dir_all(self.path().join("dist"));
        let out = Command::new(bash())
            .args(["--noprofile", "--norc", "-eo", "pipefail", "-c"])
            .arg(script())
            .current_dir(self.path())
            .env("MERGE_SHA", merge)
            .env("HEAD_SHA", head)
            .output()
            .expect("bash runs");
        let files = fs::read_to_string(self.path().join("dist/changed-files.txt"))
            .map(|s| s.lines().map(str::to_string).collect())
            .unwrap_or_default();
        let said = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        (out.status.success(), files, said)
    }
}

/// A PR opened on the first `main`, changing one Go file; `main` then takes a workflow change.
fn stale_base() -> (Repo, String, String, String) {
    let r = Repo::new();
    let old_base = r.git(&["rev-parse", "HEAD"]);
    r.git(&["checkout", "-q", "-b", "pr"]);
    r.write("go/ixctl/need.go", "package main // v2\n");
    let head = r.commit("the PR");
    r.git(&["checkout", "-q", "main"]);
    r.write(".github/workflows/ci.yml", "name: ci\n# main moved on\n");
    r.write("docs/note.md", "note v2\n");
    r.commit("main moved on");
    let merge = r.test_merge(&head);
    (r, old_base, head, merge)
}

#[test]
fn a_pr_on_an_old_base_is_reported_with_its_own_files_only() {
    let (r, old_base, head, merge) = stale_base();
    // The defect: from the base the PR was pushed on, main's own changes are in the list.
    let from_old_base = r.git(&["diff", "--name-only", &old_base, &merge]);
    assert!(
        from_old_base.contains(".github/workflows/ci.yml"),
        "the fixture does not reproduce #507: {from_old_base}"
    );
    let (ok, files, said) = r.establish(&merge, &head);
    assert!(ok, "{said}");
    assert_eq!(files, ["go/ixctl/need.go"], "{said}");
}

#[test]
fn the_shas_the_list_was_made_from_are_on_record() {
    let (r, _, head, merge) = stale_base();
    let (ok, _, said) = r.establish(&merge, &head);
    assert!(ok, "{said}");
    let main = r.git(&["rev-parse", "main"]);
    let rec: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(r.path().join("dist/changeset.json")).expect("changeset.json"),
    )
    .expect("JSON");
    assert_eq!(rec["base_sha"], main.as_str());
    assert_eq!(rec["head_sha"], head.as_str());
    assert_eq!(rec["merge_sha"], merge.as_str());
    // The patch is of the same two commits as the list.
    let patch = fs::read_to_string(r.path().join("dist/diff.patch")).unwrap();
    assert!(patch.contains("go/ixctl/need.go") && !patch.contains("ci.yml"));
}

#[test]
fn a_pr_that_merged_main_in_is_still_reported_with_its_own_files_only() {
    let (r, _, _, _) = stale_base();
    r.git(&["checkout", "-q", "pr"]);
    r.git(&["merge", "-q", "--no-ff", "-m", "main into the PR", "main"]);
    let head = r.git(&["rev-parse", "HEAD"]);
    let merge = r.test_merge(&head);
    let (ok, files, said) = r.establish(&merge, &head);
    assert!(ok, "{said}");
    assert_eq!(files, ["go/ixctl/need.go"], "{said}");
}

#[test]
fn a_protected_file_the_pr_changes_deletes_or_moves_stays_in_the_list() {
    // Changed.
    let r = Repo::new();
    r.git(&["checkout", "-q", "-b", "pr"]);
    r.write(
        ".github/workflows/ci.yml",
        "name: ci\n# changed by the PR\n",
    );
    let head = r.commit("the PR edits a workflow");
    let (ok, files, said) = r.establish(&r.test_merge(&head), &head);
    assert!(ok, "{said}");
    assert_eq!(files, [".github/workflows/ci.yml"]);

    // Deleted.
    let r = Repo::new();
    r.git(&["checkout", "-q", "-b", "pr"]);
    r.git(&["rm", "-q", ".github/workflows/ci.yml"]);
    let head = r.commit("the PR deletes a workflow");
    let (ok, files, said) = r.establish(&r.test_merge(&head), &head);
    assert!(ok, "{said}");
    assert_eq!(files, [".github/workflows/ci.yml"]);

    // Moved, content unchanged: a rename-aware diff would list the new name only.
    let r = Repo::new();
    r.git(&["checkout", "-q", "-b", "pr"]);
    r.git(&["mv", ".github/workflows/ci.yml", "docs/ci.yml.txt"]);
    let head = r.commit("the PR moves a workflow away");
    let (ok, files, said) = r.establish(&r.test_merge(&head), &head);
    assert!(ok, "{said}");
    assert_eq!(files, [".github/workflows/ci.yml", "docs/ci.yml.txt"]);
}

#[test]
fn a_changeset_that_cannot_be_established_fails_and_lists_nothing() {
    let (r, _, head, merge) = stale_base();
    let main = r.git(&["rev-parse", "main"]);
    for (why, merge_sha, head_sha) in [
        ("not a merge: one parent", head.as_str(), head.as_str()),
        ("a merge of another head", merge.as_str(), main.as_str()),
        (
            "a commit the checkout does not have",
            "0123456789012345678901234567890123456789",
            head.as_str(),
        ),
        ("no merge sha at all", "", head.as_str()),
    ] {
        let (ok, files, said) = r.establish(merge_sha, head_sha);
        assert!(!ok, "{why}: the step succeeded: {said}");
        assert!(files.is_empty(), "{why}: files were listed: {files:?}");
        assert!(
            said.contains("cannot establish what this PR changes"),
            "{why}: {said}"
        );
    }
}

/// A merge of this head and of something else is not the test merge of this PR: its diff
/// against `main` holds the other branch's changes too.
#[test]
fn a_merge_of_three_parents_is_refused_even_when_one_is_this_head() {
    let (r, _, head, _) = stale_base();
    r.git(&["checkout", "-q", "-b", "other", "main"]);
    r.write("crates/other.rs", "// another branch\n");
    r.commit("another branch");
    r.git(&["checkout", "-q", "--detach", "main"]);
    r.git(&["merge", "-q", "--no-ff", "-m", "octopus", &head, "other"]);
    let octopus = r.git(&["rev-parse", "HEAD"]);
    let (ok, files, said) = r.establish(&octopus, &head);
    assert!(!ok, "the step took a three-parent merge: {said}");
    assert!(files.is_empty(), "files were listed: {files:?}");
    assert!(said.contains("3 parent(s)"), "{said}");
}

/// No step may diff from `pull_request.base.sha`: that is the base of the last push, not of
/// the test merge.
#[test]
fn no_step_diffs_from_the_base_of_the_last_push() {
    for s in steps() {
        let run = s["run"].as_str().unwrap_or_default();
        assert!(
            !(run.contains("git diff") && run.contains("pull_request.base.sha")),
            "step {:?} diffs from pull_request.base.sha",
            s["name"].as_str().unwrap_or("?")
        );
    }
}
