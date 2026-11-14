//! `ci.yml` ends on one check, `ci-ok`, that always reports and tells the truth.
//!
//! A check can be required on `main` only if it always starts. With `paths` on `pull_request`
//! a prose-only PR started nothing, so nothing could be required and "green" had no meaning
//! GitHub could act on (auto-merge would merge at once). Now the workflow always starts on a
//! pull request, the `changes` job decides whether the jobs have anything to read, and
//! `ci-ok` sums them up.
//!
//! These tests run the two jobs' own scripts, as the runner does (`bash -eo pipefail`).
use serde_yaml::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const WORKFLOW: &str = include_str!("../../../.github/workflows/ci.yml");

fn workflow() -> Value {
    serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML")
}

fn jobs() -> Vec<String> {
    workflow()["jobs"]
        .as_mapping()
        .expect("jobs")
        .keys()
        .map(|k| k.as_str().unwrap().to_string())
        .collect()
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

fn run_bash(script: &str, dir: &Path, env: &[(&str, &str)]) -> (bool, String) {
    let mut cmd = Command::new(bash());
    cmd.args(["--noprofile", "--norc", "-eo", "pipefail", "-c"])
        .arg(script)
        .current_dir(dir);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("bash runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr),
    )
}

// ---- the structure -------------------------------------------------------------------

#[test]
fn every_job_is_summed_up_by_ci_ok() {
    let wf = workflow();
    let gate = &wf["jobs"]["ci-ok"];
    assert_eq!(
        gate["if"].as_str(),
        Some("always()"),
        "ci-ok must run when a job it needs failed or was skipped"
    );
    let needs: BTreeSet<String> = gate["needs"]
        .as_sequence()
        .expect("ci-ok needs a list")
        .iter()
        .map(|n| n.as_str().unwrap().to_string())
        .collect();
    let others: BTreeSet<String> = jobs().into_iter().filter(|j| j != "ci-ok").collect();
    assert_eq!(
        needs, others,
        "a job ci-ok does not need can fail without making ci-ok red"
    );
    let step = &gate["steps"][0];
    let run = step["run"].as_str().expect("ci-ok's script");
    for job in &others {
        let var = format!("R_{}", job.to_uppercase().replace('-', "_"));
        assert_eq!(
            step["env"][var.as_str()].as_str(),
            Some(format!("${{{{ needs.{job}.result }}}}").as_str()),
            "{var} must carry the result of {job}"
        );
        assert!(
            run.contains(&format!("\"${var}\"")),
            "ci-ok's script never looks at {var}"
        );
    }
}

#[test]
fn every_job_that_reads_code_waits_for_the_decision() {
    let wf = workflow();
    for job in jobs() {
        match job.as_str() {
            "changes" | "ci-ok" => continue,
            // Runs when the discovery, which waits for the decision, found a module.
            "go-modules" => {
                assert_eq!(
                    wf["jobs"][job.as_str()]["needs"].as_str(),
                    Some("go-modules-discover")
                );
            }
            _ => {
                let j = &wf["jobs"][job.as_str()];
                assert_eq!(j["needs"].as_str(), Some("changes"), "{job}");
                assert_eq!(
                    j["if"].as_str(),
                    Some("needs.changes.outputs.code == 'true'"),
                    "{job}"
                );
            }
        }
    }
}

// ---- ci-ok ---------------------------------------------------------------------------

const JOBS: [&str; 7] = [
    "R_CHANGES",
    "R_BUILD_AND_TEST",
    "R_GO_MODULES_DISCOVER",
    "R_GO_MODULES",
    "R_GATED_FEATURES",
    "R_DUCKDB_SQL",
    "R_COMFYUI_PACK",
];

/// Runs ci-ok's script with every job at `base`, then the overrides.
fn gate(code: &str, base: &str, overrides: &[(&str, &str)]) -> (bool, String) {
    let script = workflow()["jobs"]["ci-ok"]["steps"][0]["run"]
        .as_str()
        .unwrap()
        .to_string();
    let mut env: Vec<(&str, &str)> = vec![("CODE", code)];
    for j in JOBS {
        let v = overrides
            .iter()
            .find(|(k, _)| *k == j)
            .map(|(_, v)| *v)
            .unwrap_or(if j == "R_CHANGES" { "success" } else { base });
        env.push((j, v));
    }
    let dir = tempfile::tempdir().unwrap();
    run_bash(&script, dir.path(), &env)
}

#[test]
fn ci_ok_is_green_when_everything_ran_and_when_there_was_nothing_to_read() {
    let (ok, said) = gate("true", "success", &[]);
    assert!(ok, "{said}");
    // A prose-only PR: the decision is no, every job is skipped.
    let (ok, said) = gate("false", "skipped", &[]);
    assert!(ok, "{said}");
    // Code changed and the tree has no Go module: go-modules alone is skipped.
    let (ok, said) = gate("true", "success", &[("R_GO_MODULES", "skipped")]);
    assert!(ok, "{said}");
}

#[test]
fn ci_ok_is_red_on_a_failure_a_cancellation_or_a_job_that_should_have_run() {
    for job in &JOBS[1..] {
        for result in ["failure", "cancelled", ""] {
            let (ok, said) = gate("true", "success", &[(job, result)]);
            assert!(!ok, "{job}={result:?} left ci-ok green: {said}");
        }
    }
    // Skipped though the change has code: a job that did not run proves nothing.
    for job in [
        "R_BUILD_AND_TEST",
        "R_GO_MODULES_DISCOVER",
        "R_GATED_FEATURES",
        "R_DUCKDB_SQL",
        "R_COMFYUI_PACK",
    ] {
        let (ok, said) = gate("true", "success", &[(job, "skipped")]);
        assert!(
            !ok && said.contains("was skipped though this change needs it"),
            "{job} skipped with code=true left ci-ok green: {said}"
        );
    }
}

#[test]
fn ci_ok_is_red_when_the_decision_failed_or_is_missing() {
    for result in ["failure", "cancelled", "skipped", ""] {
        let (ok, said) = gate("true", "success", &[("R_CHANGES", result)]);
        assert!(!ok, "changes={result:?} left ci-ok green: {said}");
    }
    // No decision, every job skipped: nothing was checked.
    for code in ["", "maybe"] {
        let (ok, said) = gate(code, "skipped", &[]);
        assert!(!ok, "code={code:?} left ci-ok green: {said}");
        // Even with every job green: the decision is what says they were the right ones.
        let (ok, said) = gate(code, "success", &[]);
        assert!(
            !ok && said.contains("changes gave no decision"),
            "code={code:?}, all green, left ci-ok green: {said}"
        );
    }
}

// ---- changes -------------------------------------------------------------------------

struct Repo(tempfile::TempDir);

impl Repo {
    fn new() -> Self {
        let r = Repo(tempfile::tempdir().unwrap());
        r.git(&["init", "-q", "-b", "main"]);
        r.write("crates/a/src/lib.rs", "// a v1\n");
        r.write("go/ixctl/README.md", "ixctl v1\n");
        r.write("docs/guide.md", "guide v1\n");
        r.write("README.md", "readme v1\n");
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

    /// A PR made by `change`, on a `main` that then took a code change of its own; returns
    /// what the `changes` job decides and what it said.
    fn decide_for(change: impl Fn(&Repo)) -> (String, String) {
        let r = Repo::new();
        r.git(&["checkout", "-q", "-b", "pr"]);
        change(&r);
        let head = r.commit("the PR");
        r.git(&["checkout", "-q", "main"]);
        r.write("crates/b/src/lib.rs", "// main moved on, with code\n");
        r.commit("main moved on");
        r.git(&["merge", "-q", "--no-ff", "-m", "test merge", &head]);
        let merge = r.git(&["rev-parse", "HEAD"]);
        r.decide("pull_request", &merge, &head)
    }

    fn decide(&self, event: &str, merge: &str, head: &str) -> (String, String) {
        let wf = workflow();
        let step = wf["jobs"]["changes"]["steps"]
            .as_sequence()
            .unwrap()
            .iter()
            .find(|s| s["id"].as_str() == Some("decide"))
            .expect("a step with id `decide`")
            .clone();
        // Under .git, so that a later `git add -A` of the test does not commit it.
        let out_file = self.path().join(".git").join("github-output.txt");
        let _ = fs::remove_file(&out_file);
        let (ok, said) = run_bash(
            step["run"].as_str().unwrap(),
            self.path(),
            &[
                ("EVENT", event),
                ("MERGE_SHA", merge),
                ("HEAD_SHA", head),
                ("PATTERNS", step["env"]["PATTERNS"].as_str().unwrap()),
                ("GITHUB_OUTPUT", out_file.to_str().unwrap()),
            ],
        );
        assert!(ok, "the decision step failed: {said}");
        let written = fs::read_to_string(&out_file).expect("the step wrote GITHUB_OUTPUT");
        let code: Vec<&str> = written
            .lines()
            .filter_map(|l| l.strip_prefix("code="))
            .collect();
        assert_eq!(code.len(), 1, "one decision, got {written:?}");
        (code[0].to_string(), said)
    }
}

#[test]
fn a_prose_only_pr_skips_the_jobs_even_when_main_took_code_since() {
    let (code, said) = Repo::decide_for(|r| {
        r.write("README.md", "readme v2\n");
        r.write("docs/guide.md", "guide v2\n");
        r.write(".claude/skills/x/SKILL.md", "a skill\n");
    });
    assert_eq!(code, "false", "{said}");
}

#[test]
fn a_pr_that_changes_a_file_a_job_reads_runs_them() {
    type Change = fn(&Repo);
    let cases: [(&str, Change); 6] = [
        ("code", |r| r.write("crates/a/src/lib.rs", "// a v2\n")),
        ("prose and one code file", |r| {
            r.write("README.md", "readme v2\n");
            r.write("go/ixctl/ws.go", "package main\n");
        }),
        ("a Markdown file a job reads", |r| {
            r.write("go/ixctl/README.md", "ixctl v2\n")
        }),
        ("a schema under docs/contracts", |r| {
            r.write("docs/contracts/x.schema.json", "{}\n")
        }),
        ("a hook", |r| r.write(".claude/hooks/x.sh", "true\n")),
        ("a workflow", |r| {
            r.write(".github/workflows/ci.yml", "name: ci\n")
        }),
    ];
    for (what, change) in cases {
        let (code, said) = Repo::decide_for(change);
        assert_eq!(code, "true", "{what}: {said}");
    }
}

#[test]
fn deleting_or_moving_a_file_a_job_reads_runs_them() {
    let (code, said) = Repo::decide_for(|r| {
        r.git(&["rm", "-q", "crates/a/src/lib.rs"]);
    });
    assert_eq!(code, "true", "a deleted source file: {said}");
    // Moved to a name no job reads, content unchanged: a rename-aware diff would list the
    // new name only, and the matrix would not run on a crate that lost a file.
    let (code, said) = Repo::decide_for(|r| {
        r.git(&["mv", "crates/a/src/lib.rs", "docs/lib.rs.md"]);
    });
    assert_eq!(code, "true", "a source file moved into docs: {said}");
}

#[test]
fn when_the_decision_cannot_be_made_everything_runs() {
    let r = Repo::new();
    let base = r.git(&["rev-parse", "HEAD"]);
    r.write("README.md", "readme v2\n");
    let prose = r.commit("prose only, not a merge");
    for (why, event, merge, head) in [
        ("a push: on.push.paths decided", "push", prose.as_str(), ""),
        (
            "not a merge",
            "pull_request",
            prose.as_str(),
            prose.as_str(),
        ),
        (
            "a merge of another head",
            "pull_request",
            prose.as_str(),
            base.as_str(),
        ),
        (
            "an unknown commit",
            "pull_request",
            "0123456789012345678901234567890123456789",
            prose.as_str(),
        ),
    ] {
        let (code, said) = r.decide(event, merge, head);
        assert_eq!(code, "true", "{why}: {said}");
    }
    // A real, prose-only test merge, but of a head that is not this PR's: its file list
    // says nothing about this PR.
    r.git(&["checkout", "-q", "-b", "other", &base]);
    r.write("docs/guide.md", "guide v2\n");
    let other = r.commit("another prose-only head");
    r.git(&["checkout", "-q", "main"]);
    r.git(&["merge", "-q", "--no-ff", "-m", "test merge", &other]);
    let merge = r.git(&["rev-parse", "HEAD"]);
    let (code, said) = r.decide("pull_request", &merge, &other);
    assert_eq!(code, "false", "the control, the right head: {said}");
    let (code, said) = r.decide("pull_request", &merge, &prose);
    assert_eq!(code, "true", "a test merge of another head: {said}");
}
