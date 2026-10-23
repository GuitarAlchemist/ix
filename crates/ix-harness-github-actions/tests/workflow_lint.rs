//! The `workflow-lint` gate runs actionlint and zizmor, and a finding fails it.
//!
//! The only workflow lint ix had ran "if actionlint is available", and no runner had it: the
//! step was green and checked nothing (retro of 2026-10-22). The gate downloads each tool at a
//! pinned version, verifies the bytes before running them, and lets a finding fail the job.
use serde_yaml::Value;
use std::path::PathBuf;

const WORKFLOW: &str = include_str!("../../../.github/workflows/workflow-lint.yml");
const POLICY: &str = include_str!("../../../agent-blackbox.policy.json");

fn workflows_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows")
}

fn lint_steps(wf: &Value) -> &Vec<Value> {
    wf["jobs"]["workflow-lint"]["steps"]
        .as_sequence()
        .expect("workflow-lint steps")
}

fn run_of<'a>(wf: &'a Value, name: &str) -> &'a str {
    lint_steps(wf)
        .iter()
        .find(|s| s["name"].as_str() == Some(name))
        .and_then(|s| s["run"].as_str())
        .unwrap_or_else(|| panic!("a step named {name} with a script"))
}

fn is_sha256(s: Option<&str>) -> bool {
    s.is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

#[test]
fn each_tool_is_pinned_and_verified_before_it_runs() {
    let wf: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    for tool in ["ACTIONLINT", "ZIZMOR"] {
        let version = wf["env"][format!("{tool}_VERSION")].as_str().unwrap_or("");
        assert!(
            !version.is_empty() && version.bytes().all(|b| b.is_ascii_digit() || b == b'.'),
            "{tool}_VERSION is an exact version: {version:?}"
        );
        assert!(
            is_sha256(wf["env"][format!("{tool}_SHA256")].as_str()),
            "{tool}_SHA256 is a SHA-256"
        );
    }
    let actionlint = run_of(&wf, "actionlint");
    let verify = actionlint
        .find("sha256sum -c")
        .expect("the archive is checked");
    let lint = actionlint
        .find("actionlint\" -shellcheck= -pyflakes=")
        .expect("actionlint runs without the passes a dev machine may lack");
    assert!(verify < lint, "checked before it runs:\n{actionlint}");
    let zizmor = run_of(&wf, "zizmor");
    assert!(zizmor.contains("--require-hashes"), "{zizmor}");
    assert!(
        zizmor.contains("--offline --min-severity high --config .github/zizmor.yml"),
        "{zizmor}"
    );
}

#[test]
fn a_finding_fails_the_job() {
    let wf: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    assert!(wf["jobs"]["workflow-lint"]["continue-on-error"].is_null());
    for step in lint_steps(&wf) {
        assert!(step["continue-on-error"].is_null(), "{step:?}");
        assert!(step["if"].is_null(), "{step:?}");
        let run = step["run"].as_str().unwrap_or_default();
        assert!(
            !run.contains("|| true") && !run.contains("command -v") && !run.contains("set +e"),
            "a finding must fail the step:\n{run}"
        );
    }
}

#[test]
fn the_gate_follows_every_workflow_and_its_own_policy() {
    let wf: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    for event in ["pull_request", "push"] {
        let paths: Vec<&str> = wf["on"][event]["paths"]
            .as_sequence()
            .unwrap_or_else(|| panic!("{event} paths"))
            .iter()
            .filter_map(Value::as_str)
            .collect();
        for p in [".github/workflows/**", ".github/zizmor.yml"] {
            assert!(paths.contains(&p), "{event}: {paths:?} lacks {p}");
        }
    }
}

/// The zizmor policy decides what the gate lets through: changing it needs the review label.
#[test]
fn the_zizmor_policy_is_a_blocked_path() {
    let policy: serde_json::Value = serde_json::from_str(POLICY).expect("valid policy JSON");
    let blocked: Vec<&str> = policy["blocked_paths"]
        .as_array()
        .expect("blocked_paths")
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    assert!(blocked.contains(&".github/zizmor.yml"), "{blocked:?}");
}

#[test]
fn no_workflow_lints_only_when_the_tool_happens_to_be_there() {
    let mut seen = 0;
    for entry in std::fs::read_dir(workflows_dir()).expect("the workflows directory") {
        let path = entry.expect("a directory entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("a readable workflow");
        for tool in ["actionlint", "zizmor"] {
            assert!(
                !src.contains(&format!("command -v {tool}")),
                "{} runs {tool} only if the runner has it",
                path.display()
            );
        }
        seen += 1;
    }
    assert!(seen > 10, "read {seen} workflows");
}
