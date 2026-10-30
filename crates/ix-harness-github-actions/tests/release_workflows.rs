//! The release workflows publish only what passed, as a draft, with the least token.
//!
//! An adversarial review of PR #428 found that `ixctl-release.yml` would ship a public
//! release from a commit whose `go vet` or `go test` failed: a GitHub `pwsh` step fails only
//! on the exit code of its last native command, and `$ErrorActionPreference = 'Stop'` does not
//! cover native commands. It also gave the build job, which runs the tagged code's tests, a
//! write token, published at once, interpolated the tag into `pwsh` source, and checked out
//! a dispatched tag by a bare name a branch could shadow. These tests pin the fixes.
use serde_yaml::Value;

const IXCTL: &str = include_str!("../../../.github/workflows/ixctl-release.yml");
const COMFYUI: &str = include_str!("../../../.github/workflows/comfyui-release.yml");

fn parse(src: &str) -> Value {
    serde_yaml::from_str(src).expect("valid workflow YAML")
}

fn steps<'a>(wf: &'a Value, job: &str) -> &'a Vec<Value> {
    wf["jobs"][job]["steps"]
        .as_sequence()
        .unwrap_or_else(|| panic!("job {job} has steps"))
}

#[test]
fn every_native_go_command_is_checked_before_the_next_line_runs() {
    let wf = parse(IXCTL);
    let mut checked = 0;
    for step in steps(&wf, "build") {
        let Some(run) = step["run"].as_str() else {
            continue;
        };
        let lines: Vec<&str> = run
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        for (i, line) in lines.iter().enumerate() {
            if line.starts_with("go ") {
                let next = lines.get(i + 1).copied().unwrap_or("");
                assert!(
                    next.starts_with("if ($LASTEXITCODE)"),
                    "`{line}` is not followed by an exit-code check (next: `{next}`)"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 3,
        "vet, test and build are all checked ({checked})"
    );
}

#[test]
fn only_the_publish_job_can_write() {
    let wf = parse(IXCTL);
    assert_eq!(wf["permissions"]["contents"].as_str(), Some("read"));
    assert!(
        wf["jobs"]["build"]["permissions"].is_null(),
        "build inherits the read-only token"
    );
    assert_eq!(
        wf["jobs"]["publish"]["permissions"]["contents"].as_str(),
        Some("write")
    );
}

#[test]
fn the_release_is_a_draft_that_does_not_become_latest() {
    let wf = parse(IXCTL);
    let release = steps(&wf, "publish")
        .iter()
        .find(|s| {
            s["uses"]
                .as_str()
                .is_some_and(|u| u.starts_with("softprops/action-gh-release"))
        })
        .expect("the release step");
    let flag = |k: &str| match &release["with"][k] {
        Value::Bool(b) => Some(*b),
        Value::String(s) => s.parse().ok(),
        _ => None,
    };
    assert_eq!(
        flag("draft"),
        Some(true),
        "nothing is public until a person publishes it"
    );
    assert_eq!(flag("make_latest"), Some(false));
}

#[test]
fn the_tag_reaches_scripts_through_the_environment_only() {
    let wf = parse(IXCTL);
    for job in ["build", "publish"] {
        for step in steps(&wf, job) {
            if let Some(run) = step["run"].as_str() {
                assert!(
                    !run.contains("inputs.tag") && !run.contains("github.ref_name"),
                    "{job}: a tag is interpolated into script source:\n{run}"
                );
            }
        }
    }
}

#[test]
fn a_dispatched_tag_is_checked_out_as_a_tag_and_a_pushed_one_at_its_commit() {
    let wf = parse(IXCTL);
    let checkout = steps(&wf, "build")
        .iter()
        .find(|s| {
            s["uses"]
                .as_str()
                .is_some_and(|u| u.starts_with("actions/checkout"))
        })
        .expect("checkout");
    let r = checkout["with"]["ref"].as_str().expect("an explicit ref");
    assert!(r.contains("refs/tags/") && r.contains("github.sha"), "{r}");
}

#[test]
fn one_release_run_per_tag_at_a_time() {
    let wf = parse(IXCTL);
    assert!(
        wf["concurrency"]["group"].as_str().is_some(),
        "a concurrency group"
    );
    assert_eq!(
        wf["concurrency"]["cancel-in-progress"].as_bool(),
        Some(false)
    );
}

#[test]
fn the_comfyui_dry_run_follows_the_pack() {
    let wf = parse(COMFYUI);
    let paths: Vec<&str> = wf["on"]["pull_request"]["paths"]
        .as_sequence()
        .expect("pull_request paths")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert!(paths.contains(&"integrations/comfyui/**"), "{paths:?}");
}
