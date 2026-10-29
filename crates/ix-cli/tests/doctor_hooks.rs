//! The `hooks` check of `ix doctor`: the harness's command hooks in `.claude/settings.json`
//! point at files that exist and parse as bash, and every behavioural fixture under
//! `.claude/hooks/test-*.sh` passes.
//!
//! The hooks run under bash on every platform, msys bash on Windows, and a hook that crashes
//! or fails to parse is a silent no-op: the harness reports nothing, the hook simply does not
//! fire. Three `*.stackdump` files in the working tree were the only trace. These tests seed a
//! throwaway repo with each defect and assert the check names it.

use ix_cli::doctor::hooks;
use ix_cli::doctor::{self, Status};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

/// A throwaway repo with the given `.claude/settings.json` and hook files.
fn seeded(settings: &str, hooks: &[(&str, &str)]) -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let hooks_dir = dir.path().join(".claude/hooks");
    std::fs::create_dir_all(&hooks_dir).expect("mkdir");
    std::fs::write(dir.path().join(".claude/settings.json"), settings).expect("settings");
    for (name, body) in hooks {
        std::fs::write(hooks_dir.join(name), body).expect("hook");
    }
    dir
}

fn settings_with(commands: &[&str]) -> String {
    let entries: Vec<String> = commands
        .iter()
        .map(|c| format!(r#"{{"type":"command","command":"{c}"}}"#))
        .collect();
    format!(
        r#"{{"hooks":{{"UserPromptSubmit":[{{"hooks":[{}]}}]}}}}"#,
        entries.join(",")
    )
}

/// bash as the check finds it; the fixture-running tests need a real one.
fn bash() -> PathBuf {
    hooks::find_bash().expect("bash is needed to run the hook fixtures (Git bash on Windows)")
}

#[test]
fn a_wired_hook_whose_file_is_missing_fails_the_check_by_name() {
    let repo = seeded(
        &settings_with(&["bash $CLAUDE_PROJECT_DIR/.claude/hooks/ghost.sh"]),
        &[],
    );
    let result = hooks::check(repo.path(), None);
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(result.summary.contains("ghost.sh"), "{}", result.summary);
    assert!(result.remedy.as_deref().is_some_and(|r| !r.is_empty()));
}

#[test]
fn a_hook_that_does_not_parse_as_bash_fails_the_check_by_name() {
    let repo = seeded(
        &settings_with(&["bash $CLAUDE_PROJECT_DIR/.claude/hooks/broken.sh"]),
        &[("broken.sh", "#!/usr/bin/env bash\nif [ ; then\n")],
    );
    let result = hooks::check(repo.path(), Some(&bash()));
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(result.summary.contains("broken.sh"), "{}", result.summary);
}

#[test]
fn a_fixture_that_exits_non_zero_fails_the_check_with_its_exit_code() {
    let repo = seeded(
        &settings_with(&["bash $CLAUDE_PROJECT_DIR/.claude/hooks/fine.sh"]),
        &[
            ("fine.sh", "#!/usr/bin/env bash\nexit 0\n"),
            (
                "test-fine.sh",
                "#!/bin/bash\necho 'FAIL  something'\nexit 3\n",
            ),
        ],
    );
    let result = hooks::check(repo.path(), Some(&bash()));
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(
        result.summary.contains("test-fine.sh"),
        "{}",
        result.summary
    );
    assert!(result.summary.contains("exit 3"), "{}", result.summary);
    let tail = result.details["fixtures"][0]["tail"]
        .as_str()
        .expect("the fixture's last lines ride in details");
    assert!(tail.contains("FAIL  something"), "{tail}");
}

#[test]
fn a_passing_fixture_and_present_hooks_are_ok_and_counted() {
    let repo = seeded(
        &settings_with(&[
            "bash $CLAUDE_PROJECT_DIR/.claude/hooks/a.sh",
            "bash $CLAUDE_PROJECT_DIR/.claude/hooks/b.sh",
        ]),
        &[
            ("a.sh", "#!/usr/bin/env bash\nexit 0\n"),
            ("b.sh", "#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n"),
            (
                "test-a.sh",
                "#!/bin/bash\nbash \"$(dirname \"$0\")/a.sh\" </dev/null\n",
            ),
        ],
    );
    let result = hooks::check(repo.path(), Some(&bash()));
    assert_eq!(result.status, Status::Ok, "{result:?}");
    assert_eq!(result.details["hooks_wired"], 2);
    assert_eq!(result.details["fixtures_passed"], 1);
}

#[test]
fn without_bash_the_check_warns_instead_of_claiming_green() {
    let repo = seeded(
        &settings_with(&["bash $CLAUDE_PROJECT_DIR/.claude/hooks/a.sh"]),
        &[("a.sh", "#!/usr/bin/env bash\nexit 0\n")],
    );
    let result = hooks::check(repo.path(), None);
    assert_eq!(result.status, Status::Warn, "{result:?}");
    assert!(result.summary.contains("bash"), "{}", result.summary);
    assert!(result.remedy.as_deref().is_some_and(|r| !r.is_empty()));
}

#[test]
fn a_repo_without_settings_has_nothing_to_check() {
    let dir = tempfile::tempdir().expect("tempdir");
    let result = hooks::check(dir.path(), None);
    assert_eq!(result.status, Status::Ok, "{result:?}");
    assert_eq!(result.details["hooks_wired"], 0);
}

#[test]
fn this_workspace_wires_its_hooks_and_runs_their_fixtures() {
    let result = hooks::check(&repo_root(), Some(&bash()));
    assert_eq!(result.status, Status::Ok, "{result:?}");
    let wired = result.details["hooks_wired"].as_u64().expect("count");
    let passed = result.details["fixtures_passed"].as_u64().expect("count");
    assert!(wired >= 10, "ix wires at least ten hooks, saw {wired}");
    assert!(
        passed >= 2,
        "governance-check and correction-detector have fixtures, saw {passed}"
    );
    let report = doctor::run(&repo_root(), doctor::Options::default());
    assert!(
        report.checks.iter().any(|c| c.name == "hooks"),
        "the hooks check is part of `ix doctor`"
    );
}
