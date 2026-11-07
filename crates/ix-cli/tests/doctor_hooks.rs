//! The `hooks` check of `ix doctor`: the harness's command hooks in `.claude/settings.json`
//! point at files that exist, none runs a shell or a script interpreter, and the hooks' Go tests
//! (`go/claudehooks`) pass.
//!
//! The hooks were bash scripts until 2026-11: a crash or a parse error was a silent no-op (three
//! `*.stackdump` files in the working tree were the only trace), and each cost seconds per tool
//! call. The user's rule since 2026-11-05 is "jamais de shell dans les hooks". These tests seed a
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

/// A throwaway repo with the given `.claude/settings.json` and files (repo-relative paths).
fn seeded(settings: &str, files: &[(&str, &str)]) -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join(".claude/hooks")).expect("mkdir");
    std::fs::write(dir.path().join(".claude/settings.json"), settings).expect("settings");
    for (rel, body) in files {
        let path = dir.path().join(rel);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(path, body).expect("file");
    }
    dir
}

fn settings_with(commands: &[&str]) -> String {
    let entries: Vec<String> = commands
        .iter()
        .map(|c| serde_json::json!({"type": "command", "command": c}).to_string())
        .collect();
    format!(
        r#"{{"hooks":{{"UserPromptSubmit":[{{"hooks":[{}]}}]}}}}"#,
        entries.join(",")
    )
}

/// A one-package Go hooks module whose test passes or fails.
fn module(pass: bool) -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "go/claudehooks/go.mod",
            "module example.com/claudehooks\n\ngo 1.21\n",
        ),
        ("go/claudehooks/h.go", "package h\n"),
        (
            "go/claudehooks/h_test.go",
            if pass {
                "package h\n\nimport \"testing\"\n\nfunc TestOK(t *testing.T) {}\n"
            } else {
                "package h\n\nimport \"testing\"\n\nfunc TestRed(t *testing.T) { t.Fatal(\"HOOK-RED\") }\n"
            },
        ),
    ]
}

const IX_HOOK: &str = "\"$CLAUDE_PROJECT_DIR/.claude/hooks/bin/ix-hooks\" governance-check";

/// Go as the check finds it; the test-running cases need a real one.
fn go() -> PathBuf {
    hooks::find_go().expect("Go is needed to run the hooks' tests")
}

#[test]
fn a_wired_hook_whose_file_is_missing_fails_the_check_by_name() {
    let repo = seeded(
        &settings_with(&["\"$CLAUDE_PROJECT_DIR/.claude/hooks/ghost\""]),
        &[],
    );
    let result = hooks::check(repo.path(), None);
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(
        result.summary.contains(".claude/hooks/ghost"),
        "{}",
        result.summary
    );
}

#[test]
fn the_built_bin_directory_is_never_missing() {
    let repo = seeded(&settings_with(&[IX_HOOK]), &module(true));
    let result = hooks::check(repo.path(), Some(&go()));
    assert_eq!(result.status, Status::Ok, "{result:?}");
}

#[test]
fn a_hook_run_through_a_shell_fails_the_check_naming_its_command() {
    for command in [
        "bash $CLAUDE_PROJECT_DIR/.claude/hooks/a.sh",
        "pwsh -File C:/x/h.ps1",
        "powershell -NoProfile -File \"C:\\x\\h.ps1\" session",
        "python3 hook.py",
        "\"C:/Program Files/Git/bin/bash.exe\" -c true",
        "~/.claude/hooks/guard.sh",
    ] {
        let repo = seeded(&settings_with(&[command]), &[]);
        let result = hooks::check(repo.path(), None);
        assert_eq!(result.status, Status::Fail, "{command}: {result:?}");
        assert!(
            result.summary.contains("through a shell") || result.summary.contains("absent"),
            "{command}: {}",
            result.summary
        );
    }
    let repo = seeded(&settings_with(&["bash -c true"]), &[]);
    let result = hooks::check(repo.path(), None);
    assert!(
        result.summary.contains("bash -c true"),
        "{}",
        result.summary
    );
}

#[test]
fn a_single_quoted_program_or_an_env_launcher_is_a_shell() {
    for command in [
        "'C:/Program Files/Git/bin/bash.exe' -c true",
        "'/bin/sh' -c true",
        "env bash -c true",
        "/usr/bin/env bash -c true",
        "\"C:/Program Files/Git/usr/bin/env.exe\" sh -c true",
        "env -i PATH=/usr/bin HOME=/tmp sh -c true",
        "env -u HOME -C /tmp zsh -c true",
        "env -- pwsh -NoProfile -Command 1",
        "env -S \"bash -c true\"",
        "env --split-string='python3 -c 1'",
    ] {
        assert!(hooks::runs_shell(command), "{command}");
    }
    let repo = seeded(&settings_with(&["env bash -c true"]), &[]);
    let result = hooks::check(repo.path(), None);
    assert_eq!(result.status, Status::Fail, "{result:?}");
}

#[test]
fn a_shell_name_as_an_ordinary_argument_is_not_a_shell() {
    for command in [
        "node hook.js env bash",
        "\"$CLAUDE_PROJECT_DIR/.claude/hooks/bin/ix-hooks\" correction-detector bash",
        "'C:/tools/ix-hooks.exe' sh",
        "env FOO=1 node hook.js bash",
        "env -u bash node hook.js",
        "env",
    ] {
        assert!(!hooks::runs_shell(command), "{command}");
    }
}

#[test]
fn wmux_style_tool_name_arguments_are_not_shells() {
    assert!(!hooks::runs_shell(
        "node \"C:/Program Files/wmux/resources/cli/wmux-hook.js\" Bash"
    ));
    assert!(!hooks::runs_shell(
        "node \"C:/Program Files/wmux/resources/cli/wmux-hook.js\" PowerShell"
    ));
    assert!(!hooks::runs_shell(IX_HOOK));
    assert!(!hooks::runs_shell(
        "go -C \"$CLAUDE_PROJECT_DIR/go/claudehooks\" run ./cmd/ix-hooks session-start"
    ));
}

#[test]
fn a_script_lying_in_the_hooks_directory_fails_the_check() {
    let mut files = module(true);
    files.push((".claude/hooks/old.sh", "echo hi\n"));
    let repo = seeded(&settings_with(&[IX_HOOK]), &files);
    let result = hooks::check(repo.path(), None);
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(result.summary.contains("old.sh"), "{}", result.summary);
}

#[test]
fn red_hook_tests_fail_the_check_with_their_tail() {
    let repo = seeded(&settings_with(&[IX_HOOK]), &module(false));
    let result = hooks::check(repo.path(), Some(&go()));
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(result.summary.contains("go test"), "{}", result.summary);
    let tail = result.details["go_tests"]["tail"].as_str().expect("tail");
    assert!(tail.contains("HOOK-RED"), "{tail}");
}

#[test]
fn ix_hooks_without_its_module_fails_the_check() {
    let repo = seeded(&settings_with(&[IX_HOOK]), &[]);
    let result = hooks::check(repo.path(), None);
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(
        result.summary.contains("go/claudehooks"),
        "{}",
        result.summary
    );
}

#[test]
fn without_go_the_check_warns_instead_of_claiming_green() {
    let repo = seeded(&settings_with(&[IX_HOOK]), &module(true));
    let result = hooks::check(repo.path(), None);
    assert_eq!(result.status, Status::Warn, "{result:?}");
}

#[test]
fn a_repo_without_settings_has_nothing_to_check() {
    let dir = tempfile::tempdir().expect("tempdir");
    let result = hooks::check(dir.path(), None);
    assert_eq!(result.status, Status::Ok, "{result:?}");
    assert_eq!(result.details["hooks_wired"], 0);
}

#[test]
fn this_workspace_wires_its_hooks_in_go_and_their_tests_pass() {
    let result = hooks::check(&repo_root(), Some(&go()));
    assert_eq!(result.status, Status::Ok, "{result:?}");
    let wired = result.details["hooks_wired"].as_u64().expect("count");
    assert!(wired >= 10, "ix wires at least ten hooks, saw {wired}");
    assert_eq!(result.details["shell_wired"], serde_json::json!([]));
    assert_eq!(result.details["go_tests"]["exit"], 0);
    let report = doctor::run(&repo_root(), doctor::Options::default());
    assert!(
        report.checks.iter().any(|c| c.name == "hooks"),
        "the hooks check is part of `ix doctor`"
    );
}
