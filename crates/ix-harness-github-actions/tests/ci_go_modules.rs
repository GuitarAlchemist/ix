//! `ci.yml` finds the Go modules it tests, under `go/` and under `prototypes/`.
//!
//! PR #390 changed only `prototypes/probes` (a Go module with tests): CI ran the whole Rust
//! matrix and `go/ixctl` and `go/ixmux`, and not one of the prototype's own tests. These tests
//! run the workflow's own discovery script, as the runner does (`bash -eo pipefail`), over
//! synthetic trees.
use serde_yaml::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const WORKFLOW: &str = include_str!("../../../.github/workflows/ci.yml");

fn workflow() -> Value {
    serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML")
}

/// The `run` script of the discovery job's `find` step.
fn discovery_script() -> String {
    let wf = workflow();
    let steps = wf["jobs"]["go-modules-discover"]["steps"]
        .as_sequence()
        .expect("go-modules-discover has steps");
    let find = steps
        .iter()
        .find(|s| s["id"].as_str() == Some("find"))
        .expect("a step with id `find`");
    assert_eq!(find["shell"].as_str(), Some("bash"));
    find["run"].as_str().expect("a run script").to_string()
}

/// Runs the discovery script in `root` and returns the `modules=` line it wrote.
fn discover(root: &Path) -> String {
    let out = root.join("github-output.txt");
    let status = Command::new(bash())
        .args(["--noprofile", "--norc", "-eo", "pipefail", "-c"])
        .arg(discovery_script())
        .current_dir(root)
        .env("GITHUB_OUTPUT", &out)
        .status()
        .expect("bash runs");
    assert!(status.success(), "the discovery step failed: {status}");
    let written = fs::read_to_string(&out).expect("the step wrote GITHUB_OUTPUT");
    written
        .lines()
        .find_map(|l| l.strip_prefix("modules="))
        .expect("a modules= line")
        .to_string()
}

/// Git's bash on Windows (the `bash` a GitHub Windows runner uses; `System32ash.exe` is
/// WSL's), `bash` elsewhere; `IX_BASH` overrides. None found is a failure, not a skip.
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

fn module(root: &Path, dir: &str) {
    let d = root.join(dir);
    fs::create_dir_all(&d).unwrap();
    fs::write(d.join("go.mod"), "module example.test/m\n\ngo 1.27\n").unwrap();
}

#[test]
fn modules_under_go_and_under_prototypes_are_found_by_their_directory() {
    let t = tempfile::tempdir().unwrap();
    module(t.path(), "go/ixctl");
    module(t.path(), "go/ixmux");
    module(t.path(), "prototypes/probes");
    // Not modules: a directory without go.mod, and a go.mod one level too deep.
    fs::create_dir_all(t.path().join("prototypes/notes")).unwrap();
    module(t.path(), "prototypes/deep/inner");
    assert_eq!(
        discover(t.path()),
        r#"["go/ixctl","go/ixmux","prototypes/probes"]"#
    );
}

#[test]
fn a_tree_with_no_prototype_module_still_succeeds() {
    // main today: `prototypes/*/go.mod` matches nothing. Under `-eo pipefail` a loop that ends
    // on a false `[ -f ]` fails the step, and so every Go job.
    let t = tempfile::tempdir().unwrap();
    module(t.path(), "go/ixctl");
    assert_eq!(discover(t.path()), r#"["go/ixctl"]"#);
}

#[test]
fn a_tree_with_no_module_at_all_gives_the_empty_list_the_job_is_skipped_on() {
    let t = tempfile::tempdir().unwrap();
    assert_eq!(discover(t.path()), "[]");
    let wf = workflow();
    assert_eq!(
        wf["jobs"]["go-modules"]["if"].as_str(),
        Some("needs.go-modules-discover.outputs.modules != '[]'")
    );
}

/// The matrix value is a directory, so nothing in the job may prefix it with `go/`.
#[test]
fn the_job_works_in_the_discovered_directory_on_both_systems() {
    let wf = workflow();
    let job = &wf["jobs"]["go-modules"];
    assert_eq!(
        job["defaults"]["run"]["working-directory"].as_str(),
        Some("${{ matrix.module }}")
    );
    let os: Vec<&str> = job["strategy"]["matrix"]["os"]
        .as_sequence()
        .expect("an os axis")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert_eq!(os, ["ubuntu-latest", "windows-latest"]);
    let setup = job["steps"]
        .as_sequence()
        .unwrap()
        .iter()
        .find(|s| {
            s["uses"]
                .as_str()
                .is_some_and(|u| u.starts_with("actions/setup-go@"))
        })
        .expect("a setup-go step");
    assert_eq!(
        setup["with"]["go-version-file"].as_str(),
        Some("${{ matrix.module }}/go.mod")
    );
    assert_eq!(
        setup["with"]["cache-dependency-path"].as_str(),
        Some("${{ matrix.module }}/go.*")
    );
}
