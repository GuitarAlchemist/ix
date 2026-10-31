//! Real kills of `ix pipeline run --run-id`. The ix process stops at a publish
//! point through `IX_RECOVER_PAUSE` (debug builds only), the test kills it
//! there (`Child::kill`: TerminateProcess on Windows, SIGKILL on Unix), and the
//! next run under the same id reconciles what it left. A recoverable run
//! writes nothing outside its store, so a kill leaves nothing elsewhere.
//!
//! The pause exists only where `debug_assertions` is on, so the kill tests
//! run in that profile and are ignored, with that reason, in a release
//! profile; there a smoke test checks instead that the variable is ignored
//! and the run publishes normally.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use std::{fs, thread};

const SPEC: &str = r#"version: "1"
stages:
  s:
    skill: stats
    args:
      data: [1.0, 2.0, 3.0]
"#;

/// What a paused ix exits with when its deadline passes (`PAUSE_DEADLINE_EXIT`).
const DEADLINE_EXIT: i32 = 86;

fn tempdir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("ix-recover-kill-{}-{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("ix.yaml"), SPEC).unwrap();
    path
}

fn ix(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ix"));
    cmd.current_dir(dir)
        .env(
            "IX_GOVERNANCE_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../governance/demerzel"),
        )
        .env_remove("IX_RECOVER_PAUSE")
        .args(["--format", "json", "pipeline", "run"])
        .args(["--run-id", "r1", "--runs-dir", "runs"]);
    cmd
}

/// Kills and reaps its child on every way out of a test, a panic included,
/// until the test has reaped the child itself.
struct Reaper(Option<Child>);

impl Reaper {
    fn child(&mut self) -> &mut Child {
        self.0.as_mut().expect("the child was reaped")
    }

    /// Kill the child and reap it. The guard stays armed until the wait succeeds.
    fn kill(&mut self) -> ExitStatus {
        self.child().kill().unwrap();
        self.reap()
    }

    fn reap(&mut self) -> ExitStatus {
        let status = self.child().wait().unwrap();
        self.0 = None;
        status
    }
}

impl Drop for Reaper {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Whether the OS still lists a process with this id, asked of its own tool.
/// A query that fails fails the test; it is never read as absence.
fn pid_alive(pid: u32) -> bool {
    #[cfg(windows)]
    let alive = {
        let out = Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .expect("tasklist runs");
        assert!(
            out.status.success(),
            "tasklist failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).contains(&format!(",\"{pid}\","))
    };
    // LC_ALL=C keeps kill's diagnostic in English, so "No such process" can be matched.
    #[cfg(unix)]
    let alive = {
        let out = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .env("LC_ALL", "C")
            .output()
            .expect("kill runs");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() || err.contains("No such process"),
            "kill -0 {pid} failed: {err}"
        );
        out.status.success()
    };
    alive
}

/// Start a recoverable run that pauses at `point` for up to `secs` seconds,
/// and return once it is there.
fn paused_at(dir: &Path, point: &str, secs: u64) -> Reaper {
    let marker = dir.join(format!("{point}.reached"));
    let mut ix = Reaper(Some(
        ix(dir)
            .env(
                "IX_RECOVER_PAUSE",
                format!("{point}|{}|{secs}", marker.display()),
            )
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    ));
    let deadline = Instant::now() + Duration::from_secs(120);
    while !marker.exists() {
        if let Some(status) = ix.child().try_wait().unwrap() {
            panic!("ix exited before {point}: {status} (the pause exists in debug builds only)");
        }
        assert!(Instant::now() < deadline, "ix never reached {point}");
        thread::sleep(Duration::from_millis(20));
    }
    ix
}

fn kill(mut ix: Reaper) {
    let pid = ix.child().id();
    assert!(pid_alive(pid), "pid_alive does not see the live ix {pid}");
    let status = ix.kill();
    assert!(
        !status.success() && status.code() != Some(DEADLINE_EXIT),
        "ix was not killed: {status}"
    );
    assert!(!pid_alive(pid), "ix ({pid}) outlived its kill");
}

/// Run to completion and return (stdout, stderr). The OS drops a dead
/// owner's lock as part of process rundown, so `unknown` is retried briefly.
fn rerun(dir: &Path) -> (Vec<u8>, String) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let out = ix(dir).output().unwrap();
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        if out.status.success() {
            return (out.stdout, err);
        }
        assert!(
            err.contains("unknown") && Instant::now() < deadline,
            "{err}"
        );
        thread::sleep(Duration::from_millis(50));
    }
}

fn attempt_ids(dir: &Path) -> Vec<String> {
    let mut ids: Vec<String> = fs::read_dir(dir.join("runs").join("r1").join("attempts"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    ids.sort();
    ids
}

fn assert_nothing_outside_the_store(dir: &Path) {
    assert!(!dir.join("ix.lock").exists(), "ix.lock beside the spec");
    assert!(
        !dir.join("state/thinking-machine/provenance.jsonl").exists(),
        "a provenance line"
    );
}

#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "IX_RECOVER_PAUSE exists in debug builds only; release builds run the smoke test below"
)]
fn test_kill_before_the_commit_leaves_nothing_outside_the_store() {
    let dir = tempdir("staged");
    kill(paused_at(&dir, "staged", 120));
    let run = dir.join("runs").join("r1");
    assert!(!run.join("committed.json").exists());
    let killed = attempt_ids(&dir);
    assert_eq!(killed.len(), 1);
    assert_nothing_outside_the_store(&dir);

    let (first, err) = rerun(&dir);
    assert!(err.contains("published attempt"), "{err}");
    assert!(
        !err.contains(&killed[0]),
        "the killed attempt was committed: {err}"
    );
    let (again, err) = rerun(&dir);
    assert!(
        err.contains("replayed") && err.contains("nothing executed"),
        "{err}"
    );
    assert_eq!(again, first);
    assert_eq!(attempt_ids(&dir).len(), 2);
    assert_nothing_outside_the_store(&dir);
    fs::remove_dir_all(&dir).ok();
}

#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "IX_RECOVER_PAUSE exists in debug builds only; release builds run the smoke test below"
)]
fn test_kill_after_the_commit_is_repaired_without_executing_again() {
    let dir = tempdir("committed");
    kill(paused_at(&dir, "committed", 120));
    let run = dir.join("runs").join("r1");
    assert!(run.join("committed.json").exists());
    assert!(!run.join("receipt.json").exists());
    let killed = attempt_ids(&dir);

    let (first, err) = rerun(&dir);
    assert!(
        err.contains("receipt repaired") && err.contains(&killed[0]),
        "{err}"
    );
    let summary: serde_json::Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(summary["stages"]["s"]["output"]["count"], 3, "{summary}");
    let (again, err) = rerun(&dir);
    assert!(err.contains("replayed"), "{err}");
    assert_eq!(again, first);
    assert_eq!(attempt_ids(&dir), killed);
    assert_nothing_outside_the_store(&dir);
    fs::remove_dir_all(&dir).ok();
}

#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "IX_RECOVER_PAUSE exists in debug builds only; release builds run the smoke test below"
)]
fn test_a_paused_ix_left_alone_stops_itself_at_its_deadline() {
    let dir = tempdir("deadline");
    let mut ix = paused_at(&dir, "staged", 1);
    let deadline = Instant::now() + Duration::from_secs(60);
    while ix.child().try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "ix outlived its pause deadline");
        thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(ix.reap().code(), Some(DEADLINE_EXIT));
    fs::remove_dir_all(&dir).ok();
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds honour IX_RECOVER_PAUSE; this checks that release builds ignore it"
)]
fn test_a_release_ix_ignores_the_pause_and_publishes() {
    let dir = tempdir("release");
    let marker = dir.join("staged.reached");
    let out = ix(&dir)
        .env(
            "IX_RECOVER_PAUSE",
            format!("staged|{}|60", marker.display()),
        )
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    assert!(err.contains("published attempt"), "{err}");
    assert!(!marker.exists(), "a release ix paused");
    let run = dir.join("runs").join("r1");
    assert!(run.join("committed.json").is_file() && run.join("receipt.json").is_file());
    let (again, err) = rerun(&dir);
    assert!(err.contains("replayed"), "{err}");
    assert_eq!(again, out.stdout);
    assert_nothing_outside_the_store(&dir);
    fs::remove_dir_all(&dir).ok();
}
