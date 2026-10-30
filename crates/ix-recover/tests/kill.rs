//! Real process kills. The parent test re-executes this test binary as a
//! child, which runs the ignored test `kill_child_entry`: it begins a run,
//! stops at one publish point, writes a marker and waits. The parent kills it there
//! (`Child::kill`: TerminateProcess on Windows, SIGKILL on Unix), then
//! reconciles the run in its own process.
//!
//! A child does not outlive its test: a `Reaper` kills and reaps it on every
//! way out of the parent, a panic included, and a child left waiting exits by
//! itself after its deadline, should the parent die without unwinding.

use ix_recover::{Begin, Error, Point, Store};
use serde_json::{json, Value};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::time::{Duration, Instant};
use std::{env, fs, thread};

const CHILD_ENV: &str = "IX_RECOVER_KILL_CHILD";
const SUMMARY: &[u8] = b"{\"mean\":2}\n";
/// How long a child waits for the parent's kill before it exits by itself.
const WAIT_SECS: u64 = 120;
/// What a child exits with when its deadline passes.
const DEADLINE_EXIT: i32 = 86;

fn bindings() -> Value {
    json!({ "spec_sha256": "sha256:aa", "ix_exe_sha256": "sha256:bb" })
}

/// The child's entry point, run only by `spawn`. Ignored in an ordinary
/// test run; run by hand it fails, since it has no parent to kill it.
/// POINT is `staged` or `committed`, where it writes its marker and waits, or
/// `hold`, where it holds the run and waits without ever writing one.
#[test]
#[ignore = "the child half of the kill tests, which run it themselves"]
fn kill_child_entry() {
    let spec = env::var(CHILD_ENV).expect("kill_child_entry runs only as the kill tests' child");
    let mut parts = spec.rsplitn(3, '|');
    let (Some(secs), Some(point), Some(base)) = (parts.next(), parts.next(), parts.next()) else {
        panic!("BASE|POINT|SECS, got {spec}");
    };
    let deadline = Duration::from_secs(secs.parse().expect("SECS is a number of seconds"));
    let stop_at = match point {
        "staged" => Some(Point::Staged),
        "committed" => Some(Point::Committed),
        "hold" => None,
        other => panic!("unknown point {other}"),
    };
    let store = Store::open(Path::new(base).join("runs")).unwrap();
    let Begin::Fresh(attempt) = store.begin("run", &bindings()).unwrap() else {
        panic!("the child expected a fresh run");
    };
    let Some(stop_at) = stop_at else {
        wait_out(deadline)
    };
    let marker = marker(Path::new(base), point);
    attempt
        .publish_observed(&[("summary.json", SUMMARY)], &mut |p| {
            if p == stop_at {
                fs::write(&marker, std::process::id().to_string()).unwrap();
                wait_out(deadline);
            }
        })
        .unwrap();
    panic!("the child passed its stop point");
}

/// Wait to be killed; exit by itself once `deadline` has passed.
fn wait_out(deadline: Duration) -> ! {
    thread::sleep(deadline);
    std::process::exit(DEADLINE_EXIT);
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

fn marker(base: &Path, point: &str) -> PathBuf {
    base.join(format!("{point}.reached"))
}

fn spawn(base: &Path, point: &str, secs: u64) -> Reaper {
    Reaper(Some(
        Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "kill_child_entry",
                "--ignored",
                "--nocapture",
                "--test-threads",
                "1",
            ])
            .env(CHILD_ENV, format!("{}|{point}|{secs}", base.display()))
            .spawn()
            .unwrap(),
    ))
}

/// Wait until the child reaches `point`; fail if it exits first or takes
/// longer than `within`.
fn wait_for(child: &mut Reaper, base: &Path, point: &str, within: Duration) {
    let marker = marker(base, point);
    let deadline = Instant::now() + within;
    while !marker.exists() {
        if let Some(status) = child.child().try_wait().unwrap() {
            panic!("the child exited before {point}: {status}");
        }
        assert!(Instant::now() < deadline, "the child never reached {point}");
        thread::sleep(Duration::from_millis(20));
    }
}

/// Run a child to `point`, kill it there, and return the store it left.
fn kill_at(point: &str) -> (tempfile::TempDir, Store, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let mut child = spawn(t.path(), point, WAIT_SECS);
    let pid = child.child().id();
    wait_for(&mut child, t.path(), point, Duration::from_secs(120));
    assert!(
        pid_alive(pid),
        "pid_alive does not see the live child {pid}"
    );
    let status = child.kill();
    assert!(
        !status.success() && status.code() != Some(DEADLINE_EXIT),
        "the child was not killed: {status}"
    );
    assert!(!pid_alive(pid), "the child ({pid}) outlived its kill");
    let store = Store::open(t.path().join("runs")).unwrap();
    let run = store.root().join("run");
    (t, store, run)
}

/// The OS drops a dead owner's lock as part of process rundown; allow it a
/// moment rather than asserting it is instantaneous.
fn begin_after_kill(store: &Store) -> Begin {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match store.begin("run", &bindings()) {
            Err(Error::Unknown(_)) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(50))
            }
            other => return other.unwrap(),
        }
    }
}

fn attempt_count(run: &Path) -> usize {
    fs::read_dir(run.join("attempts")).unwrap().count()
}

#[test]
fn test_kill_after_commit_repairs_the_receipt_without_publishing_again() {
    let (_t, store, run) = kill_at("committed");
    assert!(run.join("committed.json").exists());
    assert!(!run.join("receipt.json").exists());
    match begin_after_kill(&store) {
        Begin::Repaired(rec) => assert_eq!(rec.file("summary.json"), Some(SUMMARY)),
        other => panic!("expected Repaired, got {other:?}"),
    }
    assert_eq!(attempt_count(&run), 1);
    assert!(matches!(begin_after_kill(&store), Begin::Done(_)));
}

#[test]
fn test_kill_before_commit_leaves_residue_and_a_fresh_attempt() {
    let (_t, store, run) = kill_at("staged");
    assert!(!run.join("committed.json").exists());
    let Begin::Fresh(attempt) = begin_after_kill(&store) else {
        panic!("expected Fresh after a kill before the commit");
    };
    let rec = attempt.publish(&[("summary.json", SUMMARY)]).unwrap();
    // The killed attempt's staged files stay; only the new one is committed.
    assert_eq!(attempt_count(&run), 2);
    match begin_after_kill(&store) {
        Begin::Done(again) => assert_eq!(again, rec),
        other => panic!("expected Done, got {other:?}"),
    }
}

#[test]
fn test_a_wait_that_times_out_still_kills_and_reaps_the_child() {
    let t = tempfile::tempdir().unwrap();
    let base = t.path().to_path_buf();
    let mut child = spawn(&base, "hold", WAIT_SECS);
    let pid = child.child().id();
    assert!(
        pid_alive(pid),
        "pid_alive does not see the live child {pid}"
    );
    let failed = panic::catch_unwind(AssertUnwindSafe(move || {
        wait_for(&mut child, &base, "hold", Duration::from_secs(2));
    }))
    .expect_err("the wait had to time out: a holding child writes no marker");
    let message = failed
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| failed.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(message.contains("never reached hold"), "{message}");
    // Unwinding dropped the reaper, which killed and reaped the child.
    assert!(
        !pid_alive(pid),
        "the child ({pid}) outlived the failed wait"
    );
    let store = Store::open(t.path().join("runs")).unwrap();
    assert!(matches!(begin_after_kill(&store), Begin::Fresh(_)));
}

#[test]
fn test_a_child_left_at_its_point_stops_itself_at_its_deadline() {
    let t = tempfile::tempdir().unwrap();
    let mut child = spawn(t.path(), "staged", 1);
    wait_for(&mut child, t.path(), "staged", Duration::from_secs(120));
    let deadline = Instant::now() + Duration::from_secs(60);
    while child.child().try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "the child outlived its deadline");
        thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(child.reap().code(), Some(DEADLINE_EXIT));
}
