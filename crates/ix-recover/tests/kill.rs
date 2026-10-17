//! Real process kills. The parent test re-executes this test binary as a
//! child, which runs the ignored test `kill_child_entry`: it begins a run,
//! stops at one publish point, writes a marker and waits. The parent kills it there
//! (`Child::kill`: TerminateProcess on Windows, SIGKILL on Unix), then
//! reconciles the run in its own process.

use ix_recover::{Begin, Error, Point, Store};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use std::{env, fs, thread};

const CHILD_ENV: &str = "IX_RECOVER_KILL_CHILD";
const SUMMARY: &[u8] = b"{\"mean\":2}\n";

fn bindings() -> Value {
    json!({ "spec_sha256": "sha256:aa", "ix_exe_sha256": "sha256:bb" })
}

/// The child's entry point, run only by `kill_at`. Ignored in an ordinary
/// test run; run by hand it fails, since it has no parent to kill it.
#[test]
#[ignore = "the child half of the kill tests, which run it themselves"]
fn kill_child_entry() {
    let spec = env::var(CHILD_ENV).expect("kill_child_entry runs only as the kill tests' child");
    let (base, point) = spec.rsplit_once('|').expect("BASE|POINT");
    let stop_at = match point {
        "staged" => Point::Staged,
        "committed" => Point::Committed,
        other => panic!("unknown point {other}"),
    };
    let store = Store::open(Path::new(base).join("runs")).unwrap();
    let Begin::Fresh(attempt) = store.begin("run", &bindings()).unwrap() else {
        panic!("the child expected a fresh run");
    };
    let marker = Path::new(base).join(format!("{point}.reached"));
    attempt
        .publish_observed(&[("summary.json", SUMMARY)], &mut |p| {
            if p == stop_at {
                fs::write(&marker, std::process::id().to_string()).unwrap();
                loop {
                    thread::sleep(Duration::from_secs(1));
                }
            }
        })
        .unwrap();
    panic!("the child passed its stop point");
}

/// Run a child to `point`, kill it there, and return the store it left.
fn kill_at(point: &str) -> (tempfile::TempDir, Store, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let mut child = Command::new(env::current_exe().unwrap())
        .args([
            "--exact",
            "kill_child_entry",
            "--ignored",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(CHILD_ENV, format!("{}|{point}", t.path().display()))
        .spawn()
        .unwrap();
    let marker = t.path().join(format!("{point}.reached"));
    let deadline = Instant::now() + Duration::from_secs(120);
    while !marker.exists() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("the child exited before {point}: {status}");
        }
        assert!(Instant::now() < deadline, "the child never reached {point}");
        thread::sleep(Duration::from_millis(20));
    }
    child.kill().unwrap();
    let status = child.wait().unwrap();
    assert!(!status.success(), "the child was not killed: {status}");
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
