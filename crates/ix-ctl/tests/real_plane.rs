//! #453 B4 from the Rust client, against a real `ixctl up` process killed outright. The Go
//! proof (`TestAWaitAfterAHardPlaneStopEndsUnknownOtherEpoch`) drives the plane over raw HTTP;
//! this one drives it through `Ctl`, so the client's decoding of the answers is what is proved.
//!
//! It needs a built `ixctl` (`go build -o <exe> .` in `go/ixctl`) named by `IXCTL_BIN`, so it
//! is ignored by default; run with `--ignored`, it fails when `IXCTL_BIN` is not set rather
//! than passing without a plane.

use ix_ctl::{Ctl, CtlError, LaunchOutcome, RunOutcome, UnknownReason};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// `ixctl up` as its own process, read through its ready line: the process and its state file.
struct Plane {
    child: Child,
    state_file: PathBuf,
}

impl Drop for Plane {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn up(exe: &Path, spec: &Path, state: &Path) -> Plane {
    let mut child = Command::new(exe)
        .arg("up")
        .arg(spec)
        .arg("--state")
        .arg(state)
        .args(["--name", "hard"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("start ixctl up");
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .expect("ready line");
    let ready: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"));
    assert_eq!(Some("ready"), ready["event"].as_str(), "{ready}");
    assert_eq!(
        Some(u64::from(child.id())),
        ready["pid"].as_u64(),
        "{ready}"
    );
    let state_file = PathBuf::from(ready["state_file"].as_str().unwrap());
    Plane { child, state_file }
}

/// A handle on this boot only: [`Ctl::from_state`] would re-read the state file and follow a
/// restarted plane, and the wait in flight must stay with the plane it was sent to.
fn this_boot(plane: &Plane) -> Ctl {
    let st: Value = serde_json::from_str(&std::fs::read_to_string(&plane.state_file).unwrap())
        .expect("state file");
    Ctl::new(st["addr"].as_str().unwrap(), st["token"].as_str().unwrap())
}

fn job(status: &Value) -> &Value {
    status["services"]
        .as_array()
        .and_then(|s| s.iter().find(|s| s["name"] == "job"))
        .unwrap_or_else(|| panic!("no job in {status}"))
}

fn alive(pid: u64) -> bool {
    if cfg!(windows) {
        let out = Command::new("tasklist")
            .args(["/NH", "/FI", &format!("PID eq {pid}")])
            .output()
            .expect("tasklist");
        String::from_utf8_lossy(&out.stdout).contains(&format!(" {pid} "))
    } else {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .expect("kill -0")
            .success()
    }
}

/// A job that runs well past the test, on any OS. The plane resolves a bare command against
/// the spec's folder, so the path is absolute.
fn long_job() -> Value {
    if cfg!(windows) {
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let ping = Path::new(&root).join("System32").join("PING.EXE");
        json!({"name": "job", "command": ping, "args": ["-n", "120", "127.0.0.1"], "restart": "never"})
    } else {
        json!({"name": "job", "command": "/bin/sleep", "args": ["120"], "restart": "never"})
    }
}

/// The built plane, a fresh folder for this test, and the spec of one long job in it.
fn setup(test: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let exe =
        PathBuf::from(std::env::var_os("IXCTL_BIN").expect("IXCTL_BIN must name a built ixctl"));
    let dir = std::env::temp_dir().join(format!("ix-ctl-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let spec = dir.join("spec.json");
    std::fs::write(
        &spec,
        json!({"name": "hard", "services": [long_job()]}).to_string(),
    )
    .unwrap();
    let state = dir.join("state");
    (exe, dir, spec, state)
}

/// #450 against the real plane: it announces `run/1`, acknowledges `markers=run`, and the
/// stream carries the job's `ixctl.run.begin` with the run the launch wait names.
#[test]
#[ignore = "needs a built ixctl named by IXCTL_BIN; run with --ignored"]
fn test_a_real_plane_acknowledges_and_delivers_run_markers() {
    let (exe, dir, spec, state) = setup("markers");
    let plane = up(&exe, &spec, &state);
    let ctl = this_boot(&plane);
    let mut events = ctl
        .run_events("svc/", None, None, Some(Duration::from_secs(10)))
        .unwrap();
    let applied = ctl
        .apply(&json!({"services": {"job": {"state": "running"}}}), None)
        .unwrap();
    let epoch = applied["epoch"].as_str().unwrap();
    let generation = applied["generation"].as_u64().unwrap();
    let run = match ctl
        .wait_launch(epoch, generation, "job", Duration::from_secs(10), None)
        .unwrap()
    {
        LaunchOutcome::Launched(run) => run,
        other => panic!("launch: {other:?}"),
    };
    let begin = events
        .find(|e| e.as_ref().map_or(true, |e| e.kind == "ixctl.run.begin"))
        .expect("the stream ended before ixctl.run.begin")
        .unwrap();
    assert_eq!(Some(&run), begin.run.as_ref(), "{begin:?}");
    drop(events);
    drop(plane);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs a built ixctl named by IXCTL_BIN; run with --ignored"]
fn test_a_wait_after_a_hard_plane_stop_ends_unknown_other_epoch() {
    let (exe, dir, spec, state) = setup("b4");

    let mut first = up(&exe, &spec, &state);
    let ctl = this_boot(&first);
    let applied = ctl
        .apply(&json!({"services": {"job": {"state": "running"}}}), None)
        .unwrap();
    assert_eq!(
        "start-pending", applied["results"]["job"]["result"],
        "{applied}"
    );
    let epoch = applied["epoch"].as_str().unwrap().to_string();
    let generation = applied["generation"].as_u64().unwrap();
    let run = match ctl
        .wait_launch(&epoch, generation, "job", Duration::from_secs(10), None)
        .unwrap()
    {
        LaunchOutcome::Launched(run) => run,
        other => panic!("launch: {other:?}"),
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    let pid = loop {
        let status = ctl.status().unwrap();
        if let Some(pid) = job(&status)["pid"].as_u64() {
            break pid;
        }
        assert!(
            Instant::now() < deadline,
            "the job never reported a pid: {}",
            job(&status)
        );
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(alive(pid), "the job {pid} is not running");

    // A wait in flight on the plane about to die: it ends with its plane, with no outcome.
    let inflight = {
        let ctl = this_boot(&first);
        let run = run.clone();
        std::thread::spawn(move || ctl.wait_run(&run, Duration::from_secs(30), None))
    };
    std::thread::sleep(Duration::from_millis(200)); // the wait is admitted before the kill
    first.child.kill().unwrap();
    first.child.wait().unwrap();
    let began = Instant::now();
    match inflight.join().unwrap() {
        Err(CtlError::Io(_)) => {}
        other => panic!("the wait in flight on the killed plane answered {other:?}"),
    }
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "the wait outlived its plane"
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while alive(pid) {
        assert!(
            Instant::now() < deadline,
            "job {pid} outlived its killed plane"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(first);

    let again = up(&exe, &spec, &state);
    let ctl = Ctl::from_state(&again.state_file).unwrap();
    let status = ctl.status().unwrap();
    assert_ne!(Some(epoch.as_str()), status["epoch"].as_str(), "{status}");
    assert_eq!(Some(0), status["generation"].as_u64(), "{status}");
    assert_eq!("stopped", job(&status)["state"], "{status}");

    assert_eq!(
        RunOutcome::Unknown(UnknownReason::OtherEpoch),
        ctl.wait_run(&run, Duration::from_secs(5), None).unwrap()
    );
    assert_eq!(
        ix_ctl::GenerationOutcome::Unknown(UnknownReason::OtherEpoch),
        ctl.wait_generation_outcome(&epoch, generation, Duration::from_secs(5), None)
            .unwrap()
    );
    drop(again);
    let _ = std::fs::remove_dir_all(&dir);
}
