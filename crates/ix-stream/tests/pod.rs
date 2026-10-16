//! The `pod` example as `ixctl` sees it: JSON events on stdout, a clean exit when stdin
//! closes, and a failing one with `--crash-after`.

use serde_json::Value;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// `cargo test` builds the examples beside the test binaries: `target/<profile>/examples`.
fn pod() -> Command {
    let exe = std::env::current_exe().unwrap();
    let path: PathBuf = exe
        .parent()
        .and_then(|deps| deps.parent())
        .unwrap()
        .join("examples")
        .join(format!("pod{}", std::env::consts::EXE_SUFFIX));
    assert!(
        path.exists(),
        "{} is missing: `cargo test -p ix-stream` builds it, `--test pod` alone does not",
        path.display()
    );
    Command::new(path)
}

/// Waits for `child` to exit, killing it and failing after `limit`.
fn exit_within(child: &mut Child, limit: Duration) -> ExitStatus {
    let deadline = Instant::now() + limit;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("the pod did not exit within {limit:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_pod_prints_its_shape_then_stats_and_stops_when_stdin_closes() {
    let mut child = pod()
        .args(["--name", "t", "--nodes", "5", "--rate", "200"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, lines) = mpsc::channel::<Value>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            let v = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("not one JSON event per line: {line:?}: {e}"));
            if tx.send(v).is_err() {
                return;
            }
        }
    });
    let next = || {
        lines
            .recv_timeout(Duration::from_secs(10))
            .expect("an event")
    };

    let meta = next();
    assert_eq!("meta", meta["kind"]);
    assert_eq!("t", meta["data"]["name"]);
    assert_eq!(5, meta["data"]["shape"]["nodes"].as_array().unwrap().len());

    // A `stats` event whose sink has received values: the pod's five members, flowing.
    let stats = loop {
        let e = next();
        if e["kind"] == "stats" && e["data"]["nodes"][4]["received"].as_u64() > Some(0) {
            break e;
        }
    };
    let names: Vec<&str> = stats["data"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["name"].as_str().unwrap())
        .collect();
    assert_eq!(vec!["src", "s1", "s2", "s3", "sink"], names);

    drop(child.stdin.take());
    let status = exit_within(&mut child, Duration::from_secs(10));
    assert!(status.success(), "{status:?}");
    // Up to the end of stdout, which the reader reaches after the exit.
    let mut rest = Vec::new();
    while let Ok(e) = lines.recv_timeout(Duration::from_secs(5)) {
        rest.push(e);
    }
    let last = rest.last().expect("events after the stop");
    assert_eq!("done", last["kind"]);
    assert!(last["data"]["values"].as_u64() > Some(0));
}

#[test]
fn crash_after_exits_with_a_failure() {
    let mut child = pod()
        .args(["--nodes", "2", "--crash-after", "0.3"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let status = exit_within(&mut child, Duration::from_secs(10));
    assert_eq!(Some(101), status.code());
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(stderr.contains("crashing"), "{stderr}");
}
