//! Run markers on the Rust client (#450): `Ctl::run_events` asks the plane for
//! `ixctl.run.begin`/`ixctl.run.end`, only from a plane whose status announces contract
//! `run/1`, and only when the plane acknowledges the request; a plane that ignores it is an
//! error, never a stream silently without markers. The fake plane is a loopback listener that
//! keeps each request line and answers status and events as the test says.

use ix_ctl::{Ctl, CtlError, RunId};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

/// How the fake answers `GET /v1/events`.
#[derive(Clone, Copy)]
enum Events {
    /// With `Ixctl-Markers: run`, then one `ixctl.run.begin` event.
    Acknowledged,
    /// Like a plane older than run markers: no header, one ordinary event.
    Ignored,
    /// A refusal, as the plane answers a markers value it does not know.
    Refused,
}

fn fake(contract: Option<&'static str>, events: Events) -> (Ctl, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { return };
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let mut first = String::new();
            r.read_line(&mut first).unwrap();
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
            }
            let first = first.trim_end().to_string();
            log.lock().unwrap().push(first.clone());
            if first.starts_with("GET /v1/status") {
                let mut body = json!({ "name": "svc", "epoch": "e-1", "services": [] });
                if let Some(c) = contract {
                    body["contract"] = json!(c);
                }
                let body = body.to_string();
                let _ = write!(
                    conn,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                continue;
            }
            let (head, kind) = match events {
                Events::Acknowledged => ("Ixctl-Markers: run\r\n", "ixctl.run.begin"),
                Events::Ignored => ("", "service.started"),
                Events::Refused => {
                    let body = r#"{"error":"markers must be \"run\""}"#;
                    let _ = write!(
                        conn,
                        "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    );
                    continue;
                }
            };
            let envelope = json!({
                "ch": "svc/job", "seq": 7, "epoch": "e-1", "origin": "svc",
                "run": { "origin": "svc", "epoch": "e-1", "seq": 6 },
                "ev": { "kind": kind, "t": 1, "data": { "generation": 1 } }
            });
            let _ = write!(
                conn,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{head}\r\n\
                 id: e-1:7\nevent: {kind}\ndata: {envelope}\n\n"
            );
        }
    });
    (Ctl::new(addr, TOKEN), seen)
}

fn events_line(seen: &Arc<Mutex<Vec<String>>>) -> Option<String> {
    seen.lock()
        .unwrap()
        .iter()
        .find(|l| l.starts_with("GET /v1/events"))
        .cloned()
}

#[test]
fn test_acknowledged_run_markers_are_delivered() {
    let (ctl, seen) = fake(Some("run/1"), Events::Acknowledged);
    let mut events = ctl
        .run_events("svc/", None, None, Some(Duration::from_secs(2)))
        .unwrap();
    let first = events.next().unwrap().unwrap();
    assert_eq!("ixctl.run.begin", first.kind);
    // The run is the envelope's, not the event's own seq: the plane counts them apart.
    let run = RunId {
        origin: "svc".into(),
        epoch: "e-1".into(),
        seq: 6,
    };
    assert_eq!(Some(&run), first.run.as_ref());
    let flat = Value::from(&first);
    assert_eq!(
        json!({"origin": "svc", "epoch": "e-1", "seq": 6}),
        flat["run"]
    );
    let line = events_line(&seen).unwrap();
    assert!(line.contains("markers=run"), "{line}");
    // The contract is read before the stream is asked for.
    let lines = seen.lock().unwrap().clone();
    assert!(lines[0].starts_with("GET /v1/status"), "{lines:?}");
}

#[test]
fn test_a_plane_that_ignores_the_request_is_an_error_not_a_quiet_stream() {
    let (ctl, _) = fake(Some("run/1"), Events::Ignored);
    match ctl.run_events("svc/", None, None, Some(Duration::from_secs(2))) {
        Err(CtlError::Protocol(m)) => assert!(m.contains("acknowledge"), "{m}"),
        Err(other) => panic!("{other:?}"),
        Ok(_) => panic!("a stream without its acknowledgement was handed over"),
    }
}

#[test]
fn test_without_contract_run_1_no_markers_are_asked_for() {
    for contract in [None, Some("run/2")] {
        let (ctl, seen) = fake(contract, Events::Acknowledged);
        match ctl.run_events("svc/", None, None, Some(Duration::from_secs(2))) {
            Err(CtlError::Protocol(m)) => assert!(m.contains("run/1"), "{m}"),
            Err(other) => panic!("{contract:?}: {other:?}"),
            Ok(_) => panic!("{contract:?}: markers relied on without contract run/1"),
        }
        assert_eq!(None, events_line(&seen), "{contract:?}");
    }
}

#[test]
fn test_a_refused_markers_request_is_the_planes_refusal() {
    let (ctl, _) = fake(Some("run/1"), Events::Refused);
    match ctl.run_events("svc/", None, None, Some(Duration::from_secs(2))) {
        Err(CtlError::Http { status, .. }) => assert_eq!(400, status),
        Err(other) => panic!("{other:?}"),
        Ok(_) => panic!("a refused request was handed over as a stream"),
    }
}

#[test]
fn test_plain_events_never_ask_for_markers() {
    let (ctl, seen) = fake(Some("run/1"), Events::Ignored);
    let mut events = ctl
        .events_within("svc/", None, None, Duration::from_secs(2))
        .unwrap();
    assert_eq!("service.started", events.next().unwrap().unwrap().kind);
    let line = events_line(&seen).unwrap();
    assert!(!line.contains("markers"), "{line}");
}
