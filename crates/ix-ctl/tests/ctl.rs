//! The client and the `ix.ctl.*` capabilities against a fake plane: a std `TcpListener` that
//! answers `/v1/status`, `/v1/plan`, `/v1/apply` and `/v1/stop` with the contract's JSON, and
//! checks the token, the `Host` and the `Origin` the way `ixctl` does.
//!
//! The fake observes an applied generation, and a started pod's nodes, only after a delay, so
//! the waits have to poll. Its status is sent in chunks, the way Go's server streams an answer
//! past its buffer.

use ix_ctl::{Ctl, CtlError};
use ix_ixql::{Executor, MemoryHost};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const PODS: &str = include_str!("../examples/pods.ixql");
const TOKEN: &str = "5f1c0de2a9b84e7d93c6f0a1b2c3d4e5f60718293a4b5c6d7e8f9011223344aa";
/// How long the fake takes to observe an applied generation.
const CONVERGE: Duration = Duration::from_millis(120);
/// How long a started pod takes to report its nodes.
const NODES: Duration = Duration::from_millis(200);

#[derive(Default)]
struct Svc {
    desired: &'static str,
    since: Option<Instant>,
    last_exit: Value,
}

struct Plane {
    port: u16,
    generation: u64,
    observed: u64,
    pending: Option<(u64, Instant)>,
    services: BTreeMap<String, Svc>,
    /// The status's `diverged`; `None` leaves the field out, as a plane older than it does.
    diverged: Option<Value>,
    /// `METHOD PATH` of every request that passed the checks.
    log: Vec<String>,
}

impl Plane {
    fn status(&mut self) -> Value {
        if let Some((g, at)) = self.pending {
            if at.elapsed() >= CONVERGE {
                self.observed = g;
                self.pending = None;
            }
        }
        let services: Vec<Value> = self
            .services
            .iter()
            .map(|(name, s)| {
                let (state, nodes) = match s.since {
                    None => ("stopped", vec![]),
                    Some(t) if t.elapsed() < NODES => ("starting", vec![]),
                    Some(_) => ("running", pod_nodes()),
                };
                json!({
                    "name": name, "desired": s.desired, "state": state,
                    "pid": s.since.map(|_| 4242), "restarts": 0,
                    "started_at": s.since.map(|_| 1_760_000_000_000u64),
                    "last_exit": s.last_exit, "nodes": nodes,
                })
            })
            .collect();
        let mut status = json!({
            "name": "fake", "generation": self.generation,
            "observed_generation": self.observed, "services": services, "peers": [],
        });
        if let Some(d) = &self.diverged {
            status["diverged"] = d.clone();
        }
        status
    }

    /// The actions a desired-state document asks for; any undeclared name refuses all of it.
    fn plan(&self, doc: &Value) -> Result<(Vec<Value>, Vec<String>), String> {
        let wanted = doc
            .get("services")
            .and_then(Value::as_object)
            .ok_or("the document has no `services` record")?;
        if let Some(name) = wanted.keys().find(|n| !self.services.contains_key(*n)) {
            return Err(format!("service `{name}` is not declared"));
        }
        let (mut actions, mut noop) = (Vec::new(), Vec::new());
        for (name, want) in wanted {
            let running = self.services[name].since.is_some();
            match (want.get("state").and_then(Value::as_str), running) {
                (Some("running"), false) => actions.push(json!({"op": "start", "service": name})),
                (Some("stopped"), true) => actions.push(json!({"op": "stop", "service": name})),
                (Some("running" | "stopped"), _) => noop.push(name.clone()),
                _ => return Err(format!("`{name}`: state is `running` or `stopped`")),
            }
        }
        Ok((actions, noop))
    }

    fn stop(&mut self, name: &str) -> Value {
        let s = self.services.get_mut(name).expect("checked by the caller");
        s.desired = "stopped";
        s.since = None;
        s.last_exit = json!({"code": 0, "at": unix_ms()});
        s.last_exit.clone()
    }

    fn route(&mut self, method: &str, path: &str, body: &[u8]) -> (u16, Value) {
        let doc: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
        match (method, path) {
            ("GET", "/v1/status") => (200, self.status()),
            ("POST", "/v1/plan") => match self.plan(&doc) {
                Ok((actions, noop)) => (
                    200,
                    json!({"generation": self.generation, "actions": actions, "noop": noop}),
                ),
                Err(e) => (400, json!({ "error": e })),
            },
            ("POST", "/v1/apply") => match self.plan(&doc) {
                Ok((actions, _)) => {
                    self.generation += 1;
                    for a in &actions {
                        let name = a["service"].as_str().unwrap();
                        if a["op"] == "start" {
                            let s = self.services.get_mut(name).unwrap();
                            s.desired = "running";
                            s.since = Some(Instant::now());
                        } else {
                            self.stop(name);
                        }
                    }
                    self.pending = Some((self.generation, Instant::now()));
                    (
                        200,
                        json!({"generation": self.generation, "actions": actions}),
                    )
                }
                Err(e) => (400, json!({ "error": e })),
            },
            ("POST", "/v1/stop") => {
                let name = doc["service"].as_str().unwrap_or("");
                if !self.services.contains_key(name) {
                    return (
                        400,
                        json!({"error": format!("service `{name}` is not declared")}),
                    );
                }
                if doc.get("peer").is_some() {
                    return (400, json!({"error": "this plane has no peers"}));
                }
                self.generation += 1;
                let exit = self.stop(name);
                // A stop answers once the tree is gone, so it is observed at once.
                self.observed = self.generation;
                self.pending = None;
                (
                    200,
                    json!({"stopped": name, "generation": self.generation, "exit": exit}),
                )
            }
            _ => (404, json!({"error": format!("no {method} {path}")})),
        }
    }
}

fn pod_nodes() -> Vec<Value> {
    ["src", "s1", "s2", "sink"]
        .iter()
        .map(|n| {
            json!({"name": n, "queued": 0, "cap": 8, "received": 10, "emitted": 10,
                   "busy_ms": 0.1, "blocked_ms": 0.0, "workers": 1})
        })
        .collect()
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// A fake plane declaring `pod-a` and `pod-b`, both stopped, and a client for it.
fn plane() -> (Ctl, Arc<Mutex<Plane>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let services = ["pod-a", "pod-b"]
        .into_iter()
        .map(|n| {
            let svc = Svc {
                desired: "stopped",
                last_exit: Value::Null,
                ..Svc::default()
            };
            (n.to_string(), svc)
        })
        .collect();
    let plane = Arc::new(Mutex::new(Plane {
        port,
        generation: 0,
        observed: 0,
        pending: None,
        services,
        diverged: None,
        log: Vec::new(),
    }));
    let shared = plane.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = answer(stream, &shared);
        }
    });
    (Ctl::new(format!("127.0.0.1:{port}"), TOKEN), plane)
}

fn answer(stream: TcpStream, plane: &Mutex<Plane>) -> std::io::Result<()> {
    let mut r = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut words = line.split_whitespace();
    let (method, path) = (
        words.next().unwrap_or("").to_string(),
        words.next().unwrap_or("").to_string(),
    );
    let mut headers = BTreeMap::new();
    loop {
        let mut h = String::new();
        r.read_line(&mut h)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let length = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0; length];
    r.read_exact(&mut body)?;

    let mut plane = plane.lock().unwrap();
    let port = plane.port;
    let hosts = [
        format!("127.0.0.1:{port}"),
        format!("localhost:{port}"),
        format!("[::1]:{port}"),
    ];
    let origins = [
        format!("http://127.0.0.1:{port}"),
        format!("http://localhost:{port}"),
    ];
    let (code, value) = if !headers.get("host").is_some_and(|h| hosts.contains(h)) {
        (403, json!({"error": "foreign Host"}))
    } else if headers.get("origin").is_some_and(|o| !origins.contains(o)) {
        (403, json!({"error": "foreign Origin"}))
    } else if headers.get("authorization") != Some(&format!("Bearer {TOKEN}")) {
        (401, json!({"error": "missing or wrong token"}))
    } else {
        plane.log.push(format!("{method} {path}"));
        plane.route(&method, &path, &body)
    };
    drop(plane);

    let text = value.to_string();
    let mut w = stream;
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        _ => "Not Found",
    };
    write!(
        w,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nConnection: close\r\n"
    )?;
    if path == "/v1/status" && code == 200 {
        w.write_all(b"Transfer-Encoding: chunked\r\n\r\n")?;
        for chunk in text.as_bytes().chunks(37) {
            write!(w, "{:x}\r\n", chunk.len())?;
            w.write_all(chunk)?;
            w.write_all(b"\r\n")?;
        }
        w.write_all(b"0\r\n\r\n")?;
    } else {
        write!(w, "Content-Length: {}\r\n\r\n{text}", text.len())?;
    }
    w.flush()
}

fn executor(ctl: Ctl) -> Executor {
    let mut exec = Executor::new(Arc::new(MemoryHost::frozen()));
    ix_ctl::register(exec.capabilities(), ctl).unwrap();
    exec
}

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

#[test]
fn pods_ixql_plans_applies_waits_and_stops_through_the_state_file() {
    let (ctl, plane) = plane();
    let state = tmp(&format!("ix-ctl-pods-{}.json", plane.lock().unwrap().port));
    let doc =
        json!({"name": "fake", "addr": ctl.addr(), "pid": std::process::id(), "token": TOKEN});
    std::fs::write(&state, doc.to_string()).unwrap();
    let ctl = Ctl::from_state(&state).expect("the state file names the plane");
    let _ = std::fs::remove_file(&state);

    let began = Instant::now();
    let out = executor(ctl).run_source(PODS).expect("pods.ixql runs");
    let took = began.elapsed();
    let b = |name: &str| out.binding(name).unwrap_or_else(|| panic!("no `{name}`"));

    assert_eq!(
        &json!([{"op": "start", "service": "pod-a"}, {"op": "start", "service": "pod-b"}]),
        &b("planned")["actions"]
    );
    assert_eq!(json!(0), b("planned")["generation"], "plan changes nothing");
    assert_eq!(json!(1), b("applied")["generation"]);
    assert_eq!(json!(1), b("converged")["observed_generation"]);
    assert_eq!(json!("running"), b("pod_a")["state"]);
    assert_eq!(4, b("pod_a")["nodes"].as_array().unwrap().len());
    let states: Vec<&str> = b("status")["services"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["state"].as_str().unwrap())
        .collect();
    assert_eq!(vec!["running", "running"], states);
    assert_eq!(
        json!([]),
        b("replanned")["actions"],
        "converged: nothing left"
    );
    assert_eq!(json!(["pod-a", "pod-b"]), b("replanned")["noop"]);
    assert_eq!(json!("pod-a"), b("stopped_a")["stopped"]);
    assert_eq!(json!(0), b("stopped_b")["exit"]["code"]);
    // The waits had to outlast the fake's delays, polling rather than failing the first time.
    assert!(took >= NODES, "{took:?}");

    let plane = plane.lock().unwrap();
    assert_eq!(["POST /v1/plan", "POST /v1/apply"], plane.log[..2]);
    let polls = plane.log.iter().filter(|r| *r == "GET /v1/status").count();
    assert!(polls >= 4, "{:?}", plane.log);
    assert_eq!(
        ["POST /v1/plan", "POST /v1/stop", "POST /v1/stop"],
        plane.log[plane.log.len() - 3..]
    );
    assert!(plane.services.values().all(|s| s.since.is_none()));
}

#[test]
fn a_missing_or_incomplete_state_file_is_refused() {
    let missing = tmp("ix-ctl-no-such-plane.json");
    let _ = std::fs::remove_file(&missing);
    match Ctl::from_state(&missing) {
        Err(CtlError::State { path, reason }) => {
            assert_eq!(missing, path);
            assert!(reason.contains("not found"), "{reason}");
        }
        other => panic!("{other:?}"),
    }

    let tokenless = tmp("ix-ctl-tokenless.json");
    std::fs::write(
        &tokenless,
        r#"{"name": "x", "addr": "127.0.0.1:1", "pid": 1}"#,
    )
    .unwrap();
    let err = Ctl::from_state(&tokenless).unwrap_err();
    let _ = std::fs::remove_file(&tokenless);
    assert!(err.to_string().contains("no `token`"), "{err}");
}

#[test]
fn a_wrong_token_surfaces_as_a_401() {
    let (ctl, plane) = plane();
    match Ctl::new(ctl.addr(), "not-the-token").status() {
        Err(CtlError::Http { status, message }) => {
            assert_eq!(401, status);
            assert!(message.contains("token"), "{message}");
        }
        other => panic!("{other:?}"),
    }
    let err = executor(Ctl::new(ctl.addr(), ""))
        .run_source("s <- ix.ctl.status()")
        .unwrap_err();
    assert!(err.to_string().contains("401"), "{err}");
    assert!(plane.lock().unwrap().log.is_empty());
}

#[test]
fn an_undeclared_service_surfaces_as_a_400_and_nothing_is_applied() {
    let (ctl, plane) = plane();
    match ctl.plan(&json!({"services": {"nope": {"state": "running"}}})) {
        Err(CtlError::Http { status, message }) => {
            assert_eq!(400, status);
            assert!(message.contains("nope"), "{message}");
        }
        other => panic!("{other:?}"),
    }
    let err = executor(ctl.clone())
        .run_source(r#"a <- ix.ctl.apply({ services: { "pod-a": { state: "running" }, nope: { state: "running" } } })"#)
        .unwrap_err();
    assert!(err.to_string().contains("400"), "{err}");
    assert!(err.to_string().contains("nope"), "{err}");
    let err = ctl.stop("nope", None).unwrap_err();
    assert!(matches!(err, CtlError::Http { status: 400, .. }), "{err}");

    let plane = plane.lock().unwrap();
    assert_eq!(0, plane.generation);
    assert!(plane.services.values().all(|s| s.since.is_none()));
}

#[test]
fn argument_mistakes_fail_the_run_before_any_request() {
    let (ctl, plane) = plane();
    let exec = executor(ctl);
    for (src, why) in [
        ("a <- ix.ctl.wait(generation: 1)", "timeout_ms"),
        (
            r#"a <- ix.ctl.wait(service: "pod-a", timeout_ms: 10)"#,
            "nodes",
        ),
        (
            "a <- ix.ctl.wait(generation: 1.5, timeout_ms: 10)",
            "whole number",
        ),
        ("a <- ix.ctl.stop()", "service"),
        (
            r#"a <- ix.ctl.stop(service: "pod-a", force: true)"#,
            "force",
        ),
        ("a <- ix.ctl.plan()", "document"),
        ("a <- ix.ctl.apply({ pods: {} })", "services"),
        ("a <- { services: {} } → ix.ctl.status", "piped"),
        (r#"a <- ix.ctl.status("pod-a")"#, "named"),
    ] {
        let err = exec.run_source(src).unwrap_err().to_string();
        assert!(err.contains(why), "{src}: {err}");
    }
    assert!(plane.lock().unwrap().log.is_empty());
}

#[test]
fn a_wait_that_runs_out_says_what_it_saw() {
    let (ctl, _plane) = plane();
    let began = Instant::now();
    let err = ctl
        .wait_nodes("pod-b", 4, Duration::from_millis(100))
        .unwrap_err();
    assert!(began.elapsed() >= Duration::from_millis(100));
    assert!(began.elapsed() < Duration::from_secs(5));
    assert!(matches!(err, CtlError::Wait(_)), "{err:?}");
    assert!(
        err.to_string().contains("0 of 4 nodes in state `stopped`"),
        "{err}"
    );

    let err = ctl
        .wait_generation(7, Duration::from_millis(50))
        .unwrap_err();
    assert!(err.to_string().contains("observed 0"), "{err}");

    let err = ctl
        .wait_nodes("ghost", 1, Duration::from_millis(50))
        .unwrap_err();
    assert!(
        matches!(err, CtlError::NotDeclared(ref n) if n == "ghost"),
        "{err:?}"
    );
}

#[test]
fn a_diverged_generation_fails_the_wait_at_once() {
    let (ctl, plane) = plane();
    {
        let mut p = plane.lock().unwrap();
        p.generation = 2;
        p.observed = 1;
        p.diverged =
            Some(json!({"generation": 2, "service": "pod-b", "reason": "launch: not found"}));
    }
    let began = Instant::now();
    let err = ctl.wait_generation(2, Duration::from_secs(5)).unwrap_err();
    let took = began.elapsed();
    assert!(
        took < Duration::from_secs(1),
        "waited {took:?} for a failure"
    );
    match &err {
        CtlError::Diverged {
            generation,
            service,
            reason,
        } => assert_eq!(
            (2, "pod-b", "launch: not found"),
            (*generation, service.as_str(), reason.as_str())
        ),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        "generation 2 diverged on pod-b: launch: not found",
        err.to_string()
    );
    let err = executor(ctl.clone())
        .run_source("a <- ix.ctl.wait(generation: 2, timeout_ms: 5000)")
        .unwrap_err();
    assert!(
        err.to_string().contains("generation 2 diverged on pod-b"),
        "{err}"
    );

    // An older divergence, or none, leaves a later generation pending: the wait runs out.
    for diverged in [
        json!({"generation": 1, "service": "pod-a", "reason": "exited 1"}),
        Value::Null,
    ] {
        plane.lock().unwrap().diverged = Some(diverged);
        let err = ctl
            .wait_generation(2, Duration::from_millis(60))
            .unwrap_err();
        assert!(matches!(err, CtlError::Wait(_)), "{err:?}");
    }
}

#[test]
fn the_token_goes_to_loopback_only_and_never_into_debug_output() {
    let ctl = Ctl::new("192.0.2.1:80", TOKEN);
    assert!(matches!(ctl.status(), Err(CtlError::NotLoopback(_))));
    assert!(!format!("{ctl:?}").contains(TOKEN));
}
