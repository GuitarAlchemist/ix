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
    /// The status's `settled_generation` and `failed`; `None` leaves both out, as a plane
    /// older than them does.
    settled: Option<(u64, Vec<Value>)>,
    /// `METHOD PATH` of every request that passed the checks.
    log: Vec<String>,
    /// The boot id in status and apply replies; `None` leaves it out, as an older plane does.
    epoch: Option<String>,
    /// What `GET /v1/status?peer=b` answers: the peer's own status, relayed. `None` means
    /// this plane has no peers, and every routed op is refused.
    peer_status: Option<Value>,
    /// How many more routed status reads answer 502, the link to b being down.
    peer_down_for: u32,
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
        if let Some((settled, failed)) = &self.settled {
            status["settled_generation"] = json!(settled);
            status["failed"] = json!(failed);
        }
        if let Some(e) = &self.epoch {
            status["epoch"] = json!(e);
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
        // Routing rides beside the document, as `?peer=`; this plane has no peers.
        let (path, query) = path.split_once('?').unwrap_or((path, ""));
        if query.starts_with("peer=") {
            if let (("GET", "/v1/status", "peer=b"), Some(s)) =
                ((method, path, query), &self.peer_status)
            {
                if self.peer_down_for > 0 {
                    self.peer_down_for -= 1;
                    return (
                        502,
                        json!({"error": "peer \"b\" is down: link reconnecting"}),
                    );
                }
                return (200, s.clone());
            }
            return (400, json!({"error": "this plane has no peers"}));
        }
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
                    let mut reply = json!({"generation": self.generation, "actions": actions});
                    if let Some(e) = &self.epoch {
                        reply["epoch"] = json!(e);
                    }
                    (200, reply)
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
        settled: None,
        log: Vec::new(),
        epoch: Some("3f9a1c0b7d2e".into()),
        peer_status: None,
        peer_down_for: 0,
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
    match ctl.plan(&json!({"services": {"nope": {"state": "running"}}}), None) {
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

// F2.3: routing is not state. The peer is named beside the document, as `?peer=` on the
// request and `peer:` in IXQL, and the document itself goes out without it.
#[test]
fn a_peer_is_named_beside_the_document() {
    let (ctl, plane) = plane();
    let doc = json!({"services": {"pod-a": {"state": "running"}}});
    match ctl.apply(&doc, Some("b")) {
        Err(CtlError::Http {
            status: 400,
            message,
        }) => {
            assert!(message.contains("no peers"), "{message}")
        }
        other => panic!("{other:?}"),
    }
    let err = executor(ctl.clone())
        .run_source(
            r#"a <- ix.ctl.plan({ services: { "pod-a": { state: "running" } } }, peer: "b")"#,
        )
        .unwrap_err();
    assert!(err.to_string().contains("no peers"), "{err}");
    assert_eq!(
        0,
        ctl.apply(&doc, None).unwrap()["generation"]
            .as_u64()
            .unwrap()
            - 1
    );

    let plane = plane.lock().unwrap();
    assert_eq!(
        [
            "POST /v1/apply?peer=b",
            "POST /v1/plan?peer=b",
            "POST /v1/apply"
        ],
        plane.log[..]
    );
    assert_eq!(1, plane.generation);
}

// F2.4: a generation means nothing across boots. A wait that names the apply reply's epoch
// ends at once when the plane reports another boot, instead of waiting for the new boot to
// reach the same number; against a plane that reports no epoch it waits as before.
#[test]
fn a_wait_scoped_to_a_boot_ends_at_once_after_a_restart() {
    let (ctl, plane) = plane();
    let doc = json!({"services": {"pod-a": {"state": "running"}}});
    let applied = ctl.apply(&doc, None).unwrap();
    let (g, epoch) = (
        applied["generation"].as_u64().unwrap(),
        applied["epoch"].as_str().unwrap().to_string(),
    );
    assert_eq!("3f9a1c0b7d2e", epoch);
    ctl.wait_generation_of(&epoch, g, Duration::from_secs(5))
        .expect("same boot: the wait converges");

    plane.lock().unwrap().epoch = Some("0badc0ffee11".into());
    let began = Instant::now();
    let err = ctl
        .wait_generation_of(&epoch, g, Duration::from_secs(5))
        .unwrap_err();
    assert!(began.elapsed() < Duration::from_secs(1), "ended at once");
    match &err {
        CtlError::Restarted {
            generation,
            epoch: was,
            now,
        } => assert_eq!(
            (g, epoch.as_str(), "0badc0ffee11"),
            (*generation, was.as_str(), now.as_str())
        ),
        other => panic!("{other:?}"),
    }
    assert!(err.to_string().contains("restarted"), "{err}");
    let err = executor(ctl.clone())
        .run_source(&format!(
            "a <- ix.ctl.wait(generation: {g}, epoch: \"{epoch}\", timeout_ms: 5000)"
        ))
        .unwrap_err();
    assert!(err.to_string().contains("restarted"), "{err}");
    // The unscoped wait still answers by number alone.
    ctl.wait_generation(g, Duration::from_secs(5)).unwrap();

    // An older plane reports no epoch: the scoped wait cannot tell, and waits as before.
    plane.lock().unwrap().epoch = None;
    ctl.wait_generation_of(&epoch, g, Duration::from_secs(5))
        .expect("no epoch reported: waited by number");
}

// A generation applied through a peer is that peer's: its reply carries the peer's epoch and
// number, so the wait reads the peer's status (`?peer=`) too. Read from this plane instead,
// the epochs differ and the wait would report a restart that never happened.
#[test]
fn a_wait_for_a_generation_applied_through_a_peer_asks_that_peer() {
    let (ctl, plane) = plane();
    plane.lock().unwrap().peer_status = Some(json!({
        "name": "b", "epoch": "b00b00b00b00", "generation": 3, "observed_generation": 3,
        "settled_generation": 3, "failed": [], "services": [
            {"name": "pod-x", "desired": "running", "state": "running", "pid": 7, "restarts": 0,
             "started_at": 1_760_000_000_000u64, "last_exit": null, "nodes": pod_nodes()}],
        "peers": [],
    }));
    let status = ctl
        .wait_generation_at("b", Some("b00b00b00b00"), 3, Duration::from_secs(2))
        .expect("b's generation 3 has settled on b");
    assert_eq!("b", status["name"]);
    let entry = ctl
        .wait_nodes_at("b", "pod-x", 4, Duration::from_secs(2))
        .unwrap();
    assert_eq!("pod-x", entry["name"]);
    assert_eq!("b", ctl.status_of(Some("b")).unwrap()["name"]);

    // The same wait asked of this plane compares b's epoch with this plane's.
    let err = ctl
        .wait_generation_of("b00b00b00b00", 3, Duration::from_secs(2))
        .unwrap_err();
    assert!(matches!(err, CtlError::Restarted { .. }), "{err}");

    let out = executor(ctl.clone())
        .run_source(
            "done <- ix.ctl.wait(generation: 3, epoch: \"b00b00b00b00\", peer: \"b\", timeout_ms: 2000)\n\
             pod <- ix.ctl.wait(service: \"pod-x\", nodes: 4, peer: \"b\", timeout_ms: 2000)\n\
             bst <- ix.ctl.status(peer: \"b\")\n",
        )
        .expect("the IXQL waits read b through the peer");
    assert_eq!(json!("b"), out.binding("bst").unwrap()["name"]);
    assert_eq!(json!("pod-x"), out.binding("pod").unwrap()["name"]);
    {
        let log = &plane.lock().unwrap().log;
        assert!(
            log.iter().filter(|r| *r == "GET /v1/status?peer=b").count() >= 5,
            "{log:?}"
        );
    }

    // The link to b is down for a moment (b, or this plane, restarted): the plane answers
    // 502 until it is back, and the wait rides that out instead of failing on the first one.
    plane.lock().unwrap().peer_down_for = 3;
    let status = ctl
        .wait_generation_at("b", Some("b00b00b00b00"), 3, Duration::from_secs(2))
        .expect("three 502s, then b's status");
    assert_eq!("b", status["name"]);
    assert_eq!(0, plane.lock().unwrap().peer_down_for);
    // Past the deadline the 502 is what the wait reports; a 400 (no such peer) is final at once.
    plane.lock().unwrap().peer_down_for = 1000;
    let began = Instant::now();
    let err = ctl
        .wait_generation_at("b", Some("b00b00b00b00"), 3, Duration::from_millis(150))
        .unwrap_err();
    assert!(matches!(err, CtlError::Http { status: 502, .. }), "{err}");
    assert!(began.elapsed() >= Duration::from_millis(150));
    // The node wait through a peer rides out the same moment.
    plane.lock().unwrap().peer_down_for = 3;
    let entry = ctl
        .wait_nodes_at("b", "pod-x", 4, Duration::from_secs(2))
        .expect("three 502s, then pod-x on b");
    assert_eq!("pod-x", entry["name"]);
    plane.lock().unwrap().peer_down_for = 0;
    let began = Instant::now();
    let err = ctl
        .wait_generation_at("zz", Some("b00b00b00b00"), 3, Duration::from_secs(5))
        .unwrap_err();
    assert!(matches!(err, CtlError::Http { status: 400, .. }), "{err}");
    assert!(began.elapsed() < Duration::from_secs(1));
}

// A plane that restarts while a scoped wait polls it listens at a new address with a new
// token. A handle made from the state file re-reads the file when the old plane stops
// answering, reaches the new boot and reports the restart; a handle made from an address
// alone has nothing to re-read, and the connection error stands.
#[test]
fn a_scoped_wait_follows_the_state_file_across_a_restart() {
    // The new boot, already listening; the state file will name it a little later.
    let (live, plane) = plane();
    let live_addr = live.addr().to_string();
    // The old boot's address and token: nothing listens there any more, and its token is a
    // 401 on the new boot, so a status the new boot logs proves the handle was re-read twice.
    let gone = TcpListener::bind("127.0.0.1:0").unwrap();
    let dead_addr = format!("127.0.0.1:{}", gone.local_addr().unwrap().port());
    drop(gone);
    assert_ne!(dead_addr, live_addr);
    let old_token = "0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0";
    let state = tmp(&format!("ix-ctl-restart-{}.json", std::process::id()));
    let state_doc =
        |addr: &str, token: &str| json!({"name": "fake", "addr": addr, "pid": 1, "token": token});
    std::fs::write(&state, state_doc(&dead_addr, old_token).to_string()).unwrap();
    let ctl = Ctl::from_state(&state).unwrap();

    // The file moves twice: first to the new address with the old token (the new boot answers
    // 401, the second cause for a re-read), then to the new token.
    let rewrite = std::thread::spawn({
        let state = state.clone();
        move || {
            std::thread::sleep(Duration::from_millis(300));
            std::fs::write(&state, state_doc(&live_addr, old_token).to_string()).unwrap();
            std::thread::sleep(Duration::from_millis(300));
            std::fs::write(&state, state_doc(&live_addr, TOKEN).to_string()).unwrap();
        }
    });
    let began = Instant::now();
    let err = ctl
        .wait_generation_of("0ld0ld0ld0ld", 3, Duration::from_secs(20))
        .unwrap_err();
    let took = began.elapsed();
    rewrite.join().unwrap();
    let _ = std::fs::remove_file(&state);
    assert!(
        matches!(&err, CtlError::Restarted { now, .. } if now == "3f9a1c0b7d2e"),
        "{err} after {took:?}"
    );
    assert!(took < Duration::from_secs(12), "{took:?}");
    assert!(plane
        .lock()
        .unwrap()
        .log
        .contains(&"GET /v1/status".to_string()));

    let err = Ctl::new(dead_addr, TOKEN)
        .wait_generation_of("0ld0ld0ld0ld", 3, Duration::from_millis(50))
        .unwrap_err();
    assert!(matches!(err, CtlError::Io(_)), "{err}");
}

#[test]
fn argument_mistakes_fail_the_run_before_any_request() {
    let (ctl, plane) = plane();
    let exec = executor(ctl);
    for (src, why) in [
        ("a <- ix.ctl.wait(generation: 1)", "timeout_ms"),
        (
            r#"a <- ix.ctl.wait(service: "pod-a", nodes: 2, epoch: "x", timeout_ms: 10)"#,
            "epoch",
        ),
        (
            "a <- ix.ctl.wait(generation: 1, epoch: 7, timeout_ms: 10)",
            "string",
        ),
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
        ("a <- ix.ctl.apply({ services: {} }, peer: 3)", "string"),
        ("a <- ix.ctl.apply({ services: {} }, to: \"b\")", "to"),
        ("a <- { services: {} } → ix.ctl.status", "piped"),
        (r#"a <- ix.ctl.status("pod-a")"#, "named"),
        ("a <- ix.ctl.status(peer: 1)", "string"),
        (
            r#"a <- ix.ctl.wait(generation: 1, peer: 2, timeout_ms: 10)"#,
            "string",
        ),
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

/// F2.1 of the one-way-doors doc. Generation 1 failed, generation 2 converged: the old summary
/// (`observed_generation` 2, `diverged` 1) reads 1 as done, while `settled_generation` and
/// `failed` name it failed. And a generation that converged ahead of a pending one does not
/// settle it.
#[test]
fn a_later_generation_does_not_hide_a_failed_one() {
    let (ctl, plane) = plane();
    let launch = json!({"generation": 1, "service": "pod-a", "reason": "launch: not found"});
    {
        let mut p = plane.lock().unwrap();
        p.generation = 2;
        p.observed = 2;
        p.diverged = Some(launch.clone());
        p.settled = Some((2, vec![launch.clone()]));
    }
    let began = Instant::now();
    let err = ctl.wait_generation(1, Duration::from_secs(5)).unwrap_err();
    assert!(
        began.elapsed() < Duration::from_secs(1),
        "{:?}",
        began.elapsed()
    );
    assert_eq!(
        "generation 1 diverged on pod-a: launch: not found",
        err.to_string()
    );
    let done = ctl.wait_generation(2, Duration::from_secs(5)).unwrap();
    assert_eq!(json!(2), done["settled_generation"]);
    let err = executor(ctl.clone())
        .run_source("a <- ix.ctl.wait(generation: 1, timeout_ms: 5000)")
        .unwrap_err();
    assert!(err.to_string().contains("generation 1 diverged"), "{err}");

    // Generation 2 converged while 1 is still pending: nothing is settled yet, so neither
    // wait is done, where `observed_generation` 2 alone would have called both done.
    {
        let mut p = plane.lock().unwrap();
        p.diverged = None;
        p.settled = Some((0, vec![]));
    }
    for g in [1, 2] {
        let err = ctl
            .wait_generation(g, Duration::from_millis(60))
            .unwrap_err();
        assert!(matches!(err, CtlError::Wait(_)), "generation {g}: {err:?}");
    }
}

#[test]
fn the_token_goes_to_loopback_only_and_never_into_debug_output() {
    let ctl = Ctl::new("192.0.2.1:80", TOKEN);
    assert!(matches!(ctl.status(), Err(CtlError::NotLoopback(_))));
    assert!(!format!("{ctl:?}").contains(TOKEN));
}
