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
const JOB: &str = include_str!("../examples/job.ixql");
const PROBE: &str = include_str!("../examples/probe.ixql");
const TOKEN: &str = "5f1c0de2a9b84e7d93c6f0a1b2c3d4e5f60718293a4b5c6d7e8f9011223344aa";
/// How long the fake takes to observe an applied generation.
const CONVERGE: Duration = Duration::from_millis(120);
/// How long a started pod takes to report its nodes.
const NODES: Duration = Duration::from_millis(200);
/// How long the fake's supervisor takes to launch a service after the apply that asked.
const LAUNCH: Duration = Duration::from_millis(80);

#[derive(Default)]
struct Svc {
    desired: &'static str,
    since: Option<Instant>,
    last_exit: Value,
    /// Gave up restarting: state `failed`, whatever `since` says.
    failed: bool,
    /// The generation whose apply launched the current or last run.
    start_gen: u64,
    /// An apply asked for a start the supervisor has not made yet: until then the status still
    /// shows the service stopped, with the earlier run's exit, as the real plane's does.
    launch_at: Option<Instant>,
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
    /// How long every answer is held back after the request is read: a plane that accepts
    /// and stalls.
    stall: Duration,
    /// What `GET /v1/events` streams: `pings` keep-alive comments `ping_every` apart, the
    /// envelopes after `since` (none without a `since`: the real plane starts live), then `gap`
    /// if set, then the end of the stream.
    events: Vec<Value>,
    gap: Option<Value>,
    pings: u32,
    ping_every: Duration,
    /// When set, every byte of a frame is written on its own, this far apart.
    trickle: Option<Duration>,
    /// While set, no pending launch is made, however long ago the apply asked: a test holds
    /// the window between an apply and its launch open for as long as it needs it.
    hold_launch: bool,
    /// The probe tree, by node name, as `probe.create` declared it and the ops moved it.
    probes: BTreeMap<String, Node>,
    /// What a live read (no `since`) is sent: the envelopes after this sequence number. `None`
    /// sends nothing, as a plane whose channels stay quiet; the real plane sends what is
    /// published after the stream opens.
    live_from: Option<u64>,
}

/// One node of the fake's probe tree: the spec as declared, and the state the ops set.
struct Node {
    spec: Value,
    state: &'static str,
}

/// The kinds the fake has "compiled in", as `kinds_reply` lists them.
const KINDS: [&str; 4] = ["cpu@1", "heartbeat@1", "mem@1", "mux@1"];

impl Plane {
    fn status(&mut self) -> Value {
        if let Some((g, at)) = self.pending {
            if at.elapsed() >= CONVERGE {
                self.observed = g;
                self.pending = None;
            }
        }
        for s in self.services.values_mut() {
            if let Some(at) = s
                .launch_at
                .filter(|at| !self.hold_launch && Instant::now() >= *at)
            {
                s.since = Some(at);
                s.launch_at = None;
            }
        }
        let services: Vec<Value> = self
            .services
            .iter()
            .map(|(name, s)| {
                let (state, nodes) = match s.since {
                    _ if s.failed => ("failed", vec![]),
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
            let svc = &self.services[name];
            let running = svc.since.is_some() || svc.launch_at.is_some();
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
        s.launch_at = None;
        s.last_exit = json!({"code": 0, "at": unix_ms(), "generation": s.start_gen});
        s.last_exit.clone()
    }

    /// The service's process ended on its own with `code`, nobody having asked: the way a job
    /// ends. The plane records the exit, stamped with the generation that launched the run,
    /// and, with `restart: never`, leaves `desired` as it was.
    fn exit(&mut self, name: &str, code: i64) {
        let s = self.services.get_mut(name).expect("declared");
        s.since = None;
        s.last_exit = json!({"code": code, "at": unix_ms(), "generation": s.start_gen});
    }

    /// The plane killed the run for a limit of the spec: the exit names it (`reason`).
    fn kill_for(&mut self, name: &str, code: i64, reason: &str) {
        let s = self.services.get_mut(name).expect("declared");
        s.since = None;
        s.last_exit =
            json!({"code": code, "at": unix_ms(), "generation": s.start_gen, "reason": reason});
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
                            s.launch_at = Some(Instant::now() + LAUNCH);
                            s.start_gen = self.generation;
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
            ("GET", "/v1/probes") => (200, kinds_reply()),
            ("GET", "/v1/probe") => {
                let nodes: Vec<Value> = self
                    .probes
                    .iter()
                    .map(|(name, n)| {
                        let children = self
                            .probes
                            .values()
                            .filter(|c| c.spec["parent"] == *name)
                            .count();
                        let ttl = n.spec.get("ttl_ms").cloned().unwrap_or(json!(0));
                        let mut entry = json!({
                            "name": name, "kind": n.spec["kind"], "state": n.state,
                            "every_ms": n.spec.get("every_ms").cloned().unwrap_or(json!(0)),
                            "ttl_ms": ttl.clone(), "seq": 0, "errors": 0, "children": children,
                        });
                        if n.state == "running" && ttl != 0 {
                            entry["expires_in_ms"] = ttl;
                        }
                        for f in ["parent", "filter"] {
                            if let Some(v) = n.spec.get(f) {
                                entry[f] = v.clone();
                            }
                        }
                        entry
                    })
                    .collect();
                (
                    200,
                    json!({"nodes": nodes, "count": nodes.len(), "max_nodes": 256}),
                )
            }
            ("POST", "/v1/probe/create") => match self.probe_create(&doc) {
                Ok(reply) => (200, reply),
                Err(e) => (400, json!({ "error": e })),
            },
            (
                "POST",
                "/v1/probe/start" | "/v1/probe/stop" | "/v1/probe/touch" | "/v1/probe/remove",
            ) => {
                if doc.get("peer").is_some() {
                    return (400, json!({"error": "this plane has no peers"}));
                }
                let op = path.rsplit('/').next().unwrap();
                match self.probe_op(op, doc["node"].as_str().unwrap_or("")) {
                    Ok(reply) => (200, reply),
                    Err(e) => (400, json!({ "error": e })),
                }
            }
            _ => (404, json!({"error": format!("no {method} {path}")})),
        }
    }

    /// The node ops as the real plane answers them: `start` runs the node (a lease starts
    /// counting), `stop` leaves it stopped, `touch` renews a lease and refuses a node without
    /// one, `remove` takes the subtree and counts it. A name nobody declared is refused.
    fn probe_op(&mut self, op: &str, name: &str) -> Result<Value, String> {
        let (ttl, state) = {
            let n = self
                .probes
                .get_mut(name)
                .ok_or_else(|| format!("node {name:?} does not exist"))?;
            let ttl = n.spec["ttl_ms"].as_u64();
            match op {
                "start" => n.state = "running",
                "stop" => n.state = "stopped",
                "touch" if ttl.is_none() => return Err(format!("node {name:?} has no lease")),
                _ => {}
            }
            (ttl, n.state)
        };
        Ok(match op {
            "start" => {
                let mut reply = json!({"node": name, "state": "running"});
                if let Some(ttl) = ttl {
                    reply["expires_in_ms"] = json!(ttl);
                }
                reply
            }
            "stop" => json!({"node": name, "state": "stopped"}),
            "touch" => {
                let mut reply = json!({"node": name, "state": state, "ttl_ms": ttl});
                if state == "running" {
                    reply["expires_in_ms"] = json!(ttl);
                }
                reply
            }
            _ => {
                let mut gone = vec![name.to_string()];
                let mut i = 0;
                while i < gone.len() {
                    let parent = gone[i].clone();
                    gone.extend(
                        self.probes
                            .iter()
                            .filter(|(_, n)| n.spec["parent"] == parent)
                            .map(|(k, _)| k.clone()),
                    );
                    i += 1;
                }
                for g in &gone {
                    self.probes.remove(g);
                }
                json!({"node": name, "removed": gone.len()})
            }
        })
    }

    /// `probe.create` as the real plane checks it: a known kind, a fresh name, a mux as parent,
    /// an interval within the limits for a leaf and none for a mux; every refusal names what it
    /// hit. The node is created stopped; the reply gives its `depth`, the muxes above it.
    fn probe_create(&mut self, spec: &Value) -> Result<Value, String> {
        let name = spec["name"].as_str().filter(|n| !n.is_empty());
        let Some(name) = name else {
            return Err("a node needs a `name`".into());
        };
        let kind = spec["kind"].as_str().unwrap_or("");
        if !KINDS.contains(&kind) {
            return Err(format!("unknown kind {kind:?} (probe.kinds lists them)"));
        }
        if self.probes.contains_key(name) {
            return Err(format!("node {name:?} exists"));
        }
        let every = spec["every_ms"].as_u64();
        match (kind, every) {
            ("mux@1", Some(_)) => return Err("a mux@1 takes no `every_ms`".into()),
            ("mux@1", None) => {}
            (_, None) => return Err(format!("{kind} needs `every_ms`")),
            (_, Some(e)) if e < 10 => return Err(format!("every_ms {e} is under min_every_ms 10")),
            _ => {}
        }
        let parent = spec["parent"].as_str();
        if let Some(p) = parent {
            match self.probes.get(p) {
                Some(n) if n.spec["kind"] == "mux@1" => {}
                Some(_) => return Err(format!("parent {p:?} is not a mux@1")),
                None => return Err(format!("parent {p:?} does not exist")),
            }
        }
        let mut depth = 0;
        let mut up = parent;
        while let Some(p) = up {
            depth += 1;
            up = self.probes.get(p).and_then(|n| n.spec["parent"].as_str());
        }
        let reply = json!({"node": name, "kind": kind, "state": "stopped", "depth": depth});
        self.probes.insert(
            name.to_string(),
            Node {
                spec: spec.clone(),
                state: "stopped",
            },
        );
        Ok(reply)
    }
}

/// What `probe.kinds` answers on the real plane: the compiled kinds sorted by name, the limits
/// by the names the contract gives them, and the producer's revision.
fn kinds_reply() -> Value {
    json!({
        "kinds": [
            {"kind": "cpu@1", "version": 1, "fields": ["busy_pct", "cores"],
             "doc": "the host's processor busy share over the last interval"},
            {"kind": "heartbeat@1", "version": 1, "fields": ["n"], "doc": "its own tick count"},
            {"kind": "mem@1", "version": 1, "fields": ["total_bytes", "avail_bytes", "used_pct"],
             "doc": "the host's physical memory"},
            {"kind": "mux@1", "version": 1, "fields": [], "doc": "carries what its children pass it"},
        ],
        "limits": {"max_nodes": 256, "max_depth": 8, "min_every_ms": 10, "max_every_ms": 3_600_000,
                   "max_ttl_ms": 9_223_372_036_854u64, "ring": 4096, "max_where": 16, "max_keep": 32},
        "version": "fake-1",
    })
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
        stall: Duration::ZERO,
        events: Vec::new(),
        gap: None,
        pings: 1,
        ping_every: Duration::ZERO,
        trickle: None,
        hold_launch: false,
        probes: BTreeMap::new(),
        live_from: None,
    }));
    let shared = plane.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = answer(stream, &shared);
        }
    });
    (Ctl::new(format!("127.0.0.1:{port}"), TOKEN), plane)
}

/// A query value as Go's `URL.Query()` reads it: `+` is a space, `%XX` a byte, and a
/// malformed escape makes the pair worthless (`None`: Go drops it).
fn unpercent(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' => {
                let hex = b
                    .get(i + 1..i + 3)
                    .and_then(|h| std::str::from_utf8(h).ok())?;
                out.push(u8::from_str_radix(hex, 16).ok()?);
                i += 2;
            }
            c => out.push(c),
        }
        i += 1;
    }
    Some(String::from_utf8_lossy(&out).into_owned())
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
    } else if method == "GET" && path.starts_with("/v1/events") {
        // The event stream, as Go's server sends it: chunked, a comment first, one SSE event
        // per envelope under the `ch` prefix and after `since`, then the gap if any, then the end.
        plane.log.push(format!("{method} {path}"));
        let query = path.split_once('?').map_or("", |(_, q)| q);
        let arg = |k: &str| {
            query
                .split('&')
                .find_map(|kv| kv.strip_prefix(&format!("{k}=")).and_then(unpercent))
        };
        let prefix = arg("ch").unwrap_or_default();
        // Absent `since` is live: nothing of the past is sent, as the real plane does, only
        // what the fake is told comes after the open (`live_from`).
        let since = arg("since")
            .and_then(|s| s.parse::<u64>().ok())
            .or(plane.live_from);
        let (pings, ping_every, trickle, stall) =
            (plane.pings, plane.ping_every, plane.trickle, plane.stall);
        let mut frames = Vec::new();
        for e in plane.events.iter().filter(|e| {
            since.is_some_and(|since| {
                e["ch"].as_str().unwrap_or("").starts_with(&prefix)
                    && e["seq"].as_u64().unwrap_or(0) > since
            })
        }) {
            frames.push(format!(
                "id: {}:{}\nevent: {}\ndata: {e}\n\n",
                e["epoch"].as_str().unwrap_or(""),
                e["seq"],
                e["ev"]["kind"].as_str().unwrap_or("")
            ));
        }
        if let Some(g) = &plane.gap {
            frames.push(format!("event: gap\ndata: {g}\n\n"));
        }
        drop(plane);
        let mut w = stream;
        // A stalling plane accepts and withholds its headers.
        std::thread::sleep(stall);
        write!(
            w,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
        )?;
        for _ in 0..pings {
            std::thread::sleep(ping_every);
            w.write_all(b"8\r\n: ping\n\n\r\n")?;
            w.flush()?;
        }
        for frame in frames {
            let wire = format!("{:x}\r\n{frame}\r\n", frame.len());
            match trickle {
                Some(every) => {
                    for b in wire.as_bytes() {
                        w.write_all(std::slice::from_ref(b))?;
                        w.flush()?;
                        std::thread::sleep(every);
                    }
                }
                None => {
                    w.write_all(wire.as_bytes())?;
                    w.flush()?;
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
        w.write_all(b"0\r\n\r\n")?;
        return w.flush();
    } else {
        plane.log.push(format!("{method} {path}"));
        plane.route(&method, &path, &body)
    };
    let stall = plane.stall;
    drop(plane);
    if !stall.is_zero() {
        std::thread::sleep(stall);
    }

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

// D7 (ix#405), S4: one IXQL capability per probe op, named after the wire's. `probe.kinds` is
// the first: a program refuses early when the kind it wants is not compiled in, and reads the
// limits it will be held to instead of guessing them.
#[test]
fn probe_kinds_lists_the_kinds_and_the_limits() {
    let (ctl, plane) = plane();
    let out = executor(ctl.clone())
        .run_source("kinds <- ix.ctl.probe.kinds()")
        .unwrap();
    let kinds = out.binding("kinds").unwrap();
    assert_eq!(kinds_reply(), *kinds);
    let names: Vec<&str> = kinds["kinds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["kind"].as_str().unwrap())
        .collect();
    assert_eq!(["cpu@1", "heartbeat@1", "mem@1", "mux@1"], names[..]);
    assert_eq!(json!(256), kinds["limits"]["max_nodes"]);

    // The client method is the same call; a peer rides beside it and this plane has none.
    assert_eq!(kinds_reply(), ctl.probe_kinds(None).unwrap());
    match ctl.probe_kinds(Some("b")) {
        Err(CtlError::Http {
            status: 400,
            message,
        }) => assert!(message.contains("no peers"), "{message}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        ["GET /v1/probes", "GET /v1/probes", "GET /v1/probes?peer=b"],
        plane.lock().unwrap().log[..]
    );
}

// D7, S4: `probe.create` takes the node record under `probe:`, as the wire does, and answers
// the node stopped; a refusal names what it hit and the run ends there, nothing half-made.
#[test]
fn probe_create_declares_a_node_under_a_mux_and_refusals_name_their_cause() {
    let (ctl, plane) = plane();
    let out = executor(ctl.clone())
        .run_source(
            r#"agg <- ix.ctl.probe.create(probe: { name: "agg", kind: "mux@1" })
               cpu <- ix.ctl.probe.create(probe: {
                   name: "cpu", kind: "cpu@1", every_ms: 100, parent: "agg",
                   filter: { keep: ["busy_pct"] }
                 })"#,
        )
        .unwrap();
    assert_eq!(
        json!({"node": "agg", "kind": "mux@1", "state": "stopped", "depth": 0}),
        *out.binding("agg").unwrap()
    );
    assert_eq!(
        json!({"node": "cpu", "kind": "cpu@1", "state": "stopped", "depth": 1}),
        *out.binding("cpu").unwrap()
    );
    {
        let plane = plane.lock().unwrap();
        assert_eq!(
            json!({"keep": ["busy_pct"]}),
            plane.probes["cpu"].spec["filter"],
            "the record reaches the plane as written"
        );
        assert_eq!(
            ["POST /v1/probe/create", "POST /v1/probe/create"],
            plane.log[..]
        );
    }

    for (src, why) in [
        (
            r#"x <- ix.ctl.probe.create(probe: { name: "x", kind: "gpu@1", every_ms: 100 })"#,
            "unknown kind",
        ),
        (
            r#"x <- ix.ctl.probe.create(probe: { name: "x", kind: "cpu@1", every_ms: 5 })"#,
            "min_every_ms",
        ),
        (
            r#"x <- ix.ctl.probe.create(probe: { name: "x", kind: "cpu@1", every_ms: 100, parent: "cpu" })"#,
            "not a mux",
        ),
        (
            r#"x <- ix.ctl.probe.create(probe: { name: "agg", kind: "mux@1" })"#,
            "exists",
        ),
    ] {
        let err = executor(ctl.clone())
            .run_source(src)
            .unwrap_err()
            .to_string();
        assert!(err.contains(why), "{src}: {err}");
    }
    assert_eq!(2, plane.lock().unwrap().probes.len(), "nothing half-made");

    // A peer rides beside the record, as `?peer=`; this plane has none.
    match ctl.probe_create(&json!({"name": "p", "kind": "mux@1"}), Some("b")) {
        Err(CtlError::Http {
            status: 400,
            message,
        }) => assert!(message.contains("no peers"), "{message}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        Some(&"POST /v1/probe/create?peer=b".to_string()),
        plane.lock().unwrap().log.last()
    );
}

// D7, S4: the node ops take the name under `node:`, as the wire does. A mux starts before its
// leaf, a lease is renewed by `touch` and refused without one, a remove takes the subtree and
// says how many went, and a name nobody declared is refused by the plane, not guessed at.
#[test]
fn probe_node_ops_start_stop_touch_and_remove_the_tree() {
    let (ctl, plane) = plane();
    let out = executor(ctl.clone())
        .run_source(
            r#"agg <- ix.ctl.probe.create(probe: { name: "agg", kind: "mux@1", ttl_ms: 5000 })
               cpu <- ix.ctl.probe.create(probe: { name: "cpu", kind: "cpu@1", every_ms: 100, parent: "agg" })
               agg_on <- ix.ctl.probe.start(node: "agg")
               cpu_on <- ix.ctl.probe.start(node: "cpu")
               renewed <- ix.ctl.probe.touch(node: "agg")
               cpu_off <- ix.ctl.probe.stop(node: "cpu")
               gone <- ix.ctl.probe.remove(node: "agg")"#,
        )
        .unwrap();
    let b = |name: &str| out.binding(name).unwrap_or_else(|| panic!("no `{name}`"));
    assert_eq!(
        json!({"node": "agg", "state": "running", "expires_in_ms": 5000}),
        *b("agg_on")
    );
    assert_eq!(json!({"node": "cpu", "state": "running"}), *b("cpu_on"));
    assert_eq!(
        json!({"node": "agg", "state": "running", "ttl_ms": 5000, "expires_in_ms": 5000}),
        *b("renewed")
    );
    assert_eq!(json!({"node": "cpu", "state": "stopped"}), *b("cpu_off"));
    assert_eq!(json!({"node": "agg", "removed": 2}), *b("gone"));
    {
        let plane = plane.lock().unwrap();
        assert!(plane.probes.is_empty(), "the subtree went with the mux");
        assert_eq!(
            [
                "POST /v1/probe/create",
                "POST /v1/probe/create",
                "POST /v1/probe/start",
                "POST /v1/probe/start",
                "POST /v1/probe/touch",
                "POST /v1/probe/stop",
                "POST /v1/probe/remove",
            ],
            plane.log[..]
        );
    }

    for (src, why) in [
        (
            r#"x <- ix.ctl.probe.start(node: "ghost")"#,
            "does not exist",
        ),
        (
            r#"x <- ix.ctl.probe.create(probe: { name: "hb", kind: "heartbeat@1", every_ms: 20 })
               t <- ix.ctl.probe.touch(node: "hb")"#,
            "no lease",
        ),
    ] {
        let err = executor(ctl.clone())
            .run_source(src)
            .unwrap_err()
            .to_string();
        assert!(err.contains(why), "{src}: {err}");
    }

    // The client methods are the same calls; a peer rides in the body and this plane has none.
    assert_eq!(
        json!({"node": "hb", "removed": 1}),
        ctl.probe_remove("hb", None).unwrap()
    );
    match ctl.probe_start("hb", Some("b")) {
        Err(CtlError::Http {
            status: 400,
            message,
        }) => assert!(message.contains("no peers"), "{message}"),
        other => panic!("{other:?}"),
    }
}

// D7, S4: `probe.list` is the tree as the plane holds it, with the usage against its bound.
#[test]
fn probe_list_shows_the_tree_and_its_usage() {
    let (ctl, plane) = plane();
    let out = executor(ctl.clone())
        .run_source(
            r#"agg <- ix.ctl.probe.create(probe: { name: "agg", kind: "mux@1" })
               cpu <- ix.ctl.probe.create(probe: { name: "cpu", kind: "cpu@1", every_ms: 100, parent: "agg" })
               on <- ix.ctl.probe.start(node: "agg")
               tree <- ix.ctl.probe.list()"#,
        )
        .unwrap();
    let tree = out.binding("tree").unwrap();
    assert_eq!(json!(2), tree["count"]);
    assert_eq!(json!(256), tree["max_nodes"]);
    let nodes = tree["nodes"].as_array().unwrap();
    assert_eq!(
        json!({"name": "agg", "kind": "mux@1", "state": "running", "every_ms": 0, "ttl_ms": 0, "seq": 0, "errors": 0,
               "children": 1}),
        nodes[0]
    );
    assert_eq!(json!("stopped"), nodes[1]["state"]);
    assert_eq!(json!("agg"), nodes[1]["parent"]);
    assert_eq!(json!(100), nodes[1]["every_ms"]);
    assert_eq!(json!(0), nodes[1]["ttl_ms"]);
    assert_eq!(json!(0), nodes[1]["children"]);

    assert_eq!(*tree, ctl.probe_list(None).unwrap());
    match ctl.probe_list(Some("b")) {
        Err(CtlError::Http {
            status: 400,
            message,
        }) => assert!(message.contains("no peers"), "{message}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        Some(&"GET /v1/probe?peer=b".to_string()),
        plane.lock().unwrap().log.last()
    );
}

// D7, S4: the example program end to end against the fake. The ring holds twelve of the mux's
// records (one of them a read error, which reaches the pipe whatever the filter) and a leaf's
// own record on another pipe: the read takes ten from `probe/agg` alone, in seq order, and says
// it stopped at `max:`; the remove takes the tree. Run again over a thinner ring, the read is
// shorter than asked and says so by its length: a partial read is never dressed as a full one.
#[test]
fn probe_ixql_samples_a_cpu_probe_under_a_mux_and_removes_the_tree() {
    let (ctl, plane) = plane();
    let env = |seq: u64, ch: &str, data: Value| {
        json!({"ch": ch, "seq": seq, "epoch": "3f9a1c0b7d2e", "origin": "fake", "t": 1_760_000_000_000u64 + seq,
               "ev": {"kind": "sample", "t": 1_760_000_000_000u64 + seq, "data": data}})
    };
    let agg = |seq: u64, n: u64, busy: f64| {
        env(
            seq,
            "probe/agg",
            json!({"node": "agg", "kind": "mux@1", "seq": n, "src": "cpu", "busy_pct": busy}),
        )
    };
    let mut ring = vec![env(
        1,
        "probe/cpu",
        json!({"node": "cpu", "kind": "cpu@1", "seq": 1, "busy_pct": 3.0}),
    )];
    for n in 1..=12u64 {
        ring.push(if n == 4 {
            env(n + 1, "probe/agg", json!({"node": "agg", "kind": "mux@1", "seq": n, "src": "cpu", "error": "GetSystemTimes: boom"}))
        } else {
            agg(n + 1, n, n as f64)
        });
    }
    {
        let mut p = plane.lock().unwrap();
        p.events = ring.clone();
        // The program reads live; the fake hands a live read these records as "what came after
        // the open".
        p.live_from = Some(0);
    }

    let out = executor(ctl.clone())
        .run_source(PROBE)
        .expect("probe.ixql runs");
    let b = |name: &str| out.binding(name).unwrap_or_else(|| panic!("no `{name}`"));
    assert_eq!(kinds_reply(), *b("kinds"));
    assert_eq!(
        json!({"node": "agg", "state": "running", "expires_in_ms": 30000}),
        *b("agg_on"),
        "the lease is the cleanup: the mux carries one"
    );
    assert_eq!(json!("running"), b("cpu_on")["state"]);
    let samples = b("samples");
    let events = samples["events"].as_array().unwrap();
    assert_eq!(10, events.len(), "{samples}");
    assert_eq!(
        json!(false),
        samples["ended"],
        "stopped at `max:`, not at the stream's end"
    );
    assert_eq!(Value::Null, samples["gap"]);
    // The client flattens an envelope to `{channel, seq, epoch, origin, kind, t, data}`.
    let seqs: Vec<u64> = events
        .iter()
        .map(|e| e["data"]["seq"].as_u64().unwrap())
        .collect();
    assert_eq!(
        (1..=10).collect::<Vec<u64>>(),
        seqs,
        "the mux's records, in order"
    );
    assert!(events.iter().all(|e| e["channel"] == "probe/agg"
        && e["kind"] == "sample"
        && e["data"]["src"] == "cpu"));
    assert_eq!(
        json!("GetSystemTimes: boom"),
        events[3]["data"]["error"],
        "a read error is a record"
    );
    assert_eq!(json!({"node": "agg", "removed": 2}), *b("gone"));
    {
        let plane = plane.lock().unwrap();
        assert!(plane.probes.is_empty(), "the tree is gone");
        assert!(
            plane
                .log
                .iter()
                .any(|r| r == "GET /v1/events?ch=probe%2Fagg"),
            "a live read, no `since`: {:?}",
            plane.log
        );
    }

    // A thinner ring: four records, then the stream ends. The read answers four, not ten.
    plane.lock().unwrap().events = ring[..5].to_vec();
    let out = executor(ctl)
        .run_source(PROBE)
        .expect("probe.ixql runs again");
    let samples = out.binding("samples").unwrap();
    assert_eq!(4, samples["events"].as_array().unwrap().len(), "{samples}");
    assert_eq!(
        json!({"node": "agg", "removed": 2}),
        *out.binding("gone").unwrap()
    );
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

// The plane's event stream is read as it comes: chunks undone, keep-alive comments skipped,
// only the channels under the prefix and after `since`, a gap as an error item, and the end of
// the stream as the end of the iterator. A wrong token is the usual 401.
#[test]
fn the_event_stream_is_read_as_it_comes() {
    let (ctl, plane) = plane();
    let env = |seq: u64, ch: &str, kind: &str, data: Value| {
        json!({"ch": ch, "seq": seq, "epoch": "3f9a1c0b7d2e", "origin": "fake", "t": 1_760_000_000_000u64 + seq,
               "ev": {"kind": kind, "t": 1_760_000_000_000u64 + seq, "data": data}})
    };
    {
        let mut p = plane.lock().unwrap();
        p.events = vec![
            env(1, "ctl", "boot", json!({})),
            env(2, "svc/pod-a", "service.started", json!({"pid": 7})),
            env(3, "ctl", "converged", json!({"generation": 1})),
            env(4, "ctl", "converged", json!({"generation": 2})),
        ];
        p.gap = Some(json!({"reason": "overflow", "from": 5, "epoch": "3f9a1c0b7d2e"}));
    }
    let items: Vec<_> = ctl.events("ctl", Some(1), None).unwrap().collect();
    assert_eq!(3, items.len(), "{items:?}");
    let got: Vec<(u64, &str)> = items[..2]
        .iter()
        .map(|i| i.as_ref().unwrap())
        .map(|e| (e.seq, e.kind.as_str()))
        .collect();
    assert_eq!(vec![(3, "converged"), (4, "converged")], got);
    let e = items[1].as_ref().unwrap();
    assert_eq!(
        ("ctl", "3f9a1c0b7d2e", "fake", Some(1_760_000_000_004i64)),
        (e.channel.as_str(), e.epoch.as_str(), e.origin.as_str(), e.t)
    );
    assert_eq!(json!({"generation": 2}), e.data);
    assert!(
        matches!(&items[2], Err(CtlError::Gap { reason, from: 5, .. }) if reason == "overflow"),
        "{:?}",
        items[2]
    );
    // Every channel, from the start.
    let all: Vec<_> = ctl.events("", Some(0), None).unwrap().collect();
    assert_eq!(5, all.len());
    assert_eq!("svc/pod-a", all[1].as_ref().unwrap().channel);
    // A service's own event whose kind is `gap` is an event: the plane's gap is told apart by
    // its shape, not by the `event:` field. The channel prefix travels percent-encoded.
    plane
        .lock()
        .unwrap()
        .events
        .push(env(5, "svc/pod-a", "gap", json!({"lost": 2})));
    let svc: Vec<_> = ctl.events("svc/", Some(0), None).unwrap().collect();
    assert_eq!(3, svc.len(), "{svc:?}");
    let own = svc[1].as_ref().unwrap();
    assert_eq!(
        ("gap", 5, "svc/pod-a"),
        (own.kind.as_str(), own.seq, own.channel.as_str())
    );
    assert!(matches!(&svc[2], Err(CtlError::Gap { from: 5, .. })));
    // Without `since` the stream is live: nothing of the past comes, here only the gap.
    let live: Vec<_> = ctl.events("ctl", None, None).unwrap().collect();
    assert_eq!(1, live.len(), "{live:?}");
    assert!(matches!(&live[0], Err(CtlError::Gap { .. })));
    {
        let log = &plane.lock().unwrap().log;
        for path in [
            "GET /v1/events?ch=ctl&since=1",
            "GET /v1/events?ch=svc%2F&since=0",
            "GET /v1/events?ch=ctl",
        ] {
            assert!(log.contains(&path.to_string()), "{path} in {log:?}");
        }
    }
    // `epoch` qualifies `since`: alone it is refused before any request.
    let err = ctl.events("ctl", None, Some("3f9a1c0b7d2e")).err().unwrap();
    assert!(matches!(err, CtlError::Argument(_)), "{err}");
    let err = Ctl::new(
        ctl.addr(),
        "0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0ld0",
    )
    .events("ctl", None, None)
    .err()
    .expect("a wrong token is refused");
    assert!(matches!(err, CtlError::Http { status: 401, .. }), "{err}");

    // Through IXQL: `max:` stops the read early; the gap and the end are reported.
    let out = executor(ctl.clone())
        .run_source(
            "two <- ix.ctl.events(channel: \"ctl\", since: 1, max: 2, timeout_ms: 2000)\n\
             all <- ix.ctl.events(channel: \"ctl\", since: 1, max: 10, timeout_ms: 2000)\n\
             none <- ix.ctl.events(channel: \"ctl\", since: 1, max: 0, timeout_ms: 2000)\n",
        )
        .expect("the IXQL reads");
    let two = out.binding("two").unwrap();
    assert_eq!(2, two["events"].as_array().unwrap().len());
    assert_eq!(
        (json!(false), &Value::Null),
        (two["ended"].clone(), &two["gap"])
    );
    // `max: 0` asks for nothing and opens nothing: it cannot block.
    assert_eq!(
        &json!({"events": [], "gap": null, "ended": false}),
        out.binding("none").unwrap()
    );
    let opened = plane
        .lock()
        .unwrap()
        .log
        .iter()
        .filter(|l| l.starts_with("GET /v1/events?ch=ctl&since=1"))
        .count();
    assert_eq!(3, opened, "the direct read and the two IXQL reads, no more");
    let all = out.binding("all").unwrap();
    let seqs: Vec<Value> = all["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["seq"].clone())
        .collect();
    assert_eq!(vec![json!(3), json!(4)], seqs);
    assert_eq!(json!("overflow"), all["gap"]["reason"]);
    assert_eq!(json!(true), all["ended"]);
}

// `events_within` bounds the wait for one event, not one read: a plane that only ever sends
// keep-alive comments ends the read about on time instead of being kept alive by them.
#[test]
fn a_bounded_event_read_is_not_kept_alive_by_pings() {
    let (ctl, plane) = plane();
    {
        let mut p = plane.lock().unwrap();
        p.pings = 20;
        p.ping_every = Duration::from_millis(60);
    }
    let started = Instant::now();
    let items: Vec<_> = ctl
        .events_within("ctl", Some(0), None, Duration::from_millis(250))
        .unwrap()
        .collect();
    let took = started.elapsed();
    assert_eq!(1, items.len(), "{items:?}");
    assert!(
        matches!(&items[0], Err(CtlError::Io(e))
            if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock)),
        "{items:?}"
    );
    assert!(
        took >= Duration::from_millis(250) && took < Duration::from_millis(1000),
        "took {took:?}"
    );
}

// The bound holds under the readers too: a frame that arrives a byte at a time (every socket
// read gets bytes, none completes a line) ends the read at the deadline, not when the frame
// is whole; and a plane that accepts the connection and withholds its headers is bounded by
// the same deadline, not by the 40 s idle timeout.
#[test]
fn a_bounded_event_read_ends_mid_frame_and_mid_open_on_time() {
    let (ctl, plane) = plane();
    {
        let mut p = plane.lock().unwrap();
        p.pings = 0;
        p.trickle = Some(Duration::from_millis(20)); // a 300-byte frame takes 6 s
        p.events = vec![
            json!({"ch": "ctl", "seq": 1, "epoch": "e", "origin": "fake",
            "ev": {"kind": "converged", "data": {"note": "x".repeat(200)}}}),
        ];
    }
    let started = Instant::now();
    let items: Vec<_> = ctl
        .events_within("ctl", Some(0), None, Duration::from_millis(250))
        .unwrap()
        .collect();
    let took = started.elapsed();
    assert!(
        matches!(&items[..], [Err(CtlError::Io(_))]) && took < Duration::from_millis(1000),
        "{items:?} after {took:?}"
    );

    let (stalled, stalling) = self::plane();
    stalling.lock().unwrap().stall = Duration::from_secs(3);
    let started = Instant::now();
    let err = stalled
        .events_within("ctl", Some(0), None, Duration::from_millis(200))
        .err()
        .expect("the open is bounded");
    let took = started.elapsed();
    assert!(
        matches!(err, CtlError::Io(_)) && took < Duration::from_millis(1000),
        "{err} after {took:?}"
    );
}

// A wait's timeout bounds the request in flight too: against a plane that accepts and then
// stalls, a short wait ends about when it said it would, not after the client's 15 s read
// timeout, and a zero wait still asks once.
#[test]
fn a_wait_ends_on_time_against_a_plane_that_stalls() {
    let (ctl, plane) = plane();
    plane.lock().unwrap().stall = Duration::from_secs(5);
    for (timeout, kind) in [
        (Duration::from_millis(200), "generation"),
        (Duration::from_millis(200), "nodes"),
        (Duration::from_millis(200), "exit"),
        (Duration::ZERO, "generation"),
    ] {
        let began = Instant::now();
        let err = match kind {
            "generation" => ctl.wait_generation(1, timeout).unwrap_err(),
            "exit" => ctl.wait_exit("pod-a", timeout).unwrap_err(),
            _ => ctl.wait_nodes("pod-a", 1, timeout).unwrap_err(),
        };
        let took = began.elapsed();
        assert!(matches!(err, CtlError::Io(_)), "{kind}: {err}");
        assert!(
            took < timeout + Duration::from_millis(400),
            "{kind} with {timeout:?} took {took:?}"
        );
    }
    // The requests did reach the plane: the first is logged before its stall; the fake plane
    // serves one connection at a time, so the others are read, and logged, only once the earlier
    // stalls have been slept off.
    assert!(plane
        .lock()
        .unwrap()
        .log
        .contains(&"GET /v1/status".to_string()));
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
            r#"e <- ix.ctl.events(channel: "ctl", max: 1)"#,
            "timeout_ms",
        ),
        (
            r#"e <- ix.ctl.events(channel: "ctl", timeout_ms: 10)"#,
            "max",
        ),
        (
            r#"e <- ix.ctl.events(channel: "ctl", max: 1, since: "a", timeout_ms: 10)"#,
            "whole number",
        ),
        (
            r#"e <- ix.ctl.events(channel: "ctl", epoch: "x", max: 1, timeout_ms: 10)"#,
            "needs `since`",
        ),
        (
            r#"e <- ix.ctl.events(channel: "ctl", epoch: "x", max: 0, timeout_ms: 10)"#,
            "needs `since`",
        ),
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
            r#"a <- ix.ctl.wait(service: "pod-a", nodes: 2, exit: true, timeout_ms: 10)"#,
            "`exit: true`",
        ),
        (
            "a <- ix.ctl.wait(generation: 1, exit: true, timeout_ms: 10)",
            "`exit: true`",
        ),
        (
            r#"a <- ix.ctl.wait(service: "pod-a", exit: false, timeout_ms: 10)"#,
            "`true`",
        ),
        (
            r#"a <- ix.ctl.wait(service: "pod-a", exit: 1, timeout_ms: 10)"#,
            "boolean",
        ),
        (r#"a <- ix.ctl.wait(exit: true, timeout_ms: 10)"#, "service"),
        (
            r#"a <- ix.ctl.wait(service: "pod-a", exit: true, generation: "x", timeout_ms: 10)"#,
            "whole number",
        ),
        (
            r#"a <- ix.ctl.wait(service: "pod-a", exit: true, generation: 1, epoch: 7, timeout_ms: 10)"#,
            "string",
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
        (r#"k <- ix.ctl.probe.kinds(node: "hb")"#, "node"),
        ("k <- ix.ctl.probe.kinds(peer: 1)", "string"),
        ("k <- {} → ix.ctl.probe.kinds", "piped"),
        (r#"k <- ix.ctl.probe.kinds("b")"#, "named"),
        ("c <- ix.ctl.probe.create()", "probe"),
        (r#"c <- ix.ctl.probe.create(probe: "agg")"#, "record"),
        (
            r#"c <- ix.ctl.probe.create(probe: { name: "a", kind: "mux@1" }, node: "a")"#,
            "node",
        ),
        (
            r#"c <- { name: "a", kind: "mux@1" } → ix.ctl.probe.create"#,
            "piped",
        ),
        (
            r#"c <- ix.ctl.probe.create(probe: { name: "a", kind: "mux@1" }, peer: 1)"#,
            "string",
        ),
        ("s <- ix.ctl.probe.start()", "node"),
        ("s <- ix.ctl.probe.stop(node: 1)", "string"),
        (
            r#"s <- ix.ctl.probe.touch(node: "a", ttl_ms: 10)"#,
            "ttl_ms",
        ),
        (
            r#"s <- ix.ctl.probe.remove(node: "a", force: true)"#,
            "force",
        ),
        (r#"s <- "a" → ix.ctl.probe.remove"#, "piped"),
        (r#"s <- ix.ctl.probe.remove(node: "a", peer: 2)"#, "string"),
        (r#"l <- ix.ctl.probe.list(node: "a")"#, "node"),
        ("l <- {} → ix.ctl.probe.list", "piped"),
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

// A run the plane killed for a limit of the spec (`limits.wall_ms`) ends like any other, its
// exit naming the limit: the exit wait passes `last_exit.reason` through, so a program can tell
// a job that ran out of time from one that failed on its own.
#[test]
fn a_run_killed_at_its_wall_limit_says_so_in_its_exit() {
    let (ctl, plane) = plane();
    let applied = ctl
        .apply(&json!({"services": {"pod-a": {"state": "running"}}}), None)
        .unwrap();
    let g = applied["generation"].as_u64().unwrap();
    ctl.wait_generation(g, Duration::from_secs(2))
        .expect("the start converged");
    plane.lock().unwrap().kill_for("pod-a", 1, "wall_limit");
    let entry = ctl
        .wait_exit_of("pod-a", g, None, Duration::from_secs(2))
        .expect("the run ended");
    assert_eq!(json!("stopped"), entry["state"]);
    assert_eq!(json!("wall_limit"), entry["last_exit"]["reason"]);
    assert_eq!(json!(1), entry["last_exit"]["code"]);
    assert_eq!(json!(g), entry["last_exit"]["generation"]);
    // The IXQL form carries the reason too.
    let out = executor(ctl.clone())
        .run_source(r#"j <- ix.ctl.wait(service: "pod-a", exit: true, timeout_ms: 50)"#)
        .unwrap();
    assert_eq!(
        json!("wall_limit"),
        out.binding("j").unwrap()["last_exit"]["reason"]
    );
}

// A job ends on its own: the plane records its exit and, with `restart: never`, leaves it
// stopped with `desired` still `running`. The exit wait answers that entry; a service that gave
// up restarting (`failed`) answers too; one still running, or never run, runs the wait out.
#[test]
fn a_job_is_waited_to_its_exit() {
    let (ctl, plane) = plane();
    let applied = ctl
        .apply(&json!({"services": {"pod-a": {"state": "running"}}}), None)
        .unwrap();
    ctl.wait_generation(
        applied["generation"].as_u64().unwrap(),
        Duration::from_secs(2),
    )
    .expect("the start converged");
    // Still starting, or running: the wait runs out and says so.
    let err = ctl
        .wait_exit("pod-a", Duration::from_millis(60))
        .unwrap_err();
    let text = err.to_string();
    assert!(
        matches!(err, CtlError::Wait(_))
            && text.contains("`pod-a` has not exited: state `")
            && text.contains("after 60 ms")
            && !text.contains("stopped"),
        "{err}"
    );
    let ender = std::thread::spawn({
        let plane = plane.clone();
        move || {
            std::thread::sleep(Duration::from_millis(150));
            plane.lock().unwrap().exit("pod-a", 3);
        }
    });
    let began = Instant::now();
    let entry = ctl
        .wait_exit("pod-a", Duration::from_secs(2))
        .expect("the job exited");
    ender.join().unwrap();
    assert!(began.elapsed() >= Duration::from_millis(150));
    assert_eq!(json!("stopped"), entry["state"]);
    assert_eq!(
        json!("running"),
        entry["desired"],
        "nobody asked for the stop"
    );
    assert_eq!(json!(3), entry["last_exit"]["code"]);
    assert!(entry["pid"].is_null());
    // The exit stays in the status: a wait that comes later answers it at once. That is why
    // a job's exit is waited for after the generation that started it has converged.
    let again = ctl
        .wait_exit("pod-a", Duration::from_millis(50))
        .expect("the same exit, again");
    assert_eq!(entry["last_exit"], again["last_exit"]);
    // The IXQL form answers the same entry.
    let out = executor(ctl.clone())
        .run_source(r#"j <- ix.ctl.wait(service: "pod-a", exit: true, timeout_ms: 50)"#)
        .unwrap();
    assert_eq!(&entry, out.binding("j").unwrap());

    // Applied again, the job runs again under generation 2. Until the supervisor launches it
    // the status still shows it stopped with generation 1's exit: the unscoped wait answers that
    // exit at once and takes it for this run's; the wait scoped to generation 2 does not, and
    // answers once that run has ended. The fake holds the launch until the stale answer is in,
    // so the window is as long as this test needs, not a race against the clock.
    plane.lock().unwrap().hold_launch = true;
    let applied = ctl
        .apply(&json!({"services": {"pod-a": {"state": "running"}}}), None)
        .unwrap();
    assert_eq!(json!(2), applied["generation"]);
    let stale = ctl
        .wait_exit("pod-a", Duration::from_millis(20))
        .expect("the first run's exit, taken for this one");
    assert_eq!(json!(1), stale["last_exit"]["generation"]);
    assert_eq!(json!("stopped"), stale["state"]);
    plane.lock().unwrap().hold_launch = false;
    let err = ctl
        .wait_exit_of("pod-a", 2, Some("3f9a1c0b7d2e"), Duration::from_millis(150))
        .unwrap_err();
    let text = err.to_string();
    assert!(
        matches!(err, CtlError::Wait(_)) && text.contains("has not exited: state `")
            || text.contains("generation 1's run, not generation 2's"),
        "{err}"
    );
    let ender = std::thread::spawn({
        let plane = plane.clone();
        move || {
            std::thread::sleep(Duration::from_millis(150));
            plane.lock().unwrap().exit("pod-a", 0);
        }
    });
    let out = executor(ctl.clone())
        .run_source(
            "j <- ix.ctl.wait(service: \"pod-a\", exit: true, generation: 2, \
             epoch: \"3f9a1c0b7d2e\", timeout_ms: 2000)",
        )
        .expect("the second run's exit");
    ender.join().unwrap();
    let second = out.binding("j").unwrap();
    assert_eq!(json!(2), second["last_exit"]["generation"]);
    assert_eq!(json!(0), second["last_exit"]["code"]);
    // Scoped to a boot, the wait ends at once when the plane reports another one.
    plane.lock().unwrap().epoch = Some("0ther0ther00".into());
    let err = ctl
        .wait_exit_of("pod-a", 2, Some("3f9a1c0b7d2e"), Duration::from_secs(2))
        .unwrap_err();
    assert!(
        matches!(&err, CtlError::Restarted { generation: 2, now, .. } if now == "0ther0ther00"),
        "{err}"
    );
    // Without the epoch, or on a plane whose exits name no generation, the number alone counts.
    let again = ctl
        .wait_exit_of("pod-a", 2, None, Duration::from_millis(50))
        .unwrap();
    assert_eq!(second["last_exit"], again["last_exit"]);
    plane
        .lock()
        .unwrap()
        .services
        .get_mut("pod-a")
        .unwrap()
        .last_exit = json!({"code": 0, "at": 1});
    ctl.wait_exit_of("pod-a", 9, None, Duration::from_millis(50))
        .expect("an exit without a generation is taken as it is");
    plane.lock().unwrap().epoch = Some("3f9a1c0b7d2e".into());

    // A service the plane gave up on has exited for good.
    {
        let mut p = plane.lock().unwrap();
        p.exit("pod-b", 1);
        p.services.get_mut("pod-b").unwrap().failed = true;
    }
    let failed = ctl
        .wait_exit("pod-b", Duration::from_millis(50))
        .expect("failed is an exit");
    assert_eq!(json!("failed"), failed["state"]);
    assert_eq!(json!(1), failed["last_exit"]["code"]);

    // One that never ran has no exit to answer; one the plane does not declare is refused.
    let (fresh, _plane) = self::plane();
    let err = fresh
        .wait_exit("pod-b", Duration::from_millis(50))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("`pod-b` has not exited: state `stopped` after 50 ms"),
        "{err}"
    );
    let err = fresh
        .wait_exit("ghost", Duration::from_millis(50))
        .unwrap_err();
    assert!(
        matches!(err, CtlError::NotDeclared(ref n) if n == "ghost"),
        "{err:?}"
    );
}

// job.ixql against the fake: the job is launched, converges, exits while the program waits for
// it, and its output is asked for from the start of the boot.
#[test]
fn job_ixql_runs_a_job_to_its_end() {
    let (ctl, plane) = plane();
    plane.lock().unwrap().services.insert(
        "job".into(),
        Svc {
            desired: "stopped",
            last_exit: Value::Null,
            ..Svc::default()
        },
    );
    let ender = std::thread::spawn({
        let plane = plane.clone();
        move || {
            std::thread::sleep(CONVERGE + Duration::from_millis(200));
            plane.lock().unwrap().exit("job", 0);
        }
    });
    let out = executor(ctl).run_source(JOB).expect("job.ixql runs");
    ender.join().unwrap();
    let b = |name: &str| out.binding(name).unwrap_or_else(|| panic!("no `{name}`"));
    assert_eq!(
        json!([{"op": "start", "service": "job"}]),
        b("applied")["actions"]
    );
    assert_eq!(
        b("applied")["generation"],
        b("started")["observed_generation"]
    );
    assert_eq!(json!("stopped"), b("exited")["state"]);
    assert_eq!(json!("running"), b("exited")["desired"]);
    assert_eq!(json!(0), b("exited")["last_exit"]["code"]);
    // The fake has no ring: the replay is empty and the stream ends.
    assert_eq!(
        &json!({"events": [], "gap": null, "ended": true}),
        b("output")
    );
    let log = plane.lock().unwrap().log.clone();
    assert!(
        log.iter()
            .any(|r| r.starts_with("GET /v1/events?ch=svc%2Fjob&since=0&epoch=3f9a1c0b7d2e")),
        "{log:?}"
    );
}

// A job on a peer is waited for through this plane, and the wait rides out a blinking link.
#[test]
fn a_peers_job_is_waited_to_its_exit() {
    let (ctl, plane) = plane();
    {
        let mut p = plane.lock().unwrap();
        p.peer_status = Some(json!({
            "name": "b", "epoch": "b00b00b00b00", "generation": 1, "observed_generation": 1,
            "settled_generation": 1, "failed": [], "services": [
                {"name": "job", "desired": "running", "state": "stopped", "pid": null, "restarts": 0,
                 "started_at": 1_760_000_000_000u64, "generation": 1,
                 "last_exit": {"code": 0, "at": 1_760_000_001_000u64, "generation": 1}, "nodes": []}],
            "peers": [],
        }));
        p.peer_down_for = 2;
    }
    let entry = ctl
        .wait_exit_at("b", "job", Duration::from_secs(2))
        .expect("two 502s, then b's job");
    assert_eq!(json!(0), entry["last_exit"]["code"]);
    // Scoped to b's generation and boot: b's epoch is compared, not this plane's.
    let scoped = ctl
        .wait_exit_of_at("b", "job", 1, Some("b00b00b00b00"), Duration::from_secs(2))
        .unwrap();
    assert_eq!(entry, scoped);
    let err = ctl
        .wait_exit_of_at(
            "b",
            "job",
            2,
            Some("b00b00b00b00"),
            Duration::from_millis(60),
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("generation 1's run, not generation 2's"),
        "{err}"
    );
    let out = executor(ctl)
        .run_source(
            "j <- ix.ctl.wait(service: \"job\", exit: true, peer: \"b\", timeout_ms: 2000)\n\
             k <- ix.ctl.wait(service: \"job\", exit: true, generation: 1, epoch: \"b00b00b00b00\", \
             peer: \"b\", timeout_ms: 2000)\n",
        )
        .unwrap();
    assert_eq!(&entry, out.binding("j").unwrap());
    assert_eq!(&entry, out.binding("k").unwrap());
    let log = &plane.lock().unwrap().log;
    assert!(
        log.iter().filter(|r| *r == "GET /v1/status?peer=b").count() >= 6,
        "{log:?}"
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
