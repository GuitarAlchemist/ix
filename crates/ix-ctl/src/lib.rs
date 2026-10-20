//! `ix-ctl` — the Rust and IXQL client of `ixctl`, IX's control plane.
//!
//! The plane (`go/ixctl`) supervises declared services and reconciles them toward a
//! desired-state document, the way a local Terraform would: `plan` says what would change,
//! `apply` bumps a generation and acts, and status reports the generation it has observed.
//! This crate reaches it over loopback HTTP/1.1 with the bearer token from the plane's state
//! file:
//!
//! ```no_run
//! use ix_ctl::Ctl;
//! use serde_json::json;
//! use std::time::Duration;
//!
//! let ctl = Ctl::from_state("C:/Users/me/AppData/Local/ixctl/local.json")?;
//! let desired = json!({ "services": { "pod-a": { "state": "running" } } });
//! let applied = ctl.apply(&desired, None)?;
//! let g = applied["generation"].as_u64().unwrap();
//! ctl.wait_generation(g, Duration::from_secs(10))?;
//! let pod = ctl.wait_nodes("pod-a", 4, Duration::from_secs(10))?;
//! println!("{}", pod["nodes"]);
//! ctl.stop("pod-a", None)?;
//! # Ok::<(), ix_ctl::CtlError>(())
//! ```
//!
//! [`register`] exposes the same verbs to IXQL as `ix.ctl.status`, `ix.ctl.plan`,
//! `ix.ctl.apply`, `ix.ctl.stop` and `ix.ctl.wait`.
//!
//! # What the client refuses
//!
//! The token is only ever sent to a loopback address, in a header. Both waits poll
//! `/v1/status` for now; [`Ctl::events`] reads the plane's event stream, and the waits move to
//! it once a wait built on it has been measured against the polling one.

mod http;
mod ixql;

pub use ixql::register;

use serde_json::{json, Value};
use std::fmt;
use std::io::{BufRead, Read};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// How often a wait reads the status again.
const POLL: Duration = Duration::from_millis(25);

/// What went wrong talking to the plane.
#[derive(Debug)]
pub enum CtlError {
    /// The state file is missing or does not name a plane.
    State { path: PathBuf, reason: String },
    /// The address is not a loopback one, so the token is not sent there.
    NotLoopback(String),
    /// No connection, or it broke.
    Io(std::io::Error),
    /// The plane refused the request: 400 for an undeclared name, 401 for a token.
    Http { status: u16, message: String },
    /// The plane's answer is not what the contract says it is.
    Protocol(String),
    /// The service is not one the plane declares.
    NotDeclared(String),
    /// A wait ended without its condition: it ran out of time, or the service failed.
    Wait(String),
    /// The plane gave up on the generation a wait was for: one of its actions failed.
    Diverged {
        generation: u64,
        service: String,
        reason: String,
    },
    /// The plane restarted since the generation was applied: it now reports boot `now`, and
    /// the generation belonged to boot `epoch`, so its outcome is gone with that boot.
    Restarted {
        generation: u64,
        epoch: String,
        now: String,
    },
    /// The event stream skipped events: `from` is the first sequence number not delivered,
    /// `reason` the plane's (`overflow`, `since-too-old`, `epoch-changed`, `since-ahead`) and
    /// `epoch` the boot it speaks of. After `overflow` the plane ends the stream.
    Gap {
        reason: String,
        from: u64,
        epoch: String,
    },
    /// The call's arguments do not fit together; nothing was sent.
    Argument(String),
}

impl fmt::Display for CtlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CtlError::State { path, reason } => {
                write!(f, "state file {}: {reason}", path.display())
            }
            CtlError::NotLoopback(addr) => write!(
                f,
                "{addr} is not a loopback address; the plane's token is only sent to one"
            ),
            CtlError::Io(e) => write!(f, "plane unreachable: {e}"),
            CtlError::Http { status, message } => write!(f, "plane answered {status}: {message}"),
            CtlError::Protocol(m) => write!(f, "unexpected answer from the plane: {m}"),
            CtlError::NotDeclared(name) => {
                write!(f, "service `{name}` is not declared on the plane")
            }
            CtlError::Wait(m) => f.write_str(m),
            CtlError::Diverged {
                generation,
                service,
                reason,
            } => write!(f, "generation {generation} diverged on {service}: {reason}"),
            CtlError::Restarted {
                generation,
                epoch,
                now,
            } => write!(
                f,
                "the plane restarted: generation {generation} belonged to boot {epoch}, the plane is now boot {now}"
            ),
            CtlError::Gap {
                reason,
                from,
                epoch,
            } => write!(
                f,
                "the event stream skipped events from seq {from} of boot {epoch}: {reason}"
            ),
            CtlError::Argument(m) => f.write_str(m),
        }
    }
}

/// One event from the plane's stream (`GET /v1/events`): the envelope's fields, flattened.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// The channel it was published on: `ctl`, `svc/<name>`, `peer/<name>/...`.
    pub channel: String,
    /// Its sequence number, counted from 1 at each boot of the plane.
    pub seq: u64,
    /// The boot it belongs to.
    pub epoch: String,
    /// The instance that produced it (a peer's name for a mirrored event).
    pub origin: String,
    /// What happened: `converged`, `diverged`, `service.started`, ...
    pub kind: String,
    /// When, in Unix milliseconds on the emitter's clock, when the emitter said.
    pub t: Option<i64>,
    /// The event's own payload.
    pub data: Value,
}

impl From<&Event> for Value {
    /// The event as JSON, the envelope's fields flattened: `{channel, seq, epoch, origin,
    /// kind, t, data}`.
    fn from(e: &Event) -> Value {
        json!({
            "channel": e.channel, "seq": e.seq, "epoch": e.epoch, "origin": e.origin,
            "kind": e.kind, "t": e.t, "data": e.data,
        })
    }
}

/// The plane's event stream, read as it comes. Each item is an [`Event`], or a
/// [`CtlError::Gap`] where the plane skipped some; the iterator ends when the plane closes the
/// stream (after an `overflow` gap, or at shutdown) or the connection breaks.
pub struct Events {
    body: Box<dyn std::io::BufRead>,
    /// The socket again, to bound a read; `None` when the stream was never opened.
    sock: Option<std::net::TcpStream>,
    /// How long each [`Iterator::next`] may wait for its event, pings and partial frames
    /// included; `None` waits as long as the plane keeps the connection alive.
    next_within: Option<Duration>,
    done: bool,
}

/// How long one read of the stream may wait: past the plane's 15 s keep-alive ping.
const STREAM_IDLE: Duration = Duration::from_secs(40);

impl Iterator for Events {
    type Item = Result<Event, CtlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // One deadline for the whole event: a keep-alive comment or a chunk's worth of bytes
        // does not start the wait over.
        let deadline = self.next_within.map(|d| Instant::now() + d);
        let mut kind = String::new();
        let mut data: Vec<String> = Vec::new();
        let mut line = String::new();
        loop {
            if let Some(deadline) = deadline {
                let rest = left(deadline);
                if rest.is_zero() {
                    self.done = true;
                    return Some(Err(CtlError::Io(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "no event within the time given",
                    ))));
                }
                if let Some(s) = &self.sock {
                    let _ = s.set_read_timeout(Some(rest.max(Duration::from_millis(1))));
                }
            }
            line.clear();
            match self.body.read_line(&mut line) {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(_) => {}
                Err(e) => {
                    self.done = true;
                    return Some(Err(CtlError::Io(e)));
                }
            }
            let text = line.trim_end_matches(['\r', '\n']);
            if text.is_empty() {
                if data.is_empty() {
                    continue; // a keep-alive comment's blank line, or stray
                }
                break;
            }
            if let Some(rest) = text.strip_prefix(':') {
                let _ = rest; // a comment: the plane's ping
                continue;
            }
            let (field, value) = text.split_once(':').unwrap_or((text, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "event" => kind = value.to_string(),
                "data" => data.push(value.to_string()),
                _ => {} // `id` repeats the envelope's epoch and seq; `retry` is for browsers
            }
        }
        let text = data.join("\n");
        let doc: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                return Some(Err(CtlError::Protocol(format!(
                    "event data is not JSON: {e}"
                ))))
            }
        };
        // Told apart by shape, not by the `event:` field: a service may publish an event whose
        // kind is `gap`, and the plane writes every kind as `event: <kind>`.
        let text = |k: &str| doc.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let seq = match (doc.get("seq").and_then(Value::as_u64), doc.get("ev")) {
            (Some(seq), Some(_)) => seq,
            _ if doc.get("reason").is_some() && doc.get("from").is_some() => {
                return Some(Err(CtlError::Gap {
                    reason: text("reason"),
                    from: doc["from"].as_u64().unwrap_or(0),
                    epoch: text("epoch"),
                }));
            }
            _ => {
                return Some(Err(CtlError::Protocol(format!(
                    "`event: {kind}` carries neither an envelope (seq, ev) nor a gap (reason, from)"
                ))));
            }
        };
        Some(Ok(Event {
            channel: text("ch"),
            seq,
            epoch: text("epoch"),
            origin: text("origin"),
            kind: doc["ev"]["kind"].as_str().unwrap_or("").to_string(),
            t: doc["ev"].get("t").and_then(Value::as_i64),
            data: doc["ev"].get("data").cloned().unwrap_or(Value::Null),
        }))
    }
}

impl std::error::Error for CtlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CtlError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CtlError {
    fn from(e: std::io::Error) -> Self {
        CtlError::Io(e)
    }
}

/// A handle on one plane: its address and token. Cheap to clone; every call opens its own
/// connection.
#[derive(Clone)]
pub struct Ctl {
    addr: String,
    token: String,
    /// The state file this handle came from, if any: a wait scoped to a boot re-reads it when
    /// the plane stops answering, since a restarted plane has a new address and token.
    state: Option<PathBuf>,
}

impl fmt::Debug for Ctl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The token is a credential: it never reaches a log through `{:?}`.
        f.debug_struct("Ctl")
            .field("addr", &self.addr)
            .field("token", &"<redacted>")
            .field("state", &self.state)
            .finish()
    }
}

impl Ctl {
    /// A plane at `addr` (`127.0.0.1:PORT`) that accepts `token`.
    pub fn new(addr: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            token: token.into(),
            state: None,
        }
    }

    /// The plane described by its state file, `{"name", "addr", "pid", "token"}`, which
    /// `ixctl up` writes on ready and removes on a clean exit. The handle remembers the path:
    /// see [`Ctl::wait_generation_of`].
    pub fn from_state(path: impl AsRef<Path>) -> Result<Self, CtlError> {
        let mut ctl = Self::read_state(path.as_ref())?;
        ctl.state = Some(path.as_ref().to_path_buf());
        Ok(ctl)
    }

    fn read_state(path: &Path) -> Result<Self, CtlError> {
        let state = |reason: String| CtlError::State {
            path: path.to_path_buf(),
            reason,
        };
        let text = std::fs::read_to_string(path).map_err(|e| {
            state(if e.kind() == std::io::ErrorKind::NotFound {
                "not found; is `ixctl up` running?".to_string()
            } else {
                e.to_string()
            })
        })?;
        let v: Value = serde_json::from_str(&text).map_err(|e| state(format!("not JSON: {e}")))?;
        let field = |name: &str| {
            v.get(name)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .ok_or_else(|| state(format!("no `{name}`")))
        };
        Ok(Self::new(field("addr")?, field("token")?))
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// `GET /v1/status`: the generations, every service with its state and nodes, the peers.
    pub fn status(&self) -> Result<Value, CtlError> {
        self.status_of(None)
    }

    /// [`Ctl::status`], or with `peer` that peer's status as this plane relays it
    /// (`GET /v1/status?peer=`): its own name, epoch and generations.
    pub fn status_of(&self, peer: Option<&str>) -> Result<Value, CtlError> {
        self.status_within(peer, None)
    }

    /// [`Ctl::status_of`] with every socket timeout capped at `within`: a wait's poll, bounded
    /// by what is left of its deadline.
    fn status_within(
        &self,
        peer: Option<&str>,
        within: Option<Duration>,
    ) -> Result<Value, CtlError> {
        self.call_within("GET", &routed("/v1/status", peer), None, within)
    }

    /// `GET /v1/events`: the plane's events on channels under `channel` (a prefix: `ctl`,
    /// `svc/`, `peer/b/`), as they come. With `since`, the events after that sequence number
    /// are replayed first, from the plane's ring; `epoch` says which boot `since` belongs to,
    /// so a number kept across a restart is reported as a [`CtlError::Gap`] instead of being
    /// trusted. The stream stays open until the plane closes it.
    pub fn events(
        &self,
        channel: &str,
        since: Option<u64>,
        epoch: Option<&str>,
    ) -> Result<Events, CtlError> {
        self.open_events(channel, since, epoch, None)
    }

    /// [`Ctl::events`] with each event waited for at most `next_within`: past it, the next
    /// item is a [`CtlError::Io`] timeout and the iterator ends. The plane's keep-alive pings
    /// and a frame's first bytes do not start the wait over.
    pub fn events_within(
        &self,
        channel: &str,
        since: Option<u64>,
        epoch: Option<&str>,
        next_within: Duration,
    ) -> Result<Events, CtlError> {
        self.open_events(channel, since, epoch, Some(next_within))
    }

    fn open_events(
        &self,
        channel: &str,
        since: Option<u64>,
        epoch: Option<&str>,
        next_within: Option<Duration>,
    ) -> Result<Events, CtlError> {
        if since.is_none() && epoch.is_some() {
            return Err(CtlError::Argument(
                "`epoch` says which boot `since` belongs to: it needs `since`".into(),
            ));
        }
        let mut path = format!("/v1/events?ch={}", percent(channel));
        if let Some(s) = since {
            path.push_str(&format!("&since={s}"));
        }
        if let Some(e) = epoch {
            path.push_str(&format!("&epoch={}", percent(e)));
        }
        let addr = self.loopback()?;
        let mut s = http::open_stream(addr, &path, &self.token, STREAM_IDLE)?;
        if s.status != 200 {
            let mut body = Vec::new();
            s.body.read_to_end(&mut body)?;
            let message = serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string))
                .unwrap_or_else(|| String::from_utf8_lossy(&body).trim().to_string());
            return Err(CtlError::Http {
                status: s.status,
                message,
            });
        }
        Ok(Events {
            body: s.body,
            sock: s.sock,
            next_within,
            done: false,
        })
    }

    /// `POST /v1/plan`: the actions `desired` would take, changing nothing. With `peer`, the
    /// plane forwards it to that peer (`?peer=` beside the document), which decides.
    pub fn plan(&self, desired: &Value, peer: Option<&str>) -> Result<Value, CtlError> {
        self.call("POST", &routed("/v1/plan", peer), Some(desired))
    }

    /// `POST /v1/apply`: answers the new generation and its actions at once; they run in the
    /// background, and [`Ctl::wait_generation`] tells when they are done.
    ///
    /// The client never retries. A [`CtlError::Io`] here leaves the outcome unknown: the plane
    /// may or may not have taken this document, and nothing in the answer-less exchange says
    /// which. [`Ctl::status`]'s `generation` settles it only when this client is known to be the
    /// plane's one writer; with other writers a moved generation may be theirs. Applying the
    /// same document again leaves the services in the state it names, but it is a second
    /// apply, not a confirmation of the first.
    ///
    /// With `peer`, the plane forwards the document to that peer (`?peer=` beside it), which
    /// decides. The document itself never names a peer: it says what is wanted, not where.
    pub fn apply(&self, desired: &Value, peer: Option<&str>) -> Result<Value, CtlError> {
        self.call("POST", &routed("/v1/apply", peer), Some(desired))
    }

    /// `POST /v1/stop`: answers once the service's process tree is gone. With `peer`, the
    /// plane forwards it to that peer, which decides.
    pub fn stop(&self, service: &str, peer: Option<&str>) -> Result<Value, CtlError> {
        let mut body = json!({ "service": service });
        if let Some(peer) = peer {
            body["peer"] = json!(peer);
        }
        self.call("POST", "/v1/stop", Some(&body))
    }

    /// Waits until the plane has settled `generation`, and answers the status that showed it.
    ///
    /// Each poll of the status decides one of three things. With a plane that reports
    /// `settled_generation` and `failed` (every generation `<= settled_generation` has ended,
    /// `failed` lists the diverged ones since boot):
    /// - failed, [`CtlError::Diverged`] at once, when `failed` names `generation`;
    /// - done, `Ok(status)`, once `settled_generation >= generation`;
    /// - pending otherwise, until `timeout` ends the wait with [`CtlError::Wait`].
    ///
    /// An older plane has only `observed_generation` (the highest that converged) and
    /// `diverged` (the highest that failed); there the generation is done once
    /// `observed_generation >= generation`, and failed while `observed_generation < generation`
    /// and `diverged` names `generation` or a later one. That summary lets a later converged
    /// generation hide an earlier one (F2.1 in the one-way-doors doc); the settled one does not.
    /// A `diverged` that is null, or absent, never fails a wait.
    ///
    /// This wait does not know which boot `generation` belongs to: after a restart it waits
    /// for the new boot to reach that number, or times out. [`Ctl::wait_generation_of`] takes
    /// the epoch the apply reply gave and fails at once instead.
    pub fn wait_generation(&self, generation: u64, timeout: Duration) -> Result<Value, CtlError> {
        self.wait_gen(None, None, generation, timeout)
    }

    /// [`Ctl::wait_generation`], scoped to the boot `epoch` (the apply reply's `epoch`, also
    /// in status and `/health`). When the status reports another epoch the plane has
    /// restarted and the generation's outcome is gone with its boot: the wait ends at once
    /// with [`CtlError::Restarted`] instead of waiting for the new boot to reach the same
    /// number. A plane that reports no epoch is waited on as [`Ctl::wait_generation`] does.
    ///
    /// A restarted plane listens at a new address with a new token, so a handle made with
    /// [`Ctl::from_state`] re-reads its state file when the plane stops answering (no
    /// connection, or a 401) and goes on against what the file names now; that is how the
    /// restart is seen rather than reported as `plane unreachable`. A handle made with
    /// [`Ctl::new`] has nothing to re-read, and the connection error stands.
    pub fn wait_generation_of(
        &self,
        epoch: &str,
        generation: u64,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        self.wait_gen(None, Some(epoch), generation, timeout)
    }

    /// [`Ctl::wait_generation`] / [`Ctl::wait_generation_of`] for a generation that was
    /// applied through `peer` ([`Ctl::apply`] with a peer): the status read is that peer's
    /// (`GET /v1/status?peer=`), whose generations and epoch the apply reply carried. Waiting
    /// on this plane instead would compare the peer's epoch with this plane's and end with a
    /// false [`CtlError::Restarted`], or accept an unrelated local generation.
    ///
    /// The peer's status comes through this plane's link to it; while that link is down (the
    /// peer, or this plane, restarted a moment ago) the plane answers 502 or 503, and the wait
    /// keeps asking until its deadline instead of failing on the first one. A 400, the peer is
    /// not declared, is final.
    pub fn wait_generation_at(
        &self,
        peer: &str,
        epoch: Option<&str>,
        generation: u64,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        self.wait_gen(Some(peer), epoch, generation, timeout)
    }

    /// Whether `err` is one a restarted plane produces: nothing listening at the old
    /// address, or the old token refused. Only then is the state file worth re-reading.
    fn plane_may_have_moved(&self, err: &CtlError) -> bool {
        self.state.is_some() && matches!(err, CtlError::Io(_) | CtlError::Http { status: 401, .. })
    }

    /// The handle the state file names now, if it is readable and names another plane.
    fn refreshed(&self) -> Option<Ctl> {
        let path = self.state.as_deref()?;
        let fresh = Self::read_state(path).ok()?;
        (fresh.addr != self.addr || fresh.token != self.token).then(|| Ctl {
            state: self.state.clone(),
            ..fresh
        })
    }

    fn wait_gen(
        &self,
        peer: Option<&str>,
        epoch: Option<&str>,
        generation: u64,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        let deadline = Instant::now() + timeout;
        let mut me = self.clone();
        loop {
            let status = match me.status_within(peer, Some(left(deadline))) {
                Ok(status) => status,
                Err(e) if epoch.is_some() && me.plane_may_have_moved(&e) => match me.refreshed() {
                    Some(fresh) if Instant::now() < deadline => {
                        me = fresh;
                        continue;
                    }
                    Some(_) => return Err(e),
                    None if pause(deadline) => return Err(e),
                    None => continue,
                },
                // A peer's status comes through this plane's link to it, which is down for
                // a moment when either restarts (502/503). The peer's new boot is behind that
                // moment, so the wait keeps asking until its deadline; a 400 (no such peer)
                // stays final.
                Err(
                    e @ CtlError::Http {
                        status: 502 | 503, ..
                    },
                ) if peer.is_some() => {
                    if pause(deadline) {
                        return Err(e);
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };
            if let (Some(want), Some(now)) = (epoch, status.get("epoch").and_then(Value::as_str)) {
                if now != want {
                    return Err(CtlError::Restarted {
                        generation,
                        epoch: want.to_string(),
                        now: now.to_string(),
                    });
                }
            }
            let observed = status
                .get("observed_generation")
                .and_then(Value::as_u64)
                .ok_or_else(|| CtlError::Protocol("status has no `observed_generation`".into()))?;
            let divergence = |d: &Value| -> Result<CtlError, CtlError> {
                let field = |name: &str| {
                    d.get(name).ok_or_else(|| {
                        CtlError::Protocol(format!("status `diverged` has no `{name}`"))
                    })
                };
                let text = |name: &str| {
                    field(name)?.as_str().map(str::to_string).ok_or_else(|| {
                        CtlError::Protocol(format!("status `diverged.{name}` is not a string"))
                    })
                };
                Ok(CtlError::Diverged {
                    generation: field("generation")?.as_u64().ok_or_else(|| {
                        CtlError::Protocol("status `diverged.generation` is not a count".into())
                    })?,
                    service: text("service")?,
                    reason: text("reason")?,
                })
            };
            if let Some(settled) = status.get("settled_generation").and_then(Value::as_u64) {
                let failed = status
                    .get("failed")
                    .and_then(Value::as_array)
                    .ok_or_else(|| CtlError::Protocol("status has no `failed` list".into()))?;
                if let Some(d) = failed
                    .iter()
                    .find(|d| d.get("generation").and_then(Value::as_u64) == Some(generation))
                {
                    return Err(divergence(d)?);
                }
                if settled >= generation {
                    return Ok(status);
                }
            } else {
                if observed >= generation {
                    return Ok(status);
                }
                if let Some(d) = status.get("diverged").filter(|d| !d.is_null()) {
                    let err = divergence(d)?;
                    if matches!(err, CtlError::Diverged { generation: g, .. } if g >= generation) {
                        return Err(err);
                    }
                }
            }
            if pause(deadline) {
                return Err(CtlError::Wait(format!(
                    "generation {generation} not observed after {} ms (observed {observed})",
                    timeout.as_millis()
                )));
            }
        }
    }

    /// Waits until `service` runs with at least `nodes` nodes in its last `stats` event, and
    /// answers its entry in the status. A service in state `failed` ends the wait at once.
    pub fn wait_nodes(
        &self,
        service: &str,
        nodes: usize,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        self.wait_nodes_on(None, service, nodes, timeout)
    }

    /// [`Ctl::wait_nodes`] for a service of `peer`, read from that peer's status. A 502 or
    /// 503 (the link to the peer is down for a moment) is asked again until the deadline.
    pub fn wait_nodes_at(
        &self,
        peer: &str,
        service: &str,
        nodes: usize,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        self.wait_nodes_on(Some(peer), service, nodes, timeout)
    }

    fn wait_nodes_on(
        &self,
        peer: Option<&str>,
        service: &str,
        nodes: usize,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        let deadline = Instant::now() + timeout;
        loop {
            let status = match self.status_within(peer, Some(left(deadline))) {
                Ok(status) => status,
                // The link to the peer is down for a moment (see `wait_gen`): ask again.
                Err(
                    e @ CtlError::Http {
                        status: 502 | 503, ..
                    },
                ) if peer.is_some() => {
                    if pause(deadline) {
                        return Err(e);
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };
            let entry = status
                .get("services")
                .and_then(Value::as_array)
                .ok_or_else(|| CtlError::Protocol("status has no `services`".into()))?
                .iter()
                .find(|s| s.get("name").and_then(Value::as_str) == Some(service))
                .ok_or_else(|| CtlError::NotDeclared(service.to_string()))?;
            let state = entry.get("state").and_then(Value::as_str).unwrap_or("");
            let have = entry
                .get("nodes")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if state == "running" && have >= nodes {
                return Ok(entry.clone());
            }
            if state == "failed" {
                return Err(CtlError::Wait(format!(
                    "service `{service}` failed with {have} of {nodes} nodes"
                )));
            }
            if pause(deadline) {
                return Err(CtlError::Wait(format!(
                    "service `{service}` has {have} of {nodes} nodes in state `{state}` after {} ms",
                    timeout.as_millis()
                )));
            }
        }
    }

    fn call(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value, CtlError> {
        self.call_within(method, path, body, None)
    }

    fn call_within(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        within: Option<Duration>,
    ) -> Result<Value, CtlError> {
        let addr = self.loopback()?;
        let body = body.map(|b| b.to_string().into_bytes());
        let r = http::request(addr, method, path, &self.token, body.as_deref(), within)?;
        let parsed = serde_json::from_slice::<Value>(&r.body);
        if !(200..300).contains(&r.status) {
            let message = match &parsed {
                Ok(v) => v
                    .get("error")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| v.to_string()),
                Err(_) => String::from_utf8_lossy(&r.body).trim().to_string(),
            };
            return Err(CtlError::Http {
                status: r.status,
                message,
            });
        }
        parsed.map_err(|e| CtlError::Protocol(format!("{method} {path}: not JSON: {e}")))
    }

    /// The address to connect to, if every address `addr` names is a loopback one.
    fn loopback(&self) -> Result<SocketAddr, CtlError> {
        let addrs: Vec<SocketAddr> = self.addr.to_socket_addrs()?.collect();
        match addrs.first() {
            Some(first) if addrs.iter().all(|a| a.ip().is_loopback()) => Ok(*first),
            _ => Err(CtlError::NotLoopback(self.addr.clone())),
        }
    }
}

/// `path`, with `?peer=<name>` when a peer is named. A peer name is `[a-z0-9-]` on the plane;
/// anything else is percent-encoded so the query stays one argument.
fn routed(path: &str, peer: Option<&str>) -> String {
    let Some(peer) = peer else {
        return path.to_string();
    };
    format!("{path}?peer={}", percent(peer))
}

/// `s` as one query argument: unreserved bytes as they are, the rest percent-encoded.
fn percent(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// What is left until `deadline`; zero once it has passed. A poll's socket timeouts are
/// capped at it, so the request in flight ends with the wait, not seconds after it.
fn left(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

/// Sleeps one poll, or less near `deadline`. True when the deadline has passed, before or
/// after the sleep: a wait never asks again past its timeout.
fn pause(deadline: Instant) -> bool {
    let now = Instant::now();
    if now >= deadline {
        return true;
    }
    std::thread::sleep(POLL.min(deadline - now));
    Instant::now() >= deadline
}
