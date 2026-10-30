//! `ix-ctl` — the Rust and IXQL client of `ixctl`, IX's control plane.
//!
//! The plane (`go/ixctl`) supervises declared services and reconciles them toward a
//! desired-state document, the way a local Terraform would: `plan` says what would change,
//! `apply` bumps a generation and acts, and status reports the generation it has observed.
//! This crate reaches it over loopback HTTP/1.1 with the bearer token from the plane's state
//! file:
//!
//! ```no_run
//! use ix_ctl::{Ctl, GenerationOutcome};
//! use serde_json::json;
//! use std::time::Duration;
//!
//! let ctl = Ctl::from_state("C:/Users/<you>/AppData/Local/ixctl/local.json")?;
//! let desired = json!({ "services": { "pod-a": { "state": "running" } } });
//! let applied = ctl.apply(&desired, None)?;
//! let g = applied["generation"].as_u64().unwrap();
//! let epoch = applied["epoch"].as_str().unwrap();
//! let outcome = ctl.wait_generation_outcome(epoch, g, Duration::from_secs(10), None)?;
//! assert_eq!(GenerationOutcome::Converged, outcome);
//! let pod = ctl.wait_nodes("pod-a", 4, Duration::from_secs(10))?;
//! println!("{}", pod["nodes"]);
//! ctl.stop("pod-a", None)?;
//! # Ok::<(), ix_ctl::CtlError>(())
//! ```
//!
//! A job is a service declared with `restart: never`: it runs to its end and the plane leaves
//! it `stopped`, its `last_exit` saying how. [`Ctl::wait_exit_of`] waits for the end of the run
//! a generation started; the generation's convergence only says the job was launched
//! (`examples/job.ixql` runs the real `ix pipeline run` that way).
//!
//! # Plane-side waits
//!
//! [`Ctl::wait_generation_outcome`], [`Ctl::wait_launch`] and [`Ctl::wait_run`] ask the plane's
//! `POST /v1/wait` and answer one value of a closed set per wait ([`GenerationOutcome`],
//! [`LaunchOutcome`], [`RunOutcome`]): a `timeout` is a value, a refusal stays an error. A
//! handle made with [`Ctl::from_state`] follows its state file when the plane stops answering,
//! so a wait cut by a hard stop is asked again of the new boot, which answers
//! `Unknown(OtherEpoch)`. In IXQL they are `ix.ctl.wait(wait: "generation" | "launch" | "run",
//! ...)`. [`Ctl::wait_generation`] is deprecated: by number alone it cannot tell a restart.
//!
//! [`register`] exposes the same verbs to IXQL as `ix.ctl.status`, `ix.ctl.plan`,
//! `ix.ctl.apply`, `ix.ctl.stop` and `ix.ctl.wait`.
//!
//! # What the client refuses
//!
//! The token is only ever sent to a loopback address, in a header. The polling waits read
//! `/v1/status` every 25 ms; [`Ctl::events`] reads the plane's event stream. The gate's M1
//! measures what a wait on the stream would gain: on a real start, `converged` reaches a live
//! reader 2–3 ms after the apply reply, the poll sees it 12–28 ms after, depending on where in
//! its period it lands. The waits move to the stream when that period matters to a caller.

mod http;
mod ixql;

pub use ixql::register;

use serde_json::{json, Value};
use std::cell::Cell;
use std::fmt;
use std::io::{BufRead, Read};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

/// How often a polling wait reads the status again.
const POLL: Duration = Duration::from_millis(25);
/// Time left after the plane's wait deadline for its HTTP answer to cross loopback.
const WAIT_REPLY_GRACE: Duration = Duration::from_millis(250);

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
    /// On a mirrored event (a `peer/<name>/...` channel): the source's own seq, so a consumer
    /// here keys a series (`origin:src_epoch:node:<src_seq of probe.created>`) and dedups
    /// (`origin:src_epoch:src_seq`) as one on the origin would. `None` on the plane's own events,
    /// among them the `link.gap` continuity markers the plane publishes on a mirrored channel.
    pub src_seq: Option<u64>,
    /// On a mirrored event: the source's own boot id.
    pub src_epoch: Option<String>,
}

impl From<&Event> for Value {
    /// The event as JSON, the envelope's fields flattened: `{channel, seq, epoch, origin,
    /// kind, t, data}`, plus `src_seq` and `src_epoch` on a mirrored event only.
    fn from(e: &Event) -> Value {
        let mut v = json!({
            "channel": e.channel, "seq": e.seq, "epoch": e.epoch, "origin": e.origin,
            "kind": e.kind, "t": e.t, "data": e.data,
        });
        if let (Some(seq), Some(epoch)) = (e.src_seq, &e.src_epoch) {
            v["src_seq"] = json!(seq);
            v["src_epoch"] = json!(epoch);
        }
        v
    }
}

/// The plane's event stream, read as it comes. Each item is an [`Event`], or a
/// [`CtlError::Gap`] where the plane skipped some; the iterator ends when the plane closes the
/// stream (after an `overflow` gap, or at shutdown) or the connection breaks.
pub struct Events {
    body: Box<dyn std::io::BufRead>,
    /// The instant the socket's reads must not wait past; set per [`Iterator::next`] when
    /// `next_within` is `Some`, else never.
    deadline: http::Deadline,
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
        // One deadline for the whole event, enforced on the socket under the readers: a
        // keep-alive comment, a chunk's worth of bytes or a line in pieces does not start the
        // wait over, and bytes already buffered are read whatever the clock says. Each call
        // gets its own: the open had the same time, and a slow open does not eat the first
        // event's.
        if let Some(d) = self.next_within {
            self.deadline.set(Some(Instant::now() + d));
        }
        let mut kind = String::new();
        let mut data: Vec<String> = Vec::new();
        let mut line = String::new();
        loop {
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
        // Provenance is a pair or nothing: half a pair is not a mirrored envelope's (the plane
        // never writes one), so it is read as none rather than shown by half.
        let src = match (
            doc.get("src_seq").and_then(Value::as_u64),
            doc.get("src_epoch").and_then(Value::as_str),
        ) {
            (Some(seq), Some(epoch)) if seq != 0 && !epoch.is_empty() => {
                Some((seq, epoch.to_string()))
            }
            _ => None,
        };
        Some(Ok(Event {
            channel: text("ch"),
            seq,
            epoch: text("epoch"),
            origin: text("origin"),
            kind: doc["ev"]["kind"].as_str().unwrap_or("").to_string(),
            t: doc["ev"].get("t").and_then(Value::as_i64),
            data: doc["ev"].get("data").cloned().unwrap_or(Value::Null),
            src_seq: src.as_ref().map(|s| s.0),
            src_epoch: src.map(|s| s.1),
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

/// One service launch attempt: the origin plane, its boot, and the sequence of the
/// `ixctl.run.begin` marker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunId {
    pub origin: String,
    pub epoch: String,
    pub seq: u64,
}

impl RunId {
    fn decode(value: &Value, field: &str) -> Result<Self, CtlError> {
        let record = value
            .as_object()
            .ok_or_else(|| CtlError::Protocol(format!("wait `{field}` is not a run record")))?;
        let text = |name: &str| {
            record
                .get(name)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    CtlError::Protocol(format!("wait `{field}.{name}` is not a non-empty string"))
                })
        };
        let seq = record
            .get("seq")
            .and_then(Value::as_u64)
            .filter(|seq| *seq > 0)
            .ok_or_else(|| {
                CtlError::Protocol(format!("wait `{field}.seq` is not a positive count"))
            })?;
        Ok(Self {
            origin: text("origin")?,
            epoch: text("epoch")?,
            seq,
        })
    }

    fn json(&self) -> Value {
        json!({"origin": self.origin, "epoch": self.epoch, "seq": self.seq})
    }
}

/// Why the plane no longer has a wait's answer. This set is closed for `run/1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    Evicted,
    OtherEpoch,
}

impl UnknownReason {
    fn wire(self) -> &'static str {
        match self {
            Self::Evicted => "evicted",
            Self::OtherEpoch => "other-epoch",
        }
    }
}

/// The closed result of a generation wait.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenerationOutcome {
    Converged,
    Diverged { service: String, reason: String },
    Timeout,
    Unknown(UnknownReason),
}

/// The closed result of waiting for one generation's launch decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaunchOutcome {
    Launched(RunId),
    NotLaunched(Option<RunId>),
    Timeout,
    Unknown(UnknownReason),
}

/// The closed result of waiting for one exact run.
///
/// `Ended.end` is kept as [`serde_json::Value`], field-for-field with the plane's `run.end`
/// data rather than a second Rust schema that could discard a field.
#[derive(Clone, Debug, PartialEq)]
pub enum RunOutcome {
    Ended { run: RunId, end: Value },
    Timeout,
    Unknown(UnknownReason),
}

impl From<GenerationOutcome> for Value {
    fn from(outcome: GenerationOutcome) -> Self {
        match outcome {
            GenerationOutcome::Converged => json!({"outcome": "converged"}),
            GenerationOutcome::Diverged { service, reason } => {
                json!({"outcome": "diverged", "service": service, "reason": reason})
            }
            GenerationOutcome::Timeout => json!({"outcome": "timeout"}),
            GenerationOutcome::Unknown(reason) => {
                json!({"outcome": "unknown", "reason": reason.wire()})
            }
        }
    }
}

impl From<LaunchOutcome> for Value {
    fn from(outcome: LaunchOutcome) -> Self {
        match outcome {
            LaunchOutcome::Launched(run) => json!({"outcome": "launched", "run": run.json()}),
            LaunchOutcome::NotLaunched(current_run) => json!({
                "outcome": "not-launched",
                "current_run": current_run.map_or(Value::Null, |run| run.json())
            }),
            LaunchOutcome::Timeout => json!({"outcome": "timeout"}),
            LaunchOutcome::Unknown(reason) => {
                json!({"outcome": "unknown", "reason": reason.wire()})
            }
        }
    }
}

impl From<RunOutcome> for Value {
    fn from(outcome: RunOutcome) -> Self {
        match outcome {
            RunOutcome::Ended { run, end } => {
                json!({"outcome": "ended", "run": run.json(), "end": end})
            }
            RunOutcome::Timeout => json!({"outcome": "timeout"}),
            RunOutcome::Unknown(reason) => json!({"outcome": "unknown", "reason": reason.wire()}),
        }
    }
}

fn wait_outcome<'a>(reply: &'a Value, wait: &str) -> Result<&'a str, CtlError> {
    reply
        .get("outcome")
        .and_then(Value::as_str)
        .ok_or_else(|| CtlError::Protocol(format!("{wait} wait has no string `outcome`")))
}

fn wait_text(reply: &Value, wait: &str, field: &str) -> Result<String, CtlError> {
    reply
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| CtlError::Protocol(format!("{wait} wait has no string `{field}`")))
}

fn unknown_reason(reply: &Value, wait: &str) -> Result<UnknownReason, CtlError> {
    match reply.get("reason").and_then(Value::as_str) {
        Some("evicted") => Ok(UnknownReason::Evicted),
        Some("other-epoch") => Ok(UnknownReason::OtherEpoch),
        Some(reason) => Err(CtlError::Protocol(format!(
            "{wait} wait has unknown reason `{reason}`"
        ))),
        None => Err(CtlError::Protocol(format!(
            "{wait} wait has no string `reason`"
        ))),
    }
}

fn decode_generation(reply: &Value) -> Result<GenerationOutcome, CtlError> {
    match wait_outcome(reply, "generation")? {
        "converged" => Ok(GenerationOutcome::Converged),
        "diverged" => Ok(GenerationOutcome::Diverged {
            service: wait_text(reply, "generation", "service")?,
            reason: wait_text(reply, "generation", "reason")?,
        }),
        "timeout" => Ok(GenerationOutcome::Timeout),
        "unknown" => match unknown_reason(reply, "generation")? {
            UnknownReason::OtherEpoch => Ok(GenerationOutcome::Unknown(UnknownReason::OtherEpoch)),
            UnknownReason::Evicted => Err(CtlError::Protocol(
                "generation wait cannot be unknown because it was evicted".into(),
            )),
        },
        outcome => Err(CtlError::Protocol(format!(
            "generation wait has unknown outcome `{outcome}`"
        ))),
    }
}

fn decode_launch(reply: &Value) -> Result<LaunchOutcome, CtlError> {
    match wait_outcome(reply, "launch")? {
        "launched" => Ok(LaunchOutcome::Launched(RunId::decode(
            reply
                .get("run")
                .ok_or_else(|| CtlError::Protocol("launch wait has no `run`".into()))?,
            "run",
        )?)),
        "not-launched" => {
            let current = reply
                .get("current_run")
                .ok_or_else(|| CtlError::Protocol("launch wait has no `current_run`".into()))?;
            Ok(LaunchOutcome::NotLaunched(if current.is_null() {
                None
            } else {
                Some(RunId::decode(current, "current_run")?)
            }))
        }
        "timeout" => Ok(LaunchOutcome::Timeout),
        "unknown" => Ok(LaunchOutcome::Unknown(unknown_reason(reply, "launch")?)),
        outcome => Err(CtlError::Protocol(format!(
            "launch wait has unknown outcome `{outcome}`"
        ))),
    }
}

fn decode_run(reply: &Value) -> Result<RunOutcome, CtlError> {
    match wait_outcome(reply, "run")? {
        "ended" => {
            let run = RunId::decode(
                reply
                    .get("run")
                    .ok_or_else(|| CtlError::Protocol("run wait has no `run`".into()))?,
                "run",
            )?;
            let end = reply
                .get("end")
                .filter(|end| end.is_object())
                .cloned()
                .ok_or_else(|| CtlError::Protocol("run wait has no object `end`".into()))?;
            Ok(RunOutcome::Ended { run, end })
        }
        "timeout" => Ok(RunOutcome::Timeout),
        "unknown" => Ok(RunOutcome::Unknown(unknown_reason(reply, "run")?)),
        outcome => Err(CtlError::Protocol(format!(
            "run wait has unknown outcome `{outcome}`"
        ))),
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
        self.call_within("GET", &routed("/v1/status", peer), None, within, None)
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
        // A bounded stream is bounded from the connect on: a plane that accepts and withholds
        // its headers fails the open within `next_within`, not after the idle timeout.
        let deadline: http::Deadline = Rc::new(Cell::new(next_within.map(|d| Instant::now() + d)));
        let mut s = http::open_stream(addr, &path, &self.token, STREAM_IDLE, &deadline)?;
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
            deadline,
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

    /// `GET /v1/probes`: the probe kinds this plane has compiled in, the limits its tree
    /// applies (`max_nodes`, `min_every_ms`, …, by the names its refusals use) and the
    /// producer's revision (`version`). With `peer`, that peer's. A program reads this first:
    /// a kind it wants that is not listed is a refusal to make before any node exists.
    pub fn probe_kinds(&self, peer: Option<&str>) -> Result<Value, CtlError> {
        self.call("GET", &routed("/v1/probes", peer), None)
    }

    /// `GET /v1/probe`: the probe tree as the plane holds it, `{nodes: [{name, kind, state,
    /// every_ms, ttl_ms, parent?, filter?, seq, errors, bytes, children, rate_per_s,
    /// max_rate_per_s?, export?, expires_in_ms?}], count, max_nodes, rate_per_s, max_rate_per_s,
    /// tree_rate_per_s}`: each node's declared rate and bytes emitted, each mux's budget, the
    /// tree's (D10, ix#416), `export: true` on a node whose pipe federation may mirror (D4,
    /// ix#402), and `created_seq`, the seq of the node's `probe.created` (the incarnation's
    /// identity under its name, S2 slice 4). With `peer`, that peer's.
    pub fn probe_list(&self, peer: Option<&str>) -> Result<Value, CtlError> {
        self.probe_list_at(peer, false)
    }

    /// `GET /v1/probe?with_cursor=1`: `probe_list` bound to a cursor of the plane's own stream
    /// (S2 slice 4, criteria 3 and 4 of ix#408): `cursor: {seq, epoch}`, the head and the boot the
    /// list stands at, read with the nodes under the tree's lock where every lifecycle event is
    /// published, so the list is exactly what `ctl` says up to that seq and nothing of what
    /// follows; a leaf whose stop is in progress (its `probe.stopped` not out yet) says `state:
    /// "stopping"`. Without the cursor, the plain list. With `peer`, that peer's.
    pub fn probe_list_at(&self, peer: Option<&str>, with_cursor: bool) -> Result<Value, CtlError> {
        let mut path = routed("/v1/probe", peer);
        if with_cursor {
            path.push_str(if path.contains('?') {
                "&with_cursor=1"
            } else {
                "?with_cursor=1"
            });
        }
        self.call("GET", &path, None)
    }

    /// `POST /v1/probe/create`: declares `spec` (`{name, kind, every_ms?, parent?, filter?,
    /// ttl_ms?}`) as a node of the plane's probe tree, stopped, and answers `{node, kind,
    /// state, depth}` (`depth`, the muxes above it). A refusal names what it hit (an unknown kind, a taken name, a bound
    /// of `probe_kinds`'s `limits`) and leaves the tree as it was. With `peer`, that peer's
    /// tree (`?peer=` beside the record).
    pub fn probe_create(&self, spec: &Value, peer: Option<&str>) -> Result<Value, CtlError> {
        self.call("POST", &routed("/v1/probe/create", peer), Some(spec))
    }

    /// `POST /v1/probe/start`: runs the node; its lease, if it has one, starts counting.
    /// Answers `{node, state: "running", expires_in_ms?}`, a noop on a running node.
    pub fn probe_start(&self, node: &str, peer: Option<&str>) -> Result<Value, CtlError> {
        self.probe_node("start", node, peer)
    }

    /// `POST /v1/probe/stop`: stops the node and leaves its children running. Answers
    /// `{node, state: "stopped"}` once the node's last tick is out.
    pub fn probe_stop(&self, node: &str, peer: Option<&str>) -> Result<Value, CtlError> {
        self.probe_node("stop", node, peer)
    }

    /// `POST /v1/probe/touch`: renews the node's lease (`ttl_ms`), refused on a node without
    /// one. Answers `{node, state, ttl_ms, expires_in_ms?}`, the time left only on a running node.
    pub fn probe_touch(&self, node: &str, peer: Option<&str>) -> Result<Value, CtlError> {
        self.probe_node("touch", node, peer)
    }

    /// `POST /v1/probe/remove`: stops and forgets the node and its whole subtree. Answers
    /// `{node, removed}`, the subtree counted.
    pub fn probe_remove(&self, node: &str, peer: Option<&str>) -> Result<Value, CtlError> {
        self.probe_node("remove", node, peer)
    }

    /// The four node ops share one wire shape: `POST /v1/probe/<op>` with `{node, peer?}`, the
    /// peer in the body as `stop`'s is. A name nobody declared is the plane's refusal.
    pub(crate) fn probe_node(
        &self,
        op: &str,
        node: &str,
        peer: Option<&str>,
    ) -> Result<Value, CtlError> {
        let mut body = json!({ "node": node });
        if let Some(peer) = peer {
            body["peer"] = json!(peer);
        }
        self.call("POST", &format!("/v1/probe/{op}"), Some(&body))
    }

    /// `POST /v1/wait` for one generation of one boot. `timeout` is an `Ok(Timeout)` outcome;
    /// transport and HTTP refusals remain errors. With `peer`, that origin decides the wait.
    pub fn wait_generation_outcome(
        &self,
        epoch: &str,
        generation: u64,
        timeout: Duration,
        peer: Option<&str>,
    ) -> Result<GenerationOutcome, CtlError> {
        let request = json!({"wait": "generation", "epoch": epoch, "generation": generation});
        let reply = self.plane_wait(request, timeout, peer)?;
        decode_generation(&reply)
    }

    /// `POST /v1/wait` for the launch decision of `service` in one generation and boot.
    pub fn wait_launch(
        &self,
        epoch: &str,
        generation: u64,
        service: &str,
        timeout: Duration,
        peer: Option<&str>,
    ) -> Result<LaunchOutcome, CtlError> {
        let request = json!({
            "wait": "launch", "epoch": epoch, "generation": generation, "service": service,
        });
        let reply = self.plane_wait(request, timeout, peer)?;
        decode_launch(&reply)
    }

    /// `POST /v1/wait` for one exact run. [`RunOutcome::Ended`] keeps the plane's `end`
    /// object as [`serde_json::Value`].
    pub fn wait_run(
        &self,
        run: &RunId,
        timeout: Duration,
        peer: Option<&str>,
    ) -> Result<RunOutcome, CtlError> {
        let request = json!({"wait": "run", "run": run.json()});
        let reply = self.plane_wait(request, timeout, peer)?;
        decode_run(&reply)
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
    #[deprecated(
        note = "number-only wait is unsafe across restart; use wait_generation_of or wait_generation_outcome"
    )]
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

    /// A plane-side wait follows the remembered state file only for errors that can denote the
    /// old boot: I/O, its new token's 401, or its shutdown 503. Through a peer a 503 is the link
    /// to that peer, so it does not count there.
    fn plane_wait_may_have_moved(&self, err: &CtlError, peer: Option<&str>) -> bool {
        self.plane_may_have_moved(err)
            || (self.state.is_some()
                && peer.is_none()
                && matches!(err, CtlError::Http { status: 503, .. }))
    }

    /// Calls the plane's bounded wait. The semantic deadline remains `timeout`; the socket read
    /// gets a short grace so the plane can serialize its `timeout` value after that deadline.
    fn plane_wait(
        &self,
        mut request: Value,
        timeout: Duration,
        peer: Option<&str>,
    ) -> Result<Value, CtlError> {
        let timeout_ms = duration_millis(timeout)?;
        let now = Instant::now();
        let deadline = now
            .checked_add(timeout)
            .ok_or_else(|| CtlError::Argument("wait timeout is too large".into()))?;
        let read_deadline = deadline
            .checked_add(WAIT_REPLY_GRACE)
            .ok_or_else(|| CtlError::Argument("wait timeout is too large".into()))?;
        let mut me = self.clone();
        let mut first = true;
        loop {
            let request_timeout = if first {
                timeout_ms
            } else {
                let remaining = left(deadline);
                if remaining.is_zero() {
                    return Err(CtlError::Io(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "plane-side wait deadline passed while following the state file",
                    )));
                }
                duration_millis(remaining)?.max(1)
            };
            first = false;
            request["timeout_ms"] = json!(request_timeout);
            let budget = left(read_deadline);
            match me.call_within(
                "POST",
                &routed("/v1/wait", peer),
                Some(&request),
                Some(budget),
                Some(budget),
            ) {
                Ok(reply) => return Ok(reply),
                Err(e) if me.plane_wait_may_have_moved(&e, peer) => {
                    me.follow_state(e, deadline, true)?
                }
                Err(e) => return Err(e),
            }
        }
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

    /// After `err` from a plane that may have moved: `Ok` once this handle names what the state
    /// file names now. Polling waits pause once before retrying the same plane; plane-side waits
    /// keep following until the file moves. `Err(err)` is returned at the deadline.
    fn follow_state(
        &mut self,
        err: CtlError,
        deadline: Instant,
        wait_for_move: bool,
    ) -> Result<(), CtlError> {
        loop {
            match self.refreshed() {
                Some(fresh) if Instant::now() < deadline => {
                    *self = fresh;
                    return Ok(());
                }
                Some(_) => return Err(err),
                None if pause(deadline) => return Err(err),
                None if wait_for_move => continue,
                None => return Ok(()),
            }
        }
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
                Err(e) if epoch.is_some() && me.plane_may_have_moved(&e) => {
                    me.follow_state(e, deadline, false)?;
                    continue;
                }
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
        self.wait_service_on(None, service, None, timeout, |e| nodes_decision(e, nodes))
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
        self.wait_service_on(Some(peer), service, None, timeout, |e| {
            nodes_decision(e, nodes)
        })
    }

    /// Waits until `service` has exited, and answers its entry in the status: state `stopped`
    /// or `failed`, with `last_exit` saying how (`code`, `at` in Unix milliseconds, what the
    /// run cost as the OS counted it, `peak_memory_bytes`, `cpu_ms` and `peak_processes`, and
    /// `reason` when the run went past a limit of its spec: `wall_limit` for a run that lasted
    /// past `limits.wall_ms`, `memory_limit` for one that held more than `limits.memory_mb`,
    /// `cpu_limit` for one that used more processor time than `limits.cpu_ms`, `process_limit`
    /// for one that held more processes at once than `limits.processes`. The plane kills such
    /// a run; one that went past `cpu_ms` or `processes` and ended before the plane's next
    /// poll carries the reason too, read from the figure it reports. `reason` is absent when
    /// the run ended on its own within its limits, or was stopped on request).
    ///
    /// This is how a job is waited for: a service declared with `restart: never` runs to its
    /// end and the plane leaves it `stopped`, its `desired` still `running`. Converging the
    /// generation that started it ([`Ctl::wait_generation`]) only says it was launched; this
    /// wait says it ended. A service that restarts on failure is `restarting` between two
    /// runs and is waited through; it has exited for good once it is `failed`.
    ///
    /// The exit answered is the last one the plane saw for the service, whichever run it
    /// ended: between an apply and its launch that is still the earlier run's. To wait for one
    /// run's exit, use [`Ctl::wait_exit_of`] with the generation that started it. A service
    /// that never ran has no `last_exit`, and the wait runs out with [`CtlError::Wait`].
    pub fn wait_exit(&self, service: &str, timeout: Duration) -> Result<Value, CtlError> {
        self.wait_service_on(None, service, None, timeout, |e| exit_decision(e, None))
    }

    /// [`Ctl::wait_exit`] for a service of `peer`, read from that peer's status. A 502 or 503
    /// (the link to the peer is down for a moment) is asked again until the deadline.
    pub fn wait_exit_at(
        &self,
        peer: &str,
        service: &str,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        self.wait_service_on(Some(peer), service, None, timeout, |e| {
            exit_decision(e, None)
        })
    }

    /// [`Ctl::wait_exit`] for the run that `generation` started: the plane stamps each exit
    /// with the generation of the apply that launched the run it ends (`last_exit.generation`,
    /// kept across the run's restarts), so an earlier run's exit, still in the status while
    /// this run is about to be launched, is pending rather than answered. That closes the
    /// window between an apply and its launch that [`Ctl::wait_exit`] leaves open. With
    /// `epoch` (the apply reply's), a plane that reports another boot ends the wait with
    /// [`CtlError::Restarted`], as [`Ctl::wait_generation_of`] does. A plane whose exits name
    /// no generation is waited on as [`Ctl::wait_exit`] does.
    ///
    /// With `epoch`, a handle made with [`Ctl::from_state`] follows its state file across the
    /// restart, as [`Ctl::wait_generation_of`] does, so the restart is reported rather than
    /// `plane unreachable`.
    ///
    /// Only a generation whose apply had a `start` action for the service launched a run: an
    /// apply that found the service already starting, running or restarting says noop and
    /// launches nothing, so a wait on that generation's run stays pending until `timeout`.
    /// Check `actions` in the apply reply, or the service's `generation` in the status.
    pub fn wait_exit_of(
        &self,
        service: &str,
        generation: u64,
        epoch: Option<&str>,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        let scope = epoch.map(|e| (generation, e));
        self.wait_service_on(None, service, scope, timeout, |e| {
            exit_decision(e, Some(generation))
        })
    }

    /// [`Ctl::wait_exit_of`] for a service of `peer` (the generation and epoch are the
    /// peer's, from an apply made through it), read from that peer's status.
    pub fn wait_exit_of_at(
        &self,
        peer: &str,
        service: &str,
        generation: u64,
        epoch: Option<&str>,
        timeout: Duration,
    ) -> Result<Value, CtlError> {
        let scope = epoch.map(|e| (generation, e));
        self.wait_service_on(Some(peer), service, scope, timeout, |e| {
            exit_decision(e, Some(generation))
        })
    }

    /// Polls the status until `decide` is done with `service`'s entry, which is answered;
    /// `Failed` ends the wait at once, `Pending` says what the timeout message reports. With
    /// `scope`, a generation and the boot it belongs to, a status that reports another boot
    /// ends the wait with [`CtlError::Restarted`], and the state file is followed across a
    /// restart as in `wait_gen`.
    fn wait_service_on(
        &self,
        peer: Option<&str>,
        service: &str,
        scope: Option<(u64, &str)>,
        timeout: Duration,
        decide: impl Fn(&Value) -> Decision,
    ) -> Result<Value, CtlError> {
        let deadline = Instant::now() + timeout;
        let mut me = self.clone();
        loop {
            let status = match me.status_within(peer, Some(left(deadline))) {
                Ok(status) => status,
                Err(e) if scope.is_some() && me.plane_may_have_moved(&e) => {
                    me.follow_state(e, deadline, false)?;
                    continue;
                }
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
            if let (Some((generation, want)), Some(now)) =
                (scope, status.get("epoch").and_then(Value::as_str))
            {
                if now != want {
                    return Err(CtlError::Restarted {
                        generation,
                        epoch: want.to_string(),
                        now: now.to_string(),
                    });
                }
            }
            let entry = status
                .get("services")
                .and_then(Value::as_array)
                .ok_or_else(|| CtlError::Protocol("status has no `services`".into()))?
                .iter()
                .find(|s| s.get("name").and_then(Value::as_str) == Some(service))
                .ok_or_else(|| CtlError::NotDeclared(service.to_string()))?;
            let pending = match decide(entry) {
                Decision::Done => return Ok(entry.clone()),
                Decision::Failed(why) => {
                    return Err(CtlError::Wait(format!("service `{service}` {why}")))
                }
                Decision::Pending(what) => what,
            };
            if pause(deadline) {
                return Err(CtlError::Wait(format!(
                    "service `{service}` {pending} after {} ms",
                    timeout.as_millis()
                )));
            }
        }
    }

    fn call(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value, CtlError> {
        self.call_within(method, path, body, None, None)
    }

    /// One call; `within` and `read` are [`http::request`]'s.
    fn call_within(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        within: Option<Duration>,
        read: Option<Duration>,
    ) -> Result<Value, CtlError> {
        let addr = self.loopback()?;
        let body = body.map(|b| b.to_string().into_bytes());
        let r = http::request(
            addr,
            method,
            path,
            &self.token,
            body.as_deref(),
            within,
            read,
        )?;
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

/// What one status read says about the service a wait is for.
enum Decision {
    /// The condition holds: the wait answers this entry.
    Done,
    /// It never will: the wait fails with this reason.
    Failed(String),
    /// Not yet: what the wait would report if it ran out now.
    Pending(String),
}

/// [`Ctl::wait_nodes`]: running with at least `nodes` nodes; `failed` never will.
fn nodes_decision(entry: &Value, nodes: usize) -> Decision {
    let state = entry.get("state").and_then(Value::as_str).unwrap_or("");
    let have = entry
        .get("nodes")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    match state {
        "running" if have >= nodes => Decision::Done,
        "failed" => Decision::Failed(format!("failed with {have} of {nodes} nodes")),
        _ => Decision::Pending(format!("has {have} of {nodes} nodes in state `{state}`")),
    }
}

/// [`Ctl::wait_exit`]: `stopped` or `failed` with a `last_exit`. A `stopped` service without
/// one never ran, and a `restarting` one is between two runs: both are pending. With
/// `generation`, an exit the plane stamps with another generation is an earlier run's and is
/// pending too; an exit without one (an older plane) is taken as it is.
fn exit_decision(entry: &Value, generation: Option<u64>) -> Decision {
    let state = entry.get("state").and_then(Value::as_str).unwrap_or("");
    let exit = entry.get("last_exit").filter(|e| e.is_object());
    match (state, exit) {
        ("stopped" | "failed", Some(exit)) => {
            match (generation, exit.get("generation").and_then(Value::as_u64)) {
                (Some(want), Some(got)) if got != want => Decision::Pending(format!(
                    "last exited from generation {got}'s run, not generation {want}'s"
                )),
                _ => Decision::Done,
            }
        }
        _ => Decision::Pending(format!("has not exited: state `{state}`")),
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
fn duration_millis(duration: Duration) -> Result<u64, CtlError> {
    u64::try_from(duration.as_millis())
        .map_err(|_| CtlError::Argument("wait timeout does not fit `timeout_ms`".into()))
}

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
