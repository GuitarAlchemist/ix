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
//! `/v1/status` for now; they move to a subscription when the plane's events are read here.

mod http;
mod ixql;

pub use ixql::register;

use serde_json::{json, Value};
use std::fmt;
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
        }
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
}

impl fmt::Debug for Ctl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The token is a credential: it never reaches a log through `{:?}`.
        f.debug_struct("Ctl")
            .field("addr", &self.addr)
            .field("token", &"<redacted>")
            .finish()
    }
}

impl Ctl {
    /// A plane at `addr` (`127.0.0.1:PORT`) that accepts `token`.
    pub fn new(addr: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            token: token.into(),
        }
    }

    /// The plane described by its state file, `{"name", "addr", "pid", "token"}`, which
    /// `ixctl up` writes on ready and removes on a clean exit.
    pub fn from_state(path: impl AsRef<Path>) -> Result<Self, CtlError> {
        let path = path.as_ref();
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
        self.call("GET", "/v1/status", None)
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
    pub fn wait_generation(&self, generation: u64, timeout: Duration) -> Result<Value, CtlError> {
        let deadline = Instant::now() + timeout;
        loop {
            let status = self.status()?;
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
        let deadline = Instant::now() + timeout;
        loop {
            let status = self.status()?;
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
        let addr = self.loopback()?;
        let body = body.map(|b| b.to_string().into_bytes());
        let r = http::request(addr, method, path, &self.token, body.as_deref())?;
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
    let mut out = format!("{path}?peer=");
    for b in peer.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Sleeps one poll, or less near `deadline`. True when the deadline has passed.
fn pause(deadline: Instant) -> bool {
    let now = Instant::now();
    if now >= deadline {
        return true;
    }
    std::thread::sleep(POLL.min(deadline - now));
    false
}
