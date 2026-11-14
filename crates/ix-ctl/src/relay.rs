//! A request/reply adapter for ixrelay (ix#473): one bounded request to a plane, one typed
//! [`Outcome`], for a future MCP tool or the TUI to hand on as it is.
//!
//! It is a client only. The plane owns the operations, the identities and any queue; this
//! module keeps nothing and never retries.
//!
//! - `ping` is real. It reads `GET /v1/status`, or with a target that peer's status as the plane
//!   relays it (`?peer=`), and checks the consumer contract the status announces. The plane's
//!   own `ping` op is spoken only on its WebSocket, which this crate does not speak.
//! - Every other op, `deliver` included, is [`Outcome::Unsupported`] and sends nothing: the
//!   plane has no operation that hands a request to a Claude session.
//!
//! The caller's `id` is echoed, not sent: the plane's HTTP ops carry no request id, and one
//! connection per request is what ties the answer to the request.

use crate::{http, Ctl, CtlError};
use serde_json::{json, Value};
use std::time::Duration;

/// The consumer contract `ping` accepts in the plane's status.
pub const PLANE_CONTRACT: &str = "run/1";
/// The name of this adapter's own reply shape, in [`Reply::to_json`].
pub const REPLY_CONTRACT: &str = "ixrelay.adapter/0";
/// The largest answer the adapter reads; past it the answer is [`Outcome::TooLarge`].
pub const MAX_REPLY_BYTES: usize = 1 << 20;
/// The longest deadline a request may ask for.
pub const MAX_DEADLINE: Duration = Duration::from_secs(30);
/// The longest caller id, and the longest message an outcome carries.
pub const MAX_TEXT_BYTES: usize = 200;

/// One request: the caller's correlation id, the op, the peer it is for (`None`: the plane
/// itself), and how long the whole exchange may take.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: String,
    pub op: String,
    pub target: Option<String>,
    pub deadline: Duration,
}

/// What came of one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The target answered: its own name and the boot it is in.
    Answered { name: String, epoch: String },
    /// No backend for this op; nothing was sent.
    Unsupported { reason: String },
    /// The request itself is not admissible; nothing was sent.
    Invalid { reason: String },
    /// The connection never opened (nothing listens, or the connect ran out); nothing was sent.
    Unreachable,
    /// The plane answered with a refusal.
    Refused { status: u16, message: String },
    /// The plane keeps another consumer contract (`None`: it names none).
    WrongContract { got: Option<String> },
    /// The answer is not what the contract says.
    Malformed { reason: String },
    /// The answer is larger than [`MAX_REPLY_BYTES`]; it was not read.
    TooLarge,
    /// No complete answer: the deadline passed or the link was lost after the request may
    /// have left. What the plane did is not known, and the adapter does not ask again.
    Indeterminate { reason: String },
}

/// The outcome with the request's identity beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub id: String,
    pub op: String,
    pub target: Option<String>,
    pub outcome: Outcome,
}

impl Reply {
    /// The reply as one JSON object: `contract`, `id`, `op`, `target` and `outcome`, with the
    /// outcome's fields beside it.
    pub fn to_json(&self) -> Value {
        let (outcome, fields) = match &self.outcome {
            Outcome::Answered { name, epoch } => {
                ("answered", json!({ "name": name, "epoch": epoch }))
            }
            Outcome::Unsupported { reason } => ("unsupported", json!({ "reason": reason })),
            Outcome::Invalid { reason } => ("invalid", json!({ "reason": reason })),
            Outcome::Unreachable => ("unreachable", json!({})),
            Outcome::Refused { status, message } => {
                ("refused", json!({ "status": status, "message": message }))
            }
            Outcome::WrongContract { got } => ("wrong_contract", json!({ "got": got })),
            Outcome::Malformed { reason } => ("malformed", json!({ "reason": reason })),
            Outcome::TooLarge => ("too_large", json!({})),
            Outcome::Indeterminate { reason } => ("indeterminate", json!({ "reason": reason })),
        };
        let mut v = json!({
            "contract": REPLY_CONTRACT,
            "id": self.id,
            "op": self.op,
            "target": self.target,
            "outcome": outcome,
        });
        if let Value::Object(fields) = fields {
            for (k, x) in fields {
                v[k] = x;
            }
        }
        v
    }
}

/// Runs one request against `ctl`'s plane.
pub fn request(ctl: &Ctl, req: &Request) -> Reply {
    Reply {
        id: req.id.clone(),
        op: req.op.clone(),
        target: req.target.clone(),
        outcome: outcome(ctl, req),
    }
}

fn outcome(ctl: &Ctl, req: &Request) -> Outcome {
    let invalid = |reason: &str| Outcome::Invalid {
        reason: reason.into(),
    };
    if req.id.is_empty() || req.id.len() > MAX_TEXT_BYTES {
        return invalid("id must be 1 to 200 bytes");
    }
    if req.deadline.is_zero() || req.deadline > MAX_DEADLINE {
        return invalid("deadline must be above zero and at most 30 s");
    }
    if req.target.as_deref() == Some("") {
        return invalid("a named target must not be empty");
    }
    if req.op != "ping" {
        return Outcome::Unsupported {
            reason: bounded(&format!(
                "no plane operation behind `{}`: only `ping` is wired",
                req.op
            )),
        };
    }
    let path = crate::routed("/v1/status", req.target.as_deref());
    match ctl.call_bounded(
        "GET",
        &path,
        None,
        Some(req.deadline),
        None,
        MAX_REPLY_BYTES,
    ) {
        Ok(status) => answered(&status),
        Err(e) => failed(e),
    }
}

/// `ping`'s answer: the contract the status announces, then who answered.
fn answered(status: &Value) -> Outcome {
    match status.get("contract") {
        None | Some(Value::Null) => return Outcome::WrongContract { got: None },
        Some(Value::String(c)) if c == PLANE_CONTRACT => {}
        Some(Value::String(c)) => {
            return Outcome::WrongContract {
                got: Some(bounded(c)),
            }
        }
        Some(_) => {
            return Outcome::Malformed {
                reason: "status `contract` is not a string".into(),
            }
        }
    }
    let text = |name: &str| {
        status
            .get(name)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(bounded)
    };
    match (text("name"), text("epoch")) {
        (Some(name), Some(epoch)) => Outcome::Answered { name, epoch },
        _ => Outcome::Malformed {
            reason: "status has no non-empty string `name` and `epoch`".into(),
        },
    }
}

fn failed(e: CtlError) -> Outcome {
    match e {
        CtlError::Io(e) if e.get_ref().is_some_and(|i| i.is::<http::NotConnected>()) => {
            Outcome::Unreachable
        }
        CtlError::Io(e) if e.get_ref().is_some_and(|i| i.is::<http::TooLarge>()) => {
            Outcome::TooLarge
        }
        CtlError::Io(e) => Outcome::Indeterminate {
            reason: bounded(&e.to_string()),
        },
        CtlError::Http { status, message } => Outcome::Refused {
            status,
            message: bounded(&message),
        },
        CtlError::NotLoopback(_) => Outcome::Invalid {
            reason: "the plane's address is not a loopback one".into(),
        },
        other => Outcome::Malformed {
            reason: bounded(&other.to_string()),
        },
    }
}

/// `s` cut to [`MAX_TEXT_BYTES`] on a character boundary.
fn bounded(s: &str) -> String {
    let mut end = s.len().min(MAX_TEXT_BYTES);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}
