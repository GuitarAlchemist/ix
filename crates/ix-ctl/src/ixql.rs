//! The `ix.ctl.*` capabilities: the plane's verbs as IXQL calls.
//!
//! ```text
//! desired <- { services: { "pod-a": { state: "running" } } }
//! planned <- desired → ix.ctl.plan
//! applied <- ix.ctl.apply(desired)
//! on_b    <- ix.ctl.apply(desired, peer: "b")
//! done    <- ix.ctl.wait(generation: applied.generation, epoch: applied.epoch, timeout_ms: 10000)
//! pod     <- ix.ctl.wait(service: "pod-a", nodes: 4, timeout_ms: 10000)
//! status  <- ix.ctl.status()
//! stopped <- ix.ctl.stop(service: "pod-a")
//! seen    <- ix.ctl.events(channel: "ctl", since: 0, max: 20, timeout_ms: 2000)
//! ```
//!
//! `ix.ctl.events` reads the plane's event stream and stops at `max:` events, at the end of
//! the stream, or after `timeout_ms:` without a next event (the plane's keep-alive pings do
//! not count as one): `{events[], gap, ended}`. `max: 0` opens nothing; `epoch:` needs `since:`.
//!
//! Arguments are checked before any request: a misspelt name, a missing timeout or a value
//! piped into a call that would ignore it fails the run instead of being dropped.

use crate::{Ctl, CtlError};
use ix_ixql::{CallArgs, Capabilities, Produced, RegistrationError};
use serde_json::{json, Value};
use std::time::Duration;

/// Registers `ix.ctl.status`, `ix.ctl.plan`, `ix.ctl.apply`, `ix.ctl.stop`, `ix.ctl.wait` and
/// `ix.ctl.events` against `ctl`.
pub fn register(caps: &mut Capabilities, ctl: Ctl) -> Result<(), RegistrationError> {
    let c = ctl.clone();
    caps.register("ix.ctl.status", move |a: CallArgs| {
        expect(&a, &["peer"])?;
        answer(c.status_of(string(&a, "peer")?.as_deref()))
    })?;

    let c = ctl.clone();
    caps.register("ix.ctl.plan", move |a: CallArgs| {
        let (doc, peer) = document(a)?;
        answer(c.plan(&doc, peer.as_deref()))
    })?;

    let c = ctl.clone();
    caps.register("ix.ctl.apply", move |a: CallArgs| {
        let (doc, peer) = document(a)?;
        answer(c.apply(&doc, peer.as_deref()))
    })?;

    let c = ctl.clone();
    caps.register("ix.ctl.stop", move |a: CallArgs| {
        expect(&a, &["service", "peer"])?;
        let service = string(&a, "service")?.ok_or("needs `service:`")?;
        answer(c.stop(&service, string(&a, "peer")?.as_deref()))
    })?;

    let c = ctl.clone();
    caps.register("ix.ctl.events", move |a: CallArgs| {
        expect(&a, &["channel", "since", "epoch", "max", "timeout_ms"])?;
        let idle = Duration::from_millis(count(&a, "timeout_ms")?.ok_or("needs `timeout_ms:`")?);
        let max = count(&a, "max")?.ok_or("needs `max:`")? as usize;
        let channel = string(&a, "channel")?.unwrap_or_default();
        let since = count(&a, "since")?;
        let epoch = string(&a, "epoch")?;
        if epoch.is_some() && since.is_none() {
            return Err(
                "`epoch:` needs `since`: it says which boot that sequence number belongs to".into(),
            );
        }
        if max == 0 {
            // Nothing asked for: nothing opened, nothing waited for.
            return Ok(Produced::plain(
                json!({"events": [], "gap": Value::Null, "ended": false}),
            ));
        }
        let stream = c
            .events_within(&channel, since, epoch.as_deref(), idle)
            .map_err(|e| e.to_string())?;
        let (mut events, mut gap, mut ended) = (Vec::new(), Value::Null, true);
        for item in stream {
            match item {
                Ok(e) => {
                    events.push(Value::from(&e));
                    if events.len() >= max {
                        ended = false;
                        break;
                    }
                }
                Err(CtlError::Gap {
                    reason,
                    from,
                    epoch,
                }) => gap = json!({"reason": reason, "from": from, "epoch": epoch}),
                // The plane went quiet for `timeout_ms`: what came so far is the answer.
                Err(CtlError::Io(e))
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    ended = false;
                    break;
                }
                Err(e) => return Err(e.to_string()),
            }
        }
        Ok(Produced::plain(
            json!({"events": events, "gap": gap, "ended": ended}),
        ))
    })?;

    caps.register("ix.ctl.wait", move |a: CallArgs| {
        expect(&a, &["generation", "epoch", "service", "nodes", "peer", "timeout_ms"])?;
        let timeout = Duration::from_millis(count(&a, "timeout_ms")?.ok_or("needs `timeout_ms:`")?);
        let peer = string(&a, "peer")?;
        match (
            count(&a, "generation")?,
            string(&a, "epoch")?,
            string(&a, "service")?,
            count(&a, "nodes")?,
        ) {
            (Some(g), epoch, None, None) => answer(match (peer.as_deref(), epoch.as_deref()) {
                (Some(p), epoch) => ctl.wait_generation_at(p, epoch, g, timeout),
                (None, Some(e)) => ctl.wait_generation_of(e, g, timeout),
                (None, None) => ctl.wait_generation(g, timeout),
            }),
            (None, None, Some(s), Some(n)) => answer(match peer.as_deref() {
                Some(p) => ctl.wait_nodes_at(p, &s, n as usize, timeout),
                None => ctl.wait_nodes(&s, n as usize, timeout),
            }),
            _ => Err("takes `generation:` (with `epoch:`), or `service:` with `nodes:`, and `timeout_ms:`; `peer:` reads that peer".into()),
        }
    })?;
    Ok(())
}

fn answer(r: Result<Value, crate::CtlError>) -> Result<Produced, String> {
    r.map(Produced::plain).map_err(|e| e.to_string())
}

/// No piped value, no positional one, and no named argument outside `named`.
fn expect(a: &CallArgs, named: &[&str]) -> Result<(), String> {
    if a.piped.is_some() {
        return Err("does not take a piped value".into());
    }
    if !a.positional.is_empty() {
        return Err("takes named arguments only".into());
    }
    expect_named(a, named)
}

/// The desired-state document, the piped value or the one positional value, and the peer it
/// is for (`peer:`), if any. The peer rides beside the document, never inside it.
fn document(a: CallArgs) -> Result<(Value, Option<String>), String> {
    expect_named(&a, &["peer"])?;
    let peer = string(&a, "peer")?;
    let doc = match (a.piped, a.positional.len()) {
        (Some(v), 0) => v,
        (None, 1) => a.positional.into_iter().next().expect("one value"),
        (None, 0) => return Err("needs a desired-state document, piped or as its argument".into()),
        _ => return Err("takes one desired-state document, piped or as its argument".into()),
    };
    if !doc.get("services").is_some_and(Value::is_object) {
        return Err("the desired-state document needs a `services` record".into());
    }
    Ok((doc, peer))
}

/// No named argument outside `named`.
fn expect_named(a: &CallArgs, named: &[&str]) -> Result<(), String> {
    match a.named.keys().find(|k| !named.contains(&k.as_str())) {
        Some(k) => Err(format!("has no argument `{k}`")),
        None => Ok(()),
    }
}

fn string(a: &CallArgs, name: &str) -> Result<Option<String>, String> {
    match a.named.get(name) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!("`{name}:` must be a string")),
    }
}

fn count(a: &CallArgs, name: &str) -> Result<Option<u64>, String> {
    match a.named.get(name) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("`{name}:` must be a whole number, not {v}")),
    }
}
