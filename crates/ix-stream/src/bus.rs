//! The algedonic bus: events that must never hold the graph back.
//!
//! A publisher never waits. Each subscriber has its own bounded queue, and an event that finds
//! it full is lost for that subscriber and counted. Signals of pleasure and pain (an answer
//! judged right or wrong, money spent, a budget running ahead of its pace) travel here, beside
//! the monitor's stats, so feedback can go from a judge back to a gate without a channel cycle
//! that could wedge.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// One event: its kind, when it was published, and what it says.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub kind: String,
    /// Seconds since the bus was made.
    pub t: f64,
    pub data: Value,
}

impl Event {
    pub fn json(&self) -> Value {
        json!({ "kind": self.kind, "t": (self.t * 1000.0).round() / 1000.0, "data": self.data })
    }
}

/// A signal of pleasure (`valence` +1) or pain (−1), from a node, about an item or not.
#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    pub node: String,
    pub valence: i8,
    /// How strong, on a scale the emitter states in `reason`; 1 for an answer judged.
    pub intensity: f64,
    pub reason: String,
    pub item: Option<String>,
}

impl Signal {
    pub fn pleasure(node: &str, intensity: f64, reason: &str) -> Self {
        Self::new(node, 1, intensity, reason)
    }

    pub fn pain(node: &str, intensity: f64, reason: &str) -> Self {
        Self::new(node, -1, intensity, reason)
    }

    fn new(node: &str, valence: i8, intensity: f64, reason: &str) -> Self {
        Self {
            node: node.to_string(),
            valence,
            intensity,
            reason: reason.to_string(),
            item: None,
        }
    }

    pub fn about(mut self, item: &str) -> Self {
        self.item = Some(item.to_string());
        self
    }

    pub fn json(&self) -> Value {
        json!({
            "node": self.node, "valence": self.valence, "intensity": self.intensity,
            "reason": self.reason, "item": self.item,
        })
    }

    pub fn from_json(v: &Value) -> Option<Self> {
        Some(Self {
            node: v.get("node")?.as_str()?.to_string(),
            valence: v.get("valence")?.as_i64()?.signum() as i8,
            intensity: v.get("intensity")?.as_f64()?,
            reason: v.get("reason")?.as_str()?.to_string(),
            item: v.get("item").and_then(Value::as_str).map(str::to_string),
        })
    }
}

struct Sub {
    kinds: Vec<String>,
    tx: SyncSender<Event>,
}

struct Inner {
    t0: Instant,
    subs: Mutex<Vec<Sub>>,
    published: AtomicU64,
    lost: AtomicU64,
}

/// The bus. Clones share it.
#[derive(Clone)]
pub struct Bus(Arc<Inner>);

impl Default for Bus {
    fn default() -> Self {
        Self::new()
    }
}

impl Bus {
    pub fn new() -> Self {
        Self(Arc::new(Inner {
            t0: Instant::now(),
            subs: Mutex::new(Vec::new()),
            published: AtomicU64::new(0),
            lost: AtomicU64::new(0),
        }))
    }

    /// Events of `kinds` (every kind when empty), queued up to `cap` for this subscriber.
    /// Dropping the receiver ends the subscription.
    pub fn subscribe(&self, kinds: &[&str], cap: usize) -> Receiver<Event> {
        let (tx, rx) = sync_channel(cap.max(1));
        self.0
            .subs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Sub {
                kinds: kinds.iter().map(|k| k.to_string()).collect(),
                tx,
            });
        rx
    }

    /// Publishes without waiting: a subscriber whose queue is full loses this event.
    pub fn publish(&self, kind: &str, data: Value) {
        let event = Event {
            kind: kind.to_string(),
            t: self.elapsed(),
            data,
        };
        self.0.published.fetch_add(1, Ordering::Relaxed);
        let mut subs = self.0.subs.lock().unwrap_or_else(|e| e.into_inner());
        subs.retain(|s| {
            if !s.kinds.is_empty() && !s.kinds.contains(&event.kind) {
                return true;
            }
            match s.tx.try_send(event.clone()) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    self.0.lost.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(TrySendError::Disconnected(_)) => false,
            }
        });
    }

    pub fn signal(&self, s: &Signal) {
        self.publish("signal", s.json());
    }

    /// Seconds since the bus was made.
    pub fn elapsed(&self) -> f64 {
        self.0.t0.elapsed().as_secs_f64()
    }

    pub fn published(&self) -> u64 {
        self.0.published.load(Ordering::Relaxed)
    }

    /// Events lost to full subscribers, all subscribers together.
    pub fn lost(&self) -> u64 {
        self.0.lost.load(Ordering::Relaxed)
    }
}
