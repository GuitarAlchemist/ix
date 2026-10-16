//! JSON streams through a graph of threads joined by bounded channels.
//!
//! Each node is one or more threads, and its input is a channel that holds at most `cap`
//! values. A send to a full channel waits: that wait is the back-pressure. A slow node holds
//! back everything upstream of it, down to the source, instead of letting a queue grow. It is
//! the `request(n)` of Reactive Streams in its plainest form, `n` being the free slots.
//!
//! Cancellation runs the other way. When a node stops ([`Out::stop`]) or its workers all end,
//! its input is dropped and every send to it fails. A node whose every link leads to nodes
//! that ended stops too, so the stop runs up to the source. A
//! [`source::duckdb`] source then kills its `duckdb` process, so a query nobody reads any more
//! stops computing.
//!
//! Nothing flows back through the channels. Feedback (quality, cost, budget) goes on the
//! [`Bus`], which never blocks a publisher, so a loop from a judge back to a gate cannot wedge
//! the graph. [`petri::check`] tells, before a run, whether the channels themselves can.

pub mod bus;
pub mod gate;
pub mod petri;
pub mod source;
pub mod sse;

pub use bus::{Bus, Event, Signal};

use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long an idle worker waits before it looks at its node's stop flag again.
const POLL: Duration = Duration::from_millis(50);

/// A node's counters, read by the monitor while the graph runs.
#[derive(Debug)]
struct Meter {
    name: String,
    cap: usize,
    workers: usize,
    queued: AtomicI64,
    max_queued: AtomicI64,
    received: AtomicU64,
    emitted: AtomicU64,
    busy_ns: AtomicU64,
    blocked_ns: AtomicU64,
    refused: AtomicU64,
    stop: AtomicBool,
    /// Workers still running.
    live: AtomicUsize,
    /// Every worker ended: nothing sent here will be read.
    gone: AtomicBool,
}

impl Meter {
    fn new(name: &str, cap: usize, workers: usize) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            cap,
            workers,
            queued: AtomicI64::new(0),
            max_queued: AtomicI64::new(0),
            received: AtomicU64::new(0),
            emitted: AtomicU64::new(0),
            busy_ns: AtomicU64::new(0),
            blocked_ns: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            live: AtomicUsize::new(workers),
            gone: AtomicBool::new(false),
        })
    }

    /// One worker ended; the last one marks the node gone.
    fn left(&self) {
        if self.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.gone.store(true, Ordering::Release);
        }
    }

    fn arrived(&self) {
        let now = self.queued.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_queued.fetch_max(now, Ordering::Relaxed);
    }

    fn stats(&self) -> NodeStats {
        NodeStats {
            name: self.name.clone(),
            cap: self.cap,
            workers: self.workers,
            queued: self.queued.load(Ordering::Relaxed).max(0),
            max_queued: self.max_queued.load(Ordering::Relaxed),
            received: self.received.load(Ordering::Relaxed),
            emitted: self.emitted.load(Ordering::Relaxed),
            busy_ms: self.busy_ns.load(Ordering::Relaxed) as f64 / 1e6,
            blocked_ms: self.blocked_ns.load(Ordering::Relaxed) as f64 / 1e6,
            refused: self.refused.load(Ordering::Relaxed),
        }
    }
}

/// What a node did, at one moment.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeStats {
    pub name: String,
    /// The capacity of its input; 0 for a source.
    pub cap: usize,
    pub workers: usize,
    /// Values waiting in its input.
    pub queued: i64,
    /// The most that ever waited there. A send counts after it lands, so this can pass `cap`
    /// by the one value a worker has just taken out and not yet counted.
    pub max_queued: i64,
    pub received: u64,
    pub emitted: u64,
    /// Time its workers spent on values.
    pub busy_ms: f64,
    /// Time its sends spent waiting for a full channel downstream.
    pub blocked_ms: f64,
    /// Values it could not place: `try_emit` to a full or closed node, or a lossy port.
    pub refused: u64,
}

impl NodeStats {
    pub fn json(&self) -> Value {
        json!({
            "name": self.name, "cap": self.cap, "workers": self.workers,
            "queued": self.queued, "max_queued": self.max_queued,
            "received": self.received, "emitted": self.emitted,
            "busy_ms": round(self.busy_ms), "blocked_ms": round(self.blocked_ms),
            "refused": self.refused,
        })
    }
}

fn round(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// The node downstream has stopped: nothing sent to it will be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed;

/// Why [`Out::try_emit`] did not place a value; the value comes back.
#[derive(Debug)]
pub enum Refused {
    Full(Value),
    Closed(Value),
}

#[derive(Clone)]
struct Port {
    tx: SyncSender<Value>,
    to: Arc<Meter>,
    lossy: bool,
}

/// A node's way out: its ports, in the order they were linked, and the bus.
#[derive(Clone)]
pub struct Out {
    ports: Vec<Port>,
    meter: Arc<Meter>,
    bus: Bus,
}

impl Out {
    pub fn name(&self) -> &str {
        &self.meter.name
    }

    pub fn ports(&self) -> usize {
        self.ports.len()
    }

    pub fn bus(&self) -> &Bus {
        &self.bus
    }

    pub fn signal(&self, s: &Signal) {
        self.bus.signal(s);
    }

    /// Sends `v` on `port`, waiting while the node there is full. On a lossy port it does not
    /// wait: a full node loses the value, which counts as refused. Panics on a port the node
    /// does not have.
    pub fn emit(&self, port: usize, v: Value) -> Result<(), Closed> {
        let p = &self.ports[port];
        if p.lossy {
            return match self.try_emit(port, v) {
                Err(Refused::Closed(_)) => Err(Closed),
                _ => Ok(()),
            };
        }
        match p.tx.try_send(v) {
            Ok(()) => {
                self.landed(p);
                Ok(())
            }
            Err(TrySendError::Full(v)) => {
                let waited = Instant::now();
                let sent = p.tx.send(v);
                self.meter
                    .blocked_ns
                    .fetch_add(waited.elapsed().as_nanos() as u64, Ordering::Relaxed);
                match sent {
                    Ok(()) => {
                        self.landed(p);
                        Ok(())
                    }
                    Err(_) => Err(Closed),
                }
            }
            Err(TrySendError::Disconnected(_)) => Err(Closed),
        }
    }

    /// Sends `v` on `port` if the node there has room now, else hands it back.
    pub fn try_emit(&self, port: usize, v: Value) -> Result<(), Refused> {
        let p = &self.ports[port];
        match p.tx.try_send(v) {
            Ok(()) => {
                self.landed(p);
                Ok(())
            }
            Err(TrySendError::Full(v)) => {
                self.meter.refused.fetch_add(1, Ordering::Relaxed);
                Err(Refused::Full(v))
            }
            Err(TrySendError::Disconnected(v)) => {
                self.meter.refused.fetch_add(1, Ordering::Relaxed);
                Err(Refused::Closed(v))
            }
        }
    }

    /// Stops this node: its workers end after the value in hand, its input is dropped, and
    /// the nodes upstream stop when they next send to it.
    pub fn stop(&self) {
        self.meter.stop.store(true, Ordering::Relaxed);
    }

    pub fn stopped(&self) -> bool {
        self.meter.stop.load(Ordering::Relaxed)
    }

    /// True when the node has links and every node they reach has ended.
    pub fn downstream_gone(&self) -> bool {
        !self.ports.is_empty() && self.ports.iter().all(|p| p.to.gone.load(Ordering::Acquire))
    }

    fn landed(&self, p: &Port) {
        p.to.arrived();
        self.meter.emitted.fetch_add(1, Ordering::Relaxed);
    }
}

type Work = Box<dyn FnMut(Value, &Out) + Send>;

enum Kind {
    Source(Box<dyn FnOnce(&Out) + Send>),
    Stage(Vec<Work>),
}

struct Node {
    kind: Kind,
    meter: Arc<Meter>,
    outs: Vec<(usize, bool)>,
}

/// A node of a [`Graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(pub usize);

/// What a run did.
#[derive(Debug, Clone)]
pub struct Report {
    pub elapsed: Duration,
    pub nodes: Vec<NodeStats>,
    /// Nodes whose worker panicked; the graph still ran down.
    pub panicked: Vec<String>,
    pub bus_lost: u64,
}

impl Report {
    pub fn node(&self, name: &str) -> Option<&NodeStats> {
        self.nodes.iter().find(|n| n.name == name)
    }
}

/// Nodes and the channels between them, run by [`Graph::run`].
pub struct Graph {
    nodes: Vec<Node>,
    bus: Bus,
}

impl Graph {
    pub fn new(bus: &Bus) -> Self {
        Self {
            nodes: Vec::new(),
            bus: bus.clone(),
        }
    }

    pub fn bus(&self) -> &Bus {
        &self.bus
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn add(&mut self, name: &str, cap: usize, workers: usize, kind: Kind) -> NodeId {
        self.nodes.push(Node {
            kind,
            meter: Meter::new(name, cap, workers),
            outs: Vec::new(),
        });
        NodeId(self.nodes.len() - 1)
    }

    /// A source: one thread that sends what it has and returns. A send that fails means
    /// everything downstream stopped; the source should then stop too.
    pub fn source(&mut self, name: &str, f: impl FnOnce(&Out) + Send + 'static) -> NodeId {
        self.add(name, 0, 1, Kind::Source(Box::new(f)))
    }

    /// A stage of `workers` threads sharing one input of `cap` values. Each value goes to one
    /// worker, so values can leave in another order than they came.
    pub fn stage(
        &mut self,
        name: &str,
        cap: usize,
        workers: usize,
        f: impl Fn(Value, &Out) + Send + Sync + 'static,
    ) -> NodeId {
        assert!(
            cap >= 1 && workers >= 1,
            "{name}: cap and workers are at least 1"
        );
        let f = Arc::new(f);
        let works = (0..workers)
            .map(|_| {
                let f = f.clone();
                Box::new(move |v: Value, out: &Out| f(v, out)) as Work
            })
            .collect();
        self.add(name, cap, workers, Kind::Stage(works))
    }

    /// A stage of one thread that keeps state between values.
    pub fn stage_mut(
        &mut self,
        name: &str,
        cap: usize,
        f: impl FnMut(Value, &Out) + Send + 'static,
    ) -> NodeId {
        assert!(cap >= 1, "{name}: cap is at least 1");
        self.add(name, cap, 1, Kind::Stage(vec![Box::new(f)]))
    }

    /// Links `from` to `to` and answers the port's number on `from`.
    pub fn link(&mut self, from: NodeId, to: NodeId) -> usize {
        self.link_as(from, to, false)
    }

    /// A link that never waits: a value sent while `to` is full is lost.
    pub fn link_lossy(&mut self, from: NodeId, to: NodeId) -> usize {
        self.link_as(from, to, true)
    }

    fn link_as(&mut self, from: NodeId, to: NodeId, lossy: bool) -> usize {
        assert!(
            matches!(self.nodes[to.0].kind, Kind::Stage(_)),
            "{}: a source has no input",
            self.nodes[to.0].meter.name
        );
        let outs = &mut self.nodes[from.0].outs;
        outs.push((to.0, lossy));
        outs.len() - 1
    }

    /// The graph's shape, for [`petri::check`].
    pub fn shape(&self) -> petri::Shape {
        petri::Shape {
            nodes: self
                .nodes
                .iter()
                .map(|n| petri::ShapeNode {
                    name: n.meter.name.clone(),
                    source: matches!(n.kind, Kind::Source(_)),
                    cap: n.meter.cap,
                    workers: n.meter.workers,
                    outs: n
                        .outs
                        .iter()
                        .map(|&(to, lossy)| petri::ShapeEdge { to, lossy })
                        .collect(),
                })
                .collect(),
        }
    }

    /// Runs every node to its end. With `tick`, a monitor publishes `stats` on the bus at that
    /// pace, and once more at the end.
    pub fn run(self, tick: Option<Duration>) -> Report {
        let started = Instant::now();
        let mut senders: Vec<Option<SyncSender<Value>>> = Vec::new();
        let mut receivers: Vec<Option<Receiver<Value>>> = Vec::new();
        for node in &self.nodes {
            if let Kind::Stage(_) = node.kind {
                let (tx, rx) = sync_channel(node.meter.cap);
                senders.push(Some(tx));
                receivers.push(Some(rx));
            } else {
                senders.push(None);
                receivers.push(None);
            }
        }
        let meters: Vec<Arc<Meter>> = self.nodes.iter().map(|n| n.meter.clone()).collect();
        let outs: Vec<Out> = self
            .nodes
            .iter()
            .map(|node| Out {
                ports: node
                    .outs
                    .iter()
                    .map(|&(to, lossy)| Port {
                        tx: senders[to].clone().expect("links reach stages"),
                        to: meters[to].clone(),
                        lossy,
                    })
                    .collect(),
                meter: node.meter.clone(),
                bus: self.bus.clone(),
            })
            .collect();
        // From here a channel stays open only while a node that sends to it runs.
        drop(senders);

        let done = Arc::new(AtomicBool::new(false));
        let monitor = tick.map(|tick| {
            let (meters, bus, done) = (meters.clone(), self.bus.clone(), done.clone());
            std::thread::spawn(move || {
                let mut next = Instant::now();
                while !done.load(Ordering::Relaxed) {
                    if Instant::now() >= next {
                        publish_stats(&bus, &meters, false);
                        next += tick;
                    }
                    std::thread::sleep(POLL.min(tick));
                }
            })
        });

        let mut workers = Vec::new();
        for ((node, out), rx) in self.nodes.into_iter().zip(outs).zip(receivers) {
            let name = node.meter.name.clone();
            match node.kind {
                Kind::Source(f) => {
                    let meter = node.meter.clone();
                    let source = std::thread::spawn(move || {
                        let _left = Left(&meter);
                        f(&out);
                    });
                    workers.push((name, source));
                }
                Kind::Stage(works) => {
                    let rx = Arc::new(Mutex::new(rx.expect("a stage has an input")));
                    for mut work in works {
                        let (rx, out, meter) = (rx.clone(), out.clone(), node.meter.clone());
                        let worker = std::thread::spawn(move || {
                            let _left = Left(&meter);
                            while let Some(v) = next(&rx, &meter, &out) {
                                meter.queued.fetch_sub(1, Ordering::Relaxed);
                                meter.received.fetch_add(1, Ordering::Relaxed);
                                let began = Instant::now();
                                work(v, &out);
                                meter.busy_ns.fetch_add(
                                    began.elapsed().as_nanos() as u64,
                                    Ordering::Relaxed,
                                );
                            }
                        });
                        workers.push((name.clone(), worker));
                    }
                }
            }
        }
        let mut panicked = Vec::new();
        for (name, w) in workers {
            if w.join().is_err() && !panicked.contains(&name) {
                panicked.push(name);
            }
        }
        done.store(true, Ordering::Relaxed);
        if let Some(m) = monitor {
            let _ = m.join();
        }
        if tick.is_some() {
            publish_stats(&self.bus, &meters, true);
        }
        Report {
            elapsed: started.elapsed(),
            nodes: meters.iter().map(|m| m.stats()).collect(),
            panicked,
            bus_lost: self.bus.lost(),
        }
    }
}

/// Marks a worker ended when it returns or unwinds.
struct Left<'a>(&'a Meter);

impl Drop for Left<'_> {
    fn drop(&mut self) {
        self.0.left();
    }
}

/// The next value for a worker, or `None` once its node stopped, its input closed, or every
/// node downstream ended.
fn next(rx: &Mutex<Receiver<Value>>, meter: &Meter, out: &Out) -> Option<Value> {
    loop {
        if meter.stop.load(Ordering::Relaxed) || out.downstream_gone() {
            return None;
        }
        let got = rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .recv_timeout(POLL);
        match got {
            Ok(v) => return Some(v),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
}

fn publish_stats(bus: &Bus, meters: &[Arc<Meter>], last: bool) {
    let nodes: Vec<Value> = meters.iter().map(|m| m.stats().json()).collect();
    bus.publish("stats", json!({ "nodes": nodes, "final": last }));
}
