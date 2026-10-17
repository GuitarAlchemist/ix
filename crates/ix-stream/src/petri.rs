//! The graph as a Petri net, checked for deadlocks before it runs.
//!
//! ix-petri has no place capacities, so each channel is two places: the values in it, and its
//! free slots, which start at its capacity. Every send takes a free slot and every receive
//! gives one back, so the pair always holds `cap` tokens and the net is bounded by
//! construction. A node's workers are tokens in its `idle` place.
//!
//! A worker takes a value and holds it (`take`), may send nothing for it (`skip`, as a filter
//! does), or commits to one port (`to`) and waits there for a free slot (`put`). A lossy port
//! can drop what it holds instead (`lose`). Committing at any moment is more than a node does
//! (the gate commits to its fallback only once the others are full), so the net has every
//! behaviour the threads have, and a deadlock-free net means deadlock-free threads. The
//! converse does not hold: a reported deadlock can be one the threads never reach. A source is
//! a worker that always has a next value; a sink takes and is done. A batching node (a window)
//! is both a stage and a source: it takes values, and it can send at any moment what it held
//! back (`make`), which covers the many rows a window may answer for one value.
//!
//! When the waiting links have no cycle, no exploration is needed: a dead marking would need a
//! worker waiting on a full node whose workers all wait further down, and following the
//! waiting links down an acyclic graph ends at a sink, which always takes. [`check`] says so
//! without exploring ([`Method::Acyclic`]), so a chain of hundreds of nodes is answered at
//! once. With a cycle it explores the net ([`Method::Reachability`]). The state space grows
//! with the capacities; past `max_states` the verdict is unknown, and an [`Abstraction`] can
//! clamp capacities and workers to keep it small, for a weaker claim: the verdict then holds
//! for the clamped graph, not for the real one.

use ix_petri::analysis::{analyze, Limits, Verdict};
use ix_petri::net::PetriNet;
use ix_pipeline::dag::Dag;
use serde_json::{json, Value};
use std::collections::HashMap;

/// A node of a graph, as the net sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeNode {
    pub name: String,
    pub source: bool,
    /// Holds values back and sends when it chooses ([`crate::Batch`]).
    pub batching: bool,
    pub cap: usize,
    pub workers: usize,
    pub outs: Vec<ShapeEdge>,
}

/// A link, by the index of the node it reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeEdge {
    pub to: usize,
    pub lossy: bool,
}

/// A graph's nodes and links, from [`crate::Graph::shape`] or written by hand.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Shape {
    pub nodes: Vec<ShapeNode>,
}

impl Shape {
    pub fn node(&mut self, name: &str, cap: usize, workers: usize) -> usize {
        self.nodes.push(ShapeNode {
            name: name.to_string(),
            source: false,
            batching: false,
            cap,
            workers,
            outs: Vec::new(),
        });
        self.nodes.len() - 1
    }

    pub fn source(&mut self, name: &str) -> usize {
        let i = self.node(name, 0, 1);
        self.nodes[i].source = true;
        i
    }

    pub fn link(&mut self, from: usize, to: usize, lossy: bool) {
        self.nodes[from].outs.push(ShapeEdge { to, lossy });
    }

    /// The nodes in index order and the links between them, by index, for a reader that
    /// draws the graph: `{ "nodes": [{ name, source, batching, cap, workers }], "edges":
    /// [{ from, to, lossy }] }`.
    pub fn json(&self) -> Value {
        let nodes: Vec<Value> = self
            .nodes
            .iter()
            .map(|n| {
                json!({ "name": n.name, "source": n.source, "batching": n.batching,
                        "cap": n.cap, "workers": n.workers })
            })
            .collect();
        let edges: Vec<Value> = self
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| {
                n.outs
                    .iter()
                    .map(move |e| json!({ "from": i, "to": e.to, "lossy": e.lossy }))
            })
            .collect();
        json!({ "nodes": nodes, "edges": edges })
    }

    /// The shape [`Shape::json`] writes, read back. A field left out takes its default: not a
    /// source, not batching, one worker, and a capacity of 8 (0 for a source). The links of a
    /// node keep the order they are listed in, which numbers its ports.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let list = |key: &str| {
            v.get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("`{key}` must be an array"))
        };
        let count = |n: &Value, key: &str, default: usize| match n.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(x) => x
                .as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| format!("`{key}` must be a whole number, got {x}")),
        };
        let flag = |n: &Value, key: &str| match n.get(key) {
            None | Some(Value::Null) => Ok(false),
            Some(x) => x
                .as_bool()
                .ok_or_else(|| format!("`{key}` must be true or false, got {x}")),
        };
        let mut shape = Shape::default();
        for (i, n) in list("nodes")?.iter().enumerate() {
            let name = n
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("nodes[{i}] has no `name`"))?;
            let source = flag(n, "source").map_err(|e| format!("nodes[{i}]: {e}"))?;
            shape.nodes.push(ShapeNode {
                name: name.to_string(),
                source,
                batching: flag(n, "batching").map_err(|e| format!("nodes[{i}]: {e}"))?,
                cap: count(n, "cap", if source { 0 } else { 8 })
                    .map_err(|e| format!("nodes[{i}]: {e}"))?,
                workers: count(n, "workers", 1).map_err(|e| format!("nodes[{i}]: {e}"))?,
                outs: Vec::new(),
            });
        }
        let n = shape.nodes.len();
        for (i, e) in list("edges")?.iter().enumerate() {
            let end = |key: &str| {
                count(e, key, usize::MAX)
                    .ok()
                    .filter(|&k| k < n)
                    .ok_or_else(|| format!("edges[{i}]: `{key}` must be the index of a node"))
            };
            let (from, to) = (end("from")?, end("to")?);
            let lossy = flag(e, "lossy").map_err(|e| format!("edges[{i}]: {e}"))?;
            shape.link(from, to, lossy);
        }
        Ok(shape)
    }
}

/// Clamps on the graph before it becomes a net. `None` keeps the real value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Abstraction {
    pub max_cap: Option<usize>,
    pub max_workers: Option<usize>,
}

impl Abstraction {
    pub fn exact() -> Self {
        Self::default()
    }

    pub fn clamp(cap: usize, workers: usize) -> Self {
        Self {
            max_cap: Some(cap),
            max_workers: Some(workers),
        }
    }

    pub fn is_exact(&self) -> bool {
        self.max_cap.is_none() && self.max_workers.is_none()
    }
}

/// How a verdict was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// The waiting links have no cycle; nothing was explored.
    Acyclic,
    /// The net's reachable markings were explored.
    Reachability,
}

/// Do the links that wait (all but the lossy ones) form an acyclic graph?
pub fn acyclic(shape: &Shape) -> bool {
    let mut dag: Dag<()> = Dag::new();
    for i in 0..shape.nodes.len() {
        dag.add_node(i.to_string(), ()).expect("fresh ids");
    }
    shape.nodes.iter().enumerate().all(|(i, n)| {
        n.outs.iter().filter(|e| !e.lossy).all(|e| {
            e.to < shape.nodes.len() && dag.add_edge(i.to_string(), e.to.to_string()).is_ok()
        })
    })
}

/// What the check found.
#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    pub method: Method,
    /// `Some(true)` when no dead marking is reachable, `Some(false)` when one is, `None` when
    /// the search stopped first.
    pub deadlock_free: Option<bool>,
    pub bounded: Option<bool>,
    pub states: usize,
    pub truncated: bool,
    /// The shortest firing sequence to a dead marking, as node actions.
    pub witness: Vec<String>,
    /// That marking: where the values and the workers are.
    pub dead_marking: Option<String>,
    pub abstraction: Abstraction,
    pub places: usize,
    pub transitions: usize,
}

impl Check {
    pub fn json(&self) -> Value {
        json!({
            "method": match self.method { Method::Acyclic => "acyclic", Method::Reachability => "reachability" },
            "deadlock_free": self.deadlock_free, "bounded": self.bounded,
            "states": self.states, "truncated": self.truncated,
            "witness": self.witness, "dead_marking": self.dead_marking,
            "exact": self.abstraction.is_exact(),
            "max_cap": self.abstraction.max_cap, "max_workers": self.abstraction.max_workers,
            "places": self.places, "transitions": self.transitions,
        })
    }
}

/// The net of `shape` under `abs`, and a label for each transition id.
pub fn net(shape: &Shape, abs: Abstraction) -> Result<(PetriNet, HashMap<String, String>), String> {
    let cap = |c: usize| abs.max_cap.map_or(c, |m| c.min(m)).max(1) as u64;
    let workers = |w: usize| abs.max_workers.map_or(w, |m| w.min(m)).max(1) as u64;
    let mut b = PetriNet::builder().name("ix-stream graph");
    let mut labels = HashMap::new();
    let mut t = |b: ix_petri::net::PetriNetBuilder, id: String, label: String| {
        labels.insert(id.clone(), label.clone());
        b.named_transition(id, label)
    };
    for (i, n) in shape.nodes.iter().enumerate() {
        if n.outs.iter().any(|e| e.to >= shape.nodes.len()) {
            return Err(format!("{}: a link to no node", n.name));
        }
        if n.outs.iter().any(|e| shape.nodes[e.to].source) {
            return Err(format!("{}: a link into a source", n.name));
        }
        if !n.source {
            b = b
                .named_place(format!("q{i}"), format!("{} queued", n.name), 0)
                .named_place(format!("free{i}"), format!("{} free", n.name), cap(n.cap));
        }
        if n.source || !n.outs.is_empty() {
            let idle = if n.source { 1 } else { workers(n.workers) };
            b = b
                .named_place(format!("idle{i}"), format!("{} idle", n.name), idle)
                .named_place(format!("held{i}"), format!("{} holding", n.name), 0);
        }
        if n.source {
            b = t(b, format!("make{i}"), format!("{} reads", n.name));
            b = b.arc(format!("idle{i}"), format!("make{i}"));
            b = b.arc(format!("make{i}"), format!("held{i}"));
        } else if n.outs.is_empty() {
            b = t(
                b,
                format!("take{i}"),
                format!("{} takes and is done", n.name),
            );
            b = b.arc(format!("q{i}"), format!("take{i}"));
            b = b.arc(format!("take{i}"), format!("free{i}"));
            continue;
        } else {
            b = t(b, format!("take{i}"), format!("{} takes", n.name));
            b = t(b, format!("skip{i}"), format!("{} sends nothing", n.name));
            b = b
                .arc(format!("q{i}"), format!("take{i}"))
                .arc(format!("idle{i}"), format!("take{i}"))
                .arc(format!("take{i}"), format!("free{i}"))
                .arc(format!("take{i}"), format!("held{i}"))
                .arc(format!("held{i}"), format!("skip{i}"))
                .arc(format!("skip{i}"), format!("idle{i}"));
            if n.batching {
                b = t(
                    b,
                    format!("make{i}"),
                    format!("{} sends what it held", n.name),
                );
                b = b
                    .arc(format!("idle{i}"), format!("make{i}"))
                    .arc(format!("make{i}"), format!("held{i}"));
            }
        }
        for (p, e) in n.outs.iter().enumerate() {
            let (to, u) = (&shape.nodes[e.to].name, e.to);
            let held = format!("held{i}.{p}");
            b = b.named_place(held.clone(), format!("{} waiting on {to}", n.name), 0);
            b = t(b, format!("to{i}.{p}"), format!("{} picks {to}", n.name));
            b = t(
                b,
                format!("put{i}.{p}"),
                format!("{} sends to {to}", n.name),
            );
            b = b
                .arc(format!("held{i}"), format!("to{i}.{p}"))
                .arc(format!("to{i}.{p}"), held.clone())
                .arc(held.clone(), format!("put{i}.{p}"))
                .arc(format!("free{u}"), format!("put{i}.{p}"))
                .arc(format!("put{i}.{p}"), format!("q{u}"))
                .arc(format!("put{i}.{p}"), format!("idle{i}"));
            if e.lossy {
                b = t(
                    b,
                    format!("lose{i}.{p}"),
                    format!("{} drops for {to}", n.name),
                );
                b = b
                    .arc(held, format!("lose{i}.{p}"))
                    .arc(format!("lose{i}.{p}"), format!("idle{i}"));
            }
        }
    }
    let net = b.build().map_err(|e| e.to_string())?;
    Ok((net, labels))
}

/// The verdict for `shape`: at once when its waiting links are acyclic, else by exploring its
/// net under `abs`, up to `max_states` markings.
pub fn check(shape: &Shape, abs: Abstraction, max_states: usize) -> Result<Check, String> {
    // Built even when it is not explored: building checks the shape.
    let (net, _) = net(shape, abs)?;
    if acyclic(shape) {
        return Ok(Check {
            method: Method::Acyclic,
            deadlock_free: Some(true),
            // Every channel's two places always hold its capacity.
            bounded: Some(true),
            states: 0,
            truncated: false,
            witness: Vec::new(),
            dead_marking: None,
            abstraction: Abstraction::exact(),
            places: net.places().len(),
            transitions: net.transitions().len(),
        });
    }
    explore(shape, abs, max_states)
}

/// The verdict for `shape` by exploring its net under `abs`, even when it is acyclic.
pub fn explore(shape: &Shape, abs: Abstraction, max_states: usize) -> Result<Check, String> {
    let (net, labels) = net(shape, abs)?;
    let a = analyze(
        &net,
        Limits {
            max_states,
            max_reported_deadlocks: 1,
        },
    );
    let (deadlock_free, witness, dead_marking) = match &a.deadlock_free {
        Verdict::Holds(_) => (Some(true), Vec::new(), None),
        Verdict::Fails(d) => {
            let first = d.first();
            (
                Some(false),
                first.map_or(Vec::new(), |d| {
                    d.witness
                        .iter()
                        .map(|id| labels.get(id).cloned().unwrap_or_else(|| id.clone()))
                        .collect()
                }),
                first.map(|d| d.marking.clone()),
            )
        }
        Verdict::Unknown { .. } => (None, Vec::new(), None),
    };
    let bounded = match a.bounded {
        Verdict::Holds(_) => Some(true),
        Verdict::Fails(_) => Some(false),
        Verdict::Unknown { .. } => None,
    };
    Ok(Check {
        method: Method::Reachability,
        deadlock_free,
        bounded,
        states: a.states,
        truncated: a.truncated,
        witness,
        dead_marking,
        abstraction: abs,
        places: net.places().len(),
        transitions: net.transitions().len(),
    })
}
