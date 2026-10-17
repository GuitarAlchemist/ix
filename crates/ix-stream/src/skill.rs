//! `stream.run`, the ix-pipeline skill: one stage of an `ix.yaml` that runs an ix-stream graph.
//!
//! ```yaml
//! stages:
//!   chain:
//!     skill: stream.run
//!     args:
//!       graph:
//!         nodes: [{ name: src, source: true }, { name: work, cap: 4, workers: 2 }, { name: sink }]
//!         edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }]
//!       max_events: 100
//!       budget: 1000
//! ```
//!
//! `graph` is a [`Shape`] in the form [`Shape::json`] writes, the `shape` of a `meta` event. Its
//! nodes do no work of their own yet: a source sends `{ source, i }` values, a stage sends each
//! value it takes on every link, a batching node sends one `{ node, count }` when its input
//! ends, and a node with no link is a sink.
//!
//! Two bounds stop the stream: `max_events`, the most values the sources send in all, and
//! `budget`, the most values the nodes take in all, one step each. Before the run the skill
//! works out from the shape how many values each source can send within both, serving the
//! sources in the order of `nodes`. The run then has to take and send exactly that, and the
//! [`Ledger`] that holds the budget refuses any step past it. So the answer depends on the input
//! alone, never on thread timing: lossy links, which drop values by timing, are refused, and so
//! is a cycle, on which a value would never end.
//!
//! The answer is `{ stopped_by, events, steps, receipts }`. The receipts are the evidence: the
//! Petri check that cleared the graph, the budget and what the run spent of it, and what each
//! node took and sent. A run whose counts differ from what its shape predicts is an error.

use crate::gate::{Budget, Ledger};
use crate::petri::{self, Abstraction, Shape};
use crate::{Batch, Bus, Closed, Graph, NodeId, Out, Signal};
use ix_pipeline::dag::Dag;
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};
use std::sync::Arc;

/// The most nodes a graph may have.
const MAX_NODES: usize = 256;
/// The most links a graph may have.
const MAX_LINKS: usize = 2048;
/// The most threads a run may start: one per source, one per worker.
const MAX_THREADS: usize = 512;
/// The largest input a node may have. Its channel holds that many slots from the start.
const MAX_CAP: usize = 1024;
/// The largest `max_events` and `budget`.
const MAX_BOUND: u64 = 1_000_000;
/// `max_events` when the caller gives none.
const DEFAULT_EVENTS: u64 = 1_000;

pub fn stream_run_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "graph": {
                "type": "object",
                "description": "The graph, in the form ix-stream writes its shape: the nodes in index order, and the links between them by index. A node with no link is a sink.",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "source": { "type": "boolean", "default": false, "description": "Sends { source, i } values and has no input." },
                                "batching": { "type": "boolean", "default": false, "description": "Counts what it takes and sends one { node, count } when its input ends." },
                                "cap": { "type": "integer", "minimum": 1, "maximum": MAX_CAP, "default": 8, "description": "The values its input holds; a full input holds back the nodes before it." },
                                "workers": { "type": "integer", "minimum": 1, "default": 1, "description": "Threads sharing its input." }
                            },
                            "required": ["name"]
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": { "type": "integer", "minimum": 0 },
                                "to": { "type": "integer", "minimum": 0 },
                                "lossy": { "type": "boolean", "description": "Refused: a lossy link drops values by timing." }
                            },
                            "required": ["from", "to"]
                        }
                    }
                },
                "required": ["nodes", "edges"]
            },
            "max_events": { "type": "integer", "minimum": 1, "maximum": MAX_BOUND, "default": DEFAULT_EVENTS, "description": "The most values the sources send, in all." },
            "budget": { "type": "integer", "minimum": 1, "maximum": MAX_BOUND, "default": MAX_BOUND, "description": "The most values the nodes take, in all: one step each." }
        },
        "required": ["graph"]
    })
}

pub fn stream_run_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "stopped_by": { "type": "string", "enum": ["max_events", "budget"], "description": "The bound that ended the stream." },
            "events": { "type": "integer", "description": "The values the sources sent." },
            "steps": { "type": "integer", "description": "The values the nodes took: what the run spent of its budget." },
            "receipts": {
                "type": "object",
                "description": "`petri`: the check that cleared the graph; `budget`: { cap, spent }; `nodes`: what each node took and sent, { name, received, emitted }, in the order of `nodes`."
            }
        }
    })
}

/// Runs an ix-stream graph as a pipeline stage, bounded by `max_events` (the values its sources
/// send) and `budget` (the values its nodes take). Deterministic: no files, no processes, no
/// network, and an answer that does not depend on thread timing.
///
/// `graph` is the shape ix-stream writes, `{ nodes: [{ name, source, batching, cap, workers }],
/// edges: [{ from, to }] }`. Sources send `{ source, i }`, stages send what they take on every
/// link, batching nodes send `{ node, count }` at the end. Cycles and lossy links are refused.
/// The answer is `{ stopped_by, events, steps, receipts }`, the receipts holding the Petri
/// check, the budget spent, and each node's counts.
#[ix_skill(
    domain = "stream",
    name = "stream.run",
    governance = "safety,deterministic",
    schema_fn = "crate::skill::stream_run_schema",
    output_schema_fn = "crate::skill::stream_run_output_schema"
)]
pub fn stream_run(params: Value) -> Result<Value, String> {
    let graph = params
        .get("graph")
        .ok_or("give the graph to run as `graph`")?;
    let shape = Shape::from_json(graph)?;
    let max_events = bound(&params, "max_events", DEFAULT_EVENTS)?;
    let budget = bound(&params, "budget", MAX_BOUND)?;
    let order = admit(&shape)?;
    // Built even though an acyclic graph needs no exploring: building checks the shape.
    let check = petri::check(&shape, Abstraction::exact(), 1)?;
    if check.deadlock_free != Some(true) {
        return Err(format!(
            "the Petri check did not clear the graph: {}",
            check.json()
        ));
    }

    let sent = plan(&shape, &order, max_events, budget);
    let (taken, sends) = predict(&shape, &order, &sent);
    let ledger = Ledger::new(vec![Budget::new("steps", "values taken", budget as f64)]);
    let mut g = Graph::new(&Bus::new());
    for (i, n) in shape.nodes.iter().enumerate() {
        if n.source {
            let (name, count) = (n.name.clone(), sent[i]);
            g.source(&n.name, move |out| {
                for k in 0..count {
                    if send_all(out, &json!({ "source": name, "i": k })).is_err() {
                        return;
                    }
                }
            });
        } else if n.batching {
            let count = Count {
                ledger: ledger.clone(),
                taken: 0,
            };
            g.batch(&n.name, n.cap, count);
        } else {
            let ledger = ledger.clone();
            g.stage(&n.name, n.cap, n.workers, move |v, out| {
                if step(&ledger, out) {
                    let _ = send_all(out, &v);
                }
            });
        }
    }
    for (i, n) in shape.nodes.iter().enumerate() {
        for e in &n.outs {
            g.link(NodeId(i), NodeId(e.to));
        }
    }
    let report = g.run(None);

    if !report.panicked.is_empty() {
        return Err(format!("nodes panicked: {}", report.panicked.join(", ")));
    }
    for (i, s) in report.nodes.iter().enumerate() {
        if s.received != taken[i] || s.emitted != sends[i] {
            return Err(format!(
                "{} took {} values and sent {}, where its shape predicts {} and {}",
                s.name, s.received, s.emitted, taken[i], sends[i]
            ));
        }
    }
    let spent = ledger.get("steps").map_or(0.0, |b| b.spent) as u64;
    if spent != steps(&taken) {
        return Err(format!(
            "the nodes spent {spent} steps, where the shape predicts {}",
            steps(&taken)
        ));
    }
    let events: u64 = sent.iter().sum();
    let nodes: Vec<Value> = report
        .nodes
        .iter()
        .map(|s| json!({ "name": s.name, "received": s.received, "emitted": s.emitted }))
        .collect();
    Ok(json!({
        "stopped_by": if events == max_events { "max_events" } else { "budget" },
        "events": events,
        "steps": spent,
        "receipts": {
            "petri": check.json(),
            "budget": { "cap": budget, "spent": spent },
            "nodes": nodes,
        },
    }))
}

/// `params[key]` as a whole number from 1 to [`MAX_BOUND`], `default` when it is absent.
fn bound(params: &Value, key: &str, default: u64) -> Result<u64, String> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .filter(|n| (1..=MAX_BOUND).contains(n))
            .ok_or_else(|| format!("{key} must be a whole number from 1 to {MAX_BOUND}, got {v}")),
    }
}

/// Refuses a graph this skill cannot run, or cannot run deterministically, and answers its
/// nodes in an order where every node comes after those that send to it.
fn admit(shape: &Shape) -> Result<Vec<usize>, String> {
    let nodes = &shape.nodes;
    if nodes.is_empty() || nodes.len() > MAX_NODES {
        return Err(format!(
            "a graph has 1 to {MAX_NODES} nodes, this one {}",
            nodes.len()
        ));
    }
    if !nodes.iter().any(|n| n.source) {
        return Err("the graph has no source: nothing would flow".into());
    }
    let links: usize = nodes.iter().map(|n| n.outs.len()).sum();
    if links > MAX_LINKS {
        return Err(format!(
            "a graph has at most {MAX_LINKS} links, this one {links}"
        ));
    }
    let mut threads = 0usize;
    for n in nodes {
        if n.source {
            if n.batching {
                return Err(format!("{}: a source cannot be batching", n.name));
            }
            threads += 1;
            continue;
        }
        if !(1..=MAX_CAP).contains(&n.cap) || n.workers == 0 {
            return Err(format!(
                "{}: cap is 1 to {MAX_CAP} and workers at least 1",
                n.name
            ));
        }
        threads = threads.saturating_add(if n.batching { 1 } else { n.workers });
    }
    if threads > MAX_THREADS {
        return Err(format!(
            "the graph needs {threads} threads, more than the {MAX_THREADS} a run may start"
        ));
    }
    if let Some(n) = nodes.iter().find(|n| n.outs.iter().any(|e| e.lossy)) {
        return Err(format!(
            "{}: a lossy link drops values by timing, so stream.run refuses it",
            n.name
        ));
    }
    let mut dag: Dag<()> = Dag::new();
    for i in 0..nodes.len() {
        dag.add_node(i.to_string(), ()).expect("fresh ids");
    }
    for (i, n) in nodes.iter().enumerate() {
        for e in &n.outs {
            if dag.add_edge(i.to_string(), e.to.to_string()).is_err() {
                return Err(format!(
                    "{} -> {}: the graph has a cycle, on which a value would never end",
                    n.name, nodes[e.to].name
                ));
            }
        }
    }
    Ok(dag
        .topological_sort()
        .into_iter()
        .map(|id| id.parse().expect("ids are indices"))
        .collect())
}

/// What each node takes and what it sends, in all, when each source `i` sends `sent[i]`.
fn predict(shape: &Shape, order: &[usize], sent: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut taken = vec![0u64; shape.nodes.len()];
    let mut sends = vec![0u64; shape.nodes.len()];
    for &i in order {
        let n = &shape.nodes[i];
        // What it sends on each of its links: every link carries all of it.
        let each = if n.source {
            sent[i]
        } else if n.batching {
            u64::from(taken[i] > 0)
        } else {
            taken[i]
        };
        for e in &n.outs {
            taken[e.to] = taken[e.to].saturating_add(each);
        }
        sends[i] = each.saturating_mul(n.outs.len() as u64);
    }
    (taken, sends)
}

/// The steps a run takes: one per value a node takes.
fn steps(taken: &[u64]) -> u64 {
    taken.iter().fold(0, |sum, &t| sum.saturating_add(t))
}

/// How many values each source sends. In the order of the nodes, each source sends the most it
/// can within the events left and the budget.
fn plan(shape: &Shape, order: &[usize], max_events: u64, budget: u64) -> Vec<u64> {
    let mut sent = vec![0u64; shape.nodes.len()];
    let mut left = max_events;
    for i in 0..shape.nodes.len() {
        if !shape.nodes[i].source {
            continue;
        }
        // The steps grow with what a source sends, so halving finds the most it can send.
        let (mut lo, mut hi) = (0, left);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            sent[i] = mid;
            if steps(&predict(shape, order, &sent).0) <= budget {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        sent[i] = lo;
        left -= lo;
    }
    sent
}

/// Sends `v` on every link of the node.
fn send_all(out: &Out, v: &Value) -> Result<(), Closed> {
    for port in 0..out.ports() {
        out.emit(port, v.clone())?;
    }
    Ok(())
}

/// Pays one step for a value taken. False when the budget refuses it, which a planned run never
/// meets: the ledger is the hard stop behind the plan.
fn step(ledger: &Ledger, out: &Out) -> bool {
    let paid = ledger.spend("steps", 1.0);
    if !paid {
        out.signal(&Signal::pain(out.name(), 1.0, "the budget refused a step"));
    }
    paid
}

/// A batching node: counts what it takes, and sends `{ node, count }` once its input ends.
struct Count {
    ledger: Arc<Ledger>,
    taken: u64,
}

impl Batch for Count {
    fn take(&mut self, _: Value, out: &Out) {
        if step(&self.ledger, out) {
            self.taken += 1;
        }
    }

    fn end(&mut self, out: &Out) {
        if self.taken > 0 {
            let _ = send_all(out, &json!({ "node": out.name(), "count": self.taken }));
        }
    }
}
