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
//! `graph` is a [`Shape`] in the form [`Shape::json`] writes, the `shape` of a `meta` event. A
//! source sends `{ source, i }` values, a stage sends each value it takes on every link, a
//! batching node sends one `{ node, count }` when its input ends, and a node with no link is a
//! sink. A stage with `wasm: true` (the `wasm` feature) sends instead what its `program`, a WASI
//! module run sandboxed by ix-wasm, answers to each value, within `fuel` per value:
//!
//! ```yaml
//! nodes: [{ name: src, source: true },
//!         { name: double, wasm: true, program: { go: "package main ..." }, fuel: 50000000 },
//!         { name: sink }]
//! ```
//!
//! A program that runs out of fuel, is refused a file, exits non-zero or writes anything but one
//! JSON value stops the stream: the answer is then `stopped_by: "error"` with the `error`, and
//! counts that tell how far the stream got before the stop reached it, which timing decides.
//!
//! A source with `values` sends those instead of `{ source, i }`, and no more of them. With the
//! `world` feature a node can also reach outside the process (see [`crate::world`]): `http`
//! calls an allowed loopback service, `sql` runs DuckDB over everything a batching node took,
//! and `file` writes a sink's values under `out_dir`. Such a graph is not deterministic, so the
//! paragraph below does not hold for it: its counts are reported, not predicted, the budget
//! still stops it, and the answer adds what it wrote, what went wrong, and its time.
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
use crate::petri::{self, Abstraction, Shape, ShapeNode};
use crate::{Batch, Bus, Closed, Graph, NodeId, Out, Signal};
use ix_pipeline::dag::Dag;
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(feature = "wasm")]
mod wasm;

/// Without the `wasm` feature there is no wasm node: one is refused before the run.
#[cfg(not(feature = "wasm"))]
mod wasm {
    use serde_json::Value;

    pub(super) enum Wasm {}

    impl Wasm {
        pub(super) fn new(name: &str, _: &Value) -> Result<Self, String> {
            Err(format!(
                "{name}: a wasm node needs ix-stream built with its `wasm` feature"
            ))
        }

        pub(super) fn run(&self, _: &Value) -> Result<Value, (&'static str, String)> {
            match *self {}
        }

        pub(super) fn json(&self) -> Value {
            match *self {}
        }
    }
}

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
                                "workers": { "type": "integer", "minimum": 1, "default": 1, "description": "Threads sharing its input." },
                                "wasm": { "type": "boolean", "default": false, "description": "Runs `program` on each value it takes and sends what it answers. Needs the `wasm` feature." },
                                "program": { "type": "object", "description": "A wasm node's WASI module, named as wasm.run names it: exactly one of `stage` (a cached id), `go`, `zig` (source, built once) or `wat`. It reads the value as JSON on stdin and writes one JSON value on stdout." },
                                "fuel": { "type": "integer", "minimum": 1, "default": 2000000000, "description": "A wasm node's fuel per value, about one unit per instruction." },
                                "values": { "type": "array", "description": "A source sends these values, in order, instead of { source, i }." },
                                "http": { "type": "object", "description": "(feature world) A stage that calls an allowed loopback service per value: { url, method, body, each, pick, into, timeout_ms }." },
                                "sql": { "type": "string", "description": "(feature world) A batching node that runs this DuckDB query over all it took, as the table `batch`, and sends each row." },
                                "file": { "type": "string", "description": "(feature world) A sink that writes what it takes, as JSON lines, to this path under out_dir." }
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
            "budget": { "type": "integer", "minimum": 1, "maximum": MAX_BOUND, "default": MAX_BOUND, "description": "The most values the nodes take, in all: one step each." },
            "out_dir": { "type": "string", "description": "(feature world) The folder where file sinks write and sql nodes keep their input." }
        },
        "required": ["graph"]
    })
}

pub fn stream_run_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "stopped_by": { "type": "string", "enum": ["max_events", "budget", "sources", "error"], "description": "The bound that ended the stream, `sources` when they sent all their values, or `error` when a wasm node failed and stopped it." },
            "error": { "type": "object", "description": "With `stopped_by: error`: { node, kind, message }, kind one of fuel, refused, exit, output, trap, input." },
            "events": { "type": "integer", "description": "The values the sources sent." },
            "steps": { "type": "integer", "description": "The values the nodes took: what the run spent of its budget." },
            "receipts": {
                "type": "object",
                "description": "`petri`: the check that cleared the graph; `budget`: { cap, spent }; `nodes`: what each node took and sent, { name, received, emitted }, in the order of `nodes`, and for a wasm node `wasm`: { stage, fuel_cap, out, fuel }, the values its program answered and the fuel its runs spent. A graph with world nodes adds `files` ({ node, path, rows }), `faults` and `elapsed_ms`."
            }
        }
    })
}

/// Runs an ix-stream graph as a pipeline stage, bounded by `max_events` (the values its sources
/// send) and `budget` (the values its nodes take). Deterministic: no files, no network, no
/// process but a wasm node's build, and an answer that does not depend on thread timing; only
/// `http`, `sql` and `file` nodes, under the `world` feature, reach outside.
///
/// `graph` is the shape ix-stream writes, `{ nodes: [{ name, source, batching, cap, workers }],
/// edges: [{ from, to }] }`. Sources send `{ source, i }`, stages send what they take on every
/// link, batching nodes send `{ node, count }` at the end, and a `wasm: true` stage sends what
/// its sandboxed `program` answers to each value, within `fuel`. Cycles and lossy links are
/// refused. The answer is `{ stopped_by, events, steps, receipts }`, the receipts holding the
/// Petri check, the budget spent, and each node's counts; a failed wasm node stops the stream
/// with `stopped_by: "error"` and the `error`.
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

    let wasm = load_wasm(graph, &shape)?;
    let works = works(graph, &shape)?;
    if let Some((n, _)) = shape
        .nodes
        .iter()
        .zip(&wasm)
        .zip(&works)
        .find_map(|((n, w), k)| (w.is_some() && !matches!(k, Work::Pass)).then_some((n, k)))
    {
        return Err(format!(
            "{}: a wasm node has none of values, http, sql and file",
            n.name
        ));
    }
    let world = works.iter().any(Work::is_world);

    let caps: Vec<u64> = works.iter().map(Work::cap).collect();
    let sent = if world {
        serve(&shape, &caps, max_events)
    } else {
        plan(&shape, &order, &caps, max_events, budget)
    };
    let (taken, sends) = predict(&shape, &order, &sent);
    let ledger = Ledger::new(vec![Budget::new("steps", "values taken", budget as f64)]);
    // The first wasm node to fail, as the answer's `error`.
    let failed: Arc<Mutex<Option<Value>>> = Arc::default();
    let events = Arc::new(AtomicU64::new(0));
    #[cfg(feature = "world")]
    let receipts = crate::world::Shared::default();
    #[cfg(feature = "world")]
    let dir = out_dir(&params, &works)?;
    let mut g = Graph::new(&Bus::new());
    for ((i, n), work) in shape.nodes.iter().enumerate().zip(works) {
        match work {
            Work::Values(values) => {
                let values: Vec<Value> = values.into_iter().take(sent[i] as usize).collect();
                let events = events.clone();
                g.source(&n.name, move |out| {
                    for v in &values {
                        if send_all(out, v).is_err() {
                            break;
                        }
                        events.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            Work::Pass if n.source => {
                let (name, count, events) = (n.name.clone(), sent[i], events.clone());
                g.source(&n.name, move |out| {
                    for k in 0..count {
                        if send_all(out, &json!({ "source": name, "i": k })).is_err() {
                            return;
                        }
                        events.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            Work::Pass if wasm[i].is_some() => {
                let w = wasm[i].clone().expect("a wasm node");
                let (ledger, failed) = (ledger.clone(), failed.clone());
                g.stage(&n.name, n.cap, n.workers, move |v, out| {
                    if !step(&ledger, out) {
                        return;
                    }
                    match w.run(&v) {
                        Ok(answer) => {
                            let _ = send_all(out, &answer);
                        }
                        Err((kind, message)) => {
                            let mut first = failed.lock().unwrap_or_else(|e| e.into_inner());
                            first.get_or_insert_with(
                                || json!({ "node": out.name(), "kind": kind, "message": message }),
                            );
                            // Its input drops, so the nodes upstream stop at their next send.
                            out.stop();
                        }
                    }
                });
            }
            Work::Pass if n.batching => {
                let count = Count {
                    ledger: ledger.clone(),
                    taken: 0,
                };
                g.batch(&n.name, n.cap, count);
            }
            Work::Pass => {
                let ledger = ledger.clone();
                g.stage(&n.name, n.cap, n.workers, move |v, out| {
                    if step(&ledger, out) {
                        let _ = send_all(out, &v);
                    }
                });
            }
            #[cfg(feature = "world")]
            Work::Http(http) => {
                let ledger = ledger.clone();
                g.stage(&n.name, n.cap, n.workers, move |v, out| {
                    if step(&ledger, out) {
                        let _ = send_all(out, &http.apply(v));
                    }
                });
            }
            #[cfg(feature = "world")]
            Work::Sql(sql) => {
                let inner = crate::world::Sql::new(&n.name, &sql, &dir, &receipts);
                g.batch(
                    &n.name,
                    n.cap,
                    Paid {
                        ledger: ledger.clone(),
                        inner,
                    },
                );
            }
            #[cfg(feature = "world")]
            Work::File(file) => {
                let inner = crate::world::File::new(&n.name, &file, &dir, &receipts)?;
                g.batch(
                    &n.name,
                    n.cap,
                    Paid {
                        ledger: ledger.clone(),
                        inner,
                    },
                );
            }
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
    let error = failed.lock().unwrap_or_else(|e| e.into_inner()).take();
    let spent = ledger.get("steps").map_or(0.0, |b| b.spent) as u64;
    // A stopped stream ends where the stop reached each node, which no shape predicts, and a
    // world node's counts depend on what the world answers: both are reported, not checked.
    if error.is_none() && !world {
        for (i, s) in report.nodes.iter().enumerate() {
            if s.received != taken[i] || s.emitted != sends[i] {
                return Err(format!(
                    "{} took {} values and sent {}, where its shape predicts {} and {}",
                    s.name, s.received, s.emitted, taken[i], sends[i]
                ));
            }
        }
        if spent != steps(&taken) {
            return Err(format!(
                "the nodes spent {spent} steps, where the shape predicts {}",
                steps(&taken)
            ));
        }
    }
    let events = events.load(Ordering::Relaxed);
    let drained = events == sent.iter().sum::<u64>()
        && shape
            .nodes
            .iter()
            .enumerate()
            .all(|(i, n)| !n.source || sent[i] == caps[i]);
    let nodes: Vec<Value> = report
        .nodes
        .iter()
        .zip(&wasm)
        .map(|(s, w)| {
            let mut node = json!({ "name": s.name, "received": s.received, "emitted": s.emitted });
            if let Some(w) = w {
                node["wasm"] = w.json();
            }
            node
        })
        .collect();
    let stopped_by = match error {
        Some(_) => "error",
        None if events == max_events => "max_events",
        None if drained => "sources",
        None => "budget",
    };
    let mut answer = json!({
        "stopped_by": stopped_by,
        "events": events,
        "steps": spent,
        "receipts": {
            "petri": check.json(),
            "budget": { "cap": budget, "spent": spent },
            "nodes": nodes,
        },
    });
    if let Some(error) = error {
        answer["error"] = error;
    }
    #[cfg(feature = "world")]
    if world {
        let r = receipts
            .lock()
            .map_err(|_| "a world node panicked holding the receipts")?;
        answer["receipts"]["files"] = json!(r.files);
        answer["receipts"]["faults"] = json!(r.faults);
        answer["receipts"]["elapsed_ms"] = json!(report.elapsed.as_millis() as u64);
    }
    Ok(answer)
}

/// The wasm node each node of `graph` is, loaded before the run; `None` for the other nodes.
fn load_wasm(graph: &Value, shape: &Shape) -> Result<Vec<Option<Arc<wasm::Wasm>>>, String> {
    let specs = graph["nodes"].as_array().map_or(&[][..], Vec::as_slice);
    shape
        .nodes
        .iter()
        .zip(specs)
        .map(|(n, spec)| match spec.get("wasm") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => {
                if spec.get("program").is_some() || spec.get("fuel").is_some() {
                    return Err(format!(
                        "{}: `program` and `fuel` belong to a node with `wasm: true`",
                        n.name
                    ));
                }
                Ok(None)
            }
            Some(Value::Bool(true)) => {
                if n.source || n.batching {
                    return Err(format!(
                        "{}: a wasm node is a stage, neither a source nor batching",
                        n.name
                    ));
                }
                Ok(Some(Arc::new(wasm::Wasm::new(&n.name, spec)?)))
            }
            Some(x) => Err(format!("{}: `wasm` must be true or false, got {x}", n.name)),
        })
        .collect()
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
/// can within what it has (`caps`), the events left and the budget.
fn plan(shape: &Shape, order: &[usize], caps: &[u64], max_events: u64, budget: u64) -> Vec<u64> {
    let mut sent = vec![0u64; shape.nodes.len()];
    let mut left = max_events;
    for i in 0..shape.nodes.len() {
        if !shape.nodes[i].source {
            continue;
        }
        // The steps grow with what a source sends, so halving finds the most it can send.
        let (mut lo, mut hi) = (0, left.min(caps[i]));
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

/// How many values each source of a graph with world nodes sends: in the order of the nodes,
/// what it has within the events left. Its steps cannot be predicted, so the budget is not
/// planned for: the ledger alone stops it.
fn serve(shape: &Shape, caps: &[u64], max_events: u64) -> Vec<u64> {
    let mut left = max_events;
    shape
        .nodes
        .iter()
        .zip(caps)
        .map(|(n, &cap)| {
            let sent = if n.source { left.min(cap) } else { 0 };
            left -= sent;
            sent
        })
        .collect()
}

/// What a node does beyond what its shape says, read from the fields the shape leaves.
enum Work {
    /// What the shape says.
    Pass,
    /// A source that sends these values.
    Values(Vec<Value>),
    #[cfg(feature = "world")]
    Http(Box<crate::world::Http>),
    #[cfg(feature = "world")]
    Sql(String),
    #[cfg(feature = "world")]
    File(String),
}

impl Work {
    fn is_world(&self) -> bool {
        !matches!(self, Work::Pass | Work::Values(_))
    }

    /// The most values it has to send as a source.
    fn cap(&self) -> u64 {
        match self {
            Work::Values(v) => v.len() as u64,
            _ => u64::MAX,
        }
    }
}

/// What each node of `graph` does. A node has at most one of `values` (on a source), `http`
/// (on a stage), `sql` (on a batching node) and `file` (on a sink); the last three need the
/// `world` feature.
fn works(graph: &Value, shape: &Shape) -> Result<Vec<Work>, String> {
    let raw = graph["nodes"]
        .as_array()
        .ok_or("`nodes` must be an array")?;
    shape
        .nodes
        .iter()
        .zip(raw)
        .map(|(n, j)| work(n, j))
        .collect()
}

fn work(n: &ShapeNode, j: &Value) -> Result<Work, String> {
    let given: Vec<&str> = ["values", "http", "sql", "file"]
        .into_iter()
        .filter(|k| j.get(*k).is_some_and(|v| !v.is_null()))
        .collect();
    let kind = match given[..] {
        [] => return Ok(Work::Pass),
        [kind] => kind,
        _ => {
            return Err(format!(
                "{}: a node has one of values, http, sql and file, this one {}",
                n.name,
                given.join(", ")
            ))
        }
    };
    let (fits, place) = match kind {
        "values" => (n.source, "a source"),
        "http" => (!n.source && !n.batching, "a stage that is not batching"),
        "sql" => (!n.source && n.batching, "a batching node"),
        _ => (!n.source && n.outs.is_empty(), "a sink"),
    };
    if !fits {
        return Err(format!("{}: `{kind}` goes on {place}", n.name));
    }
    let cfg = &j[kind];
    if kind == "values" {
        return cfg
            .as_array()
            .cloned()
            .map(Work::Values)
            .ok_or_else(|| format!("{}: `values` is a list", n.name));
    }
    world_work(n, kind, cfg)
}

#[cfg(feature = "world")]
fn world_work(n: &ShapeNode, kind: &str, cfg: &Value) -> Result<Work, String> {
    let text = || {
        cfg.as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{}: `{kind}` is a string", n.name))
    };
    match kind {
        "http" => crate::world::Http::from_json(&n.name, cfg).map(|h| Work::Http(Box::new(h))),
        "sql" => text().map(Work::Sql),
        _ => text().map(Work::File),
    }
}

#[cfg(not(feature = "world"))]
fn world_work(n: &ShapeNode, kind: &str, _: &Value) -> Result<Work, String> {
    Err(format!(
        "{}: a `{kind}` node reaches outside the process: build ix-stream with its `world` feature",
        n.name
    ))
}

/// `out_dir`, made if missing, when a node writes files; an empty path when none does.
#[cfg(feature = "world")]
fn out_dir(params: &Value, works: &[Work]) -> Result<std::path::PathBuf, String> {
    if !works
        .iter()
        .any(|w| matches!(w, Work::Sql(_) | Work::File(_)))
    {
        return Ok(std::path::PathBuf::new());
    }
    let dir = params
        .get("out_dir")
        .and_then(Value::as_str)
        .ok_or("sql and file nodes need `out_dir`, the folder they write in")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("out_dir {dir}: {e}"))?;
    Ok(dir.into())
}

/// A batching node that pays one step for each value it takes.
#[cfg(feature = "world")]
struct Paid<B> {
    ledger: Arc<Ledger>,
    inner: B,
}

#[cfg(feature = "world")]
impl<B: Batch> Batch for Paid<B> {
    fn take(&mut self, v: Value, out: &Out) {
        if step(&self.ledger, out) {
            self.inner.take(v, out);
        }
    }

    fn end(&mut self, out: &Out) {
        self.inner.end(out);
    }
}

/// Sends `v` on every link of the node.
pub(crate) fn send_all(out: &Out, v: &Value) -> Result<(), Closed> {
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
