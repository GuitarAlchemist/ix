//! Batch nodes, and the SQL nodes built on them.

use ix_stream::petri::{explore, Abstraction, Shape};
use ix_stream::sql::{query, ParquetSink, Window};
use ix_stream::{source, Batch, Bus, Graph, Out};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Sends the sum of what it took, every `every` values and at the end.
struct Sum {
    every: usize,
    held: Vec<i64>,
    ended: Arc<AtomicBool>,
}

impl Sum {
    fn send(&mut self, out: &Out) {
        if !self.held.is_empty() {
            let s: i64 = self.held.drain(..).sum();
            let _ = out.emit(0, json!(s));
        }
    }
}

impl Batch for Sum {
    fn take(&mut self, v: Value, out: &Out) {
        self.held.push(v.as_i64().unwrap());
        if self.held.len() == self.every {
            self.send(out);
        }
    }

    fn end(&mut self, out: &Out) {
        self.ended.store(true, Ordering::Relaxed);
        self.send(out);
    }
}

fn collect(g: &mut Graph, cap: usize) -> (ix_stream::NodeId, Arc<Mutex<Vec<Value>>>) {
    let got = Arc::new(Mutex::new(Vec::new()));
    let kept = got.clone();
    let sink = g.stage_mut("sink", cap, move |v, _out| kept.lock().unwrap().push(v));
    (sink, got)
}

#[test]
fn a_batch_sends_what_it_held_when_the_stream_ends() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("ten", |out| {
        source::iter((1..=10).map(|i| json!(i)), out);
    });
    let ended = Arc::new(AtomicBool::new(false));
    let sum = g.batch(
        "sum",
        4,
        Sum {
            every: 4,
            held: Vec::new(),
            ended: ended.clone(),
        },
    );
    let (sink, got) = collect(&mut g, 4);
    g.link(src, sum);
    g.link(sum, sink);
    g.run(None);
    assert!(ended.load(Ordering::Relaxed));
    assert_eq!(*got.lock().unwrap(), vec![json!(10), json!(26), json!(19)]);
}

#[test]
fn a_cancelled_batch_does_not_end() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("forever", |out| {
        let mut i = 0;
        while out.emit(0, json!(i)).is_ok() {
            i += 1;
        }
    });
    let ended = Arc::new(AtomicBool::new(false));
    let sum = g.batch(
        "sum",
        4,
        Sum {
            every: 3,
            held: Vec::new(),
            ended: ended.clone(),
        },
    );
    let mut n = 0;
    let take = g.stage_mut("take", 2, move |_v, out| {
        n += 1;
        if n == 2 {
            out.stop();
        }
    });
    g.link(src, sum);
    g.link(sum, take);
    g.run(None);
    assert!(
        !ended.load(Ordering::Relaxed),
        "nothing would read what end sends"
    );
}

#[test]
fn a_window_can_answer_whenever_it_chooses_and_stay_deadlock_free() {
    let mut s = Shape::default();
    let src = s.source("rows");
    let w = s.node("window", 2, 1);
    s.nodes[w].batching = true;
    let sink = s.node("sink", 1, 1);
    s.link(src, w, false);
    s.link(w, sink, false);
    let e = explore(&s, Abstraction::exact(), 100_000).unwrap();
    assert_eq!(e.deadlock_free, Some(true), "{e:?}");
    assert!(e.transitions > 0);
}

#[test]
#[ignore = "needs the duckdb CLI"]
fn a_window_runs_its_query_over_each_batch() {
    let dir = tempdir();
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("rows", |out| {
        source::iter((0..250).map(|i| json!({ "i": i, "g": i % 3 })), out);
    });
    let w = g.batch(
        "w",
        16,
        Window::new(
            "w",
            "SELECT g, count(*) AS n, sum(i)::BIGINT AS s FROM batch GROUP BY g ORDER BY g",
            100,
            &dir,
        ),
    );
    let (sink, got) = collect(&mut g, 16);
    g.link(src, w);
    g.link(w, sink);
    g.run(None);
    let got = got.lock().unwrap();
    assert_eq!(got.len(), 9, "{got:?}");
    let n: i64 = got.iter().map(|r| r["n"].as_i64().unwrap()).sum();
    let s: i64 = got.iter().map(|r| r["s"].as_i64().unwrap()).sum();
    assert_eq!((n, s), (250, (0..250).sum()));
    let last: Vec<&Value> = got.iter().filter(|r| r["window"] == 3).collect();
    let left: i64 = last.iter().map(|r| r["n"].as_i64().unwrap()).sum();
    assert_eq!(
        (last.len(), left),
        (3, 50),
        "the last window holds the 50 that were left"
    );
}

#[test]
#[ignore = "needs the duckdb CLI"]
fn a_window_with_a_bad_query_says_so_on_the_bus() {
    let dir = tempdir();
    let bus = Bus::new();
    let signals = bus.subscribe(&["signal"], 16);
    let mut g = Graph::new(&bus);
    let src = g.source("rows", |out| {
        source::iter((0..5).map(|i| json!({ "i": i })), out);
    });
    let w = g.batch("w", 4, Window::new("w", "SELECT nope FROM batch", 10, &dir));
    let (sink, got) = collect(&mut g, 4);
    g.link(src, w);
    g.link(w, sink);
    g.run(None);
    assert!(got.lock().unwrap().is_empty());
    let pain = signals.try_recv().unwrap();
    assert_eq!(pain.data["valence"], -1);
    assert!(pain.data["reason"].as_str().unwrap().contains("nope"));
}

#[test]
#[ignore = "needs the duckdb CLI"]
fn a_parquet_sink_keeps_the_stream() {
    let dir = tempdir();
    let path = dir.join("kept.parquet");
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("rows", |out| {
        source::iter(
            (0..300).map(|i| json!({ "i": i, "label": format!("x{i}") })),
            out,
        );
    });
    let sink = g.batch("parquet", 8, ParquetSink::new("parquet", &path).unwrap());
    g.link(src, sink);
    g.run(None);
    let p = path.display().to_string().replace('\\', "/");
    let rows = query(&format!(
        "SELECT count(*) AS n, max(i) AS top FROM read_parquet('{p}')"
    ))
    .unwrap();
    assert_eq!(rows, vec![json!({ "n": 300, "top": 299 })]);
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ix-stream-sql-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let dir = std::path::PathBuf::from(
        dir.display()
            .to_string()
            .replace(['(', ')'], "")
            .replace("ThreadId", "t"),
    );
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
