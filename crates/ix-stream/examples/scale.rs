//! Many nodes: a chain of wasm stages fed by a DuckDB query, to watch back-pressure and
//! cancellation at scale.
//!
//! ```text
//! cargo run -p ix-stream --example scale -- [--stages 128] [--rows 2000] [--cap 2]
//!     [--slow-every 32] [--slow-ms 2] [--take N] [--serve 127.0.0.1:8792] [--linger S]
//!     [--events PATH]
//! ```
//!
//! The source is `SELECT range AS i, ... FROM range(rows)` in DuckDB. Each stage runs one WAT
//! module in ix-wasm's sandbox that copies its stdin to its stdout, so each value is parsed,
//! sandboxed and written back `stages` times. Every `--slow-every`-th stage also sleeps
//! `--slow-ms` per value: the queues upstream of it fill and the source slows to its pace.
//! With `--take N` the sink stops after N values; the stop runs up the chain and kills DuckDB.
//! With `--events PATH` every bus event (the graph's shape in `meta`, then `stats` every
//! 200 ms) is written to PATH as JSON lines when the run ends.

use ix_stream::petri::{check, Abstraction};
use ix_stream::sse::Feed;
use ix_stream::{source, Bus, Graph};
use ix_wasm::{Limits, Program};
use serde_json::{json, Value};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const PAGE: &str = include_str!("../assets/dash.html");

/// Copies stdin to stdout, 60 000 bytes at a time.
const ECHO: &str = r#"(module
  (import "wasi_snapshot_preview1" "fd_read" (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 2)
  (func (export "_start") (local $n i32)
    (loop $more
      (i32.store (i32.const 0) (i32.const 1024))
      (i32.store (i32.const 4) (i32.const 60000))
      (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))
      (local.set $n (i32.load (i32.const 8)))
      (if (i32.gt_u (local.get $n) (i32.const 0))
        (then
          (i32.store (i32.const 16) (i32.const 1024))
          (i32.store (i32.const 20) (local.get $n))
          (drop (call $fd_write (i32.const 1) (i32.const 16) (i32.const 1) (i32.const 24)))
          (br $more))))))
"#;

fn main() {
    let mut stages = 128usize;
    let mut rows = 2000u64;
    let mut cap = 2usize;
    let mut slow_every = 32usize;
    let mut slow_ms = 2u64;
    let mut take: Option<u64> = None;
    let mut serve = None;
    let mut linger = 0u64;
    let mut events: Option<std::path::PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--stages" => stages = value().parse().expect("--stages"),
            "--rows" => rows = value().parse().expect("--rows"),
            "--cap" => cap = value().parse().expect("--cap"),
            "--slow-every" => slow_every = value().parse().expect("--slow-every"),
            "--slow-ms" => slow_ms = value().parse().expect("--slow-ms"),
            "--take" => take = Some(value().parse().expect("--take")),
            "--serve" => serve = Some(value()),
            "--linger" => linger = value().parse().expect("--linger"),
            "--events" => events = Some(value().into()),
            f => panic!("unknown flag {f}"),
        }
    }
    let echo = Arc::new(Program::new(ECHO.as_bytes()).expect("the echo module"));
    let bus = Bus::new();
    let record = events.as_ref().map(|_| bus.subscribe(&[], 1 << 20));
    let mut g = Graph::new(&bus);
    let sent = Arc::new(Mutex::new(None));
    let kept = sent.clone();
    let sql = format!("SELECT range AS i, md5(range::VARCHAR) AS h FROM range({rows})");
    let src = g.source("duckdb", move |out| {
        *kept.lock().unwrap() = Some(source::duckdb(&sql, out));
    });
    let mut prev = src;
    for k in 0..stages {
        let echo = echo.clone();
        let slow = slow_every > 0 && (k + 1) % slow_every == 0;
        let node = g.stage(&format!("s{k:03}"), cap, 1, move |v, out| {
            let stdin = serde_json::to_vec(&v).unwrap();
            let o = echo.run(&stdin, &Limits::default()).expect("echo runs");
            if slow {
                std::thread::sleep(Duration::from_millis(slow_ms));
            }
            let back: Value = serde_json::from_slice(&o.stdout).expect("echo answers JSON");
            let _ = out.emit(0, back);
        });
        g.link(prev, node);
        prev = node;
    }
    let got = Arc::new(Mutex::new(0u64));
    let count = got.clone();
    let sink = g.stage_mut("sink", cap, move |v, out| {
        assert!(v["i"].is_u64());
        let mut n = count.lock().unwrap();
        *n += 1;
        if take.is_some_and(|t| *n >= t) {
            out.stop();
        }
    });
    g.link(prev, sink);

    let shape = g.shape();
    let petri = check(&shape, Abstraction::clamp(1, 1), 1_000_000).expect("petri net");
    println!("{} nodes; petri: {}", g.len(), petri.json());
    let feed = serve.as_deref().map(|addr| {
        let f = Feed::serve(addr, PAGE, &bus, &["stats", "meta", "done", "signal"], 4000)
            .expect("feed");
        println!("dashboard: http://{}/", f.addr());
        std::thread::sleep(Duration::from_secs(2));
        f
    });
    bus.publish(
        "meta",
        json!({ "title": "scale", "policy": "chaîne wasm", "items": take.unwrap_or(rows),
                "petri": petri.json(), "shape": shape.json() }),
    );
    let report = g.run(Some(Duration::from_millis(200)));
    bus.publish("done", json!({ "elapsed_s": report.elapsed.as_secs_f64() }));
    if let (Some(path), Some(record)) = (&events, &record) {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("--events file"));
        for e in record.try_iter() {
            writeln!(f, "{}", e.json()).expect("--events write");
        }
        f.flush().expect("--events write");
        println!("events: {}", path.display());
    }
    let sent = sent.lock().unwrap().take().expect("the source ran");
    let n = *got.lock().unwrap();
    let secs = report.elapsed.as_secs_f64();
    println!(
        "{n} values through {stages} wasm stages in {secs:.2} s: {:.0} values/s, {:.0} stage runs/s",
        n as f64 / secs,
        (n * stages as u64) as f64 / secs
    );
    println!("duckdb: {sent:?}");
    let source = report.node("duckdb").unwrap();
    println!(
        "source blocked {:.0} ms of {:.0} ms",
        source.blocked_ms,
        secs * 1e3
    );
    let worst = report
        .nodes
        .iter()
        .filter(|s| s.cap > 0)
        .max_by(|a, b| a.blocked_ms.total_cmp(&b.blocked_ms))
        .unwrap();
    println!(
        "most blocked stage {}: {:.0} ms; max queued anywhere {}",
        worst.name,
        worst.blocked_ms,
        report.nodes.iter().map(|s| s.max_queued).max().unwrap_or(0)
    );
    if !report.panicked.is_empty() {
        println!("panicked: {:?}", report.panicked);
    }
    if let Some(f) = feed {
        if linger > 0 {
            std::thread::sleep(Duration::from_secs(linger));
        }
        f.close();
    }
}
