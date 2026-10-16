//! A pod: one process running an ix-stream graph of several nodes, the unit `ixctl` supervises.
//!
//! ```text
//! cargo run -p ix-stream --example pod -- [--name pod] [--nodes 4] [--rate 20] [--crash-after S]
//! ```
//!
//! A source sends `--rate` values per second through `--nodes` − 2 stages to a sink. Every bus
//! event is one JSON line on stdout: `meta` with the graph's shape first, then `stats` every
//! 200 ms (the nodes `ixctl` shows as the pod's members), and `done` at the end. The pod stops
//! when its stdin closes, which is how `ixctl` asks it to, and exits 0 once the graph has run
//! down. With `--crash-after S` it exits with code 101 after S seconds instead, so the plane's
//! restarts can be seen.

use ix_stream::{Bus, Graph};
use serde_json::json;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The longest the source sleeps before it looks at the stop flag again.
const TICK: Duration = Duration::from_millis(50);

fn main() {
    let mut name = "pod".to_string();
    let mut nodes = 4usize;
    let mut rate = 20.0f64;
    let mut crash_after: Option<f64> = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--name" => name = value(),
            "--nodes" => nodes = value().parse().expect("--nodes"),
            "--rate" => rate = value().parse().expect("--rate"),
            "--crash-after" => crash_after = Some(value().parse().expect("--crash-after")),
            f => panic!("unknown flag {f}"),
        }
    }
    assert!(
        nodes >= 2,
        "--nodes counts the source and the sink: 2 at least"
    );
    assert!(
        rate.is_finite() && rate >= 0.0,
        "--rate is values per second"
    );

    // Set when stdin closes, or when nobody reads stdout any more.
    let quit = Arc::new(AtomicBool::new(false));
    let q = quit.clone();
    std::thread::spawn(move || {
        let mut buf = [0u8; 256];
        let mut stdin = std::io::stdin().lock();
        while matches!(stdin.read(&mut buf), Ok(n) if n > 0) {}
        q.store(true, Ordering::Relaxed);
    });
    if let Some(s) = crash_after {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(s));
            eprintln!("pod: crashing after {s} s, as --crash-after asks");
            std::process::exit(101);
        });
    }

    let bus = Bus::new();
    let events = bus.subscribe(&[], 1 << 16);
    let q = quit.clone();
    let printer = std::thread::spawn(move || {
        let mut stdout = std::io::stdout().lock();
        for e in events {
            if writeln!(stdout, "{}", e.json())
                .and_then(|()| stdout.flush())
                .is_err()
            {
                q.store(true, Ordering::Relaxed);
                return;
            }
        }
    });

    let mut g = Graph::new(&bus);
    let period = (rate > 0.0).then(|| Duration::from_secs_f64(1.0 / rate));
    let q = quit.clone();
    let src = g.source("src", move |out| {
        let mut next = Instant::now();
        let mut i = 0u64;
        while !q.load(Ordering::Relaxed) {
            let now = Instant::now();
            match period {
                Some(p) if now >= next => {
                    if out.emit(0, json!({ "i": i, "hops": 0 })).is_err() {
                        return;
                    }
                    i += 1;
                    next += p;
                }
                Some(_) => std::thread::sleep((next - now).min(TICK)),
                None => std::thread::sleep(TICK),
            }
        }
    });
    let mut prev = src;
    for k in 1..nodes - 1 {
        let stage = g.stage(&format!("s{k}"), 8, 1, |mut v, out| {
            v["hops"] = json!(v["hops"].as_u64().unwrap_or(0) + 1);
            let _ = out.emit(0, v);
        });
        g.link(prev, stage);
        prev = stage;
    }
    let seen = Arc::new(AtomicU64::new(0));
    let count = seen.clone();
    let sink = g.stage("sink", 8, 1, move |_, _| {
        count.fetch_add(1, Ordering::Relaxed);
    });
    g.link(prev, sink);

    bus.publish(
        "meta",
        json!({ "name": name, "nodes": nodes, "rate": rate, "shape": g.shape().json() }),
    );
    let report = g.run(Some(Duration::from_millis(200)));
    bus.publish(
        "done",
        json!({ "values": seen.load(Ordering::Relaxed),
                "elapsed_s": report.elapsed.as_secs_f64(), "panicked": report.panicked }),
    );
    // The last clone of the bus: its subscription closes, and the printer ends after the
    // events still queued.
    drop(bus);
    let _ = printer.join();
}
