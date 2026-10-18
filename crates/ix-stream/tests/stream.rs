//! The graph: back-pressure, cancellation, the bus, the gate, the Petri check, the feed.

use ix_stream::gate::{Arm, Budget, Gate, Ledger};
use ix_stream::petri::{acyclic, check, explore, Abstraction, Method, Shape};
use ix_stream::{source, Bus, Graph, Signal};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[test]
fn a_full_channel_holds_the_source_back() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let made = Arc::new(AtomicU64::new(0));
    let counted = made.clone();
    let src = g.source("numbers", move |out| {
        for i in 0..10_000 {
            if out.emit(0, json!(i)).is_err() {
                break;
            }
            counted.fetch_add(1, Ordering::Relaxed);
        }
    });
    let a = g.stage("a", 2, 1, |v, out| {
        let _ = out.emit(0, v);
    });
    let b = g.stage("b", 3, 2, |v, out| {
        let _ = out.emit(0, v);
    });
    let (release, gate) = channel::<()>();
    let gate = Mutex::new(gate);
    let seen = Arc::new(AtomicU64::new(0));
    let total = seen.clone();
    let sink = g.stage_mut("sink", 4, move |_v, _out| {
        // The first value waits until the test lets it through; the rest flow.
        if total.fetch_add(1, Ordering::Relaxed) == 0 {
            let _ = gate.lock().unwrap().recv();
        }
    });
    g.link(src, a);
    g.link(a, b);
    g.link(b, sink);
    let run = std::thread::spawn(move || g.run(None));
    std::thread::sleep(Duration::from_millis(400));
    // Values in channels, plus one in each worker's hand, plus the source's next one.
    let bound = (2 + 1) + (3 + 2) + (4 + 1) + 1;
    let held = made.load(Ordering::Relaxed);
    assert!(
        held <= bound,
        "{held} values left the source, more than {bound}"
    );
    release.send(()).unwrap();
    let report = run.join().unwrap();
    assert_eq!(seen.load(Ordering::Relaxed), 10_000);
    assert!(report.node("numbers").unwrap().blocked_ms > 100.0);
    assert!(report.node("a").unwrap().max_queued <= 3);
    assert!(report.panicked.is_empty());
}

#[test]
fn a_stopped_sink_stops_an_endless_source() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("forever", |out| {
        let mut i = 0u64;
        while out.emit(0, json!(i)).is_ok() {
            i += 1;
        }
    });
    let mid = g.stage("mid", 4, 2, |v, out| {
        let _ = out.emit(0, v);
    });
    let mut n = 0;
    let take = g.stage_mut("take", 4, move |_v, out| {
        n += 1;
        if n == 5 {
            out.stop();
        }
    });
    g.link(src, mid);
    g.link(mid, take);
    let started = Instant::now();
    let report = g.run(None);
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(report.node("take").unwrap().received >= 5);
    assert!(report.node("forever").unwrap().emitted < 100);
}

#[test]
fn a_panicking_stage_runs_the_graph_down() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("forever", |out| while out.emit(0, json!(1)).is_ok() {});
    let bad = g.stage("bad", 2, 1, |_v, _out| panic!("a stage that fails"));
    let sink = g.stage("sink", 2, 1, |_v, _out| {});
    g.link(src, bad);
    g.link(bad, sink);
    let report = g.run(None);
    assert_eq!(report.panicked, vec!["bad".to_string()]);
}

#[test]
fn a_lossy_link_never_waits() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("burst", |out| {
        for i in 0..1000 {
            let _ = out.emit(0, json!(i));
        }
    });
    let slow = g.stage("slow", 1, 1, |_v, _out| {
        std::thread::sleep(Duration::from_millis(1))
    });
    g.link_lossy(src, slow);
    let started = Instant::now();
    let report = g.run(None);
    let burst = report.node("burst").unwrap();
    assert!(burst.refused > 500, "{burst:?}");
    assert_eq!(burst.emitted + burst.refused, 1000);
    assert!(burst.blocked_ms == 0.0);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn the_bus_never_blocks_a_publisher() {
    let bus = Bus::new();
    let _slow = bus.subscribe(&["signal"], 1);
    let other = bus.subscribe(&["stats"], 8);
    let started = Instant::now();
    for i in 0..1000 {
        bus.signal(&Signal::pain("judge", 1.0, "wrong").about(&i.to_string()));
    }
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(bus.lost(), 999);
    assert!(
        other.try_recv().is_err(),
        "a subscriber gets only its kinds"
    );
    let all = bus.subscribe(&[], 4);
    bus.publish("stats", json!({}));
    assert_eq!(all.try_recv().unwrap().kind, "stats");
}

#[test]
fn a_signal_reads_back_from_its_json() {
    let s = Signal::pleasure("judge", 0.5, "right").about("0:s-1");
    assert_eq!(Signal::from_json(&s.json()), Some(s));
}

#[test]
fn the_monitor_publishes_stats_and_a_last_one() {
    let bus = Bus::new();
    let stats = bus.subscribe(&["stats"], 1024);
    let mut g = Graph::new(&bus);
    let src = g.source("few", |out| {
        source::iter((0..20).map(|i| json!(i)), out);
    });
    let sink = g.stage("sink", 2, 1, |_v, _out| {
        std::thread::sleep(Duration::from_millis(10))
    });
    g.link(src, sink);
    g.run(Some(Duration::from_millis(20)));
    let all: Vec<Value> = stats.try_iter().map(|e| e.data).collect();
    assert!(all.len() >= 3, "{}", all.len());
    let last = all.last().unwrap();
    assert_eq!(last["final"], true);
    assert_eq!(last["nodes"][1]["received"], 20);
}

#[test]
fn the_gate_spends_a_budget_where_it_gains_most() {
    // Context 0: the cheap arm is always wrong, the paid arm always right.
    // Context 1: both are right. The budget pays for half the stream.
    let ledger = Ledger::new(vec![Budget::new("paid", "calls", 100.0)]);
    let arms = vec![Arm::free("cheap"), Arm::paid("paid", 0.02, "paid", 1.0)];
    let mut gate = Gate::new(2, arms, ledger.clone(), 200);
    let mut used = [[0u32; 2]; 2];
    for i in 0..200 {
        let ctx = i % 2;
        let choice = gate.choose(ctx);
        let arm = choice.ranked[0];
        if arm == 1 {
            assert!(ledger.spend("paid", 1.0));
        }
        used[ctx][arm] += 1;
        let item = i.to_string();
        gate.commit(&item, ctx, arm);
        let right = ctx == 1 || arm == 1;
        assert!(gate.learn(&item, if right { 1.0 } else { 0.0 }));
    }
    assert!(used[0][1] >= 90, "{used:?}");
    assert!(used[1][1] <= 15, "{used:?}");
    assert!(ledger.get("paid").unwrap().spent <= 100.0);
}

#[test]
fn a_spent_budget_takes_its_arm_out_of_the_ranking() {
    let ledger = Ledger::new(vec![Budget::new("jev", "usd", 0.001)]);
    let arms = vec![Arm::free("local"), Arm::paid("jev", 0.0, "jev", 0.001)];
    let mut gate = Gate::new(1, arms, ledger.clone(), 10);
    assert!(gate.choose(0).ranked.contains(&1));
    assert!(ledger.spend("jev", 0.001));
    assert!(!ledger.spend("jev", 0.001));
    let c = gate.choose(0);
    assert_eq!(c.ranked, vec![0]);
    assert!(c.prices[1].is_infinite());
}

#[test]
fn a_chain_is_deadlock_free() {
    let mut s = Shape::default();
    let src = s.source("rows");
    let a = s.node("a", 2, 2);
    let b = s.node("b", 2, 1);
    let sink = s.node("sink", 2, 1);
    s.link(src, a, false);
    s.link(a, b, false);
    s.link(b, sink, false);
    let c = check(&s, Abstraction::exact(), 100_000).unwrap();
    assert_eq!(c.method, Method::Acyclic);
    assert_eq!(c.deadlock_free, Some(true), "{c:?}");
    // The shortcut agrees with the exploration it skips.
    let e = explore(&s, Abstraction::exact(), 100_000).unwrap();
    assert_eq!(e.method, Method::Reachability);
    assert_eq!(e.deadlock_free, Some(true), "{e:?}");
    assert_eq!(e.bounded, Some(true));
    assert!(!e.truncated);
}

#[test]
fn acyclic_fan_out_and_fan_in_explore_deadlock_free() {
    // A gate with three lanes that meet again at a judge, as in the route example.
    let mut s = Shape::default();
    let src = s.source("rows");
    let gate = s.node("gate", 2, 1);
    let jev = s.node("jev", 1, 2);
    let claude = s.node("claude", 1, 1);
    let judge = s.node("judge", 2, 1);
    s.link(src, gate, false);
    s.link(gate, judge, false);
    s.link(gate, jev, false);
    s.link(gate, claude, false);
    s.link(jev, judge, false);
    s.link(claude, judge, false);
    assert!(acyclic(&s));
    let e = explore(&s, Abstraction::exact(), 500_000).unwrap();
    assert_eq!(e.deadlock_free, Some(true), "{e:?}");
    assert!(!e.truncated);
}

/// A gate sends to an expert, the expert to a judge, and the judge sends what it rejects back
/// to the gate. With every link waiting, the three can fill and wait on each other.
fn retry_loop(lossy_retry: bool) -> Shape {
    let mut s = Shape::default();
    let src = s.source("rows");
    let gate = s.node("gate", 1, 1);
    let expert = s.node("expert", 1, 1);
    let judge = s.node("judge", 1, 1);
    let done = s.node("done", 1, 1);
    s.link(src, gate, false);
    s.link(gate, expert, false);
    s.link(expert, judge, false);
    s.link(judge, done, false);
    s.link(judge, gate, lossy_retry);
    s
}

#[test]
fn a_retry_loop_of_waiting_links_can_wedge() {
    let c = check(&retry_loop(false), Abstraction::exact(), 200_000).unwrap();
    assert_eq!(c.method, Method::Reachability);
    assert_eq!(c.deadlock_free, Some(false), "{c:?}");
    assert!(!c.witness.is_empty());
    let dead = c.dead_marking.unwrap();
    assert!(dead.contains("judge waiting on gate"), "{dead}");
}

#[test]
fn a_lossy_retry_link_frees_the_loop() {
    // Acyclic once the lossy link is set aside, and the exploration agrees.
    let c = check(&retry_loop(true), Abstraction::exact(), 200_000).unwrap();
    assert_eq!(c.method, Method::Acyclic);
    let e = explore(&retry_loop(true), Abstraction::exact(), 200_000).unwrap();
    assert_eq!(e.deadlock_free, Some(true), "{e:?}");
    assert!(!e.truncated);
}

#[test]
fn the_graph_and_its_shape_agree() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("rows", |_out| {});
    let a = g.stage("a", 3, 2, |_v, _out| {});
    let lossy = g.stage("lossy", 1, 1, |_v, _out| {});
    g.link(src, a);
    g.link_lossy(src, lossy);
    let s = g.shape();
    assert!(s.nodes[0].source);
    assert_eq!((s.nodes[1].cap, s.nodes[1].workers), (3, 2));
    assert_eq!(s.nodes[0].outs.len(), 2);
    assert!(s.nodes[0].outs[1].lossy);
    assert_eq!(
        s.json(),
        json!({
            "nodes": [
                { "name": "rows", "source": true, "batching": false, "cap": 0, "workers": 1 },
                { "name": "a", "source": false, "batching": false, "cap": 3, "workers": 2 },
                { "name": "lossy", "source": false, "batching": false, "cap": 1, "workers": 1 },
            ],
            "edges": [
                { "from": 0, "to": 1, "lossy": false },
                { "from": 0, "to": 2, "lossy": true },
            ],
        })
    );
}

#[test]
fn the_feed_streams_bus_events_to_a_browser() {
    let bus = Bus::new();
    let feed = ix_stream::sse::Feed::serve("127.0.0.1:0", "<p>page</p>", &bus, &[], 16).unwrap();
    bus.publish("meta", json!({ "before": true }));
    let addr = feed.addr();

    let mut page = std::net::TcpStream::connect(addr).unwrap();
    page.write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        .unwrap();
    let mut body = String::new();
    page.read_to_string(&mut body).unwrap();
    assert!(
        body.starts_with("HTTP/1.1 200 OK") && body.ends_with("<p>page</p>"),
        "{body}"
    );

    let mut events = std::net::TcpStream::connect(addr).unwrap();
    events
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    events
        .write_all(b"GET /events HTTP/1.1\r\nHost: x\r\n\r\n")
        .unwrap();
    // Wait for the stream to start, then publish.
    let started = Instant::now();
    while feed.clients() == 0 && started.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(10));
    }
    bus.publish("signal", json!({ "after": true }));
    let mut got = Vec::new();
    let mut buf = [0u8; 4096];
    while !String::from_utf8_lossy(&got).contains("event: signal") {
        let n = events.read(&mut buf).unwrap();
        assert!(n > 0, "the stream ended");
        got.extend_from_slice(&buf[..n]);
    }
    let text = String::from_utf8_lossy(&got);
    assert!(text.contains("text/event-stream"));
    let run = text
        .split("id: ")
        .nth(1)
        .and_then(|rest| rest.split_once('-'))
        .map(|(run, _)| run.to_string())
        .expect("an id names its run");
    assert!(
        text.contains(&format!("id: {run}-1\nevent: meta")),
        "the history comes first: {text}"
    );
    assert!(
        text.contains(&format!("id: {run}-2\nevent: signal")),
        "{text}"
    );
    drop(events);

    let replay_after = |last: &str| {
        let mut again = std::net::TcpStream::connect(addr).unwrap();
        write!(
            again,
            "GET /events?replay HTTP/1.1\r\nHost: x\r\nLast-Event-ID: {last}\r\n\r\n"
        )
        .unwrap();
        let mut replayed = String::new();
        again.read_to_string(&mut replayed).unwrap();
        replayed
    };
    // A browser that reconnects after id 1 gets only what followed it.
    let replayed = replay_after(&format!("{run}-1"));
    assert!(!replayed.contains("event: meta"), "{replayed}");
    assert!(
        replayed.contains(&format!("id: {run}-2\nevent: signal")),
        "{replayed}"
    );
    // One that saw a run before this one, or an id with no run, gets all of this one.
    for last in ["0-5", "1"] {
        let replayed = replay_after(last);
        assert!(
            replayed.contains("event: meta") && replayed.contains("event: signal"),
            "after {last}: {replayed}"
        );
    }

    // A replay with no id answers everything kept, and closes.
    let mut all = std::net::TcpStream::connect(addr).unwrap();
    all.write_all(b"GET /events?replay HTTP/1.1\r\nHost: x\r\n\r\n")
        .unwrap();
    let mut everything = String::new();
    all.read_to_string(&mut everything).unwrap();
    assert!(everything.contains("event: meta") && everything.contains("event: signal"));
    feed.close();
}

/// Needs the DuckDB CLI (`IX_DUCKDB`, else `duckdb` on the path).
#[test]
#[ignore = "needs the duckdb CLI"]
fn a_stopped_graph_stops_the_duckdb_query() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let sent = Arc::new(Mutex::new(None));
    let result = sent.clone();
    let src = g.source("duckdb", move |out| {
        // A hundred million rows: reading them all would take minutes.
        let r = source::duckdb("SELECT range AS i FROM range(100000000)", out);
        *result.lock().unwrap() = Some(r);
    });
    let mut n = 0;
    let take = g.stage_mut("take", 8, move |v, out| {
        assert!(v["i"].is_u64());
        n += 1;
        if n == 1000 {
            out.stop();
        }
    });
    g.link(src, take);
    let started = Instant::now();
    g.run(None);
    let sent = sent.lock().unwrap().take().unwrap().unwrap();
    assert!(sent.cancelled);
    assert!(sent.rows < 100_000, "{sent:?}");
    assert!(started.elapsed() < Duration::from_secs(20));
}

#[test]
#[ignore = "needs the duckdb CLI"]
fn a_duckdb_query_streams_its_rows() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let src = g.source("duckdb", |out| {
        source::duckdb("SELECT range AS i, 'x' || range AS s FROM range(500)", out).unwrap();
    });
    let rows = Arc::new(Mutex::new(Vec::new()));
    let kept = rows.clone();
    let sink = g.stage_mut("sink", 4, move |v, _out| kept.lock().unwrap().push(v));
    g.link(src, sink);
    g.run(None);
    let rows = rows.lock().unwrap();
    assert_eq!(rows.len(), 500);
    assert_eq!(rows[7], json!({ "i": 7, "s": "x7" }));
}

#[test]
#[ignore = "needs the duckdb CLI"]
fn a_bad_query_answers_duckdb_s_error() {
    let bus = Bus::new();
    let mut g = Graph::new(&bus);
    let err = Arc::new(Mutex::new(String::new()));
    let kept = err.clone();
    let src = g.source("duckdb", move |out| {
        *kept.lock().unwrap() = source::duckdb("SELECT * FROM no_such_table", out).unwrap_err();
    });
    let sink = g.stage("sink", 1, 1, |_v, _out| {});
    g.link(src, sink);
    g.run(None);
    assert!(
        err.lock().unwrap().contains("no_such_table"),
        "{}",
        err.lock().unwrap()
    );
}
