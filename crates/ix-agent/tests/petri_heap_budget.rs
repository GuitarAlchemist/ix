//! `petri.analyze` admits a request on `ix_petri::json::heap_bound` taken for
//! `Output::McpResponse`. This measures the real peak heap of what the server
//! does after building the net — admit, analyse, render the `Value`,
//! pretty-print it into the `tools/call` result, and serialize the JSON-RPC
//! line — for adversarial nets, and checks it never exceeds that bound, and
//! that a net at the limits stays inside `HEAP_BUDGET_BYTES`.
//!
//! The allocator is `ix-petri`'s `tests/heap_budget.rs` one: every live
//! allocation is charged the way `heap_bound` charges it (its size rounded up
//! to 16, plus 16), and a reallocation's new block before the old one is
//! released. It is global to this test binary, which is why the file holds a
//! single test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use ix_agent::server_context::{tools_call_result, ServerContext};
use ix_agent::skills::petri::analyze_and_render;
use ix_petri::json::{admit, heap_bound, JsonNetError, NetSpec, Output};
use ix_petri::json::{HEAP_BUDGET_BYTES, MAX_STATES_CEILING};
use ix_petri::{Limits, PetriNet};
use serde_json::json;

struct Charging;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn charge(size: usize) -> usize {
    size.div_ceil(16) * 16 + 16
}

fn take(size: usize) {
    let now = LIVE.fetch_add(charge(size), SeqCst) + charge(size);
    PEAK.fetch_max(now, SeqCst);
}

fn give(size: usize) {
    LIVE.fetch_sub(charge(size), SeqCst);
}

unsafe impl GlobalAlloc for Charging {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        take(layout.size());
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        take(layout.size());
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        give(layout.size());
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        take(new_size);
        let moved = System.realloc(ptr, layout, new_size);
        give(if moved.is_null() {
            new_size
        } else {
            layout.size()
        });
        moved
    }
}

#[global_allocator]
static GLOBAL: Charging = Charging;

fn build(json: &str) -> PetriNet {
    serde_json::from_str::<NetSpec>(json)
        .unwrap()
        .build()
        .unwrap()
}

/// Peak heap above the baseline while the server answers one `tools/call`
/// for an already-built `net`, as `main.rs` does: the tool result, the
/// JSON-RPC response around it, and the line `ServerContext::write_value`
/// queues for the writer thread while the response is still alive. Returns
/// the peak and the line.
fn peak_of_one_response(net: &PetriNet, max_states: i64) -> (u128, String) {
    let (ctx, rx) = ServerContext::new();
    let base = LIVE.load(SeqCst);
    PEAK.store(base, SeqCst);
    let line = {
        let outcome = analyze_and_render(net, max_states);
        assert!(outcome.is_ok(), "admitted: {:?}", outcome.err());
        let resp = json!({ "jsonrpc": "2.0", "id": 1, "result": tools_call_result(outcome) });
        ctx.write_value(&resp);
        let line = rx.try_recv().expect("the response line was queued");
        drop(resp);
        line
    };
    let peak = (PEAK.load(SeqCst) - base) as u128;
    (peak, line)
}

fn bound(net: &PetriNet, max_states: i64) -> u128 {
    heap_bound(
        net,
        0,
        Limits::with_max_states(max_states as usize),
        Output::McpResponse,
    )
}

/// The largest budget `petri.analyze` admits for `net`.
fn admissible(net: &PetriNet) -> i64 {
    match admit(net, 0, MAX_STATES_CEILING, Output::McpResponse) {
        Ok(_) => MAX_STATES_CEILING,
        Err(JsonNetError::HeapBudget { admissible, .. }) => admissible as i64,
        Err(e) => panic!("{e}"),
    }
}

/// A control character, and a quote, as a JSON string spells them.
const CTRL: &str = "\\u0001";
const QUOTE: &str = "\\\"";

/// A chain of `steps` firings of one transition whose id is `id_len` copies of
/// `unit`, a JSON escape (a control character is one byte in memory, six
/// serialized and seven in the JSON-RPC line; a quote is one, two and four),
/// and whose name is `name_len` bytes, ending where eight transitions
/// are enabled and each leads to its own dead marking: eight witnesses each
/// `steps + 1` long. `idle` further places each hold a token and a long label,
/// so every one of the eight dead markings lists them all.
fn deep_witnesses(steps: u64, unit: &str, id_len: usize, name_len: usize, idle: usize) -> String {
    let long = unit.repeat(id_len);
    let name = "n".repeat(name_len);
    let mut places = vec![
        format!(r#"{{"id":"c","tokens":{steps}}}"#),
        r#"{"id":"d"}"#.to_string(),
        r#"{"id":"z","tokens":1}"#.to_string(),
    ];
    let mut transitions = vec![format!(r#"{{"id":"{long}","name":"{name}"}}"#)];
    let mut arcs = vec![
        format!(r#"{{"from":"c","to":"{long}"}}"#),
        format!(r#"{{"from":"{long}","to":"d"}}"#),
    ];
    for i in 0..8 {
        places.push(format!(r#"{{"id":"o{i}"}}"#));
        transitions.push(format!(r#"{{"id":"f{i}","name":"{name}{i}"}}"#));
        arcs.push(format!(r#"{{"from":"z","to":"f{i}"}}"#));
        arcs.push(format!(r#"{{"from":"d","to":"f{i}","weight":{steps}}}"#));
        arcs.push(format!(r#"{{"from":"f{i}","to":"o{i}"}}"#));
    }
    for i in 0..idle {
        places.push(format!(
            r#"{{"id":"i{i:05}","name":"{}","tokens":1}}"#,
            "\\u0002".repeat(16)
        ));
    }
    net(&places, &transitions, &arcs)
}

/// A chain of `steps` firings with `loops` self-loop transitions enabled in
/// every marking: `steps × loops` edges and a `states × transitions` liveness
/// table on an exact run.
fn dense_edges(steps: u64, loops: usize) -> String {
    let mut transitions = vec![r#"{"id":"t"}"#.to_string()];
    let mut arcs = vec![
        r#"{"from":"c","to":"t"}"#.to_string(),
        r#"{"from":"t","to":"d"}"#.to_string(),
    ];
    for i in 0..loops {
        transitions.push(format!(r#"{{"id":"l{i:05}"}}"#));
        arcs.push(format!(r#"{{"from":"h","to":"l{i:05}"}}"#));
        arcs.push(format!(r#"{{"from":"l{i:05}","to":"h"}}"#));
    }
    let places = [
        format!(r#"{{"id":"c","tokens":{steps}}}"#),
        r#"{"id":"d"}"#.to_string(),
        r#"{"id":"h","tokens":1}"#.to_string(),
    ];
    net(&places, &transitions, &arcs)
}

fn net(places: &[String], transitions: &[String], arcs: &[String]) -> String {
    format!(
        r#"{{"places":[{}],"transitions":[{}],"arcs":[{}]}}"#,
        places.join(","),
        transitions.join(","),
        arcs.join(",")
    )
}

fn check(case: &str, json: &str, max_states: i64) -> String {
    let net = build(json);
    let want = bound(&net, max_states);
    let (peak, line) = peak_of_one_response(&net, max_states);
    eprintln!(
        "{case}: max_states {max_states}, line {} B, peak {peak} B, bound {want} B ({:.0}%)",
        line.len(),
        100.0 * peak as f64 / want as f64,
    );
    assert!(
        peak <= want,
        "{case}: peak {peak} B exceeds heap_bound {want} B"
    );
    line
}

/// The line holds `name` as a label exactly once however deep the witnesses
/// are, and a witness step per state.
fn assert_labelled_once(line: &str, name_len: usize, steps: usize) {
    let name = "n".repeat(name_len);
    assert_eq!(line.matches(&format!("{name}\\\"")).count(), 1, "one label");
    assert!(line.contains("transition_labels"));
    assert!(line.matches("\\\\u0001").count() >= 8 * steps);
}

#[test]
fn mcp_response_peak_heap_stays_under_heap_bound_at_the_limits() {
    // Round 3's failing request: a 4 kB name on the transition every witness
    // step fires, which used to be copied into the result once per step.
    let line = check("names", &deep_witnesses(5_000, CTRL, 1, 4_096, 0), 6_000);
    assert_labelled_once(&line, 4_096, 5_000);

    // Control-character ids, whose escapes grow again in the pretty text and
    // again in the JSON-RPC line, and quote ids, whose escape doubles in each.
    let line = check("control-ids", &deep_witnesses(20_000, CTRL, 32, 0, 0), 25_000);
    assert!(line.matches("\\u0001").count() >= 32 * 8 * 20_000);
    let line = check("quote-ids", &deep_witnesses(20_000, QUOTE, 32, 0, 0), 25_000);
    assert!(line.matches("\\\\\\\"").count() >= 32 * 8 * 20_000);

    // Places times states, and eight dead markings listing every place.
    check("places", &deep_witnesses(2_000, CTRL, 1, 16, 2_000), 3_000);

    // Edges and the liveness table: states times transitions.
    check("edges", &dense_edges(2_000, 1_000), 3_000);

    // An unbounded net runs to its budget, whichever is admitted.
    let grow = net(
        &[r#"{"id":"q"}"#.to_string()],
        &[r#"{"id":"grow"}"#.to_string()],
        &[r#"{"from":"grow","to":"q"}"#.to_string()],
    );
    let at = admissible(&build(&grow)).min(200_000);
    check("unbounded", &grow, at);

    // At the budget itself: long escaped ids and long names at the largest
    // admitted max_states, deep enough that every state is enumerated.
    for (case, unit) in [("at-budget-control", CTRL), ("at-budget-quotes", QUOTE)] {
        let template = admissible(&build(&deep_witnesses(1, unit, 64, 1_024, 0)));
        let json = deep_witnesses(template as u64 - 16, unit, 64, 1_024, 0);
        let max_states = admissible(&build(&json));
        assert!(max_states >= template - 16 + 9, "the chain fits the budget");
        let line = check(case, &json, max_states);
        let name = "n".repeat(1_024);
        assert_eq!(line.matches(&format!("{name}\\\"")).count(), 1, "one label");
        assert!(bound(&build(&json), max_states) <= u128::from(HEAP_BUDGET_BYTES));
        assert!(matches!(
            admit(&build(&json), 0, max_states + 1, Output::McpResponse),
            Err(JsonNetError::HeapBudget { .. })
        ));
    }
}
