//! The JSON boundary admits a call on `ix_petri::json::heap_bound`, a
//! worst-case byte count. This measures the real peak heap of adversarial nets
//! and checks it never exceeds that count, and that a net at the limits stays
//! inside `HEAP_BUDGET_BYTES`.
//!
//! The allocator below charges every live allocation the way `heap_bound`
//! does (its size rounded up to 16, plus 16), and charges a reallocation's new
//! block before releasing the old one, since a copying reallocation holds
//! both. It is global to this test binary, which is why the file holds a single
//! test: nothing else allocates while a case is measured.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::CString;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use ix_petri::json::{admit, analyze_json, heap_bound, JsonNetError, NetSpec};
use ix_petri::json::{HEAP_BUDGET_BYTES, MAX_NET_JSON_BYTES, MAX_STATES_CEILING};
use ix_petri::{Analysis, Limits, Verdict};

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

/// Peak heap above the baseline while doing what `ix_petri_analyze` does for
/// one row: copy the text out of DuckDB, analyse it, serialize the result,
/// make it a `CString`, and let the host copy its bytes.
fn peak_of_one_row(json: &str, max_states: i64) -> (u128, Analysis) {
    let base = LIVE.load(SeqCst);
    PEAK.store(base, SeqCst);
    let analysis = {
        let row = json.to_string();
        let analysis = analyze_json(&row, max_states).expect("admitted");
        let out = CString::new(serde_json::to_string(&analysis).unwrap()).unwrap();
        let host_copy = out.as_bytes().to_vec();
        assert!(!host_copy.is_empty());
        analysis
    };
    let peak = (PEAK.load(SeqCst) - base) as u128;
    (peak, analysis)
}

fn bound(json: &str, max_states: i64) -> u128 {
    let net = serde_json::from_str::<NetSpec>(json)
        .unwrap()
        .build()
        .unwrap();
    heap_bound(
        &net,
        json.len(),
        Limits::with_max_states(max_states as usize),
    )
}

/// The largest budget the boundary admits for `json`.
fn admissible(json: &str) -> i64 {
    let net = serde_json::from_str::<NetSpec>(json)
        .unwrap()
        .build()
        .unwrap();
    match admit(&net, json.len(), MAX_STATES_CEILING) {
        Ok(_) => MAX_STATES_CEILING,
        Err(JsonNetError::HeapBudget { admissible, .. }) => admissible as i64,
        Err(e) => panic!("{e}"),
    }
}

/// A chain of `steps` firings of one transition whose id is `id_len` control
/// characters (one byte each in memory, six serialized), ending where eight
/// transitions are enabled and each leads to its own dead marking: eight
/// witnesses each `steps + 1` long. `idle` further places each hold a token
/// and a long label, so every one of the eight dead markings lists them all.
fn deep_witnesses(steps: u64, id_len: usize, idle: usize) -> String {
    let long = "\\u0001".repeat(id_len);
    let mut places = vec![
        format!(r#"{{"id":"c","tokens":{steps}}}"#),
        r#"{"id":"d"}"#.to_string(),
        r#"{"id":"z","tokens":1}"#.to_string(),
    ];
    let mut transitions = vec![format!(r#"{{"id":"{long}"}}"#)];
    let mut arcs = vec![
        format!(r#"{{"from":"c","to":"{long}"}}"#),
        format!(r#"{{"from":"{long}","to":"d"}}"#),
    ];
    for i in 0..8 {
        places.push(format!(r#"{{"id":"o{i}"}}"#));
        transitions.push(format!(r#"{{"id":"f{i}"}}"#));
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

/// As close to `MAX_NET_JSON_BYTES` as whole entries allow, of `entry(i)`.
fn filled(entry: impl Fn(usize) -> String, wrap: impl Fn(&[String]) -> String) -> String {
    let mut items = Vec::new();
    let mut len = wrap(&[]).len();
    loop {
        let next = entry(items.len());
        if len + next.len() + 1 > MAX_NET_JSON_BYTES {
            return wrap(&items);
        }
        len += next.len() + 1;
        items.push(next);
    }
}

fn net(places: &[String], transitions: &[String], arcs: &[String]) -> String {
    format!(
        r#"{{"places":[{}],"transitions":[{}],"arcs":[{}]}}"#,
        places.join(","),
        transitions.join(","),
        arcs.join(",")
    )
}

fn check(case: &str, json: &str, max_states: i64) -> Analysis {
    let want = bound(json, max_states);
    let (peak, analysis) = peak_of_one_row(json, max_states);
    eprintln!(
        "{case}: {} states, json {} B, peak {peak} B, bound {want} B ({:.0}%), {:.1} B per json byte",
        analysis.states,
        json.len(),
        100.0 * peak as f64 / want as f64,
        peak as f64 / json.len() as f64,
    );
    assert!(
        peak <= want,
        "{case}: peak {peak} B exceeds heap_bound {want} B"
    );
    analysis
}

fn dead_witness_len(a: &Analysis) -> usize {
    match &a.deadlock_free {
        Verdict::Fails(d) => {
            assert_eq!(d.len(), 8, "all eight witnesses reported");
            d.iter().map(|w| w.witness.len()).min().unwrap()
        }
        other => panic!("expected eight deadlocks, got {other:?}"),
    }
}

#[test]
fn peak_heap_stays_under_heap_bound_at_the_limits() {
    // Witness copies times id length: the review's 2.6 MB-per-id-byte case.
    let a = check("witnesses", &deep_witnesses(2_000, 512, 0), 3_000);
    assert_eq!(dead_witness_len(&a), 2_001);

    // Places times states, and eight dead markings listing every place.
    let a = check("places", &deep_witnesses(2_000, 1, 2_000), 3_000);
    assert_eq!(dead_witness_len(&a), 2_001);

    // Edges and the liveness table: states times transitions.
    let a = check("edges", &dense_edges(2_000, 1_000), 3_000);
    assert!(!a.truncated && a.transitions_fired >= 2_000 * 1_000);

    // An unbounded net runs to its budget.
    let grow = net(
        &[r#"{"id":"q"}"#.to_string()],
        &[r#"{"id":"grow"}"#.to_string()],
        &[r#"{"from":"grow","to":"q"}"#.to_string()],
    );
    let a = check("unbounded", &grow, 200_000);
    assert!(a.bounded.fails() && a.states == 200_000);

    // Parsing at the JSON limit: minimal transitions (the costliest entry per
    // byte), minimal places, and minimal arcs.
    let wrap_t = |t: &[String]| net(&[], t, &[]);
    check(
        "json-transitions",
        &filled(|i| format!(r#"{{"id":"{i:x}"}}"#), wrap_t),
        1,
    );
    let wrap_p = |p: &[String]| net(p, &[], &[]);
    check(
        "json-places",
        &filled(|i| format!(r#"{{"id":"{i:x}"}}"#), wrap_p),
        1,
    );
    let pairs = filled(
        |i| format!(r#"{{"id":"t{i:x}"}}|{{"from":"p","to":"t{i:x}"}}"#),
        |pairs| {
            let (t, a): (Vec<_>, Vec<_>) = pairs
                .iter()
                .map(|s| {
                    let (t, a) = s.split_once('|').unwrap();
                    (t.to_string(), a.to_string())
                })
                .unzip();
            net(&[r#"{"id":"p","tokens":1}"#.to_string()], &t, &a)
        },
    );
    check("json-arcs", &pairs, 2);

    // At the budget itself: the witness case at the largest admitted
    // max_states, deep enough that every one of those states is enumerated.
    let template = deep_witnesses(1, 256, 0);
    let at_limit = admissible(&template);
    let json = deep_witnesses(at_limit as u64 - 16, 256, 0);
    let max_states = admissible(&json);
    assert!(max_states >= at_limit - 16 + 9, "the chain fits the budget");
    let a = check("at-budget", &json, max_states);
    assert_eq!(dead_witness_len(&a), (at_limit - 16 + 1) as usize);
    assert!(bound(&json, max_states) <= u128::from(HEAP_BUDGET_BYTES));
    assert!(matches!(
        analyze_json(&json, max_states + 1),
        Err(JsonNetError::HeapBudget { .. })
    ));
}
