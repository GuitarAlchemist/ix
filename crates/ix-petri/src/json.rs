//! A JSON net description, for callers that reach this crate across a process
//! or language boundary rather than by linking it.
//!
//! The first such caller is SQL: `ix-duck` exposes [`analyze_json`] as the
//! DuckDB scalar `ix_petri_analyze(net VARCHAR, max_states BIGINT) -> VARCHAR`,
//! which is how a Node.js repository gets the analyses without a second Petri
//! implementation. The wire shape mirrors [`PetriNetBuilder`] one call per
//! entry, so nothing is inferred:
//!
//! ```json
//! {
//!   "name": "leaked-lock",
//!   "places": [{ "id": "lock", "tokens": 1 }, { "id": "working" }],
//!   "transitions": [{ "id": "acquire" }],
//!   "arcs": [{ "from": "lock", "to": "acquire" }, { "from": "acquire", "to": "working" }]
//! }
//! ```
//!
//! `tokens` defaults to 0, `weight` to 1, `name` to none. Unknown fields are
//! **rejected**, not ignored: a caller that spells `initial_marking` where this
//! reads `tokens` would otherwise analyse an empty net and be told, correctly
//! and uselessly, that it is dead at `m0`.
//!
//! Validation is [`PetriNetBuilder::build`]'s, unchanged, so a net refused here
//! is refused for exactly the reason the builder gives.
//!
//! # What bounds the memory
//!
//! A caller across a process boundary can hand over a net whose analysis would
//! need more memory than the host has, and a failed Rust allocation aborts the
//! process: no SQL error, no `try/catch`. So [`analyze_json`] refuses such a
//! call up front, and what it refuses on is bytes, not only a state count.
//!
//! The state count alone does not bound memory. Enumeration holds every
//! marking (`states × places`), every reachability edge and a liveness table
//! (`states × transitions`), and up to nine witnesses as long as the state
//! space is deep, each step a copy of a transition id that is serialized again
//! (`states × id length`). An unbounded net always runs to its budget, because
//! a deadlock found after the unboundedness witness is still a real result.
//!
//! So three limits apply, each refused rather than clamped:
//!
//! * the net JSON is at most [`MAX_NET_JSON_BYTES`];
//! * `max_states` is at most [`MAX_STATES_CEILING`];
//! * [`heap_bound`], a worst-case byte count derived from the net's own size
//!   and `max_states`, is at most [`HEAP_BUDGET_BYTES`].
//!
//! The bound covers parsing, enumeration, every analysis, the returned
//! [`Analysis`], one `serde_json::to_string` of it, and handing that string to a
//! C caller as a `CString` plus one copy of its bytes (which is what `ix-duck`
//! does). It does not cover what the host does with the string after that, and
//! it is per call: a host analysing several nets at once holds one budget per
//! call in flight, and the strings it has already been handed.

use serde::{Deserialize, Serialize};

use crate::analysis::{analyze, Analysis, Limits};
use crate::net::{PetriError, PetriNet, PetriNetBuilder};

/// A net as JSON. See the module docs for the shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetSpec {
    #[serde(default)]
    pub name: Option<String>,
    pub places: Vec<PlaceSpec>,
    pub transitions: Vec<TransitionSpec>,
    pub arcs: Vec<ArcSpec>,
}

/// One place and its initial token count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaceSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tokens: u64,
}

/// One transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
}

/// One arc, place to transition or transition to place.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArcSpec {
    pub from: String,
    pub to: String,
    #[serde(default = "one")]
    pub weight: u64,
}

fn one() -> u64 {
    1
}

/// The largest `max_states` [`analyze_json`] accepts, checked before the net is
/// parsed so an absurd budget costs nothing. It is a count cap only, and today
/// never the binding one: [`heap_bound`] charges at least about 1.4 kB per
/// state even for an empty net, so [`HEAP_BUDGET_BYTES`] refuses a budget this
/// large for every net.
/// A caller that needs more should link the crate and choose its own
/// [`Limits`].
pub const MAX_STATES_CEILING: i64 = 1_000_000;

/// The most heap [`heap_bound`] may predict for one [`analyze_json`] call.
pub const HEAP_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// The longest net JSON [`analyze_json`] parses. Parsing and building a net
/// costs a bounded multiple of its JSON length, so this caps that part of the
/// budget before any of it is spent.
pub const MAX_NET_JSON_BYTES: usize = 1024 * 1024;

/// Why a JSON net could not be analysed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JsonNetError {
    /// Not JSON, or not this shape (including an unknown field).
    #[error("invalid net JSON: {0}")]
    Parse(String),
    /// Well-formed JSON describing a net the builder refuses.
    #[error("invalid net: {0}")]
    Net(#[from] PetriError),
    /// `max_states` must admit at least the initial marking, and may not exceed
    /// [`MAX_STATES_CEILING`].
    #[error("max_states must be between 1 and {MAX_STATES_CEILING}, got {0}")]
    MaxStates(i64),
    /// The net JSON is longer than [`MAX_NET_JSON_BYTES`].
    #[error("net JSON is {0} bytes; the limit is {MAX_NET_JSON_BYTES}")]
    TooLong(usize),
    /// [`heap_bound`] for this net and budget exceeds [`HEAP_BUDGET_BYTES`].
    #[error(
        "max_states {max_states} could need {bound} bytes of heap for this net, over the \
         {HEAP_BUDGET_BYTES}-byte budget; the largest admissible max_states is {admissible}"
    )]
    HeapBudget {
        max_states: i64,
        bound: u128,
        /// 0 when the net is over the budget even at a single state.
        admissible: usize,
    },
}

impl NetSpec {
    /// Build the net through [`PetriNetBuilder`], so every structural check is
    /// the builder's own.
    pub fn build(&self) -> Result<PetriNet, PetriError> {
        let mut b = PetriNetBuilder::default();
        if let Some(name) = &self.name {
            b = b.name(name.clone());
        }
        for p in &self.places {
            b = match &p.name {
                Some(name) => b.named_place(p.id.clone(), name.clone(), p.tokens),
                None => b.place(p.id.clone(), p.tokens),
            };
        }
        for t in &self.transitions {
            b = match &t.name {
                Some(name) => b.named_transition(t.id.clone(), name.clone()),
                None => b.transition(t.id.clone()),
            };
        }
        for a in &self.arcs {
            b = b.weighted_arc(a.from.clone(), a.to.clone(), a.weight);
        }
        b.build()
    }
}

/// Parse a JSON net and run [`analyze`] with a `max_states` budget.
///
/// `max_states` is signed because SQL `BIGINT` is. Every limit in the module
/// docs is refused rather than clamped, since a silently changed budget is a
/// different analysis from the one asked for.
pub fn analyze_json(net_json: &str, max_states: i64) -> Result<Analysis, JsonNetError> {
    state_count(max_states)?;
    if net_json.len() > MAX_NET_JSON_BYTES {
        return Err(JsonNetError::TooLong(net_json.len()));
    }
    let spec: NetSpec =
        serde_json::from_str(net_json).map_err(|e| JsonNetError::Parse(e.to_string()))?;
    let net = spec.build()?;
    let limits = admit(&net, net_json.len(), max_states)?;
    Ok(analyze(&net, limits))
}

/// The [`Limits`] for analysing an already-built `net` under `max_states`, or
/// the refusal: `max_states` outside `1..=`[`MAX_STATES_CEILING`], or a
/// [`heap_bound`] over [`HEAP_BUDGET_BYTES`]. `json_len` is the length of the
/// JSON the net was parsed from, or 0 for a caller that built it another way,
/// whose own parsing is then outside the bound.
pub fn admit(net: &PetriNet, json_len: usize, max_states: i64) -> Result<Limits, JsonNetError> {
    let limits = Limits::with_max_states(state_count(max_states)?);
    let (per_state, fixed) = heap_terms(net, json_len, limits.max_reported_deadlocks);
    let bound = fixed + per_state * limits.max_states as u128;
    let budget = u128::from(HEAP_BUDGET_BYTES);
    if bound <= budget {
        return Ok(limits);
    }
    let admissible = budget.checked_sub(fixed).map_or(0, |room| room / per_state);
    Err(JsonNetError::HeapBudget {
        max_states,
        bound,
        admissible: usize::try_from(admissible).unwrap_or(usize::MAX),
    })
}

fn state_count(max_states: i64) -> Result<usize, JsonNetError> {
    Some(max_states)
        .filter(|m| (1..=MAX_STATES_CEILING).contains(m))
        .and_then(|m| usize::try_from(m).ok())
        .ok_or(JsonNetError::MaxStates(max_states))
}

/// A worst-case count of the heap bytes [`analyze_json`] holds at its peak for
/// `net`, parsed from `json_len` bytes, under `limits`: including one
/// `serde_json::to_string` of the result, a `CString` made from that string,
/// and one copy of its bytes.
///
/// It is `fixed + limits.max_states × per_state`, both read off `net`, and it
/// sums every structure the call ever holds as if all were live at once. Each
/// allocation is charged its size rounded up to 16 plus a 16-byte header, as
/// the system allocators add; fragmentation is not counted. The derivation is
/// in the comments of `heap_terms`, and `tests/heap_budget.rs` measures the
/// real peak of adversarial nets under that same charging against it.
// @ai:invariant analyze_json's peak heap, with one serialization, CString handoff and host copy of its result, is <= heap_bound(net, json_len, limits), and an admitted net at the limits stays <= HEAP_BUDGET_BYTES [P:test conf:0.75 src:heap_budget::peak_heap_stays_under_heap_bound_at_the_limits]
pub fn heap_bound(net: &PetriNet, json_len: usize, limits: Limits) -> u128 {
    let (per_state, fixed) = heap_terms(net, json_len, limits.max_reported_deadlocks);
    fixed + per_state * limits.max_states as u128
}

/// One allocation of `n` bytes as the allocator holds it.
fn alloc(n: u128) -> u128 {
    n.div_ceil(16) * 16 + 16
}

/// `(per_state, fixed)` for [`heap_bound`], for a 64-bit target: a `Vec` or
/// `String` header is 24 bytes, and a vector grown by pushing holds at most 3×
/// its elements' bytes while it reallocates.
fn heap_terms(net: &PetriNet, json_len: usize, max_reported_deadlocks: usize) -> (u128, u128) {
    let n = |x: usize| x as u128;
    let json = |s: &str| n(serde_json::to_string(s).map_or(6 * s.len() + 2, |j| j.len()));
    let longest = |ids: Vec<&str>| {
        ids.into_iter()
            .map(|id| (n(id.len()), json(id)))
            .fold((0, 0), |a, b| (a.0.max(b.0), a.1.max(b.1)))
    };
    let d = n(max_reported_deadlocks);
    let places = n(net.places().len());
    let transitions = n(net.transitions().len());
    let (tid, tid_json) = longest(net.transitions().iter().map(|t| t.id.as_str()).collect());
    let (pid, pid_json) = longest(net.places().iter().map(|p| p.id.as_str()).collect());

    // Per enumerated marking: its `markings`, `succ` and `parent` slots (24
    // bytes each, pushed: 72), its `index` B-tree entry (a 480-byte node holds
    // at least 5: 96), the BFS queue and `dead_states` (8 bytes, pushed: 24
    // each), Tarjan's four arrays (25) and two stacks (24 + 48), the liveness
    // `terminal` flag and `fired_in` row header (25); the marking twice (in
    // `markings` and as the index key); its successor list (16-byte edges, at
    // most one per transition, capacity at most 2× or the first 4); its
    // liveness row (a byte per transition); and one step in each of up to `d`
    // deadlock witnesses and the pumping sequence: a slot (72), a copy of the
    // id, and 3× its serialized `"id",` (the doubling `to_string` buffer, then
    // the `CString` and the copy made from it).
    let per_state = 482
        + 2 * alloc(8 * places)
        + alloc(32 * transitions + 64)
        + alloc(transitions)
        + (d + 1) * (72 + alloc(tid) + 3 * (tid_json + 1));

    // Parsing and building: the caller's copy of the text, `NetSpec`, the
    // builder's copy of every entry and id, `build()`'s sort buffers, id
    // indexes and arc set, and the net itself. Per JSON byte this is largest
    // for a net of minimal `{"id":"t"}` transitions, which `tests/heap_budget.rs`
    // measures at about 30 bytes, peak analysis and output included.
    let mut fixed = 96 * n(json_len) + 16_384;
    // A successor marking built and then found to exist already, the transient
    // `enabled` list in exploration and in the dead-marking scan, and one
    // successor list reallocating.
    fixed += alloc(8 * places) + 2 * alloc(24 * transitions) + alloc(48 * transitions + 64);

    // Per place: its `per_place` bound and its entry in up to `d` deadlock
    // `tokens` lists (slots and id copies), its `label=tokens ` part in up to
    // `d` deadlock markings and the two unbounded-witness markings (a pushed
    // part slot, the part, and its share of the joined string), and all of
    // that serialized, 3×.
    for p in net.places() {
        let (id, label) = (n(p.id.len()), n(p.label().len()));
        fixed += 8 + 32 + alloc(id) + d * (96 + alloc(id));
        fixed += (d + 2) * (72 + alloc(label + 21) + label + 22 + 16);
        fixed += 3 * ((d + 1) * (json(&p.id) + 24) + (d + 2) * (json(p.label()) + 20));
    }
    // Per transition: `fired`, the `never_fired` list, the `missing` B-tree
    // entry and the `live` list (slots and two id copies), serialized twice, 3×.
    for t in net.transitions() {
        fixed += 1 + 72 + 96 + 24 + 2 * alloc(n(t.id.len()));
        fixed += 3 * 2 * (json(&t.id) + 1);
    }
    // Five `unknown` reasons that can quote an overflowing transition and
    // place, the stored overflow error, and the net name, serialized 3×.
    fixed += 5 * (alloc(256 + tid + pid) + 3 * (256 + tid_json + pid_json));
    fixed += 2 * alloc(tid.max(pid));
    if let Some(name) = net.name() {
        fixed += alloc(n(name.len())) + 3 * json(name);
    }
    (per_state, fixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Verdict;

    const LEAKED_LOCK: &str = r#"{
        "name": "leaked-lock",
        "places": [{ "id": "lock", "tokens": 1 }, { "id": "working" }],
        "transitions": [{ "id": "acquire" }],
        "arcs": [{ "from": "lock", "to": "acquire" }, { "from": "acquire", "to": "working" }]
    }"#;

    // @ai:invariant analyze_json over a net equals analyze over the same net built with PetriNetBuilder — the JSON path adds no semantics [T:test conf:0.9 src:ix_petri::json::tests::json_matches_builder]
    #[test]
    fn json_matches_builder() {
        let built = PetriNet::builder()
            .name("leaked-lock")
            .place("lock", 1)
            .place("working", 0)
            .transition("acquire")
            .arc("lock", "acquire")
            .arc("acquire", "working")
            .build()
            .unwrap();
        let via_json = analyze_json(LEAKED_LOCK, 50_000).unwrap();
        assert_eq!(via_json, analyze(&built, Limits::with_max_states(50_000)));
        let Verdict::Fails(d) = &via_json.deadlock_free else {
            panic!("a lock nobody releases must wedge");
        };
        assert_eq!(
            (d[0].marking.as_str(), d[0].witness.as_slice()),
            ("working=1", &["acquire".to_string()][..])
        );
    }

    /// The serialized `Analysis` is the wire contract `ix_petri_analyze` returns
    /// and callers in other languages parse, so its bytes are pinned here, on
    /// the default `cargo test` path, rather than only behind ix-duck's `duck`
    /// feature, which CI never compiles.
    #[test]
    fn wire_shape_is_pinned() {
        let got = serde_json::to_string(&analyze_json(LEAKED_LOCK, 50_000).unwrap()).unwrap();
        let want = concat!(
            r#"{"net":"leaked-lock","states":2,"transitions_fired":1,"truncated":false,"truncation":null,"#,
            r#""deadlock_free":{"verdict":"fails","detail":[{"state":1,"marking":"working=1","tokens":[["working",1]],"witness":["acquire"]}]},"#,
            r#""deadlock_count":1,"#,
            r#""bounded":{"verdict":"holds","detail":{"k":1,"per_place":[["lock",1],["working",1]]}},"#,
            r#""unbounded_witness":null,"quasi_live":{"verdict":"holds","detail":[]},"#,
            r#""live":{"verdict":"fails","detail":["acquire"]},"reversible":{"verdict":"fails","detail":null}}"#,
        );
        assert_eq!(got, want);
    }

    #[test]
    fn unknown_field_is_refused_not_ignored() {
        let typo = LEAKED_LOCK.replace("\"tokens\"", "\"initial_marking\"");
        assert!(matches!(
            analyze_json(&typo, 10),
            Err(JsonNetError::Parse(_))
        ));
    }

    #[test]
    fn builder_refusals_pass_through() {
        let dup = r#"{"places":[{"id":"p"},{"id":"p"}],"transitions":[],"arcs":[]}"#;
        assert_eq!(
            analyze_json(dup, 10),
            Err(JsonNetError::Net(PetriError::DuplicatePlace("p".into())))
        );
    }

    #[test]
    fn non_positive_budget_is_refused() {
        assert_eq!(
            analyze_json(LEAKED_LOCK, 0),
            Err(JsonNetError::MaxStates(0))
        );
        assert_eq!(
            analyze_json(LEAKED_LOCK, -3),
            Err(JsonNetError::MaxStates(-3))
        );
    }

    #[test]
    fn budget_above_the_ceiling_is_refused_not_clamped() {
        // At the ceiling the count check passes and the heap budget decides:
        // even this two-place net's bound is over it at a million states.
        assert!(matches!(
            analyze_json(LEAKED_LOCK, MAX_STATES_CEILING),
            Err(JsonNetError::HeapBudget { admissible, .. }) if admissible > 50_000
        ));
        for over in [MAX_STATES_CEILING + 1, i64::MAX] {
            assert_eq!(
                analyze_json(LEAKED_LOCK, over),
                Err(JsonNetError::MaxStates(over))
            );
        }
    }

    #[test]
    fn json_longer_than_the_limit_is_refused_before_parsing() {
        let long = format!("{}{}", " ".repeat(MAX_NET_JSON_BYTES), LEAKED_LOCK);
        assert_eq!(
            analyze_json(&long, 10),
            Err(JsonNetError::TooLong(long.len()))
        );
    }

    /// The review's case: a transition id thousands of bytes long on a deep
    /// chain costs gigabytes in witness copies long before a million states.
    /// It is refused, and the refusal names the budget that would be admitted.
    #[test]
    fn a_budget_over_the_heap_bound_is_refused_with_the_admissible_one() {
        let chain = |tokens: u64| {
            let id = "x".repeat(25_000);
            format!(
                r#"{{"places":[{{"id":"c","tokens":{tokens}}},{{"id":"d"}}],"transitions":[{{"id":"{id}"}}],"arcs":[{{"from":"c","to":"{id}"}},{{"from":"{id}","to":"d"}}]}}"#
            )
        };
        let Err(JsonNetError::HeapBudget { bound, .. }) =
            analyze_json(&chain(999_999), MAX_STATES_CEILING)
        else {
            panic!("a million steps of a 25 kB id must be refused");
        };
        assert!(bound > u128::from(HEAP_BUDGET_BYTES));
        // The bound reads the net's size, not its depth, so a three-step chain
        // with the same id is refused alike, and its admitted run is cheap.
        let short = chain(3);
        let Err(JsonNetError::HeapBudget { admissible, .. }) =
            analyze_json(&short, MAX_STATES_CEILING)
        else {
            panic!("the refusal does not depend on depth");
        };
        assert!(admissible > 0 && (admissible as i64) < MAX_STATES_CEILING);
        assert!(analyze_json(&short, admissible as i64).is_ok());
        assert!(matches!(
            analyze_json(&short, admissible as i64 + 1),
            Err(JsonNetError::HeapBudget { admissible: a, .. }) if a == admissible
        ));
    }

    /// A marking already at `u64::MAX` is expressible in JSON. The analysis
    /// must say it could not finish, not report `bounded: holds` and a pump
    /// transition that is "never enabled".
    #[test]
    fn token_overflow_is_reported_as_truncation() {
        let net = r#"{"places":[{"id":"p","tokens":18446744073709551615}],"transitions":[{"id":"t"}],"arcs":[{"from":"t","to":"p"}]}"#;
        let a = analyze_json(net, 1_000).unwrap();
        assert!(a.truncated);
        assert!(
            matches!(a.bounded, Verdict::Unknown { .. }),
            "{:?}",
            a.bounded
        );
        assert!(
            matches!(a.quasi_live, Verdict::Unknown { .. }),
            "{:?}",
            a.quasi_live
        );
    }

    /// The UDF's refusal text quotes ids and field names verbatim. They can
    /// carry a NUL (JSON `\u0000`), which is what `ix-duck` must escape before
    /// DuckDB sees the message; this pins that the NUL really arrives here.
    #[test]
    fn refusal_text_carries_input_nul_verbatim() {
        let dup = "{\"places\":[{\"id\":\"p\\u0000\"},{\"id\":\"p\\u0000\"}],\"transitions\":[],\"arcs\":[]}";
        assert!(analyze_json(dup, 10)
            .unwrap_err()
            .to_string()
            .contains('\0'));
        let field = "{\"places\":[],\"transitions\":[],\"arcs\":[],\"x\\u0000\":1}";
        assert!(analyze_json(field, 10)
            .unwrap_err()
            .to_string()
            .contains('\0'));
    }
}
