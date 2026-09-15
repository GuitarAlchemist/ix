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
//! The state budget is capped at [`MAX_STATES_CEILING`]. Enumeration holds every
//! marking in memory, and an unbounded net always runs to its budget (a
//! deadlock found after the unboundedness witness is still a real result, so
//! the search does not stop at the witness). An uncapped budget arriving across
//! a process boundary would be a way to exhaust the host's memory, not an
//! analysis.

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

/// The largest `max_states` [`analyze_json`] accepts. A million markings is
/// twenty times the in-crate default and already hundreds of megabytes for a
/// net with a few dozen places; a caller that needs more should link the crate
/// and choose its own [`Limits`].
pub const MAX_STATES_CEILING: i64 = 1_000_000;

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
/// `max_states` is signed because SQL `BIGINT` is; a value below 1 or above
/// [`MAX_STATES_CEILING`] is refused rather than clamped, since a silently
/// changed budget is a different analysis from the one asked for.
pub fn analyze_json(net_json: &str, max_states: i64) -> Result<Analysis, JsonNetError> {
    let budget = Some(max_states)
        .filter(|m| (1..=MAX_STATES_CEILING).contains(m))
        .and_then(|m| usize::try_from(m).ok())
        .ok_or(JsonNetError::MaxStates(max_states))?;
    let spec: NetSpec =
        serde_json::from_str(net_json).map_err(|e| JsonNetError::Parse(e.to_string()))?;
    Ok(analyze(&spec.build()?, Limits::with_max_states(budget)))
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
            r#"{"net":"leaked-lock","states":2,"transitions_fired":1,"truncated":false,"#,
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
        assert!(analyze_json(LEAKED_LOCK, MAX_STATES_CEILING).is_ok());
        for over in [MAX_STATES_CEILING + 1, i64::MAX] {
            assert_eq!(
                analyze_json(LEAKED_LOCK, over),
                Err(JsonNetError::MaxStates(over))
            );
        }
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
