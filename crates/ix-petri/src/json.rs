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

/// Why a JSON net could not be analysed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JsonNetError {
    /// Not JSON, or not this shape (including an unknown field).
    #[error("invalid net JSON: {0}")]
    Parse(String),
    /// Well-formed JSON describing a net the builder refuses.
    #[error("invalid net: {0}")]
    Net(#[from] PetriError),
    /// `max_states` must admit at least the initial marking.
    #[error("max_states must be >= 1, got {0}")]
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
/// `max_states` is signed because SQL `BIGINT` is; a value below 1 is refused
/// rather than clamped, since a silently raised budget is a different analysis
/// from the one asked for.
pub fn analyze_json(net_json: &str, max_states: i64) -> Result<Analysis, JsonNetError> {
    let budget = usize::try_from(max_states)
        .ok()
        .filter(|&m| m >= 1)
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
            r#""deadlock_free":{"verdict":"fails","detail":[{"state":1,"marking":"working=1","witness":["acquire"]}]},"#,
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
}
