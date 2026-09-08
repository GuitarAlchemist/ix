//! The Place/Transition net itself: places, transitions, weighted arcs, the
//! initial marking, and the *deterministic* firing rule built on top of them.
//!
//! # Determinism, and why it is a property of the type rather than the caller
//!
//! Every analysis in [`crate::analysis`] enumerates markings by repeatedly
//! asking "which transitions are enabled here, and in what order do I try
//! them?". If that order depends on how the net happened to be assembled, two
//! runs over the same net produce different state numbering, different witness
//! sequences and different truncation points — and no result is reproducible.
//!
//! So the order is fixed by the type, not chosen at each call site:
//!
//! * [`PetriNetBuilder::build`] sorts places and transitions by their `id`
//!   strings and rejects duplicate ids. Ids are therefore unique, so byte
//!   comparison of two distinct ids never returns `Equal` — the order is
//!   **total**, with no residual tie left for a further rule to break. (Same
//!   discipline, and the same reason, as the tie-breaking note at the foot of
//!   `crates/ix-duck/sql/pareto_frontier.sql`.)
//! * A [`Marking`] is a `Vec<u64>` indexed by that sorted place order, so its
//!   derived `Ord` is a total order over markings too.
//! * [`PetriNet::enabled`] returns transition indices ascending, which *is*
//!   id-ascending.
//!
//! Consequence: `build()` is the only place where insertion order can matter,
//! and it discards it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A marking: token count per place, indexed by the net's sorted place order.
///
/// Ordering is lexicographic over that fixed order, which makes it a usable
/// `BTreeMap` key for the reachability graph's state index.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Marking(Vec<u64>);

impl Marking {
    /// Wrap a raw token vector. Callers normally get markings from
    /// [`PetriNet::initial_marking`] or [`PetriNet::fire`] instead.
    pub fn new(tokens: Vec<u64>) -> Self {
        Marking(tokens)
    }

    /// Token count at `place_index`, or 0 if the index is out of range.
    pub fn get(&self, place_index: usize) -> u64 {
        self.0.get(place_index).copied().unwrap_or(0)
    }

    /// The raw token vector, in the net's sorted place order.
    pub fn tokens(&self) -> &[u64] {
        &self.0
    }

    /// Number of places this marking covers.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when the marking covers no places at all (a net with no places).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Total tokens across all places.
    pub fn total(&self) -> u64 {
        self.0.iter().sum()
    }

    /// `self >= other` componentwise — "self covers other".
    ///
    /// Distinct from [`Ord`], which is lexicographic: covering is the partial
    /// order that matters for boundedness, lexicographic order is only the
    /// deterministic index key.
    pub fn covers(&self, other: &Marking) -> bool {
        debug_assert_eq!(self.0.len(), other.0.len());
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a >= b)
    }

    /// `self > other` componentwise — covers *and* differs somewhere.
    ///
    /// A pair `m < m'` with `m'` reachable from `m` is a witness that the net
    /// is unbounded.
    pub fn strictly_covers(&self, other: &Marking) -> bool {
        self.covers(other) && self != other
    }
}

/// A place. `id` is the identity used for ordering and for PNML interchange;
/// `name` is the human label and carries no semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Place {
    pub id: String,
    pub name: Option<String>,
}

impl Place {
    /// The label to show a human: the name if there is one, else the id.
    pub fn label(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }
}

/// A transition together with its weighted pre-set and post-set.
///
/// `pre` and `post` are `(place index, weight)` pairs sorted by place index,
/// with `weight >= 1` and at most one entry per place.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transition {
    pub id: String,
    pub name: Option<String>,
    pub pre: Vec<(usize, u64)>,
    pub post: Vec<(usize, u64)>,
}

impl Transition {
    /// The label to show a human: the name if there is one, else the id.
    pub fn label(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }
}

/// Everything that can go wrong turning a builder into a net, or firing one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PetriError {
    #[error("empty {kind} id")]
    EmptyId { kind: &'static str },

    #[error("duplicate place id `{0}`")]
    DuplicatePlace(String),

    #[error("duplicate transition id `{0}`")]
    DuplicateTransition(String),

    /// PNML types `id` as XML `ID`, which is document-unique across *all*
    /// objects, so a place and a transition may not share one either.
    #[error("id `{0}` is used by both a place and a transition")]
    IdCollision(String),

    #[error("arc `{arc}` refers to unknown node `{node}`")]
    UnknownNode { arc: String, node: String },

    #[error("arc `{arc}` connects two {kind}s; a P/T net arc must cross place and transition")]
    NotBipartite { arc: String, kind: &'static str },

    #[error("arc `{arc}` has weight 0; PNML types an inscription as a positive integer")]
    ZeroWeight { arc: String },

    /// Rejected rather than summed: summing would make the net's behaviour
    /// depend on how many times the caller happened to draw the same arc.
    #[error("duplicate arc `{from}` -> `{to}`; merge them into one weighted arc")]
    DuplicateArc { from: String, to: String },

    #[error("transition `{0}` is not enabled in this marking")]
    NotEnabled(String),

    #[error("firing transition `{transition}` overflows the token count of place `{place}`")]
    Overflow { transition: String, place: String },
}

/// A Place/Transition net: places, transitions, weighted arcs, initial marking.
///
/// Construct one with [`PetriNet::builder`] or read one with
/// [`crate::pnml::read_pnml`]. Once built it is immutable, which is what lets
/// the analyses index markings by position in a fixed place order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PetriNet {
    name: Option<String>,
    places: Vec<Place>,
    transitions: Vec<Transition>,
    initial: Marking,
}

impl PetriNet {
    /// Start building a net.
    pub fn builder() -> PetriNetBuilder {
        PetriNetBuilder::default()
    }

    /// The net's name, if it was given one.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Places, in the sorted id order that indexes every [`Marking`].
    pub fn places(&self) -> &[Place] {
        &self.places
    }

    /// Transitions, in sorted id order — the order [`Self::enabled`] returns.
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    /// The initial marking `m0`.
    pub fn initial_marking(&self) -> &Marking {
        &self.initial
    }

    /// Index of the place with this id.
    pub fn place_index(&self, id: &str) -> Option<usize> {
        self.places.binary_search_by(|p| p.id.as_str().cmp(id)).ok()
    }

    /// Index of the transition with this id.
    pub fn transition_index(&self, id: &str) -> Option<usize> {
        self.transitions
            .binary_search_by(|t| t.id.as_str().cmp(id))
            .ok()
    }

    /// Whether transition `t` may fire in `marking`.
    pub fn is_enabled(&self, marking: &Marking, t: usize) -> bool {
        match self.transitions.get(t) {
            Some(tr) => tr.pre.iter().all(|&(p, w)| marking.get(p) >= w),
            None => false,
        }
    }

    /// Every transition enabled in `marking`, as indices in ascending order.
    ///
    /// Ascending index is ascending transition id, which is a total order (ids
    /// are unique), so this is the same sequence on every run and every
    /// machine. An empty result means `marking` is a **dead marking**.
    // @ai:invariant build() discards builder insertion order, so two nets described in different orders are equal and every analysis over them enumerates identically [T:test conf:0.95 src:net::tests::build_sorts_nodes_by_id_regardless_of_insertion_order]
    pub fn enabled(&self, marking: &Marking) -> Vec<usize> {
        (0..self.transitions.len())
            .filter(|&t| self.is_enabled(marking, t))
            .collect()
    }

    /// Fire transition `t` in `marking`, returning the successor marking.
    ///
    /// Consumes the pre-set before producing the post-set, so a self-loop (a
    /// place in both sets) behaves as the standard P/T semantics require.
    pub fn fire(&self, marking: &Marking, t: usize) -> Result<Marking, PetriError> {
        let tr = self
            .transitions
            .get(t)
            .ok_or_else(|| PetriError::NotEnabled(format!("<index {t}>")))?;
        if !self.is_enabled(marking, t) {
            return Err(PetriError::NotEnabled(tr.id.clone()));
        }

        let mut next = marking.0.clone();
        for &(p, w) in &tr.pre {
            next[p] -= w;
        }
        for &(p, w) in &tr.post {
            next[p] = next[p].checked_add(w).ok_or_else(|| PetriError::Overflow {
                transition: tr.id.clone(),
                place: self.places[p].id.clone(),
            })?;
        }
        Ok(Marking(next))
    }

    /// Map a firing sequence of transition ids to their human labels.
    ///
    /// Witnesses carry ids because those are the stable identity
    /// [`Self::transition_index`] accepts, so a caller can replay them. This is
    /// for showing the same sequence to a person; unknown ids pass through.
    pub fn label_sequence(&self, ids: &[String]) -> Vec<String> {
        ids.iter()
            .map(|id| match self.transition_index(id) {
                Some(t) => self.transitions[t].label().to_string(),
                None => id.clone(),
            })
            .collect()
    }

    /// Render a marking as `label=tokens` pairs for the places that hold any,
    /// in place-id order. A marking with no tokens renders as `(empty)`.
    pub fn describe_marking(&self, marking: &Marking) -> String {
        let parts: Vec<String> = self
            .places
            .iter()
            .enumerate()
            .filter(|(i, _)| marking.get(*i) > 0)
            .map(|(i, p)| format!("{}={}", p.label(), marking.get(i)))
            .collect();
        if parts.is_empty() {
            "(empty)".to_string()
        } else {
            parts.join(" ")
        }
    }
}

/// Incremental net description. Nodes and arcs may be added in any order —
/// [`Self::build`] imposes the canonical one.
#[derive(Debug, Default, Clone)]
pub struct PetriNetBuilder {
    name: Option<String>,
    places: Vec<(String, Option<String>, u64)>,
    transitions: Vec<(String, Option<String>)>,
    arcs: Vec<ArcSpec>,
}

#[derive(Debug, Clone)]
struct ArcSpec {
    id: String,
    source: String,
    target: String,
    weight: u64,
}

impl PetriNetBuilder {
    /// Name the net (optional; PNML carries it as the net's `<name>`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a place holding `tokens` initially.
    pub fn place(mut self, id: impl Into<String>, tokens: u64) -> Self {
        self.places.push((id.into(), None, tokens));
        self
    }

    /// Add a place with a human label.
    pub fn named_place(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        tokens: u64,
    ) -> Self {
        self.places.push((id.into(), Some(name.into()), tokens));
        self
    }

    /// Add a transition.
    pub fn transition(mut self, id: impl Into<String>) -> Self {
        self.transitions.push((id.into(), None));
        self
    }

    /// Add a transition with a human label.
    pub fn named_transition(mut self, id: impl Into<String>, name: impl Into<String>) -> Self {
        self.transitions.push((id.into(), Some(name.into())));
        self
    }

    /// Add a weight-1 arc between a place and a transition (either direction).
    pub fn arc(self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.weighted_arc(source, target, 1)
    }

    /// Add an arc with an explicit inscription (must be >= 1).
    pub fn weighted_arc(
        self,
        source: impl Into<String>,
        target: impl Into<String>,
        weight: u64,
    ) -> Self {
        let source = source.into();
        let target = target.into();
        let id = format!("{source}->{target}");
        self.arc_with_id(id, source, target, weight)
    }

    /// Add an arc carrying its own PNML id, so a reader can report violations
    /// against the id the source document actually used.
    pub fn arc_with_id(
        mut self,
        id: impl Into<String>,
        source: impl Into<String>,
        target: impl Into<String>,
        weight: u64,
    ) -> Self {
        self.arcs.push(ArcSpec {
            id: id.into(),
            source: source.into(),
            target: target.into(),
            weight,
        });
        self
    }

    /// Validate and freeze into a [`PetriNet`].
    ///
    /// Checks run in a fixed order so the *first* error reported for a given
    /// input is stable: id well-formedness, then duplicate ids, then the
    /// place/transition id collision, then per-arc endpoints, weight and
    /// duplication in arc-insertion order.
    pub fn build(self) -> Result<PetriNet, PetriError> {
        for (id, _, _) in &self.places {
            if id.is_empty() {
                return Err(PetriError::EmptyId { kind: "place" });
            }
        }
        for (id, _) in &self.transitions {
            if id.is_empty() {
                return Err(PetriError::EmptyId { kind: "transition" });
            }
        }

        let mut places: Vec<(String, Option<String>, u64)> = self.places;
        places.sort_by(|a, b| a.0.cmp(&b.0));
        if let Some(w) = places.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(PetriError::DuplicatePlace(w[0].0.clone()));
        }

        let mut transitions: Vec<(String, Option<String>)> = self.transitions;
        transitions.sort_by(|a, b| a.0.cmp(&b.0));
        if let Some(w) = transitions.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(PetriError::DuplicateTransition(w[0].0.clone()));
        }

        let place_index: BTreeMap<&str, usize> = places
            .iter()
            .enumerate()
            .map(|(i, (id, _, _))| (id.as_str(), i))
            .collect();
        let transition_index: BTreeMap<&str, usize> = transitions
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.as_str(), i))
            .collect();

        if let Some(id) = place_index
            .keys()
            .find(|id| transition_index.contains_key(**id))
        {
            return Err(PetriError::IdCollision((*id).to_string()));
        }

        let mut pre: Vec<Vec<(usize, u64)>> = vec![Vec::new(); transitions.len()];
        let mut post: Vec<Vec<(usize, u64)>> = vec![Vec::new(); transitions.len()];
        let mut seen_arcs: BTreeMap<(&str, &str), ()> = BTreeMap::new();

        for arc in &self.arcs {
            let src_place = place_index.get(arc.source.as_str()).copied();
            let src_trans = transition_index.get(arc.source.as_str()).copied();
            let dst_place = place_index.get(arc.target.as_str()).copied();
            let dst_trans = transition_index.get(arc.target.as_str()).copied();

            if src_place.is_none() && src_trans.is_none() {
                return Err(PetriError::UnknownNode {
                    arc: arc.id.clone(),
                    node: arc.source.clone(),
                });
            }
            if dst_place.is_none() && dst_trans.is_none() {
                return Err(PetriError::UnknownNode {
                    arc: arc.id.clone(),
                    node: arc.target.clone(),
                });
            }
            if src_place.is_some() && dst_place.is_some() {
                return Err(PetriError::NotBipartite {
                    arc: arc.id.clone(),
                    kind: "place",
                });
            }
            if src_trans.is_some() && dst_trans.is_some() {
                return Err(PetriError::NotBipartite {
                    arc: arc.id.clone(),
                    kind: "transition",
                });
            }
            if arc.weight == 0 {
                return Err(PetriError::ZeroWeight {
                    arc: arc.id.clone(),
                });
            }
            if seen_arcs
                .insert((arc.source.as_str(), arc.target.as_str()), ())
                .is_some()
            {
                return Err(PetriError::DuplicateArc {
                    from: arc.source.clone(),
                    to: arc.target.clone(),
                });
            }

            match (src_place, dst_trans) {
                (Some(p), Some(t)) => pre[t].push((p, arc.weight)),
                _ => {
                    let t = src_trans.expect("source is a transition (bipartite check passed)");
                    let p = dst_place.expect("target is a place (bipartite check passed)");
                    post[t].push((p, arc.weight));
                }
            }
        }

        for v in pre.iter_mut().chain(post.iter_mut()) {
            v.sort_by_key(|&(p, _)| p);
        }

        let initial = Marking(places.iter().map(|(_, _, tokens)| *tokens).collect());
        let transitions = transitions
            .into_iter()
            .zip(pre)
            .zip(post)
            .map(|(((id, name), pre), post)| Transition {
                id,
                name,
                pre,
                post,
            })
            .collect();
        let places = places
            .into_iter()
            .map(|(id, name, _)| Place { id, name })
            .collect();

        Ok(PetriNet {
            name: self.name,
            places,
            transitions,
            initial,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Producer/consumer over a one-slot buffer.
    fn buffer_net() -> PetriNet {
        PetriNet::builder()
            .name("buffer")
            .place("empty", 1)
            .place("full", 0)
            .transition("produce")
            .transition("consume")
            .arc("empty", "produce")
            .arc("produce", "full")
            .arc("full", "consume")
            .arc("consume", "empty")
            .build()
            .expect("valid net")
    }

    #[test]
    fn build_sorts_nodes_by_id_regardless_of_insertion_order() {
        let a = PetriNet::builder()
            .place("z", 1)
            .place("a", 0)
            .transition("t2")
            .transition("t1")
            .build()
            .unwrap();
        let b = PetriNet::builder()
            .place("a", 0)
            .place("z", 1)
            .transition("t1")
            .transition("t2")
            .build()
            .unwrap();

        assert_eq!(a, b, "insertion order must not survive build()");
        assert_eq!(
            a.places().iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
            ["a", "z"]
        );
        assert_eq!(a.initial_marking().tokens(), [0, 1]);
    }

    #[test]
    fn enabled_is_ascending_transition_id_order() {
        let net = buffer_net();
        let m0 = net.initial_marking();
        assert_eq!(
            net.enabled(m0),
            vec![net.transition_index("produce").unwrap()]
        );

        let m1 = net
            .fire(m0, net.transition_index("produce").unwrap())
            .unwrap();
        assert_eq!(
            net.enabled(&m1),
            vec![net.transition_index("consume").unwrap()]
        );
        assert_eq!(
            net.fire(&m1, net.transition_index("consume").unwrap())
                .unwrap(),
            *m0
        );
    }

    #[test]
    fn firing_a_disabled_transition_is_an_error_not_a_negative_marking() {
        let net = buffer_net();
        let consume = net.transition_index("consume").unwrap();
        assert_eq!(
            net.fire(net.initial_marking(), consume),
            Err(PetriError::NotEnabled("consume".into()))
        );
    }

    #[test]
    fn self_loop_consumes_before_producing() {
        let net = PetriNet::builder()
            .place("p", 1)
            .transition("t")
            .arc("p", "t")
            .arc("t", "p")
            .build()
            .unwrap();
        let m = net.fire(net.initial_marking(), 0).unwrap();
        assert_eq!(m.tokens(), [1], "a self-loop is marking-preserving");
    }

    #[test]
    fn weighted_arcs_gate_on_the_inscription() {
        let net = PetriNet::builder()
            .place("p", 2)
            .place("q", 0)
            .transition("t")
            .weighted_arc("p", "t", 3)
            .arc("t", "q")
            .build()
            .unwrap();
        assert!(net.enabled(net.initial_marking()).is_empty());
    }

    #[test]
    fn validation_rejects_malformed_nets() {
        let dup = PetriNet::builder().place("p", 0).place("p", 1).build();
        assert_eq!(dup.unwrap_err(), PetriError::DuplicatePlace("p".into()));

        let collide = PetriNet::builder().place("x", 0).transition("x").build();
        assert_eq!(collide.unwrap_err(), PetriError::IdCollision("x".into()));

        let unknown = PetriNet::builder().place("p", 0).arc("p", "nope").build();
        assert_eq!(
            unknown.unwrap_err(),
            PetriError::UnknownNode {
                arc: "p->nope".into(),
                node: "nope".into()
            }
        );

        let flat = PetriNet::builder()
            .place("p", 0)
            .place("q", 0)
            .arc("p", "q")
            .build();
        assert_eq!(
            flat.unwrap_err(),
            PetriError::NotBipartite {
                arc: "p->q".into(),
                kind: "place"
            }
        );

        let zero = PetriNet::builder()
            .place("p", 0)
            .transition("t")
            .weighted_arc("p", "t", 0)
            .build();
        assert_eq!(
            zero.unwrap_err(),
            PetriError::ZeroWeight { arc: "p->t".into() }
        );

        let twice = PetriNet::builder()
            .place("p", 0)
            .transition("t")
            .arc("p", "t")
            .arc("p", "t")
            .build();
        assert_eq!(
            twice.unwrap_err(),
            PetriError::DuplicateArc {
                from: "p".into(),
                to: "t".into()
            }
        );
    }

    #[test]
    fn covering_is_componentwise_not_lexicographic() {
        let many_tokens = Marking::new(vec![0, 5]);
        let few_tokens = Marking::new(vec![1, 0]);

        // Lexicographic order (the deterministic index key) sorts on place 0
        // first, so the marking with *more* tokens sorts lower...
        assert!(many_tokens < few_tokens);
        // ...while covering, the order boundedness reasons with, relates
        // neither of them.
        assert!(!many_tokens.covers(&few_tokens));
        assert!(!few_tokens.covers(&many_tokens));

        assert!(Marking::new(vec![1, 5]).strictly_covers(&few_tokens));
        assert!(!few_tokens.strictly_covers(&few_tokens));
    }
}
