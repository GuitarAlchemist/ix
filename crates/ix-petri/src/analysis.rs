//! Reachability-graph construction and the behavioural properties read off it:
//! **deadlock**, **boundedness**, **dead transitions**, **liveness** and
//! **reversibility**.
//!
//! These are the point of the crate. A `struct` holding places and transitions
//! is easy; what you cannot get by reading the model is whether two lanes can
//! wedge each other, whether a queue grows without limit, or whether a
//! transition you wrote can ever fire at all.
//!
//! # What is claimed, and what is not
//!
//! Everything here is computed by *explicit enumeration* of reachable markings,
//! bounded by [`Limits::max_states`]. That bound is the honesty boundary:
//!
//! * **Exhausted** (every reachable marking was visited) — deadlock,
//!   boundedness, liveness and reversibility results are exact.
//! * **Truncated with an unboundedness witness** — the witness is a pair
//!   `m < m'` where `m'` is reachable from `m`, which *proves* the net is
//!   unbounded (fire the same sequence again and the surplus compounds). The
//!   other properties become [`Verdict::Unknown`]: they range over an infinite
//!   state space this method does not cover.
//! * **Truncated with no witness** — nothing is claimed. Every property is
//!   `Unknown` and the report says how far it got.
//!
//! No approximation is ever reported as a result. A truncated run that happens
//! to have found a deadlock still reports that deadlock — a witness firing
//! sequence is a positive existence proof and truncation cannot invalidate it —
//! but it will not report the *absence* of one.
//!
//! # Determinism
//!
//! State ids are assigned in breadth-first discovery order, and the successors
//! of a state are generated in ascending transition-id order (see
//! [`crate::net`]). Both are total orders, so the state numbering, the witness
//! sequences and the point at which a truncated run stops are identical on
//! every run and every machine.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::net::{Marking, PetriNet};

/// How many ancestors the unboundedness check walks back from a newly
/// discovered marking. See [`ReachabilityGraph::strictly_covered_ancestor`].
const MAX_COVERING_WALK: usize = 512;

/// How much of the state space the analysis is allowed to enumerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// Stop after this many distinct markings. Reaching it sets
    /// [`ReachabilityGraph::truncated`].
    pub max_states: usize,
    /// Cap on how many dead markings [`Analysis::deadlocks`] lists. The count
    /// in [`Analysis::deadlock_count`] is never capped.
    pub max_reported_deadlocks: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_states: 50_000,
            max_reported_deadlocks: 8,
        }
    }
}

impl Limits {
    /// Limits with a specific state budget, other fields defaulted.
    pub fn with_max_states(max_states: usize) -> Self {
        Limits {
            max_states,
            ..Limits::default()
        }
    }
}

/// A three-valued result. `Unknown` is not a failure — it is the analysis
/// refusing to answer beyond the part of the state space it enumerated.
/// Serialized adjacently (`{"verdict": "fails", "detail": ...}`) because the
/// payload is often a sequence, which serde's internal tagging cannot encode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", content = "detail", rename_all = "snake_case")]
pub enum Verdict<T> {
    /// The property holds, and here is what was measured.
    Holds(T),
    /// The property fails, and here is the counterexample.
    Fails(T),
    /// Not decided within [`Limits`]; `reason` says why.
    Unknown { reason: String },
}

impl<T> Verdict<T> {
    /// True only for [`Verdict::Holds`] — `Unknown` is never taken as a pass.
    pub fn holds(&self) -> bool {
        matches!(self, Verdict::Holds(_))
    }

    /// True only for [`Verdict::Fails`].
    pub fn fails(&self) -> bool {
        matches!(self, Verdict::Fails(_))
    }
}

/// A reachable dead marking, with the shortest firing sequence that reaches it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deadlock {
    /// Index into [`ReachabilityGraph::markings`].
    pub state: usize,
    /// The wedged marking, rendered as `label=tokens` pairs.
    pub marking: String,
    /// Transition ids, in firing order, from `m0` to the dead marking. Shortest
    /// such sequence, because the graph is explored breadth-first.
    pub witness: Vec<String>,
}

/// Per-place token bound over the enumerated state space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bounds {
    /// `max(k)` over all places — the net is k-bounded.
    pub k: u64,
    /// `(place id, max tokens)` in place-id order.
    pub per_place: Vec<(String, u64)>,
}

/// Proof that the net is unbounded: a marking and a strictly larger marking
/// reachable from it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnboundedWitness {
    pub smaller_state: usize,
    pub smaller: String,
    pub larger_state: usize,
    pub larger: String,
    /// The transitions fired between the two, which can be repeated forever.
    pub pumping_sequence: Vec<String>,
}

/// The full behavioural report for one net.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Analysis {
    /// Net name, if it had one.
    pub net: Option<String>,
    /// Distinct markings enumerated.
    pub states: usize,
    /// Edges in the reachability graph.
    pub transitions_fired: usize,
    /// True when [`Limits::max_states`] stopped the search early.
    pub truncated: bool,

    /// `Holds(())` when no dead marking exists; `Fails(deadlocks)` otherwise.
    pub deadlock_free: Verdict<Vec<Deadlock>>,
    /// Total dead markings found, even when `deadlock_free` lists fewer.
    pub deadlock_count: usize,

    /// `Holds(bounds)` for a bounded net. On `Fails` the payload is the maxima
    /// *observed before the search stopped* — not bounds, since there are none;
    /// the evidence for the failure is [`Analysis::unbounded_witness`].
    pub bounded: Verdict<Bounds>,
    /// Present only when the net was shown unbounded.
    pub unbounded_witness: Option<UnboundedWitness>,

    /// `Holds(())` when every transition can fire somewhere; `Fails(ids)` lists
    /// the transitions that are enabled in no reachable marking.
    pub quasi_live: Verdict<Vec<String>>,

    /// L4-liveness: from every reachable marking, every transition can still
    /// eventually fire. `Fails(ids)` lists transitions absent from some
    /// terminal strongly-connected component.
    pub live: Verdict<Vec<String>>,

    /// Whether `m0` is reachable from every reachable marking.
    pub reversible: Verdict<()>,
}

/// The enumerated reachability graph. Built by [`ReachabilityGraph::explore`].
#[derive(Debug, Clone)]
pub struct ReachabilityGraph {
    markings: Vec<Marking>,
    index: BTreeMap<Marking, usize>,
    /// `succ[s]` = `(transition index, successor state)`, ascending by
    /// transition index.
    succ: Vec<Vec<(usize, usize)>>,
    /// First-discovery parent, giving the breadth-first (shortest) path to each
    /// state. `None` for the root only.
    parent: Vec<Option<(usize, usize)>>,
    truncated: bool,
    unbounded: Option<(usize, usize)>,
}

impl ReachabilityGraph {
    /// Breadth-first enumeration of the markings reachable from `m0`.
    pub fn explore(net: &PetriNet, limits: Limits) -> ReachabilityGraph {
        let m0 = net.initial_marking().clone();
        let mut graph = ReachabilityGraph {
            markings: vec![m0.clone()],
            index: BTreeMap::from([(m0, 0usize)]),
            succ: vec![Vec::new()],
            parent: vec![None],
            truncated: false,
            unbounded: None,
        };

        let mut queue: VecDeque<usize> = VecDeque::from([0usize]);
        while let Some(s) = queue.pop_front() {
            // `enabled` is ascending by transition id, so successors are
            // generated — and therefore numbered — in a fixed order.
            for t in net.enabled(&graph.markings[s]) {
                let next = match net.fire(&graph.markings[s], t) {
                    Ok(m) => m,
                    // Only reachable via token-count overflow, which is itself
                    // evidence of unboundedness; the covering check below finds
                    // it far earlier, so treat the edge as absent.
                    Err(_) => continue,
                };

                match graph.index.get(&next) {
                    Some(&existing) => graph.succ[s].push((t, existing)),
                    None => {
                        if graph.markings.len() >= limits.max_states {
                            graph.truncated = true;
                            continue;
                        }
                        let id = graph.markings.len();
                        graph.index.insert(next.clone(), id);
                        graph.markings.push(next);
                        graph.succ.push(Vec::new());
                        graph.parent.push(Some((s, t)));
                        graph.succ[s].push((t, id));

                        if graph.unbounded.is_none() {
                            if let Some(anc) = graph.strictly_covered_ancestor(id) {
                                graph.unbounded = Some((anc, id));
                            }
                        }
                        queue.push_back(id);
                    }
                }
            }
        }

        graph
    }

    /// The nearest ancestor of `state` on its discovery path that `state`
    /// strictly covers. Such a pair proves unboundedness: the sequence between
    /// them is repeatable and adds tokens each time.
    ///
    /// The walk stops after [`MAX_COVERING_WALK`] ancestors so that exploring
    /// a deep state space stays linear-ish rather than quadratic. Cutting the
    /// walk short can only *miss* a witness, degrading the boundedness verdict
    /// to [`Verdict::Unknown`]; it can never manufacture one, because a
    /// reported witness is checked componentwise against a real ancestor.
    ///
    /// `strictly_covers` implies a strictly larger token total, so the cheap
    /// `u64` total comparison screens out most ancestors before the O(places)
    /// componentwise check runs.
    fn strictly_covered_ancestor(&self, state: usize) -> Option<usize> {
        let total = self.markings[state].total();
        let mut cursor = self.parent[state].map(|(p, _)| p);
        let mut walked = 0usize;
        while let Some(a) = cursor {
            if walked >= MAX_COVERING_WALK {
                return None;
            }
            walked += 1;
            if self.markings[a].total() < total
                && self.markings[state].strictly_covers(&self.markings[a])
            {
                return Some(a);
            }
            cursor = self.parent[a].map(|(p, _)| p);
        }
        None
    }

    /// Every enumerated marking, indexed by state id.
    pub fn markings(&self) -> &[Marking] {
        &self.markings
    }

    /// Successors of `state` as `(transition index, state id)`.
    pub fn successors(&self, state: usize) -> &[(usize, usize)] {
        &self.succ[state]
    }

    /// Whether [`Limits::max_states`] cut the search short.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Transition ids fired along the shortest path from `m0` to `state`.
    pub fn witness(&self, net: &PetriNet, state: usize) -> Vec<String> {
        let mut steps = Vec::new();
        let mut cursor = state;
        while let Some((prev, t)) = self.parent[cursor] {
            steps.push(net.transitions()[t].id.clone());
            cursor = prev;
        }
        steps.reverse();
        steps
    }

    /// Transition ids fired along the shortest path from `from` to `to`, where
    /// `from` is an ancestor of `to`. Empty when it is not.
    fn segment(&self, net: &PetriNet, from: usize, to: usize) -> Vec<String> {
        let mut steps = Vec::new();
        let mut cursor = to;
        while cursor != from {
            match self.parent[cursor] {
                Some((prev, t)) => {
                    steps.push(net.transitions()[t].id.clone());
                    cursor = prev;
                }
                None => return Vec::new(),
            }
        }
        steps.reverse();
        steps
    }

    /// Strongly-connected components, as a component id per state.
    ///
    /// Iterative Tarjan: the reachability graph can be hundreds of thousands of
    /// states deep and a recursive formulation would blow the stack on exactly
    /// the nets worth analysing.
    fn scc(&self) -> (Vec<usize>, usize) {
        let n = self.markings.len();
        let mut index = vec![usize::MAX; n];
        let mut lowlink = vec![0usize; n];
        let mut on_stack = vec![false; n];
        let mut component = vec![usize::MAX; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut next_index = 0usize;
        let mut next_component = 0usize;

        for root in 0..n {
            if index[root] != usize::MAX {
                continue;
            }
            // (state, position of the next successor to visit)
            let mut call_stack: Vec<(usize, usize)> = vec![(root, 0)];
            index[root] = next_index;
            lowlink[root] = next_index;
            next_index += 1;
            stack.push(root);
            on_stack[root] = true;

            while let Some(&(v, pos)) = call_stack.last() {
                if pos < self.succ[v].len() {
                    let w = self.succ[v][pos].1;
                    call_stack
                        .last_mut()
                        .expect("call_stack is non-empty inside this loop")
                        .1 += 1;
                    if index[w] == usize::MAX {
                        index[w] = next_index;
                        lowlink[w] = next_index;
                        next_index += 1;
                        stack.push(w);
                        on_stack[w] = true;
                        call_stack.push((w, 0));
                    } else if on_stack[w] {
                        lowlink[v] = lowlink[v].min(index[w]);
                    }
                } else {
                    if lowlink[v] == index[v] {
                        while let Some(w) = stack.pop() {
                            on_stack[w] = false;
                            component[w] = next_component;
                            if w == v {
                                break;
                            }
                        }
                        next_component += 1;
                    }
                    call_stack.pop();
                    if let Some(&(parent, _)) = call_stack.last() {
                        lowlink[parent] = lowlink[parent].min(lowlink[v]);
                    }
                }
            }
        }

        (component, next_component)
    }
}

/// Run every property analysis over `net`.
///
/// This is the crate's headline entry point; see the module docs for exactly
/// what each verdict claims and when it degrades to [`Verdict::Unknown`].
pub fn analyze(net: &PetriNet, limits: Limits) -> Analysis {
    let graph = ReachabilityGraph::explore(net, limits);
    let complete = !graph.truncated;
    let unbounded = graph.unbounded;

    // An unbounded net has an infinite state space, so an *exhaustive* claim
    // over it is not something breadth-first enumeration can make.
    let exact = complete && unbounded.is_none();
    let unknown = |what: &str| -> String {
        if unbounded.is_some() {
            format!("{what} ranges over an infinite state space (the net is unbounded)")
        } else {
            format!(
                "{what} needs the full state space; exploration stopped at {} markings (max_states)",
                graph.markings.len()
            )
        }
    };

    let transitions_fired = graph.succ.iter().map(Vec::len).sum();

    // --- deadlock ---------------------------------------------------------
    // A dead marking is one with no enabled transition. Every such marking
    // found is real regardless of truncation: the witness sequence reaches it
    // from m0 and firing is deterministic.
    let dead_states: Vec<usize> = (0..graph.markings.len())
        .filter(|&s| net.enabled(&graph.markings[s]).is_empty())
        .collect();
    let deadlock_count = dead_states.len();
    let deadlocks: Vec<Deadlock> = dead_states
        .iter()
        .take(limits.max_reported_deadlocks)
        .map(|&s| Deadlock {
            state: s,
            marking: net.describe_marking(&graph.markings[s]),
            witness: graph.witness(net, s),
        })
        .collect();
    let deadlock_free = if deadlock_count > 0 {
        Verdict::Fails(deadlocks)
    } else if complete {
        Verdict::Holds(Vec::new())
    } else {
        Verdict::Unknown {
            reason: unknown("deadlock freedom"),
        }
    };

    // --- boundedness ------------------------------------------------------
    let (bounded, unbounded_witness) = match unbounded {
        Some((small, large)) => {
            let witness = UnboundedWitness {
                smaller_state: small,
                smaller: net.describe_marking(&graph.markings[small]),
                larger_state: large,
                larger: net.describe_marking(&graph.markings[large]),
                pumping_sequence: graph.segment(net, small, large),
            };
            (Verdict::Fails(place_bounds(net, &graph)), Some(witness))
        }
        None if complete => (Verdict::Holds(place_bounds(net, &graph)), None),
        None => (
            Verdict::Unknown {
                reason: unknown("boundedness"),
            },
            None,
        ),
    };

    // --- quasi-liveness (dead transitions) --------------------------------
    let mut fired = vec![false; net.transitions().len()];
    for edges in &graph.succ {
        for &(t, _) in edges {
            fired[t] = true;
        }
    }
    let never_fired: Vec<String> = fired
        .iter()
        .enumerate()
        .filter(|(_, &f)| !f)
        .map(|(t, _)| net.transitions()[t].id.clone())
        .collect();
    let quasi_live = if never_fired.is_empty() {
        Verdict::Holds(Vec::new())
    } else if exact {
        Verdict::Fails(never_fired)
    } else {
        Verdict::Unknown {
            reason: unknown("quasi-liveness"),
        }
    };

    // --- liveness and reversibility ---------------------------------------
    let (live, reversible) = if exact {
        let (component, n_components) = graph.scc();
        liveness_and_reversibility(net, &graph, &component, n_components)
    } else {
        (
            Verdict::Unknown {
                reason: unknown("liveness"),
            },
            Verdict::Unknown {
                reason: unknown("reversibility"),
            },
        )
    };

    Analysis {
        net: net.name().map(str::to_string),
        states: graph.markings.len(),
        transitions_fired,
        truncated: graph.truncated,
        deadlock_free,
        deadlock_count,
        bounded,
        unbounded_witness,
        quasi_live,
        live,
        reversible,
    }
}

fn place_bounds(net: &PetriNet, graph: &ReachabilityGraph) -> Bounds {
    let mut per_place = vec![0u64; net.places().len()];
    for m in &graph.markings {
        for (i, slot) in per_place.iter_mut().enumerate() {
            *slot = (*slot).max(m.get(i));
        }
    }
    Bounds {
        k: per_place.iter().copied().max().unwrap_or(0),
        per_place: net
            .places()
            .iter()
            .zip(&per_place)
            .map(|(p, &k)| (p.id.clone(), k))
            .collect(),
    }
}

/// L4-liveness and reversibility, both read off the SCC decomposition.
///
/// * **Live** — for a finite reachability graph, every transition can still
///   eventually fire from every reachable marking exactly when every transition
///   occurs on some edge inside *every terminal* SCC (one with no edge leaving
///   it). A dead marking is a terminal SCC with no edges at all, so a net with
///   a deadlock and at least one transition is never live — as it should be.
/// * **Reversible** — every state is reachable from `m0` by construction, so
///   `m0` is reachable from every state exactly when the whole graph is one SCC.
fn liveness_and_reversibility(
    net: &PetriNet,
    graph: &ReachabilityGraph,
    component: &[usize],
    n_components: usize,
) -> (Verdict<Vec<String>>, Verdict<()>) {
    let mut terminal = vec![true; n_components];
    let mut fired_in: Vec<Vec<bool>> = vec![vec![false; net.transitions().len()]; n_components];
    for (s, edges) in graph.succ.iter().enumerate() {
        let c = component[s];
        for &(t, next) in edges {
            if component[next] == c {
                fired_in[c][t] = true;
            } else {
                terminal[c] = false;
            }
        }
    }

    let mut missing: BTreeMap<String, ()> = BTreeMap::new();
    for c in 0..n_components {
        if !terminal[c] {
            continue;
        }
        for (t, tr) in net.transitions().iter().enumerate() {
            if !fired_in[c][t] {
                missing.insert(tr.id.clone(), ());
            }
        }
    }

    let live = if missing.is_empty() {
        Verdict::Holds(Vec::new())
    } else {
        Verdict::Fails(missing.into_keys().collect())
    };
    let reversible = if n_components <= 1 {
        Verdict::Holds(())
    } else {
        Verdict::Fails(())
    };
    (live, reversible)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclic_buffer_is_bounded_live_and_reversible() {
        let net = PetriNet::builder()
            .name("buffer")
            .place("empty", 1)
            .place("full", 0)
            .transition("consume")
            .transition("produce")
            .arc("empty", "produce")
            .arc("produce", "full")
            .arc("full", "consume")
            .arc("consume", "empty")
            .build()
            .unwrap();

        let a = analyze(&net, Limits::default());
        assert_eq!(a.states, 2);
        assert!(!a.truncated);
        assert!(a.deadlock_free.holds());
        assert_eq!(
            a.bounded,
            Verdict::Holds(Bounds {
                k: 1,
                per_place: vec![("empty".into(), 1), ("full".into(), 1)],
            })
        );
        assert!(a.quasi_live.holds());
        assert!(a.live.holds());
        assert!(a.reversible.holds());
    }

    #[test]
    fn a_transition_that_can_never_fire_is_reported() {
        let net = PetriNet::builder()
            .place("p", 0)
            .place("q", 1)
            .transition("never")
            .transition("loop")
            .arc("p", "never")
            .arc("q", "loop")
            .arc("loop", "q")
            .build()
            .unwrap();

        let a = analyze(&net, Limits::default());
        assert_eq!(a.quasi_live, Verdict::Fails(vec!["never".to_string()]));
        assert!(
            a.deadlock_free.holds(),
            "the `loop` self-cycle never wedges"
        );
    }

    #[test]
    fn an_unbounded_producer_is_proved_unbounded_by_a_pumping_witness() {
        // `grow` needs nothing and adds a token to `queue` forever.
        let net = PetriNet::builder()
            .name("unbounded-queue")
            .place("queue", 0)
            .transition("grow")
            .arc("grow", "queue")
            .build()
            .unwrap();

        let a = analyze(&net, Limits::with_max_states(50));
        assert!(a.truncated);
        assert!(a.bounded.fails(), "an infinite queue is not bounded");
        let w = a.unbounded_witness.expect("witness proves unboundedness");
        assert_eq!(w.pumping_sequence, vec!["grow".to_string()]);
        // Everything that ranges over the infinite space refuses to answer.
        assert!(matches!(a.live, Verdict::Unknown { .. }));
        assert!(matches!(a.reversible, Verdict::Unknown { .. }));
    }

    #[test]
    fn truncation_without_a_witness_claims_nothing() {
        // A long acyclic chain: bounded, but not provably so under a tiny budget.
        let mut b = PetriNet::builder().name("chain").place("p0", 1);
        for i in 0..20 {
            b = b
                .place(format!("p{}", i + 1), 0)
                .transition(format!("t{i:02}"))
                .arc(format!("p{i}"), format!("t{i:02}"))
                .arc(format!("t{i:02}"), format!("p{}", i + 1));
        }
        let net = b.build().unwrap();

        let a = analyze(&net, Limits::with_max_states(5));
        assert!(a.truncated);
        assert!(matches!(a.bounded, Verdict::Unknown { .. }));
        assert!(a.unbounded_witness.is_none());
        assert!(matches!(a.quasi_live, Verdict::Unknown { .. }));
    }

    #[test]
    fn witness_sequences_are_shortest_and_replayable() {
        let net = PetriNet::builder()
            .place("start", 1)
            .place("mid", 0)
            .place("end", 0)
            .transition("a_short")
            .transition("b_long")
            .arc("start", "a_short")
            .arc("a_short", "end")
            .arc("start", "b_long")
            .arc("b_long", "mid")
            .arc("mid", "b_long_2")
            .transition("b_long_2")
            .arc("b_long_2", "end")
            .build()
            .unwrap();

        let a = analyze(&net, Limits::default());
        let Verdict::Fails(deadlocks) = &a.deadlock_free else {
            panic!("a token parked in `end` has nowhere to go");
        };
        let d = &deadlocks[0];
        assert_eq!(
            d.witness,
            vec!["a_short".to_string()],
            "BFS finds the short path"
        );

        // Replaying the witness must reproduce the reported marking.
        let mut m = net.initial_marking().clone();
        for id in &d.witness {
            m = net.fire(&m, net.transition_index(id).unwrap()).unwrap();
        }
        assert_eq!(net.describe_marking(&m), d.marking);
    }

    #[test]
    fn exploration_is_byte_identical_across_runs() {
        let net = crate::models::dining_philosophers(4, crate::models::ForkProtocol::LeftThenRight)
            .unwrap();
        let a = serde_json::to_string(&analyze(&net, Limits::default())).unwrap();
        let b = serde_json::to_string(&analyze(&net, Limits::default())).unwrap();
        assert_eq!(a, b);
    }
}
