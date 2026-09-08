//! Layer 4: Topological Code Structure.
//!
//! Computes persistent homology on call graphs to extract higher-order
//! structural invariants: connected components (Betti_0), cycles (Betti_1),
//! and their persistence across filtration scales.
//!
//! Edge weights in the call graph (typically call frequency or coupling
//! strength) are inverted into distances: strongly connected functions are
//! "close", loosely connected functions are "far". The resulting (symmetric)
//! distance matrix drives a Vietoris-Rips filtration, which is then fed to
//! `ix_topo::persistence::compute_persistence`.
//!
//! H2 is intentionally skipped: with n call-graph nodes, triangles alone
//! cost O(n^3), and tetrahedra would push us to O(n^4). H0 + H1 already
//! capture the features relevant to code structure (islands and cycles).

use ix_topo::persistence::{compute_persistence, PersistenceDiagram};
use ix_topo::simplex::{Simplex, SimplexStream};
use serde::{Deserialize, Serialize};

/// Hard cap on graph size. Rips at dimension 1 is O(n^2) in edges and the
/// persistence reduction is roughly O(m^3) in simplex count. At n=300 we
/// have ~45,000 edges and the boundary-matrix reduction runs in well
/// under a second; at n=500 we were hitting multi-second stalls in the
/// review, so this cap is intentionally conservative. Tighten further
/// if persistent homology ends up on a hot path.
pub const MAX_NODES: usize = 300;

/// Hard cap on edge count, independent of node count. Pathological call
/// graphs with dense cross-module coupling can produce O(n^2) edges even
/// when n is comfortably below MAX_NODES.
pub const MAX_EDGES: usize = 10_000;

/// Minimum distance floor. Prevents division-by-zero and keeps extremely
/// heavy edges from collapsing to radius 0.
const MIN_DISTANCE: f64 = 0.01;

/// Directed, weighted call graph used as input to the topology computation.
///
/// Defined locally (rather than re-exported from Phase 1) so Phase 3 can
/// build and ship independently. The shape matches the shared definition
/// one-for-one and can be swapped for an alias once Phase 1 is merged.
#[derive(Debug, Clone, Default)]
pub struct CallGraph {
    /// Node identifiers (typically fully-qualified function names).
    pub nodes: Vec<String>,
    /// Directed edges `(from, to, weight)`. Weight is coupling strength:
    /// higher = tighter coupling.
    pub edges: Vec<(String, String, f64)>,
}

/// Topological summary of a call graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTopology {
    /// Number of connected components (H_0 rank).
    pub betti_0: usize,
    /// Number of independent 1-cycles (H_1 rank).
    pub betti_1: usize,
    /// Always 0 — H_2 is not computed (O(n^3) cost).
    pub betti_2: usize,
    /// All finite (birth, death) pairs from H_0 and H_1 diagrams.
    pub persistence_pairs: Vec<(f64, f64)>,
    /// Longest finite persistence observed across all features.
    pub max_persistence: f64,
    /// Sum of finite persistences across all features.
    pub total_persistence: f64,
    /// Number of nodes that actually entered the computation.
    pub n_nodes: usize,
    /// Number of edges that actually entered the computation.
    pub n_edges: usize,
    /// Fraction in [0, 1] indicating how cleanly the graph was parsed:
    /// 1.0 for a normal run, 0.0 when the graph was rejected (empty or
    /// above `MAX_NODES`).
    pub parse_quality: f64,
}

impl CodeTopology {
    /// Empty/rejected topology.
    fn empty() -> Self {
        Self {
            betti_0: 0,
            betti_1: 0,
            betti_2: 0,
            persistence_pairs: Vec::new(),
            max_persistence: 0.0,
            total_persistence: 0.0,
            n_nodes: 0,
            n_edges: 0,
            parse_quality: 0.0,
        }
    }
}

/// Compute the topological summary of a call graph.
///
/// Rejects empty graphs and graphs with more than [`MAX_NODES`] nodes by
/// returning an empty topology with `parse_quality = 0.0`.
pub fn compute_code_topology(call_graph: &CallGraph) -> CodeTopology {
    let n = call_graph.nodes.len();
    if n == 0 || n > MAX_NODES {
        return CodeTopology::empty();
    }
    // Independent edge-count guard: pathological graphs with dense cross
    // coupling can exceed the reduction budget even when n is small.
    if call_graph.edges.len() > MAX_EDGES {
        return CodeTopology::empty();
    }

    // Node name -> index.
    let mut idx_of = std::collections::HashMap::with_capacity(n);
    for (i, name) in call_graph.nodes.iter().enumerate() {
        idx_of.insert(name.as_str(), i);
    }

    // Symmetrized distance matrix. Default to f64::INFINITY meaning
    // "no edge" — those pairs will never produce a Rips edge.
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    for (i, row) in dist.iter_mut().enumerate() {
        row[i] = 0.0;
    }

    let mut n_edges = 0usize;
    for (from, to, weight) in &call_graph.edges {
        let (Some(&i), Some(&j)) = (idx_of.get(from.as_str()), idx_of.get(to.as_str())) else {
            continue;
        };
        if i == j {
            continue;
        }
        let w = weight.abs();
        let d = if w > 0.0 {
            (1.0 / w).max(MIN_DISTANCE)
        } else {
            f64::INFINITY
        };
        // Symmetrize: keep the minimum distance (strongest link) in both
        // directions so the matrix is a valid metric-ish dissimilarity.
        if d < dist[i][j] {
            dist[i][j] = d;
            dist[j][i] = d;
        }
        n_edges += 1;
    }

    // Build the Rips filtration directly from the distance matrix.
    // We cap at dimension 1: vertices (H_0 generators) and edges (H_1
    // candidates). Feeding triangles would let H_1 features die, but at
    // O(n^3) cost — we accept slightly inflated H_1 in exchange for speed.
    let mut stream = SimplexStream::new();
    for i in 0..n {
        stream.add(Simplex::new(vec![i]), 0.0);
    }
    for (i, row) in dist.iter().enumerate() {
        for (j, &d) in row.iter().enumerate().skip(i + 1) {
            if d.is_finite() {
                stream.add(Simplex::new(vec![i, j]), d);
            }
        }
    }
    stream.sort();

    let diagrams = compute_persistence(&stream);

    let (betti_0, betti_1, persistence_pairs, max_persistence, total_persistence) =
        summarize(&diagrams);

    CodeTopology {
        betti_0,
        betti_1,
        betti_2: 0,
        persistence_pairs,
        max_persistence,
        total_persistence,
        n_nodes: n,
        n_edges,
        parse_quality: 1.0,
    }
}

/// Number of distinct **undirected** edges the filtration will actually build.
///
/// [`CodeTopology::n_edges`] counts input edges *processed*, so `a -> b` and
/// `b -> a` contribute two to it while producing one Rips edge. Anything
/// comparing against a graph invariant — the circuit rank, a density ratio —
/// needs this count instead, or it silently over-counts every reciprocal pair.
///
/// Mirrors the filter in [`compute_code_topology`]: unknown endpoints, self
/// loops and non-positive weights never become edges.
// @ai:invariant while the Rips filtration is capped at dimension 1, no triangle fills a loop, so every 1-cycle is essential and betti_1 equals the circuit rank undirected_edge_count - n_nodes + betti_0 — i.e. McCabe's cyclomatic number over the dependency graph, NOT information beyond it; the part beyond McCabe is the H0 persistence [T:test conf:0.9 src:topology::tests::betti_1_is_the_circuit_rank_while_the_filtration_stops_at_dimension_1]
pub fn undirected_edge_count(call_graph: &CallGraph) -> usize {
    let known: std::collections::HashSet<&str> =
        call_graph.nodes.iter().map(String::as_str).collect();
    let mut pairs = std::collections::HashSet::new();
    for (from, to, weight) in &call_graph.edges {
        if from == to
            || weight.abs() <= 0.0
            || !known.contains(from.as_str())
            || !known.contains(to.as_str())
        {
            continue;
        }
        let (a, b) = if from <= to {
            (from.as_str(), to.as_str())
        } else {
            (to.as_str(), from.as_str())
        };
        pairs.insert((a, b));
    }
    pairs.len()
}

/// Extract Betti numbers and persistence aggregates from a list of diagrams.
///
/// Betti_k is the count of essential (infinite-death) features in the
/// dimension-k diagram — the standard definition of the k-th Betti number
/// for the final complex in the filtration.
fn summarize(diagrams: &[PersistenceDiagram]) -> (usize, usize, Vec<(f64, f64)>, f64, f64) {
    let mut betti_0 = 0usize;
    let mut betti_1 = 0usize;
    let mut pairs = Vec::new();
    let mut max_persistence = 0.0f64;
    let mut total_persistence = 0.0f64;

    for diagram in diagrams {
        for &(birth, death) in &diagram.pairs {
            if death.is_infinite() {
                match diagram.dimension {
                    0 => betti_0 += 1,
                    1 => betti_1 += 1,
                    _ => {}
                }
            } else {
                pairs.push((birth, death));
                let p = death - birth;
                if p > max_persistence {
                    max_persistence = p;
                }
                total_persistence += p;
            }
        }
    }

    (betti_0, betti_1, pairs, max_persistence, total_persistence)
}

// ─── The seam: semantic (Layer 2) -> topology (Layer 4) ─────────────────────
//
// `compute_code_topology` has always taken the `CallGraph` defined above, and
// `semantic::extract_call_graph` has always produced a *different* `CallGraph`
// with the same name. Nothing converted one into the other, so Layer 4 had no
// producer outside its own unit tests. These two functions are that converter.

/// One compilation unit as the module-level topology sees it: a name (in
/// practice a file path), the functions it defines, and the calls it makes.
///
/// Built by the caller, because "what is a unit" is a policy question — a file,
/// a `mod`, a crate — that this crate has no business deciding.
#[cfg(all(feature = "semantic", feature = "topology"))]
#[derive(Debug, Clone)]
pub struct Unit {
    /// Unit identifier; becomes a node name in the module-level graph.
    pub name: String,
    /// Functions defined here, from [`crate::semantic::extract_definitions`].
    pub definitions: Vec<String>,
    /// Calls made here, from [`crate::semantic::extract_call_graph`].
    pub graph: crate::semantic::CallGraph,
}

/// Outcome of a module-level resolution, alongside the graph itself.
#[cfg(all(feature = "semantic", feature = "topology"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleResolution {
    /// Call sites whose callee resolved to a definition in some unit.
    pub resolved_calls: usize,
    /// Call sites whose callee is defined nowhere in the unit set — standard
    /// library, dependencies, macros, trait methods. Dropped, not guessed.
    pub external_calls: usize,
    /// Call sites whose callee name is defined in more than one unit. Also
    /// dropped: picking one would invent an edge. A high count here means the
    /// module graph below it is an undercount, so it is reported rather than
    /// swallowed.
    pub ambiguous_calls: usize,
}

/// Function-level call graph for a single unit.
///
/// Node set is taken verbatim from the semantic graph (definitions *and* call
/// targets), and parallel call sites are fused by summing their weights, so an
/// edge weight is "number of call sites", which
/// [`compute_code_topology`] then inverts into a distance.
#[cfg(all(feature = "semantic", feature = "topology"))]
pub fn call_graph_from_semantic(graph: &crate::semantic::CallGraph) -> CallGraph {
    let mut fused: std::collections::BTreeMap<(&str, &str), f64> =
        std::collections::BTreeMap::new();
    for edge in &graph.edges {
        *fused
            .entry((edge.caller.as_str(), edge.callee_name()))
            .or_insert(0.0) += f64::from(edge.weight);
    }
    CallGraph {
        nodes: graph.nodes.clone(),
        edges: fused
            .into_iter()
            .map(|((from, to), weight)| (from.to_string(), to.to_string(), weight))
            .collect(),
    }
}

/// Module-level call graph: one node per unit, one edge per resolved
/// cross-unit call, weighted by the number of call sites.
///
/// Resolution is by bare name against the union of every unit's definition
/// set. That is deliberately conservative:
///
/// * a callee defined in no unit is **external** and dropped — the graph is
///   about coupling *inside* the given set;
/// * a callee defined in more than one unit is **ambiguous** and dropped,
///   because bare names carry no module path and choosing a winner would
///   fabricate an edge;
/// * self-calls within a unit are dropped, since a self-loop cannot change
///   any Betti number.
///
/// Both drop counts come back in [`ModuleResolution`] so a caller can see how
/// much of the call traffic the graph actually explains.
#[cfg(all(feature = "semantic", feature = "topology"))]
// @ai:assumption bare-name resolution against the union of definition sets is an adequate proxy for the real intra-crate dependency graph; measured over IX's own 82 crates it explains only 20.4% of call sites (10,466 resolved / 37,189 external / 3,616 ambiguous), and the ambiguous slice biases every edge count DOWNWARD — nothing binds "good enough", so treat reported edge counts as lower bounds [U:uncertain conf:0.4 src:docs/research/2026-09-08-code-topology-over-ix.md]
pub fn module_call_graph(units: &[Unit]) -> (CallGraph, ModuleResolution) {
    // name -> owning unit index; `None` marks a name claimed by two or more.
    let mut owner: std::collections::HashMap<&str, Option<usize>> =
        std::collections::HashMap::new();
    for (i, unit) in units.iter().enumerate() {
        for def in &unit.definitions {
            owner
                .entry(def.as_str())
                .and_modify(|slot| {
                    if *slot != Some(i) {
                        *slot = None;
                    }
                })
                .or_insert(Some(i));
        }
    }

    let mut fused: std::collections::BTreeMap<(usize, usize), f64> =
        std::collections::BTreeMap::new();
    let mut resolution = ModuleResolution {
        resolved_calls: 0,
        external_calls: 0,
        ambiguous_calls: 0,
    };

    for (i, unit) in units.iter().enumerate() {
        for edge in &unit.graph.edges {
            match owner.get(edge.callee_name()) {
                None => resolution.external_calls += 1,
                Some(None) => resolution.ambiguous_calls += 1,
                Some(&Some(j)) => {
                    resolution.resolved_calls += 1;
                    if i != j {
                        *fused.entry((i, j)).or_insert(0.0) += f64::from(edge.weight);
                    }
                }
            }
        }
    }

    let graph = CallGraph {
        nodes: units.iter().map(|u| u.name.clone()).collect(),
        edges: fused
            .into_iter()
            .map(|((i, j), weight)| (units[i].name.clone(), units[j].name.clone(), weight))
            .collect(),
    };
    (graph, resolution)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn node(name: &str) -> String {
        name.to_string()
    }

    #[test]
    fn test_empty_graph() {
        let cg = CallGraph::default();
        let topo = compute_code_topology(&cg);
        assert_eq!(topo.n_nodes, 0);
        assert_eq!(topo.n_edges, 0);
        assert_eq!(topo.betti_0, 0);
        assert_eq!(topo.betti_1, 0);
        assert_eq!(topo.parse_quality, 0.0);
    }

    #[test]
    fn test_tree_has_no_cycles() {
        // Linear call chain A -> B -> C -> D. No cycles => betti_1 == 0.
        let cg = CallGraph {
            nodes: vec![node("A"), node("B"), node("C"), node("D")],
            edges: vec![
                (node("A"), node("B"), 1.0),
                (node("B"), node("C"), 1.0),
                (node("C"), node("D"), 1.0),
            ],
        };
        let topo = compute_code_topology(&cg);
        assert_eq!(topo.parse_quality, 1.0);
        assert_eq!(topo.n_nodes, 4);
        assert_eq!(topo.betti_1, 0, "linear chain must have no 1-cycles");
        // Fully connected at the largest edge distance => one component.
        assert_eq!(topo.betti_0, 1);
    }

    #[test]
    fn test_cycle_detected() {
        // A -> B -> C -> A: a single 3-cycle. Without triangles in the
        // filtration the loop is essential => betti_1 >= 1.
        let cg = CallGraph {
            nodes: vec![node("A"), node("B"), node("C")],
            edges: vec![
                (node("A"), node("B"), 1.0),
                (node("B"), node("C"), 1.0),
                (node("C"), node("A"), 1.0),
            ],
        };
        let topo = compute_code_topology(&cg);
        assert_eq!(topo.parse_quality, 1.0);
        assert!(topo.betti_1 >= 1, "A->B->C->A must expose a 1-cycle");
    }

    #[test]
    fn test_disconnected_components() {
        // Two disjoint linear chains: {A-B} and {C-D}. betti_0 == 2.
        let cg = CallGraph {
            nodes: vec![node("A"), node("B"), node("C"), node("D")],
            edges: vec![(node("A"), node("B"), 1.0), (node("C"), node("D"), 1.0)],
        };
        let topo = compute_code_topology(&cg);
        assert_eq!(topo.parse_quality, 1.0);
        assert_eq!(topo.betti_0, 2, "two disjoint components expected");
        assert_eq!(topo.betti_1, 0);
    }

    /// The identity that says what persistent homology does and does not add
    /// over McCabe here.
    ///
    /// `compute_code_topology` builds a Rips filtration capped at dimension 1,
    /// so no triangle ever fills a loop and every 1-cycle survives to infinity.
    /// For a graph with `V` vertices, `E` (symmetrized) edges and `betti_0`
    /// components, the circuit rank is `E - V + betti_0` — which is exactly
    /// McCabe's cyclomatic number applied to the dependency graph rather than
    /// to a control-flow graph. So `betti_1` here is *not* information McCabe
    /// could not reach; the information McCabe cannot reach is the H0
    /// persistence, i.e. the coupling scale at which the graph falls apart.
    ///
    /// Stated as a test rather than a comment so that if the filtration ever
    /// gains triangles (and `betti_1` starts meaning something stronger), this
    /// fails and the claim gets rewritten instead of quietly going stale.
    #[test]
    fn betti_1_is_the_circuit_rank_while_the_filtration_stops_at_dimension_1() {
        // Two 3-cycles sharing one vertex, plus an isolated pendant chain.
        let cg = CallGraph {
            nodes: ["A", "B", "C", "D", "E", "X", "Y"]
                .iter()
                .map(|n| node(n))
                .collect(),
            edges: vec![
                (node("A"), node("B"), 1.0),
                (node("B"), node("C"), 2.0),
                (node("C"), node("A"), 3.0),
                (node("C"), node("D"), 1.0),
                (node("D"), node("E"), 4.0),
                (node("E"), node("C"), 1.0),
                (node("X"), node("Y"), 1.0),
            ],
        };
        let topo = compute_code_topology(&cg);
        let v = topo.n_nodes as i64;
        let e = undirected_edge_count(&cg) as i64;
        assert_eq!(
            topo.betti_1 as i64,
            e - v + topo.betti_0 as i64,
            "betti_1 must equal the circuit rank E - V + betti_0"
        );
        assert_eq!(topo.betti_0, 2, "{{A..E}} and {{X,Y}}");
        assert_eq!(topo.betti_1, 2, "two independent cycles");

        // Reciprocal calls are the trap the identity is stated against:
        // `n_edges` counts both directions, `undirected_edge_count` counts one.
        // Using the wrong one here inflates the rank by the number of pairs.
        let mut reciprocal = cg.clone();
        reciprocal.edges.push((node("B"), node("A"), 5.0));
        reciprocal.edges.push((node("Y"), node("X"), 5.0));
        let recip_topo = compute_code_topology(&reciprocal);
        assert_eq!(
            recip_topo.n_edges,
            topo.n_edges + 2,
            "both directions are processed"
        );
        assert_eq!(
            undirected_edge_count(&reciprocal),
            undirected_edge_count(&cg),
            "but no new undirected edge exists"
        );
        assert_eq!(
            recip_topo.betti_1 as i64,
            undirected_edge_count(&reciprocal) as i64 - recip_topo.n_nodes as i64
                + recip_topo.betti_0 as i64,
            "the identity holds against the undirected count, not n_edges"
        );
        assert_eq!(recip_topo.betti_1, topo.betti_1, "no new cycle was created");
    }

    /// The H0 persistence is the part McCabe cannot express: a weakly coupled
    /// pair of clusters must merge later than a tightly coupled one, even
    /// though both graphs have identical vertex, edge and cycle counts.
    #[test]
    fn h0_persistence_separates_graphs_with_identical_cyclomatic_numbers() {
        let build = |bridge: f64| CallGraph {
            nodes: ["A", "B", "C", "D"].iter().map(|n| node(n)).collect(),
            edges: vec![
                (node("A"), node("B"), 10.0),
                (node("C"), node("D"), 10.0),
                (node("B"), node("C"), bridge),
            ],
        };
        let tight = compute_code_topology(&build(10.0));
        let weak = compute_code_topology(&build(0.1));

        // Same graph shape => same Betti numbers, so McCabe sees no difference.
        assert_eq!(tight.betti_0, weak.betti_0);
        assert_eq!(tight.betti_1, weak.betti_1);
        assert_eq!(tight.n_edges, weak.n_edges);

        // The persistence does see it: the weak bridge is 1/0.1 = 10.0 away,
        // the tight one 1/10 = 0.1 away.
        assert!(
            weak.max_persistence > tight.max_persistence * 10.0,
            "weakly coupled clusters must stay separate far longer: \
             weak={} tight={}",
            weak.max_persistence,
            tight.max_persistence
        );
    }

    #[test]
    fn test_node_limit_exceeded() {
        let nodes: Vec<String> = (0..501).map(|i| format!("f{i}")).collect();
        let cg = CallGraph {
            nodes,
            edges: Vec::new(),
        };
        let topo = compute_code_topology(&cg);
        assert_eq!(topo.parse_quality, 0.0);
        assert_eq!(topo.n_nodes, 0);
        assert_eq!(topo.n_edges, 0);
    }
}

// ─── Seam tests: real Rust source in, topology out ──────────────────────────

#[cfg(all(test, feature = "semantic", feature = "topology"))]
mod seam_tests {
    use super::*;
    use crate::semantic::{extract_call_graph, extract_definitions};

    fn unit(name: &str, source: &str) -> Unit {
        Unit {
            name: name.to_string(),
            definitions: extract_definitions(source),
            graph: extract_call_graph(source).unwrap_or_default(),
        }
    }

    const ALPHA: &str = r#"
        pub fn alpha_entry() { beta_work(); beta_work(); }
        pub fn alpha_helper() { gamma_sink(); }
    "#;
    const BETA: &str = r#"
        pub fn beta_work() { gamma_sink(); }
    "#;
    const GAMMA: &str = r#"
        pub fn gamma_sink() { println!("done"); }
        pub fn gamma_back() { alpha_entry(); }
    "#;

    #[test]
    fn definitions_are_the_defined_functions_only() {
        // `beta_work` and `gamma_sink` are *called* here, never defined, so
        // they belong to the call graph's node set but not to the definitions.
        let defs = extract_definitions(ALPHA);
        assert_eq!(defs, vec!["alpha_entry", "alpha_helper"]);
        let nodes = extract_call_graph(ALPHA).expect("parses").nodes;
        assert!(nodes.contains(&"beta_work".to_string()));
        assert!(!defs.contains(&"beta_work".to_string()));
    }

    #[test]
    fn module_resolution_drops_external_and_ambiguous_calls() {
        let units = [unit("alpha.rs", ALPHA), unit("beta.rs", BETA)];
        let (graph, resolution) = module_call_graph(&units);

        // alpha -> beta (two call sites), beta -> gamma is unresolvable because
        // gamma.rs is not in the set.
        assert_eq!(graph.nodes, vec!["alpha.rs", "beta.rs"]);
        assert_eq!(
            graph.edges,
            vec![("alpha.rs".to_string(), "beta.rs".to_string(), 2.0)]
        );
        assert_eq!(resolution.resolved_calls, 2);
        assert!(
            resolution.external_calls >= 2,
            "gamma_sink from both units is external here, got {}",
            resolution.external_calls
        );
        assert_eq!(resolution.ambiguous_calls, 0);

        // Same name defined twice => ambiguous, and the edge is NOT invented.
        let dupes = [
            unit("alpha.rs", ALPHA),
            unit("beta.rs", BETA),
            unit("beta_copy.rs", BETA),
        ];
        let (dupe_graph, dupe_resolution) = module_call_graph(&dupes);
        assert_eq!(dupe_resolution.resolved_calls, 0);
        assert_eq!(dupe_resolution.ambiguous_calls, 2);
        assert!(
            dupe_graph.edges.is_empty(),
            "an ambiguous callee must produce no edge, got {:?}",
            dupe_graph.edges
        );
    }

    /// `compute_code_topology` symmetrizes the call graph before building the
    /// filtration, so betti_1 counts **undirected** cycles in the dependency
    /// shape — not circular dependencies. A fan-in diamond registers as tangled
    /// even though its edges point strictly downhill, and a directed back-edge
    /// laid over an existing undirected edge adds nothing.
    ///
    /// This is a real limit on what the measure means, so it is pinned here
    /// rather than described in prose that could drift away from the code.
    #[test]
    fn betti_1_counts_undirected_tangle_not_dependency_cycles() {
        const GAMMA_LEAF: &str = r#"pub fn gamma_sink() { println!("done"); }"#;
        const ALPHA_CHAIN: &str = r#"pub fn alpha_entry() { beta_work(); beta_work(); }"#;

        // Chain: alpha -> beta -> gamma. A tree, so no cycles at all.
        let chain = [
            unit("alpha.rs", ALPHA_CHAIN),
            unit("beta.rs", BETA),
            unit("gamma.rs", GAMMA_LEAF),
        ];
        let (chain_graph, _) = module_call_graph(&chain);
        assert_eq!(chain_graph.edges.len(), 2);
        let chain_topo = compute_code_topology(&chain_graph);
        assert_eq!(chain_topo.betti_0, 1);
        assert_eq!(chain_topo.betti_1, 0, "a module chain has no 1-cycles");

        // Diamond: alpha -> beta, alpha -> gamma, beta -> gamma. Still a DAG in
        // the dependency sense, but an undirected triangle.
        let diamond = [
            unit("alpha.rs", ALPHA),
            unit("beta.rs", BETA),
            unit("gamma.rs", GAMMA_LEAF),
        ];
        let (diamond_graph, resolution) = module_call_graph(&diamond);
        assert_eq!(resolution.ambiguous_calls, 0);
        assert_eq!(diamond_graph.edges.len(), 3);
        let diamond_topo = compute_code_topology(&diamond_graph);
        assert_eq!(diamond_topo.betti_0, 1);
        assert_eq!(
            diamond_topo.betti_1, 1,
            "a fan-in diamond is one undirected cycle even with acyclic \
             dependency direction"
        );

        // Add gamma -> alpha, a genuine circular dependency. It lands on the
        // already-present {alpha, gamma} undirected edge, so betti_1 does not
        // move: this measure cannot see it.
        let with_back_edge = [
            unit("alpha.rs", ALPHA),
            unit("beta.rs", BETA),
            unit("gamma.rs", GAMMA),
        ];
        let (back_graph, _) = module_call_graph(&with_back_edge);
        assert_eq!(
            back_graph.edges.len(),
            4,
            "the directed back-edge is present"
        );
        let back_topo = compute_code_topology(&back_graph);
        assert_eq!(
            back_topo.betti_1, diamond_topo.betti_1,
            "symmetrization hides a directed back-edge over an existing edge"
        );
    }

    #[test]
    fn function_level_graph_fuses_parallel_call_sites() {
        let graph = call_graph_from_semantic(&extract_call_graph(ALPHA).expect("parses"));
        let beta = graph
            .edges
            .iter()
            .find(|(from, to, _)| from == "alpha_entry" && to == "beta_work")
            .expect("alpha_entry -> beta_work");
        assert_eq!(beta.2, 2.0, "two call sites fuse into weight 2");
    }
}
