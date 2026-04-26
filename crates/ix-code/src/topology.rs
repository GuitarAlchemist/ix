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
