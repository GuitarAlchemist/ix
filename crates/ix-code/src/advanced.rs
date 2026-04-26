//! Layer 6: Advanced math features for code analysis.
//!
//! This module connects the ix math stack to code analysis, providing
//! features that traditional static analysis tools cannot compute:
//!
//! - **Hyperbolic embeddings** of call hierarchies (ix-math/poincare_hierarchy)
//! - **K-theory invariants** of call graph adjacency matrices (ix-ktheory)
//! - **Spectral analysis** of metric trajectories (ix-signal/fft)
//! - **BSP similarity index** for clone/anomaly detection (ix-math/bsp)
//!
//! Gated behind the `advanced` feature flag.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use ix_ktheory::graph_k::{k0_from_adjacency, k1_from_adjacency};
use ix_math::bsp::BspTree;
use ix_math::poincare_hierarchy::{hierarchy_map_score, HierarchyDecoder, PoincareEmbedder};
use ix_signal::fft::{rfft, Complex};

use crate::metrics::CodeMetrics;

// ─── 6a. Hyperbolic Code Embeddings ──────────────────────────────────────────

/// Poincaré-ball embedding of a call hierarchy.
///
/// Nodes closer to the origin are higher in the call hierarchy. Trees embed
/// in hyperbolic space with zero distortion, so this captures architectural
/// hierarchy better than any Euclidean embedding.
#[derive(Debug, Clone)]
pub struct HyperbolicCodeMap {
    /// One embedding per input node (function), in the Poincaré ball.
    pub embeddings: Vec<Array1<f64>>,
    /// Names of the inferred root functions (closest to origin).
    pub root_functions: Vec<String>,
    /// Depth score per node (distance from origin in the Poincaré ball).
    /// Lower = higher in the hierarchy.
    pub depth_scores: Vec<f64>,
    /// Mean-average-precision of the embedded hierarchy vs the input edges.
    /// Range `[0, 1]`: higher = more tree-like / cleaner hierarchy.
    pub hierarchy_map_score: f64,
}

/// Embed a call hierarchy in the Poincaré ball.
///
/// `nodes` is the list of function names (or any labels); `edges` are
/// `(parent_idx, child_idx)` tuples indexing into `nodes`; `dim` is the
/// target hyperbolic dimension (2 for visualization, 5-16 for quality).
pub fn embed_call_hierarchy(
    nodes: &[String],
    edges: &[(usize, usize)],
    dim: usize,
) -> HyperbolicCodeMap {
    let n = nodes.len();
    if n == 0 {
        return HyperbolicCodeMap {
            embeddings: Vec::new(),
            root_functions: Vec::new(),
            depth_scores: Vec::new(),
            hierarchy_map_score: 1.0,
        };
    }

    let embedder = PoincareEmbedder::new(5, dim.max(2))
        .with_epochs(300)
        .with_learning_rate(0.01)
        .with_seed(42);
    let embeddings = embedder.fit(n, edges);

    let decoder = HierarchyDecoder::new(&embeddings);
    let depth_scores: Vec<f64> = (0..n).map(|i| decoder.norm(i)).collect();

    // Root = node closest to origin. Also include any node that never appears
    // as a child in the edge list (true structural roots).
    let mut child_set = vec![false; n];
    for &(_, c) in edges {
        if c < n {
            child_set[c] = true;
        }
    }
    let mut root_indices: Vec<usize> = (0..n).filter(|&i| !child_set[i]).collect();
    if root_indices.is_empty() {
        // Fall back to the geometric root.
        root_indices.push(decoder.root());
    }
    let root_functions: Vec<String> = root_indices.iter().map(|&i| nodes[i].clone()).collect();

    let score = if edges.is_empty() {
        1.0
    } else {
        hierarchy_map_score(&embeddings, edges)
    };

    HyperbolicCodeMap {
        embeddings,
        root_functions,
        depth_scores,
        hierarchy_map_score: score,
    }
}

// ─── 6b. K-Theory Invariants ─────────────────────────────────────────────────

/// Algebraic K-theory invariants of a call-graph adjacency matrix.
///
/// - `k0_*` describes stable equivalence classes (refactoring candidates).
/// - `k1_*` describes essential feedback cycles (architectural lock-in).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KTheoryInvariants {
    /// Free rank of K₀ = coker(I - Aᵀ).
    pub k0_rank: usize,
    /// Torsion invariant factors of K₀.
    pub k0_torsion: Vec<i64>,
    /// Free rank of K₁ = ker(I - Aᵀ).
    pub k1_rank: usize,
    /// Torsion of K₁ (always empty for a kernel, reported for symmetry).
    pub k1_torsion: Vec<i64>,
}

/// Compute K₀ and K₁ invariants from a call-graph adjacency matrix.
///
/// `adj_matrix` must be square; entries are interpreted as integers
/// (typically `0.0`/`1.0` for unweighted call edges).
pub fn compute_k_invariants(adj_matrix: &Array2<f64>) -> KTheoryInvariants {
    let k0 = k0_from_adjacency(adj_matrix)
        .map(|r| (r.rank, r.torsion))
        .unwrap_or((0, Vec::new()));
    let k1 = k1_from_adjacency(adj_matrix)
        .map(|r| (r.rank, r.torsion))
        .unwrap_or((0, Vec::new()));

    KTheoryInvariants {
        k0_rank: k0.0,
        k0_torsion: k0.1,
        k1_rank: k1.0,
        k1_torsion: k1.1,
    }
}

// ─── 6c. Spectral Analysis of Metric Trajectories ────────────────────────────

/// Frequency-domain summary of a code metric trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSpectrum {
    /// Dominant (peak) frequency in cycles per sample.
    pub dominant_frequency: f64,
    /// Dominant period in samples (`1 / dominant_frequency`, or `∞` if DC).
    pub dominant_period: f64,
    /// Shannon entropy of the normalized power spectrum (nats).
    pub spectral_entropy: f64,
    /// One-sided power spectrum (length `fft_size / 2 + 1`).
    pub power_spectrum: Vec<f64>,
}

/// Run an FFT on a trajectory and extract dominant frequency + spectral entropy.
///
/// The input is detrended (mean-subtracted) before FFT so that DC is removed
/// and the dominant frequency reflects true oscillations, not offsets.
pub fn analyze_metric_spectrum(trajectory: &[f64]) -> MetricSpectrum {
    if trajectory.len() < 2 {
        return MetricSpectrum {
            dominant_frequency: 0.0,
            dominant_period: f64::INFINITY,
            spectral_entropy: 0.0,
            power_spectrum: Vec::new(),
        };
    }

    // Detrend.
    let mean = trajectory.iter().sum::<f64>() / trajectory.len() as f64;
    let detrended: Vec<f64> = trajectory.iter().map(|x| x - mean).collect();

    let spectrum: Vec<Complex> = rfft(&detrended);
    let n = spectrum.len();
    // One-sided spectrum (Nyquist mirror symmetry).
    let half = n / 2 + 1;
    let power: Vec<f64> = spectrum
        .iter()
        .take(half)
        .map(|c| c.re * c.re + c.im * c.im)
        .collect();

    // Dominant bin (skip DC = index 0, which is ~0 after detrending anyway).
    let (peak_bin, _) = power
        .iter()
        .enumerate()
        .skip(1)
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    let dominant_frequency = if n > 0 {
        peak_bin as f64 / n as f64
    } else {
        0.0
    };
    let dominant_period = if dominant_frequency > 0.0 {
        1.0 / dominant_frequency
    } else {
        f64::INFINITY
    };

    // Spectral entropy of normalized power.
    let total: f64 = power.iter().sum();
    let spectral_entropy = if total > 0.0 {
        power
            .iter()
            .map(|&p| {
                let q = p / total;
                if q > 0.0 {
                    -q * q.ln()
                } else {
                    0.0
                }
            })
            .sum()
    } else {
        0.0
    };

    MetricSpectrum {
        dominant_frequency,
        dominant_period,
        spectral_entropy,
        power_spectrum: power,
    }
}

// ─── 6d. BSP Code Similarity Index ───────────────────────────────────────────

/// BSP-indexed similarity search over 20-D code-metric feature vectors.
///
/// Backed by [`BspTree<20>`] so queries are `O(log n)` on average.
pub struct CodeSimilarityIndex {
    tree: BspTree<20>,
    // Parallel arrays: each index in `tree` corresponds to the same index here.
    names: Vec<String>,
    points: Vec<[f64; 20]>,
}

impl CodeSimilarityIndex {
    /// Build an index from a slice of per-function [`CodeMetrics`].
    pub fn build(metrics: &[CodeMetrics]) -> Self {
        let mut names = Vec::with_capacity(metrics.len());
        let mut points: Vec<[f64; 20]> = Vec::with_capacity(metrics.len());

        for m in metrics {
            let features = m.to_features();
            let mut arr = [0.0_f64; 20];
            for (i, slot) in arr.iter_mut().enumerate() {
                *slot = features[i];
            }
            names.push(m.name.clone());
            points.push(arr);
        }

        let tree = BspTree::<20>::from_points(points.clone());

        Self {
            tree,
            names,
            points,
        }
    }

    /// Number of indexed functions.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Return the `k` functions closest to `query_features`, as
    /// `(name, squared_distance)` tuples sorted ascending by distance.
    pub fn similar_functions(&self, query_features: &[f64; 20], k: usize) -> Vec<(String, f64)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let neighbors = match self.tree.k_nearest(query_features, k) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        // Map each returned point back to its name by exact match (BSP stores
        // raw points, and the ix-math BSP API does not expose payloads).
        neighbors
            .into_iter()
            .map(|(pt, dist_sq)| {
                let name = self
                    .points
                    .iter()
                    .position(|p| p == &pt)
                    .map(|i| self.names[i].clone())
                    .unwrap_or_default();
                (name, dist_sq)
            })
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::consts::PI;

    fn make_metric(name: &str, cyclo: f64, sloc: f64) -> CodeMetrics {
        CodeMetrics {
            name: name.to_string(),
            start_line: 1,
            end_line: 10,
            cyclomatic: cyclo,
            cognitive: cyclo,
            n_exits: 1.0,
            n_args: 2.0,
            sloc,
            ploc: sloc,
            lloc: sloc,
            cloc: 0.0,
            blank: 0.0,
            h_u_ops: 3.0,
            h_u_opnds: 2.0,
            h_total_ops: 6.0,
            h_total_opnds: 4.0,
            h_vocabulary: 5.0,
            h_length: 10.0,
            h_volume: 23.0,
            h_difficulty: 3.0,
            h_effort: 69.0,
            h_bugs: 0.008,
            maintainability_index: 100.0,
        }
    }

    #[test]
    fn test_hyperbolic_embedding_basic() {
        // 4-node tree: 0 -> 1, 0 -> 2, 1 -> 3
        let nodes = vec![
            "root".to_string(),
            "a".to_string(),
            "b".to_string(),
            "a_child".to_string(),
        ];
        let edges = vec![(0, 1), (0, 2), (1, 3)];

        let map = embed_call_hierarchy(&nodes, &edges, 5);

        assert_eq!(map.embeddings.len(), 4);
        assert_eq!(map.depth_scores.len(), 4);
        assert!(
            map.hierarchy_map_score > 0.0,
            "MAP score should be positive, got {}",
            map.hierarchy_map_score
        );
        // Node 0 has no incoming edges, should be an input root.
        assert!(map.root_functions.contains(&"root".to_string()));
    }

    #[test]
    fn test_k_invariants_cycle() {
        // 2-node cycle: 0 <-> 1
        let adj = array![[0.0, 1.0], [1.0, 0.0]];
        let inv = compute_k_invariants(&adj);
        assert!(
            inv.k1_rank > 0,
            "2-cycle should have non-trivial K1 rank, got {}",
            inv.k1_rank
        );
    }

    #[test]
    fn test_k_invariants_dag() {
        // DAG: 0 -> 1 -> 2 (no cycles)
        let adj = array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]];
        let inv = compute_k_invariants(&adj);
        assert_eq!(inv.k1_rank, 0, "DAG should have zero K1 rank");
    }

    #[test]
    fn test_spectrum_pure_sine() {
        // 64 samples of sin(2π * 4 * t / 64) => 4 cycles over 64 samples.
        // Expected dominant frequency bin = 4 / 64 = 0.0625.
        let n = 64;
        let cycles = 4.0;
        let traj: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * cycles * i as f64 / n as f64).sin())
            .collect();

        let spec = analyze_metric_spectrum(&traj);
        let expected = cycles / n as f64;
        assert!(
            (spec.dominant_frequency - expected).abs() < 1e-6,
            "expected dominant freq {}, got {}",
            expected,
            spec.dominant_frequency
        );
        assert!(
            (spec.dominant_period - n as f64 / cycles).abs() < 1e-3,
            "expected period {}, got {}",
            n as f64 / cycles,
            spec.dominant_period
        );
        assert!(
            spec.spectral_entropy >= 0.0,
            "spectral entropy must be non-negative"
        );
    }

    #[test]
    fn test_spectrum_empty_and_short() {
        let empty = analyze_metric_spectrum(&[]);
        assert_eq!(empty.dominant_frequency, 0.0);
        let one = analyze_metric_spectrum(&[1.0]);
        assert_eq!(one.dominant_frequency, 0.0);
    }

    #[test]
    fn test_bsp_similarity() {
        // Two clusters of simple/complex functions.
        let metrics = vec![
            make_metric("simple_a", 1.0, 5.0),
            make_metric("simple_b", 1.0, 6.0),
            make_metric("simple_c", 2.0, 5.0),
            make_metric("complex_a", 50.0, 200.0),
            make_metric("complex_b", 55.0, 210.0),
            make_metric("complex_c", 48.0, 195.0),
        ];

        let index = CodeSimilarityIndex::build(&metrics);
        assert_eq!(index.len(), 6);

        // Query near the "simple" cluster.
        let query_features = metrics[0].to_features();
        let mut query = [0.0_f64; 20];
        for (i, slot) in query.iter_mut().enumerate() {
            *slot = query_features[i];
        }

        let neighbors = index.similar_functions(&query, 3);
        assert_eq!(neighbors.len(), 3);

        // All 3 nearest should be from the "simple_*" cluster.
        for (name, _) in &neighbors {
            assert!(
                name.starts_with("simple_"),
                "nearest neighbor {} should be in simple cluster",
                name
            );
        }
    }
}
