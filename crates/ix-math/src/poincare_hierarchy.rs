//! Poincaré hierarchy extraction: learn hyperbolic embeddings from
//! edge lists and decode tree structure from the learned geometry.
//!
//! Nodes closer to the origin are higher in the hierarchy (more general).
//! Nodes near the boundary are leaves (more specific). The exponential
//! volume growth of hyperbolic space naturally accommodates tree-like
//! branching.
//!
//! # Example: Learn a taxonomy
//!
//! ```
//! use ix_math::poincare_hierarchy::{PoincareEmbedder, HierarchyDecoder};
//!
//! // Edges: (parent, child) — "animal" contains "dog" and "cat"
//! let edges = vec![
//!     (0, 1), // animal → dog
//!     (0, 2), // animal → cat
//!     (1, 3), // dog → poodle
//!     (1, 4), // dog → labrador
//!     (2, 5), // cat → siamese
//! ];
//!
//! let mut embedder = PoincareEmbedder::new(5, 2)
//!     .with_epochs(100)
//!     .with_learning_rate(0.01)
//!     .with_neg_samples(5);
//! let embeddings = embedder.fit(6, &edges);
//!
//! // Root ("animal") should be closest to origin
//! let norms: Vec<f64> = embeddings.iter()
//!     .map(|e| e.dot(e).sqrt())
//!     .collect();
//! assert!(norms[0] < norms[3], "Root should be closer to origin than leaf");
//!
//! // Decode hierarchy
//! let decoder = HierarchyDecoder::new(&embeddings);
//! let root = decoder.root();
//! assert_eq!(root, 0, "Animal should be root");
//! let children = decoder.children(0, &edges);
//! assert!(children.contains(&1) && children.contains(&2));
//! ```

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::hyperbolic::{init_embeddings, poincare_distance, project_to_ball, riemannian_sgd_step};

/// Poincaré embedding trainer using Riemannian SGD with negative sampling.
///
/// Learns embeddings where connected nodes (parent-child) are close
/// in hyperbolic space, and unconnected nodes are pushed apart.
pub struct PoincareEmbedder {
    /// Embedding dimension.
    pub dim: usize,
    /// Number of negative samples per positive edge.
    pub neg_samples: usize,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate for Riemannian SGD.
    pub learning_rate: f64,
    /// Random seed.
    pub seed: u64,
    /// Initial embedding radius (smaller = more stable).
    pub init_radius: f64,
    /// Margin for negative sampling loss.
    pub margin: f64,
}

impl PoincareEmbedder {
    /// Create a new embedder.
    ///
    /// # Arguments
    /// - `neg_samples` — negative samples per positive edge (5-10 typical)
    /// - `dim` — embedding dimension (2 for visualization, 5-50 for quality)
    pub fn new(neg_samples: usize, dim: usize) -> Self {
        Self {
            dim,
            neg_samples,
            epochs: 50,
            learning_rate: 0.01,
            seed: 42,
            init_radius: 0.001,
            margin: 0.1,
        }
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_neg_samples(mut self, n: usize) -> Self {
        self.neg_samples = n;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Train embeddings from an edge list.
    ///
    /// `n_nodes` is the total number of nodes. `edges` are `(parent, child)` pairs.
    /// Returns embeddings where parents are closer to the origin than children.
    pub fn fit(&self, n_nodes: usize, edges: &[(usize, usize)]) -> Vec<Array1<f64>> {
        let mut embeddings = init_embeddings(n_nodes, self.dim, self.init_radius, self.seed);
        let mut rng = StdRng::seed_from_u64(self.seed + 1);

        // Count how many times each node appears as a parent (higher = more ancestral)
        let mut parent_count = vec![0usize; n_nodes];
        let mut child_set: Vec<bool> = vec![false; n_nodes];
        for &(p, c) in edges {
            parent_count[p] += 1;
            child_set[c] = true;
        }
        // Compute target radius: roots near 0, leaves near boundary
        // Depth via BFS from roots
        let roots: Vec<usize> = (0..n_nodes).filter(|&i| !child_set[i]).collect();
        let mut depth = vec![0usize; n_nodes];
        let mut queue = std::collections::VecDeque::new();
        for &r in &roots {
            queue.push_back(r);
        }
        while let Some(node) = queue.pop_front() {
            for &(p, c) in edges {
                if p == node && depth[c] == 0 && !roots.contains(&c) {
                    depth[c] = depth[node] + 1;
                    queue.push_back(c);
                }
            }
        }
        let max_depth = *depth.iter().max().unwrap_or(&1).max(&1);
        // Target radius: depth 0 → 0.1, max_depth → 0.9
        let target_radius: Vec<f64> = depth
            .iter()
            .map(|&d| 0.1 + 0.8 * d as f64 / max_depth as f64)
            .collect();

        for _epoch in 0..self.epochs {
            let lr = self.learning_rate;

            for &(u_idx, v_idx) in edges {
                // 1. Attract connected nodes
                let grad_u = self.distance_gradient(&embeddings[u_idx], &embeddings[v_idx]);
                let grad_v = self.distance_gradient(&embeddings[v_idx], &embeddings[u_idx]);

                embeddings[u_idx] = riemannian_sgd_step(&embeddings[u_idx], &grad_u, lr);
                embeddings[v_idx] = riemannian_sgd_step(&embeddings[v_idx], &grad_v, lr);

                // 2. Negative sampling: push unconnected apart
                for _ in 0..self.neg_samples {
                    let neg_idx = rng.random_range(0..n_nodes);
                    if neg_idx == u_idx || neg_idx == v_idx {
                        continue;
                    }
                    let d_pos =
                        poincare_distance(&embeddings[u_idx], &embeddings[v_idx]).unwrap_or(0.0);
                    let d_neg = poincare_distance(&embeddings[u_idx], &embeddings[neg_idx])
                        .unwrap_or(f64::INFINITY);

                    if d_neg < d_pos + self.margin {
                        let neg_grad =
                            self.distance_gradient(&embeddings[u_idx], &embeddings[neg_idx]);
                        embeddings[u_idx] =
                            riemannian_sgd_step(&embeddings[u_idx], &(-1.0 * &neg_grad), lr * 0.5);
                    }
                }
            }

            // 3. Radial positioning: push each node toward its target radius
            for i in 0..n_nodes {
                let norm = embeddings[i].dot(&embeddings[i]).sqrt();
                if norm < 1e-8 {
                    continue;
                }
                let direction = &embeddings[i] / norm;
                let target = target_radius[i];
                let radial_error = target - norm;
                // Move toward target radius
                let adjustment = radial_error * 0.1 * &direction;
                embeddings[i] = &embeddings[i] + &adjustment;
                embeddings[i] = project_to_ball(&embeddings[i]);
            }
        }

        embeddings
    }

    /// Analytical Euclidean gradient of Poincaré distance w.r.t. `u`.
    ///
    /// ∂d/∂u = (4 / (β² * sqrt(γ² - 1))) * ((||v||² - 2⟨u,v⟩ + 1) * u - (1 - ||u||²) * v)
    /// where α = 1 - ||u||², β = 1 - ||v||², γ = 1 + 2||u-v||²/(αβ)
    fn distance_gradient(&self, u: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let u_sq = u.dot(u);
        let v_sq = v.dot(v);
        let diff = u - v;
        let diff_sq = diff.dot(&diff);

        let alpha = 1.0 - u_sq;
        let beta = 1.0 - v_sq;

        if alpha.abs() < 1e-12 || beta.abs() < 1e-12 {
            return Array1::zeros(u.len());
        }

        let gamma = 1.0 + 2.0 * diff_sq / (alpha * beta);
        let gamma_term = (gamma * gamma - 1.0).max(1e-15).sqrt();

        let coeff = 4.0 / (beta * gamma_term + 1e-15);
        let term1 = (v_sq - 2.0 * u.dot(v) + 1.0) / (alpha * alpha) * u;
        let term2 = -1.0 / alpha * v;

        coeff * &(term1 + term2)
    }
}

/// Decode hierarchical structure from trained Poincaré embeddings.
///
/// Uses the geometric properties of the Poincaré ball:
/// - Nodes closer to the origin are higher in the hierarchy
/// - Parent-child relationships follow radial direction
pub struct HierarchyDecoder<'a> {
    embeddings: &'a [Array1<f64>],
    norms: Vec<f64>,
}

impl<'a> HierarchyDecoder<'a> {
    /// Create a decoder from trained embeddings.
    pub fn new(embeddings: &'a [Array1<f64>]) -> Self {
        let norms: Vec<f64> = embeddings.iter().map(|e| e.dot(e).sqrt()).collect();
        Self { embeddings, norms }
    }

    /// Find the root node (closest to origin).
    pub fn root(&self) -> usize {
        self.norms
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    /// Get the depth ranking of all nodes (0 = shallowest/root).
    ///
    /// Nodes are ranked by distance from origin. Returns `(node_idx, rank)`.
    pub fn depth_ranking(&self) -> Vec<(usize, usize)> {
        let mut indexed: Vec<(usize, f64)> = self
            .norms
            .iter()
            .enumerate()
            .map(|(i, &n)| (i, n))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        indexed
            .iter()
            .enumerate()
            .map(|(rank, &(idx, _))| (idx, rank))
            .collect()
    }

    /// Get children of a node from a known edge list.
    pub fn children(&self, parent: usize, edges: &[(usize, usize)]) -> Vec<usize> {
        edges
            .iter()
            .filter(|&&(p, _)| p == parent)
            .map(|&(_, c)| c)
            .collect()
    }

    /// Infer parent for each node based on geometry (no edge list needed).
    ///
    /// For each node, the inferred parent is the closest node that is
    /// nearer to the origin (shallower in the hierarchy).
    ///
    /// Returns `Vec<Option<usize>>` where `None` = root (no parent).
    pub fn infer_parents(&self) -> Vec<Option<usize>> {
        let n = self.embeddings.len();
        let mut parents = vec![None; n];

        for (i, parent) in parents.iter_mut().enumerate() {
            let mut best_parent = None;
            let mut best_dist = f64::INFINITY;

            for j in 0..n {
                if j == i || self.norms[j] >= self.norms[i] {
                    continue;
                }
                let d = poincare_distance(&self.embeddings[i], &self.embeddings[j])
                    .unwrap_or(f64::INFINITY);
                if d < best_dist {
                    best_dist = d;
                    best_parent = Some(j);
                }
            }

            *parent = best_parent;
        }

        parents
    }

    /// Infer the full tree structure and return as an edge list.
    ///
    /// Each edge `(parent, child)` is inferred from geometry.
    pub fn infer_tree(&self) -> Vec<(usize, usize)> {
        self.infer_parents()
            .iter()
            .enumerate()
            .filter_map(|(child, parent)| parent.map(|p| (p, child)))
            .collect()
    }

    /// Check if node `ancestor` is an ancestor of node `descendant`
    /// in the inferred tree.
    pub fn is_ancestor(&self, ancestor: usize, descendant: usize) -> bool {
        let parents = self.infer_parents();
        let mut current = Some(descendant);
        let mut depth = 0;

        while let Some(node) = current {
            if node == ancestor {
                return true;
            }
            current = parents[node];
            depth += 1;
            if depth > self.embeddings.len() {
                break; // Cycle protection
            }
        }

        false
    }

    /// Get the norm (distance from origin) for a node.
    /// Lower norm = higher in hierarchy.
    pub fn norm(&self, node: usize) -> f64 {
        self.norms[node]
    }

    /// Get all leaf nodes (nodes with no children in the inferred tree).
    pub fn leaves(&self) -> Vec<usize> {
        let tree = self.infer_tree();
        let n = self.embeddings.len();
        let has_children: std::collections::HashSet<usize> = tree.iter().map(|&(p, _)| p).collect();
        (0..n).filter(|i| !has_children.contains(i)).collect()
    }
}

/// Compute the hierarchy quality metric: mean average precision (MAP).
///
/// For each node, checks if its true parent (from edges) is the
/// geometrically closest shallower node. Higher MAP = better embedding.
pub fn hierarchy_map_score(embeddings: &[Array1<f64>], edges: &[(usize, usize)]) -> f64 {
    let decoder = HierarchyDecoder::new(embeddings);
    let inferred = decoder.infer_parents();

    // Build true parent map from edges
    let mut true_parent = vec![None; embeddings.len()];
    for &(p, c) in edges {
        true_parent[c] = Some(p);
    }

    let mut correct = 0;
    let mut total = 0;

    for (child, true_p) in true_parent.iter().enumerate() {
        if let Some(tp) = true_p {
            total += 1;
            if inferred[child] == Some(*tp) {
                correct += 1;
            }
        }
    }

    if total == 0 {
        1.0
    } else {
        correct as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperbolic::poincare_distance;

    fn simple_tree_edges() -> Vec<(usize, usize)> {
        // Tree:   0 (root)
        //        / \
        //       1   2
        //      / \
        //     3   4
        vec![(0, 1), (0, 2), (1, 3), (1, 4)]
    }

    #[test]
    fn test_embedder_root_near_origin() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01)
            .with_seed(42);
        let emb = embedder.fit(5, &edges);

        let norms: Vec<f64> = emb.iter().map(|e| e.dot(e).sqrt()).collect();
        // Root (0) should be closest to origin
        let root_norm = norms[0];
        let leaf_avg = (norms[3] + norms[4]) / 2.0;
        assert!(
            root_norm < leaf_avg,
            "Root norm ({:.4}) should be less than leaf avg ({:.4})",
            root_norm,
            leaf_avg
        );
    }

    #[test]
    fn test_embedder_parent_closer_than_child() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01);
        let emb = embedder.fit(5, &edges);

        let norm_parent = emb[1].dot(&emb[1]).sqrt();
        let norm_child_3 = emb[3].dot(&emb[3]).sqrt();
        let norm_child_4 = emb[4].dot(&emb[4]).sqrt();

        assert!(
            norm_parent < norm_child_3 || norm_parent < norm_child_4,
            "Parent should generally be closer to origin than at least one child"
        );
    }

    #[test]
    fn test_embedder_connected_closer_than_random() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01);
        let emb = embedder.fit(5, &edges);

        // Distance between connected pair (0,1) vs unconnected (0,4)
        // Note: 0→1 is direct edge, 0→4 is two hops
        let d_connected = poincare_distance(&emb[0], &emb[1]).unwrap();
        let d_two_hop = poincare_distance(&emb[0], &emb[4]).unwrap();
        assert!(
            d_connected < d_two_hop,
            "Direct edge ({:.4}) should be shorter than two-hop ({:.4})",
            d_connected,
            d_two_hop
        );
    }

    #[test]
    fn test_decoder_root() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01);
        let emb = embedder.fit(5, &edges);

        let decoder = HierarchyDecoder::new(&emb);
        let root = decoder.root();
        assert_eq!(root, 0, "Node 0 should be detected as root");
    }

    #[test]
    fn test_decoder_children() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(5, 2)
            .with_epochs(100)
            .with_learning_rate(0.01);
        let emb = embedder.fit(5, &edges);

        let decoder = HierarchyDecoder::new(&emb);
        let children_0 = decoder.children(0, &edges);
        assert_eq!(children_0.len(), 2);
        assert!(children_0.contains(&1));
        assert!(children_0.contains(&2));
    }

    #[test]
    fn test_decoder_depth_ranking() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01);
        let emb = embedder.fit(5, &edges);

        let decoder = HierarchyDecoder::new(&emb);
        let ranking = decoder.depth_ranking();
        // Root should have rank 0 (shallowest)
        let root_rank = ranking.iter().find(|&&(idx, _)| idx == 0).unwrap().1;
        assert_eq!(root_rank, 0, "Root should have depth rank 0");
    }

    #[test]
    fn test_decoder_infer_tree() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01);
        let emb = embedder.fit(5, &edges);

        let decoder = HierarchyDecoder::new(&emb);
        let inferred = decoder.infer_tree();
        // Should have n-1 edges for a tree with n nodes
        assert_eq!(inferred.len(), 4, "Tree with 5 nodes should have 4 edges");
    }

    #[test]
    fn test_decoder_leaves() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01);
        let emb = embedder.fit(5, &edges);

        let decoder = HierarchyDecoder::new(&emb);
        let leaves = decoder.leaves();
        // Nodes 2, 3, 4 are leaves (no children in inferred tree should roughly match)
        assert!(
            leaves.len() >= 2,
            "Should have at least 2 leaves, got {}",
            leaves.len()
        );
    }

    #[test]
    fn test_hierarchy_map_score() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01);
        let emb = embedder.fit(5, &edges);

        let map = hierarchy_map_score(&emb, &edges);
        assert!(map > 0.0, "MAP should be positive, got {}", map);
    }

    #[test]
    fn test_embedder_reproducible() {
        let edges = simple_tree_edges();
        let embedder = PoincareEmbedder::new(5, 2)
            .with_epochs(50)
            .with_learning_rate(0.01)
            .with_seed(99);
        let emb1 = embedder.fit(5, &edges);
        let emb2 = embedder.fit(5, &edges);

        for (a, b) in emb1.iter().zip(emb2.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-10,
                    "Same seed should give same embeddings"
                );
            }
        }
    }

    #[test]
    fn test_larger_hierarchy() {
        // Broader tree: 3-level taxonomy
        //       0
        //     / | \
        //    1  2  3
        //   /|  |
        //  4 5  6
        let edges = vec![(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6)];

        let embedder = PoincareEmbedder::new(10, 5)
            .with_epochs(500)
            .with_learning_rate(0.01);
        let emb = embedder.fit(7, &edges);

        let decoder = HierarchyDecoder::new(&emb);
        assert_eq!(decoder.root(), 0);

        // Root should be closer to origin than all leaves
        let root_norm = decoder.norm(0);
        for &leaf in &[3, 4, 5, 6] {
            assert!(
                root_norm < decoder.norm(leaf),
                "Root norm ({:.4}) should be < leaf {} norm ({:.4})",
                root_norm,
                leaf,
                decoder.norm(leaf)
            );
        }
    }
}
