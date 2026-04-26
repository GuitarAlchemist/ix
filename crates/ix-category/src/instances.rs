//! Concrete category instances.
//!
//! Provides usable category implementations:
//! - `VecCategory`: category of finite-dimensional vector spaces with linear maps
//! - `GraphCategory`: category of directed graphs with graph homomorphisms
//!
//! # Examples
//!
//! ```
//! use ix_category::instances::{VecCategory, LinearMap};
//! use ix_category::core::Category;
//!
//! // Identity linear map on R²
//! let id = VecCategory::id(&2);
//! assert_eq!(id.matrix.len(), 4); // 2×2 = 4 elements
//! assert_eq!(id.matrix[0], 1.0);  // diagonal
//! assert_eq!(id.matrix[1], 0.0);  // off-diagonal
//! ```

use crate::core::{Category, Monoidal};

// ─── VecCategory ─────────────────────────────────────────────────────────────

/// Category of finite-dimensional vector spaces over ℝ.
///
/// Objects are dimensions (usize), morphisms are linear maps (matrices).
pub struct VecCategory;

/// A linear map represented as a flat row-major matrix.
#[derive(Clone, Debug)]
pub struct LinearMap {
    pub rows: usize,
    pub cols: usize,
    /// Row-major matrix data.
    pub matrix: Vec<f64>,
}

impl LinearMap {
    /// Create an identity matrix of size n×n.
    pub fn identity(n: usize) -> Self {
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            matrix[i * n + i] = 1.0;
        }
        Self {
            rows: n,
            cols: n,
            matrix,
        }
    }

    /// Create a zero matrix.
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            matrix: vec![0.0; rows * cols],
        }
    }

    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.matrix[row * self.cols + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.matrix[row * self.cols + col] = val;
    }

    /// Apply this linear map to a vector.
    pub fn apply(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.cols);
        let mut result = vec![0.0; self.rows];
        for (i, res) in result.iter_mut().enumerate() {
            for (j, &val) in v.iter().enumerate() {
                *res += self.get(i, j) * val;
            }
        }
        result
    }
}

impl Category for VecCategory {
    type Obj = usize; // dimension
    type Mor = LinearMap;

    fn id(obj: &Self::Obj) -> Self::Mor {
        LinearMap::identity(*obj)
    }

    fn compose(f: &Self::Mor, g: &Self::Mor) -> Self::Mor {
        // g ∘ f: multiply matrices g * f
        assert_eq!(f.rows, g.cols, "dimension mismatch in composition");
        let mut result = LinearMap::zero(g.rows, f.cols);
        for i in 0..g.rows {
            for j in 0..f.cols {
                let mut sum = 0.0;
                for k in 0..g.cols {
                    sum += g.get(i, k) * f.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    fn dom(f: &Self::Mor) -> Self::Obj {
        f.cols
    }

    fn cod(f: &Self::Mor) -> Self::Obj {
        f.rows
    }
}

impl Monoidal for VecCategory {
    fn tensor_obj(a: &Self::Obj, b: &Self::Obj) -> Self::Obj {
        a + b // Direct sum: dim(V ⊕ W) = dim(V) + dim(W)
    }

    fn tensor_mor(f: &Self::Mor, g: &Self::Mor) -> Self::Mor {
        // Block diagonal matrix: f ⊕ g
        let rows = f.rows + g.rows;
        let cols = f.cols + g.cols;
        let mut result = LinearMap::zero(rows, cols);

        for i in 0..f.rows {
            for j in 0..f.cols {
                result.set(i, j, f.get(i, j));
            }
        }
        for i in 0..g.rows {
            for j in 0..g.cols {
                result.set(f.rows + i, f.cols + j, g.get(i, j));
            }
        }
        result
    }

    fn unit() -> Self::Obj {
        0 // Zero-dimensional space
    }
}

// ─── GraphCategory ───────────────────────────────────────────────────────────

/// Category of directed graphs with graph homomorphisms.
pub struct GraphCategory;

/// A directed graph represented as an adjacency list.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    /// Number of vertices.
    pub num_vertices: usize,
    /// Edges as (source, target) pairs.
    pub edges: Vec<(usize, usize)>,
}

impl Graph {
    /// Create a graph with n vertices and no edges.
    pub fn empty(n: usize) -> Self {
        Self {
            num_vertices: n,
            edges: Vec::new(),
        }
    }

    /// Create a graph with n vertices and given edges.
    pub fn new(n: usize, edges: Vec<(usize, usize)>) -> Self {
        Self {
            num_vertices: n,
            edges,
        }
    }

    /// Add an edge.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
    }

    /// Check if an edge exists.
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.edges.contains(&(from, to))
    }
}

/// A graph homomorphism: a vertex map that preserves edges.
#[derive(Clone, Debug)]
pub struct GraphHomomorphism {
    pub source: Graph,
    pub target: Graph,
    /// vertex_map[i] = image of vertex i in target graph.
    pub vertex_map: Vec<usize>,
}

impl GraphHomomorphism {
    /// Check if this is a valid homomorphism (preserves edges).
    pub fn is_valid(&self) -> bool {
        self.source.edges.iter().all(|&(u, v)| {
            let fu = self.vertex_map[u];
            let fv = self.vertex_map[v];
            self.target.has_edge(fu, fv)
        })
    }
}

impl Category for GraphCategory {
    type Obj = Graph;
    type Mor = GraphHomomorphism;

    fn id(obj: &Self::Obj) -> Self::Mor {
        let vertex_map: Vec<usize> = (0..obj.num_vertices).collect();
        GraphHomomorphism {
            source: obj.clone(),
            target: obj.clone(),
            vertex_map,
        }
    }

    fn compose(f: &Self::Mor, g: &Self::Mor) -> Self::Mor {
        // g ∘ f: compose vertex maps
        let vertex_map: Vec<usize> = f.vertex_map.iter().map(|&v| g.vertex_map[v]).collect();
        GraphHomomorphism {
            source: f.source.clone(),
            target: g.target.clone(),
            vertex_map,
        }
    }

    fn dom(f: &Self::Mor) -> Self::Obj {
        f.source.clone()
    }

    fn cod(f: &Self::Mor) -> Self::Obj {
        f.target.clone()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── VecCategory tests ──

    #[test]
    fn test_vec_identity() {
        let id = VecCategory::id(&3);
        assert_eq!(id.rows, 3);
        assert_eq!(id.cols, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(id.get(i, j), expected);
            }
        }
    }

    #[test]
    fn test_vec_identity_apply() {
        let id = VecCategory::id(&3);
        let v = vec![1.0, 2.0, 3.0];
        let result = id.apply(&v);
        assert_eq!(result, v);
    }

    #[test]
    fn test_vec_compose_identity() {
        let id = VecCategory::id(&2);
        let composed = VecCategory::compose(&id, &id);
        // id ∘ id = id
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((composed.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_vec_compose_scaling() {
        // f scales by 2, g scales by 3 → g∘f scales by 6
        let mut f = VecCategory::id(&1);
        f.set(0, 0, 2.0);
        let mut g = VecCategory::id(&1);
        g.set(0, 0, 3.0);
        let gf = VecCategory::compose(&f, &g);
        assert!((gf.get(0, 0) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_dom_cod() {
        let mut f = LinearMap::zero(3, 2);
        f.set(0, 0, 1.0);
        assert_eq!(VecCategory::dom(&f), 2);
        assert_eq!(VecCategory::cod(&f), 3);
    }

    #[test]
    fn test_vec_monoidal_tensor_obj() {
        assert_eq!(VecCategory::tensor_obj(&2, &3), 5);
    }

    #[test]
    fn test_vec_monoidal_unit() {
        assert_eq!(VecCategory::unit(), 0);
        assert_eq!(VecCategory::tensor_obj(&VecCategory::unit(), &3), 3);
    }

    #[test]
    fn test_vec_monoidal_tensor_mor() {
        let f = VecCategory::id(&2);
        let g = VecCategory::id(&3);
        let fg = VecCategory::tensor_mor(&f, &g);
        assert_eq!(fg.rows, 5);
        assert_eq!(fg.cols, 5);
        // Should be block diagonal identity
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((fg.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    // ── GraphCategory tests ──

    #[test]
    fn test_graph_identity() {
        let g = Graph::new(3, vec![(0, 1), (1, 2)]);
        let id = GraphCategory::id(&g);
        assert_eq!(id.vertex_map, vec![0, 1, 2]);
        assert!(id.is_valid());
    }

    #[test]
    fn test_graph_homomorphism_valid() {
        let source = Graph::new(2, vec![(0, 1)]);
        let target = Graph::new(3, vec![(0, 1), (1, 2), (0, 2)]);
        let hom = GraphHomomorphism {
            source,
            target,
            vertex_map: vec![0, 1], // maps 0→0, 1→1
        };
        assert!(hom.is_valid()); // edge (0,1) exists in target
    }

    #[test]
    fn test_graph_homomorphism_invalid() {
        let source = Graph::new(2, vec![(0, 1)]);
        let target = Graph::new(3, vec![(0, 2)]); // no edge (0,1)
        let hom = GraphHomomorphism {
            source,
            target,
            vertex_map: vec![0, 1], // maps 0→0, 1→1, but target has no edge (0,1)
        };
        assert!(!hom.is_valid());
    }

    #[test]
    fn test_graph_compose() {
        let a = Graph::new(2, vec![(0, 1)]);
        let b = Graph::new(3, vec![(0, 1), (1, 2)]);
        let c = Graph::new(2, vec![(0, 1)]);

        let f = GraphHomomorphism {
            source: a.clone(),
            target: b.clone(),
            vertex_map: vec![0, 1],
        };
        let g = GraphHomomorphism {
            source: b,
            target: c.clone(),
            vertex_map: vec![0, 0, 1],
        };

        let gf = GraphCategory::compose(&f, &g);
        assert_eq!(gf.vertex_map, vec![0, 0]); // 0→0→0, 1→1→0
        assert_eq!(GraphCategory::dom(&gf), a);
        assert_eq!(GraphCategory::cod(&gf), c);
    }

    #[test]
    fn test_graph_dom_cod() {
        let source = Graph::new(2, vec![]);
        let target = Graph::new(3, vec![]);
        let hom = GraphHomomorphism {
            source: source.clone(),
            target: target.clone(),
            vertex_map: vec![0, 1],
        };
        assert_eq!(GraphCategory::dom(&hom), source);
        assert_eq!(GraphCategory::cod(&hom), target);
    }

    #[test]
    fn test_linear_map_apply() {
        // Rotation by 90°: [[0, -1], [1, 0]]
        let mut rot = LinearMap::zero(2, 2);
        rot.set(0, 1, -1.0);
        rot.set(1, 0, 1.0);
        let v = vec![1.0, 0.0];
        let result = rot.apply(&v);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }
}
