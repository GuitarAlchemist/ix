//! Simplicial complexes and Vietoris-Rips construction.
//!
//! A simplicial complex is built from simplices (vertices, edges, triangles, etc.).
//! The Vietoris-Rips complex constructs a filtered complex from point cloud data,
//! where simplices appear at the distance scale at which all their vertices are
//! pairwise within that distance.
//!
//! # Examples
//!
//! ```
//! use ix_topo::simplex::{Simplex, SimplexStream, rips_complex};
//! use ndarray::array;
//!
//! // Three points forming a triangle
//! let points = vec![
//!     vec![0.0, 0.0],
//!     vec![1.0, 0.0],
//!     vec![0.5, 0.866],
//! ];
//!
//! // Build Rips complex up to dimension 2, max radius 1.5
//! let stream = rips_complex(&points, 2, 1.5);
//! assert!(stream.len() > 3); // vertices + edges + possibly triangle
//! ```

/// A k-simplex represented as a sorted set of vertex indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    /// Sorted vertex indices.
    pub vertices: Vec<usize>,
}

impl Simplex {
    /// Create a new simplex from vertex indices (will be sorted).
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.sort_unstable();
        vertices.dedup();
        Self { vertices }
    }

    /// Dimension of this simplex (0 = vertex, 1 = edge, 2 = triangle, ...).
    pub fn dimension(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.vertices.len() - 1
        }
    }

    /// Compute the boundary operator: returns all (k-1)-faces of this k-simplex.
    ///
    /// For a k-simplex [v₀, v₁, ..., vₖ], the boundary is the alternating sum
    /// of (k-1)-faces obtained by removing one vertex at a time.
    pub fn boundary(&self) -> Vec<Simplex> {
        if self.vertices.len() <= 1 {
            return vec![];
        }
        (0..self.vertices.len())
            .map(|i| {
                let mut face = self.vertices.clone();
                face.remove(i);
                Simplex { vertices: face }
            })
            .collect()
    }

    /// Check if this simplex is a face of another simplex.
    pub fn is_face_of(&self, other: &Simplex) -> bool {
        self.vertices.iter().all(|v| other.vertices.contains(v))
    }
}

impl PartialOrd for Simplex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Simplex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dimension()
            .cmp(&other.dimension())
            .then_with(|| self.vertices.cmp(&other.vertices))
    }
}

/// A filtered simplex: a simplex with a birth time (filtration value).
#[derive(Debug, Clone)]
pub struct FilteredSimplex {
    pub simplex: Simplex,
    /// The filtration value at which this simplex appears.
    pub birth: f64,
}

/// A filtered simplicial complex (stream of simplices ordered by birth time).
#[derive(Debug, Clone)]
pub struct SimplexStream {
    pub simplices: Vec<FilteredSimplex>,
}

impl SimplexStream {
    /// Create an empty stream.
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
        }
    }

    /// Number of simplices in the stream.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Whether the stream is empty.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// Add a simplex with its birth time.
    pub fn add(&mut self, simplex: Simplex, birth: f64) {
        self.simplices.push(FilteredSimplex { simplex, birth });
    }

    /// Sort simplices by birth time, then by dimension, then lexicographically.
    pub fn sort(&mut self) {
        self.simplices.sort_by(|a, b| {
            a.birth
                .partial_cmp(&b.birth)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.simplex.cmp(&b.simplex))
        });
    }

    /// Get all simplices of a given dimension.
    pub fn simplices_of_dim(&self, dim: usize) -> Vec<&FilteredSimplex> {
        self.simplices
            .iter()
            .filter(|fs| fs.simplex.dimension() == dim)
            .collect()
    }

    /// Compute Betti numbers at a given filtration radius.
    ///
    /// Returns β₀, β₁, β₂, ... up to the maximum dimension present.
    pub fn betti_numbers(&self, radius: f64) -> Vec<usize> {
        let active: Vec<&Simplex> = self
            .simplices
            .iter()
            .filter(|fs| fs.birth <= radius)
            .map(|fs| &fs.simplex)
            .collect();

        if active.is_empty() {
            return vec![];
        }

        let max_dim = active.iter().map(|s| s.dimension()).max().unwrap_or(0);
        let mut betti = Vec::with_capacity(max_dim + 1);

        for d in 0..=max_dim {
            let simplices_d: Vec<&Simplex> = active
                .iter()
                .filter(|s| s.dimension() == d)
                .copied()
                .collect();
            let simplices_d_minus_1: Vec<&Simplex> = if d > 0 {
                active
                    .iter()
                    .filter(|s| s.dimension() == d - 1)
                    .copied()
                    .collect()
            } else {
                vec![]
            };
            let simplices_d_plus_1: Vec<&Simplex> = active
                .iter()
                .filter(|s| s.dimension() == d + 1)
                .copied()
                .collect();

            // β_d = dim(ker(∂_d)) - dim(im(∂_{d+1}))
            // Approximate via rank computation using Z/2 coefficients
            let ker_d = if d == 0 {
                simplices_d.len() // All 0-simplices are cycles
            } else {
                simplices_d.len() - boundary_rank_z2(&simplices_d, &simplices_d_minus_1)
            };

            let im_d_plus_1 = if simplices_d_plus_1.is_empty() {
                0
            } else {
                boundary_rank_z2(&simplices_d_plus_1, &simplices_d)
            };

            betti.push(ker_d.saturating_sub(im_d_plus_1));
        }

        betti
    }
}

impl Default for SimplexStream {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a Vietoris-Rips complex from a point cloud.
///
/// A simplex σ = {v₀, ..., vₖ} is included at filtration value
/// max(d(vᵢ, vⱼ)) for all pairs i, j.
///
/// - `points`: list of point coordinates (each point is a `Vec<f64>`)
/// - `max_dim`: maximum simplex dimension to include
/// - `max_radius`: maximum filtration value
pub fn rips_complex(points: &[Vec<f64>], max_dim: usize, max_radius: f64) -> SimplexStream {
    let n = points.len();
    let mut stream = SimplexStream::new();

    // Pairwise distances
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&points[i], &points[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Add vertices (birth = 0)
    for i in 0..n {
        stream.add(Simplex::new(vec![i]), 0.0);
    }

    // Add higher-dimensional simplices
    for dim in 1..=max_dim {
        // Generate all (dim+1)-subsets of vertices
        let mut subset = vec![0usize; dim + 1];
        add_simplices_recursive(
            &dist,
            &mut stream,
            &mut subset,
            0,
            0,
            n,
            dim + 1,
            max_radius,
        );
    }

    stream.sort();
    stream
}

/// Compute the boundary matrix rank over Z/2.
///
/// Given d-simplices and (d-1)-simplices, compute the rank of the
/// boundary matrix ∂_d using Gaussian elimination over GF(2).
fn boundary_rank_z2(d_simplices: &[&Simplex], d_minus_1_simplices: &[&Simplex]) -> usize {
    if d_simplices.is_empty() || d_minus_1_simplices.is_empty() {
        return 0;
    }

    let nrows = d_minus_1_simplices.len();
    let ncols = d_simplices.len();

    // Build boundary matrix as bit vectors (columns)
    let mut columns: Vec<Vec<bool>> = Vec::with_capacity(ncols);

    for sigma in d_simplices {
        let mut col = vec![false; nrows];
        let faces = sigma.boundary();
        for face in &faces {
            if let Some(idx) = d_minus_1_simplices
                .iter()
                .position(|s| s.vertices == face.vertices)
            {
                col[idx] = true;
            }
        }
        columns.push(col);
    }

    // Gaussian elimination over GF(2) to find rank
    let mut rank = 0;
    let mut pivot_row = vec![None::<usize>; nrows];

    for col_idx in 0..ncols {
        // Find lowest non-zero entry
        loop {
            let lowest = columns[col_idx]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, &v)| v)
                .map(|(i, _)| i);

            match lowest {
                None => break, // Column is zero
                Some(low) => {
                    if let Some(prev_col) = pivot_row[low] {
                        // Add previous column (XOR over GF(2))
                        let prev = columns[prev_col].clone();
                        for (i, val) in columns[col_idx].iter_mut().enumerate() {
                            *val ^= prev[i];
                        }
                    } else {
                        pivot_row[low] = Some(col_idx);
                        rank += 1;
                        break;
                    }
                }
            }
        }
    }

    rank
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[allow(clippy::too_many_arguments)]
fn add_simplices_recursive(
    dist: &[Vec<f64>],
    stream: &mut SimplexStream,
    subset: &mut [usize],
    start: usize,
    idx: usize,
    n: usize,
    k: usize,
    max_radius: f64,
) {
    if idx == k {
        // Compute diameter of this subset
        let mut diameter = 0.0f64;
        for i in 0..k {
            for j in (i + 1)..k {
                let d = dist[subset[i]][subset[j]];
                if d > diameter {
                    diameter = d;
                }
            }
        }
        if diameter <= max_radius {
            stream.add(Simplex::new(subset.to_vec()), diameter);
        }
        return;
    }

    for v in start..n {
        subset[idx] = v;
        add_simplices_recursive(dist, stream, subset, v + 1, idx + 1, n, k, max_radius);
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_new_sorts() {
        let s = Simplex::new(vec![3, 1, 2]);
        assert_eq!(s.vertices, vec![1, 2, 3]);
    }

    #[test]
    fn test_simplex_dimension() {
        assert_eq!(Simplex::new(vec![0]).dimension(), 0);
        assert_eq!(Simplex::new(vec![0, 1]).dimension(), 1);
        assert_eq!(Simplex::new(vec![0, 1, 2]).dimension(), 2);
    }

    #[test]
    fn test_simplex_boundary() {
        let triangle = Simplex::new(vec![0, 1, 2]);
        let boundary = triangle.boundary();
        assert_eq!(boundary.len(), 3);
        assert!(boundary.contains(&Simplex::new(vec![1, 2])));
        assert!(boundary.contains(&Simplex::new(vec![0, 2])));
        assert!(boundary.contains(&Simplex::new(vec![0, 1])));
    }

    #[test]
    fn test_simplex_boundary_vertex() {
        let vertex = Simplex::new(vec![0]);
        assert!(vertex.boundary().is_empty());
    }

    #[test]
    fn test_simplex_is_face_of() {
        let edge = Simplex::new(vec![0, 1]);
        let triangle = Simplex::new(vec![0, 1, 2]);
        assert!(edge.is_face_of(&triangle));
        assert!(!triangle.is_face_of(&edge));
    }

    #[test]
    fn test_stream_basic() {
        let mut stream = SimplexStream::new();
        assert!(stream.is_empty());
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.add(Simplex::new(vec![1]), 0.0);
        stream.add(Simplex::new(vec![0, 1]), 1.0);
        assert_eq!(stream.len(), 3);
    }

    #[test]
    fn test_stream_sort() {
        let mut stream = SimplexStream::new();
        stream.add(Simplex::new(vec![0, 1]), 1.0);
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.sort();
        assert_eq!(stream.simplices[0].simplex.vertices, vec![0]);
        assert_eq!(stream.simplices[1].simplex.vertices, vec![0, 1]);
    }

    #[test]
    fn test_rips_complex_three_points() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866]];
        let stream = rips_complex(&points, 2, 1.5);

        // Should have 3 vertices + 3 edges + 1 triangle = 7
        let n_vertices = stream.simplices_of_dim(0).len();
        let n_edges = stream.simplices_of_dim(1).len();
        let n_triangles = stream.simplices_of_dim(2).len();
        assert_eq!(n_vertices, 3);
        assert_eq!(n_edges, 3);
        assert_eq!(n_triangles, 1);
    }

    #[test]
    fn test_rips_complex_max_radius_filters() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 0.0], // Far away
        ];
        let stream = rips_complex(&points, 1, 1.5);
        // Only edge between points 0 and 1 should appear
        let edges = stream.simplices_of_dim(1);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_betti_numbers_two_components() {
        // Two disconnected points: β₀ = 2
        let mut stream = SimplexStream::new();
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.add(Simplex::new(vec![1]), 0.0);
        stream.sort();
        let betti = stream.betti_numbers(1.0);
        assert_eq!(betti[0], 2);
    }

    #[test]
    fn test_betti_numbers_connected() {
        // Two points connected by edge: β₀ = 1
        let mut stream = SimplexStream::new();
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.add(Simplex::new(vec![1]), 0.0);
        stream.add(Simplex::new(vec![0, 1]), 0.5);
        stream.sort();
        let betti = stream.betti_numbers(1.0);
        assert_eq!(betti[0], 1);
    }

    #[test]
    fn test_betti_numbers_triangle_loop() {
        // Triangle boundary (3 vertices, 3 edges, no face): β₀ = 1, β₁ = 1
        let mut stream = SimplexStream::new();
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.add(Simplex::new(vec![1]), 0.0);
        stream.add(Simplex::new(vec![2]), 0.0);
        stream.add(Simplex::new(vec![0, 1]), 0.5);
        stream.add(Simplex::new(vec![1, 2]), 0.5);
        stream.add(Simplex::new(vec![0, 2]), 0.5);
        stream.sort();
        let betti = stream.betti_numbers(1.0);
        assert_eq!(betti[0], 1); // connected
        assert_eq!(betti[1], 1); // one loop
    }

    #[test]
    fn test_betti_filled_triangle() {
        // Filled triangle (3 vertices, 3 edges, 1 face): β₀ = 1, β₁ = 0
        let mut stream = SimplexStream::new();
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.add(Simplex::new(vec![1]), 0.0);
        stream.add(Simplex::new(vec![2]), 0.0);
        stream.add(Simplex::new(vec![0, 1]), 0.5);
        stream.add(Simplex::new(vec![1, 2]), 0.5);
        stream.add(Simplex::new(vec![0, 2]), 0.5);
        stream.add(Simplex::new(vec![0, 1, 2]), 0.5);
        stream.sort();
        let betti = stream.betti_numbers(1.0);
        assert_eq!(betti[0], 1); // connected
        assert_eq!(betti[1], 0); // loop filled in
    }

    #[test]
    fn test_boundary_rank_z2() {
        // Edge [0,1] has boundary {[0], [1]}: rank = 1
        let edge = Simplex::new(vec![0, 1]);
        let v0 = Simplex::new(vec![0]);
        let v1 = Simplex::new(vec![1]);
        let rank = boundary_rank_z2(&[&edge], &[&v0, &v1]);
        assert_eq!(rank, 1);
    }

    #[test]
    fn test_default_stream() {
        let stream: SimplexStream = SimplexStream::default();
        assert!(stream.is_empty());
    }
}
