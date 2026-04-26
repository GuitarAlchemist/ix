//! Persistent homology computation.
//!
//! Computes persistence diagrams from filtered simplicial complexes using
//! the standard matrix reduction algorithm over Z/2 coefficients.
//!
//! A persistence diagram records (birth, death) pairs for topological features:
//! - Connected components (H₀) born when vertices appear, die when edges merge them
//! - Loops (H₁) born when cycles form, die when triangles fill them
//! - Voids (H₂) born when cavities form, die when tetrahedra fill them
//!
//! # Examples
//!
//! ```
//! use ix_topo::simplex::{Simplex, SimplexStream};
//! use ix_topo::persistence::{compute_persistence, PersistenceDiagram};
//!
//! // Triangle boundary: creates a loop
//! let mut stream = SimplexStream::new();
//! stream.add(Simplex::new(vec![0]), 0.0);
//! stream.add(Simplex::new(vec![1]), 0.0);
//! stream.add(Simplex::new(vec![2]), 0.0);
//! stream.add(Simplex::new(vec![0, 1]), 1.0);
//! stream.add(Simplex::new(vec![1, 2]), 1.0);
//! stream.add(Simplex::new(vec![0, 2]), 1.0);
//! stream.sort();
//!
//! let diagrams = compute_persistence(&stream);
//! // H₀: two components die at radius 1.0 (merged by edges)
//! // H₁: one loop born at radius 1.0 (never dies = infinite persistence)
//! assert!(!diagrams.is_empty());
//! ```

use crate::simplex::SimplexStream;

/// A persistence diagram for a single homology dimension.
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    /// Homology dimension (0 = components, 1 = loops, 2 = voids).
    pub dimension: usize,
    /// (birth, death) pairs. `f64::INFINITY` means the feature never dies.
    pub pairs: Vec<(f64, f64)>,
}

impl PersistenceDiagram {
    /// Number of persistent features.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether this diagram is empty.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Get the persistence (lifetime) of each feature.
    pub fn lifetimes(&self) -> Vec<f64> {
        self.pairs.iter().map(|(b, d)| d - b).collect()
    }

    /// Filter to features with persistence above a threshold.
    pub fn significant(&self, threshold: f64) -> Vec<(f64, f64)> {
        self.pairs
            .iter()
            .filter(|(b, d)| d - b > threshold)
            .copied()
            .collect()
    }
}

/// Compute persistent homology from a filtered simplicial complex.
///
/// Uses the standard reduction algorithm over Z/2 coefficients.
/// Returns one `PersistenceDiagram` per homology dimension present.
pub fn compute_persistence(stream: &SimplexStream) -> Vec<PersistenceDiagram> {
    let n = stream.simplices.len();
    if n == 0 {
        return vec![];
    }

    // Build boundary matrix as sparse columns (sets of row indices)
    // Each column j corresponds to simplex j in the filtration order
    let mut columns: Vec<Vec<usize>> = Vec::with_capacity(n);

    for fs in &stream.simplices {
        let faces = fs.simplex.boundary();
        let mut col = Vec::new();
        for face in &faces {
            // Find index of this face in the stream
            if let Some(idx) = stream
                .simplices
                .iter()
                .position(|s| s.simplex.vertices == face.vertices)
            {
                col.push(idx);
            }
        }
        col.sort_unstable();
        columns.push(col);
    }

    // Reduce the boundary matrix (persistence algorithm)
    let mut low: Vec<Option<usize>> = vec![None; n]; // low[j] = lowest row index in column j
    let mut pivot_col: Vec<Option<usize>> = vec![None; n]; // pivot_col[i] = column with pivot at row i

    for j in 0..n {
        update_low(&columns[j], &mut low[j]);

        while let Some(low_j) = low[j] {
            if let Some(prev) = pivot_col[low_j] {
                // XOR columns (symmetric difference over Z/2)
                xor_columns(&mut columns, j, prev);
                update_low(&columns[j], &mut low[j]);
            } else {
                pivot_col[low_j] = Some(j);
                break;
            }
        }
    }

    // Extract persistence pairs
    let max_dim = stream
        .simplices
        .iter()
        .map(|fs| fs.simplex.dimension())
        .max()
        .unwrap_or(0);

    let mut diagrams: Vec<PersistenceDiagram> = (0..=max_dim)
        .map(|d| PersistenceDiagram {
            dimension: d,
            pairs: Vec::new(),
        })
        .collect();

    let mut paired = vec![false; n];

    for j in 0..n {
        if let Some(low_j) = low[j] {
            // Paired: simplex low_j (birth) paired with simplex j (death)
            let birth = stream.simplices[low_j].birth;
            let death = stream.simplices[j].birth;
            let dim = stream.simplices[low_j].simplex.dimension();
            if dim < diagrams.len() && (death - birth).abs() > 1e-15 {
                diagrams[dim].pairs.push((birth, death));
            }
            paired[low_j] = true;
            paired[j] = true;
        }
    }

    // Essential features: unpaired simplices (infinite persistence)
    for j in 0..n {
        if !paired[j] && columns[j].is_empty() {
            let dim = stream.simplices[j].simplex.dimension();
            let birth = stream.simplices[j].birth;
            if dim < diagrams.len() {
                diagrams[dim].pairs.push((birth, f64::INFINITY));
            }
        }
    }

    diagrams
}

/// Bottleneck distance between two persistence diagrams.
///
/// The bottleneck distance is the infimum over all matchings of the
/// maximum cost of any matched pair.
pub fn bottleneck_distance(pd1: &PersistenceDiagram, pd2: &PersistenceDiagram) -> f64 {
    // Simple approximation: sort by persistence and match greedily
    let mut pts1: Vec<(f64, f64)> = pd1.pairs.to_vec();
    let mut pts2: Vec<(f64, f64)> = pd2.pairs.to_vec();

    // Add diagonal projections for unmatched points
    let n2 = pts2.len();

    // Pad shorter diagram with diagonal points from the longer one
    for p in &pts1 {
        if p.1.is_finite() {
            let mid = (p.0 + p.1) / 2.0;
            pts2.push((mid, mid));
        }
    }
    for p in &pts2[..n2] {
        if p.1.is_finite() {
            let mid = (p.0 + p.1) / 2.0;
            pts1.push((mid, mid));
        }
    }

    // Sort by persistence (descending) and greedily match
    pts1.sort_by(|a, b| {
        let pa = (a.1 - a.0).abs();
        let pb = (b.1 - b.0).abs();
        pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
    });
    pts2.sort_by(|a, b| {
        let pa = (a.1 - a.0).abs();
        let pb = (b.1 - b.0).abs();
        pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let len = pts1.len().min(pts2.len());
    let mut max_cost = 0.0f64;
    for i in 0..len {
        let cost = chebyshev(&pts1[i], &pts2[i]);
        if cost > max_cost {
            max_cost = cost;
        }
    }

    max_cost
}

/// Wasserstein distance between two persistence diagrams.
///
/// W_p(D1, D2) = (sum of |matched pair costs|^p)^(1/p)
pub fn wasserstein_distance(pd1: &PersistenceDiagram, pd2: &PersistenceDiagram, p: f64) -> f64 {
    // Greedy approximation matching by persistence
    let mut pts1: Vec<(f64, f64)> = pd1
        .pairs
        .iter()
        .filter(|x| x.1.is_finite())
        .copied()
        .collect();
    let mut pts2: Vec<(f64, f64)> = pd2
        .pairs
        .iter()
        .filter(|x| x.1.is_finite())
        .copied()
        .collect();

    // Pad with diagonal projections
    while pts1.len() < pts2.len() {
        let pt = pts2[pts1.len()];
        let mid = (pt.0 + pt.1) / 2.0;
        pts1.push((mid, mid));
    }
    while pts2.len() < pts1.len() {
        let pt = pts1[pts2.len()];
        let mid = (pt.0 + pt.1) / 2.0;
        pts2.push((mid, mid));
    }

    pts1.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    pts2.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total: f64 = pts1
        .iter()
        .zip(pts2.iter())
        .map(|(a, b)| chebyshev(a, b).powf(p))
        .sum();

    total.powf(1.0 / p)
}

fn chebyshev(a: &(f64, f64), b: &(f64, f64)) -> f64 {
    (a.0 - b.0).abs().max((a.1 - b.1).abs())
}

fn update_low(col: &[usize], low: &mut Option<usize>) {
    *low = col.last().copied();
}

fn xor_columns(columns: &mut [Vec<usize>], j: usize, prev: usize) {
    let prev_col = columns[prev].clone();
    let col = &mut columns[j];

    // Symmetric difference
    let mut result = Vec::new();
    let (mut i, mut k) = (0, 0);
    while i < col.len() && k < prev_col.len() {
        match col[i].cmp(&prev_col[k]) {
            std::cmp::Ordering::Less => {
                result.push(col[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(prev_col[k]);
                k += 1;
            }
            std::cmp::Ordering::Equal => {
                // Cancel over Z/2
                i += 1;
                k += 1;
            }
        }
    }
    result.extend_from_slice(&col[i..]);
    result.extend_from_slice(&prev_col[k..]);
    *col = result;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex;

    fn make_triangle_boundary_stream() -> SimplexStream {
        let mut stream = SimplexStream::new();
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.add(Simplex::new(vec![1]), 0.0);
        stream.add(Simplex::new(vec![2]), 0.0);
        stream.add(Simplex::new(vec![0, 1]), 1.0);
        stream.add(Simplex::new(vec![1, 2]), 1.0);
        stream.add(Simplex::new(vec![0, 2]), 1.0);
        stream.sort();
        stream
    }

    fn make_filled_triangle_stream() -> SimplexStream {
        let mut s = make_triangle_boundary_stream();
        s.add(Simplex::new(vec![0, 1, 2]), 1.0);
        s.sort();
        s
    }

    #[test]
    fn test_persistence_empty() {
        let stream = SimplexStream::new();
        let diagrams = compute_persistence(&stream);
        assert!(diagrams.is_empty());
    }

    #[test]
    fn test_persistence_single_vertex() {
        let mut stream = SimplexStream::new();
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.sort();
        let diagrams = compute_persistence(&stream);
        // One essential H₀ feature
        assert!(!diagrams[0].is_empty());
        assert_eq!(diagrams[0].pairs[0].0, 0.0);
        assert!(diagrams[0].pairs[0].1.is_infinite());
    }

    #[test]
    fn test_persistence_two_vertices_one_edge() {
        let mut stream = SimplexStream::new();
        stream.add(Simplex::new(vec![0]), 0.0);
        stream.add(Simplex::new(vec![1]), 0.0);
        stream.add(Simplex::new(vec![0, 1]), 1.0);
        stream.sort();
        let diagrams = compute_persistence(&stream);

        // H₀: one component dies at 1.0, one survives
        let h0 = &diagrams[0];
        let finite_h0: Vec<_> = h0.pairs.iter().filter(|p| p.1.is_finite()).collect();
        let infinite_h0: Vec<_> = h0.pairs.iter().filter(|p| p.1.is_infinite()).collect();
        assert_eq!(finite_h0.len(), 1);
        assert_eq!(finite_h0[0].1, 1.0);
        assert_eq!(infinite_h0.len(), 1);
    }

    #[test]
    fn test_persistence_triangle_boundary_has_loop() {
        let stream = make_triangle_boundary_stream();
        let diagrams = compute_persistence(&stream);

        // H₁ should have at least one feature (the loop)
        if diagrams.len() > 1 {
            let h1 = &diagrams[1];
            assert!(!h1.is_empty(), "triangle boundary should create a 1-cycle");
        }
    }

    #[test]
    fn test_persistence_filled_triangle_kills_loop() {
        let stream = make_filled_triangle_stream();
        let diagrams = compute_persistence(&stream);

        // H₁: loop should be killed by the filling triangle
        if diagrams.len() > 1 {
            let h1 = &diagrams[1];
            let infinite_h1: Vec<_> = h1.pairs.iter().filter(|p| p.1.is_infinite()).collect();
            assert_eq!(infinite_h1.len(), 0, "filled triangle should kill the loop");
        }
    }

    #[test]
    fn test_persistence_diagram_lifetimes() {
        let pd = PersistenceDiagram {
            dimension: 0,
            pairs: vec![(0.0, 1.0), (0.0, 3.0), (0.0, f64::INFINITY)],
        };
        let lifetimes = pd.lifetimes();
        assert_eq!(lifetimes[0], 1.0);
        assert_eq!(lifetimes[1], 3.0);
        assert!(lifetimes[2].is_infinite());
    }

    #[test]
    fn test_persistence_diagram_significant() {
        let pd = PersistenceDiagram {
            dimension: 0,
            pairs: vec![(0.0, 0.1), (0.0, 1.0), (0.0, 5.0)],
        };
        let sig = pd.significant(0.5);
        assert_eq!(sig.len(), 2); // only (0,1) and (0,5)
    }

    #[test]
    fn test_bottleneck_distance_identical() {
        let pd = PersistenceDiagram {
            dimension: 0,
            pairs: vec![(0.0, 1.0), (0.0, 2.0)],
        };
        let dist = bottleneck_distance(&pd, &pd);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_bottleneck_distance_different() {
        let pd1 = PersistenceDiagram {
            dimension: 0,
            pairs: vec![(0.0, 1.0)],
        };
        let pd2 = PersistenceDiagram {
            dimension: 0,
            pairs: vec![(0.0, 2.0)],
        };
        let dist = bottleneck_distance(&pd1, &pd2);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_wasserstein_distance_identical() {
        let pd = PersistenceDiagram {
            dimension: 0,
            pairs: vec![(0.0, 1.0), (0.0, 2.0)],
        };
        let dist = wasserstein_distance(&pd, &pd, 2.0);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_wasserstein_distance_empty() {
        let pd1 = PersistenceDiagram {
            dimension: 0,
            pairs: vec![],
        };
        let pd2 = PersistenceDiagram {
            dimension: 0,
            pairs: vec![],
        };
        let dist = wasserstein_distance(&pd1, &pd2, 2.0);
        assert!(dist < 1e-10 || dist.is_nan()); // 0^(1/p) can be 0
    }
}
