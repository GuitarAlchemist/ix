//! Point cloud → persistent homology pipeline.
//!
//! Convenience functions that chain Vietoris-Rips complex construction
//! with persistent homology computation and Betti number extraction.
//!
//! # Examples
//!
//! ```
//! use ix_topo::pointcloud::{persistence_from_points, betti_curve};
//!
//! // Three points forming a triangle
//! let points = vec![
//!     vec![0.0, 0.0],
//!     vec![1.0, 0.0],
//!     vec![0.5, 0.866],
//! ];
//!
//! // Compute persistence diagram
//! let diagrams = persistence_from_points(&points, 2, 2.0);
//! assert!(!diagrams.is_empty());
//!
//! // Compute Betti curve: β₀ and β₁ at varying radii
//! let curve = betti_curve(&points, 2, 10);
//! assert_eq!(curve.len(), 10);
//! ```

use crate::persistence::{compute_persistence, PersistenceDiagram};
use crate::simplex::rips_complex;

/// Compute persistent homology from a point cloud.
///
/// Pipeline: points → Vietoris-Rips complex → persistence diagrams.
///
/// - `points`: list of coordinates (each point is `Vec<f64>`)
/// - `max_dim`: maximum homology dimension (1 = loops, 2 = voids)
/// - `max_radius`: maximum filtration radius
pub fn persistence_from_points(
    points: &[Vec<f64>],
    max_dim: usize,
    max_radius: f64,
) -> Vec<PersistenceDiagram> {
    let stream = rips_complex(points, max_dim, max_radius);
    compute_persistence(&stream)
}

/// Compute Betti numbers at a single filtration radius.
///
/// Returns β₀, β₁, ... up to max_dim.
pub fn betti_at_radius(points: &[Vec<f64>], max_dim: usize, radius: f64) -> Vec<usize> {
    let stream = rips_complex(points, max_dim, radius);
    stream.betti_numbers(radius)
}

/// Compute a Betti curve: Betti numbers at n_steps evenly spaced radii.
///
/// Returns a vector of (radius, betti_numbers) pairs.
/// The max_radius is determined from the maximum pairwise distance.
pub fn betti_curve(points: &[Vec<f64>], max_dim: usize, n_steps: usize) -> Vec<(f64, Vec<usize>)> {
    if points.is_empty() || n_steps == 0 {
        return vec![];
    }

    // Find max pairwise distance
    let mut max_dist = 0.0f64;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d: f64 = points[i]
                .iter()
                .zip(points[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if d > max_dist {
                max_dist = d;
            }
        }
    }

    let step = max_dist / n_steps as f64;
    let stream = rips_complex(points, max_dim, max_dist);

    (0..n_steps)
        .map(|i| {
            let r = (i + 1) as f64 * step;
            let betti = stream.betti_numbers(r);
            (r, betti)
        })
        .collect()
}

/// Extract the most persistent features from a set of diagrams.
///
/// Returns (dimension, birth, death, persistence) tuples sorted by persistence.
pub fn most_persistent_features(
    diagrams: &[PersistenceDiagram],
    top_k: usize,
) -> Vec<(usize, f64, f64, f64)> {
    let mut features: Vec<(usize, f64, f64, f64)> = diagrams
        .iter()
        .flat_map(|d| {
            d.pairs
                .iter()
                .filter(|(_, death)| death.is_finite())
                .map(move |(birth, death)| (d.dimension, *birth, *death, death - birth))
        })
        .collect();

    features.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    features.truncate(top_k);
    features
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_points() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866]]
    }

    fn two_clusters() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
        ]
    }

    #[test]
    fn test_persistence_from_points_triangle() {
        let diagrams = persistence_from_points(&triangle_points(), 2, 2.0);
        assert!(!diagrams.is_empty());
        // H₀ should have features
        assert!(!diagrams[0].is_empty());
    }

    #[test]
    fn test_persistence_from_points_empty() {
        let diagrams = persistence_from_points(&[], 2, 2.0);
        assert!(diagrams.is_empty());
    }

    #[test]
    fn test_betti_at_radius_triangle() {
        let betti = betti_at_radius(&triangle_points(), 2, 1.5);
        // At radius 1.5, all edges should exist → β₀ = 1
        assert!(!betti.is_empty());
        assert_eq!(betti[0], 1);
    }

    #[test]
    fn test_betti_at_radius_clusters() {
        // At small radius, two disconnected clusters → β₀ = 2
        let betti = betti_at_radius(&two_clusters(), 1, 0.5);
        assert_eq!(betti[0], 2);
    }

    #[test]
    fn test_betti_curve_length() {
        let curve = betti_curve(&triangle_points(), 1, 5);
        assert_eq!(curve.len(), 5);
    }

    #[test]
    fn test_betti_curve_radii_increasing() {
        let curve = betti_curve(&triangle_points(), 1, 5);
        for i in 1..curve.len() {
            assert!(curve[i].0 > curve[i - 1].0);
        }
    }

    #[test]
    fn test_betti_curve_empty() {
        let curve = betti_curve(&[], 1, 5);
        assert!(curve.is_empty());
    }

    #[test]
    fn test_most_persistent_features() {
        let diagrams = persistence_from_points(&triangle_points(), 2, 2.0);
        let features = most_persistent_features(&diagrams, 3);
        // Should have at least some finite features
        for (_, birth, death, persistence) in &features {
            assert!(*death > *birth);
            assert!(*persistence > 0.0);
        }
    }

    #[test]
    fn test_most_persistent_features_sorted() {
        let diagrams = persistence_from_points(&two_clusters(), 1, 10.0);
        let features = most_persistent_features(&diagrams, 10);
        // Should be sorted by persistence descending
        for i in 1..features.len() {
            assert!(features[i].3 <= features[i - 1].3);
        }
    }
}
