//! Unified geometric space distance computation.
//!
//! Provides a single `GeometricSpace` enum that wraps 10 distance metrics
//! behind one dispatch function. Inspired by TARS v1's `HyperComplexGeometricDSL`
//! module (which TARS v2 is deferring to v3+, making it a natural fit for ix).
//!
//! # Example
//!
//! ```
//! use ix_math::geometric_space::{GeometricSpace, distance};
//! use ndarray::array;
//!
//! let a = array![0.0, 0.0];
//! let b = array![3.0, 4.0];
//!
//! // Euclidean
//! let d = distance(&GeometricSpace::Euclidean, &a, &b).unwrap();
//! assert!((d - 5.0).abs() < 1e-10);
//!
//! // Manhattan
//! let d = distance(&GeometricSpace::Manhattan, &a, &b).unwrap();
//! assert!((d - 7.0).abs() < 1e-10);
//!
//! // Chebyshev
//! let d = distance(&GeometricSpace::Chebyshev, &a, &b).unwrap();
//! assert!((d - 4.0).abs() < 1e-10);
//! ```

use ndarray::{Array1, Array2};

use crate::distance;
use crate::error::MathError;
use crate::hyperbolic::poincare_distance;

/// Unified geometric space descriptor. Each variant captures the parameters
/// needed for a specific distance metric.
#[derive(Debug, Clone)]
pub enum GeometricSpace {
    /// Standard Euclidean distance: sqrt(sum((a_i - b_i)^2)).
    Euclidean,

    /// Hyperbolic distance in the Poincare ball model. Points must be inside
    /// the open unit ball (||x|| < 1). The `curvature` parameter scales the
    /// distance (typically 1.0 for the standard Poincare metric).
    Hyperbolic { curvature: f64 },

    /// Great-circle (spherical) distance. Points are normalized to the unit
    /// sphere, and the returned distance is `radius * acos(a . b / (|a||b|))`.
    Spherical { radius: f64 },

    /// Minkowski p-norm distance: (sum(|a_i - b_i|^p))^(1/p).
    Minkowski { p: f64 },

    /// Minkowski spacetime pseudo-distance with a signature vector. For
    /// signature = [-1, 1, 1, 1] this is the special-relativity interval.
    /// The absolute value is taken before the sqrt.
    MinkowskiSpacetime { signature: Vec<f64> },

    /// Mahalanobis distance with an inverse covariance matrix:
    /// sqrt((a - b)^T S^-1 (a - b)).
    Mahalanobis { inv_cov: Array2<f64> },

    /// 1-D Wasserstein (earth-mover's) distance between two sorted
    /// distributions. Vectors are treated as 1-D samples; this is equivalent
    /// to the L1 distance between sorted vectors.
    Wasserstein1D,

    /// Manhattan (L1) distance: sum(|a_i - b_i|).
    Manhattan,

    /// Chebyshev (L-infinity) distance: max(|a_i - b_i|).
    Chebyshev,

    /// Hamming distance: count of coordinates where a_i != b_i.
    Hamming,

    /// Jaccard distance on binary vectors: 1 - |A intersect B| / |A union B|.
    /// Non-zero entries are treated as set membership.
    Jaccard,
}

/// Compute the distance between two vectors in the specified geometric space.
///
/// Returns `MathError::DimensionMismatch` if the vectors have different
/// lengths, and other errors specific to the chosen space (e.g.
/// `MathError::InvalidParameter` for Hyperbolic points outside the unit ball).
pub fn distance(
    space: &GeometricSpace,
    a: &Array1<f64>,
    b: &Array1<f64>,
) -> Result<f64, MathError> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    match space {
        GeometricSpace::Euclidean => distance::euclidean(a, b),

        GeometricSpace::Hyperbolic { curvature } => {
            let d = poincare_distance(a, b)?;
            Ok(curvature.abs() * d)
        }

        GeometricSpace::Spherical { radius } => {
            let norm_a = a.dot(a).sqrt();
            let norm_b = b.dot(b).sqrt();
            if norm_a < 1e-12 || norm_b < 1e-12 {
                return Ok(0.0);
            }
            let cos_angle = (a.dot(b) / (norm_a * norm_b)).clamp(-1.0, 1.0);
            Ok(radius * cos_angle.acos())
        }

        GeometricSpace::Minkowski { p } => distance::minkowski(a, b, *p),

        GeometricSpace::MinkowskiSpacetime { signature } => {
            if signature.len() != a.len() {
                return Err(MathError::DimensionMismatch {
                    expected: a.len(),
                    got: signature.len(),
                });
            }
            let mut acc = 0.0;
            for i in 0..a.len() {
                let d = a[i] - b[i];
                acc += signature[i] * d * d;
            }
            Ok(acc.abs().sqrt())
        }

        GeometricSpace::Mahalanobis { inv_cov } => {
            let n = a.len();
            if inv_cov.shape() != [n, n] {
                return Err(MathError::DimensionMismatch {
                    expected: n,
                    got: inv_cov.shape()[0],
                });
            }
            let diff = a - b;
            let sx = inv_cov.dot(&diff);
            let acc = diff.dot(&sx);
            Ok(acc.max(0.0).sqrt())
        }

        GeometricSpace::Wasserstein1D => {
            // 1D EMD for two sorted "sample" vectors of the same length is
            // the mean L1 distance between sorted elements -- a classic
            // result for empirical distributions.
            let mut sa: Vec<f64> = a.iter().copied().collect();
            let mut sb: Vec<f64> = b.iter().copied().collect();
            sa.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            sb.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let acc: f64 = sa.iter().zip(sb.iter()).map(|(x, y)| (x - y).abs()).sum();
            Ok(acc / a.len() as f64)
        }

        GeometricSpace::Manhattan => distance::manhattan(a, b),

        GeometricSpace::Chebyshev => distance::chebyshev(a, b),

        GeometricSpace::Hamming => {
            let count = a
                .iter()
                .zip(b.iter())
                .filter(|(x, y)| (**x - **y).abs() > 1e-12)
                .count();
            Ok(count as f64)
        }

        GeometricSpace::Jaccard => {
            // Treat non-zero entries as set membership.
            let mut intersection = 0usize;
            let mut union = 0usize;
            for (x, y) in a.iter().zip(b.iter()) {
                let in_a = x.abs() > 1e-12;
                let in_b = y.abs() > 1e-12;
                if in_a && in_b {
                    intersection += 1;
                }
                if in_a || in_b {
                    union += 1;
                }
            }
            if union == 0 {
                Ok(0.0)
            } else {
                Ok(1.0 - (intersection as f64 / union as f64))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_euclidean() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let d = distance(&GeometricSpace::Euclidean, &a, &b).unwrap();
        assert!(close(d, 5.0));
    }

    #[test]
    fn test_manhattan() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let d = distance(&GeometricSpace::Manhattan, &a, &b).unwrap();
        assert!(close(d, 7.0));
    }

    #[test]
    fn test_chebyshev() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let d = distance(&GeometricSpace::Chebyshev, &a, &b).unwrap();
        assert!(close(d, 4.0));
    }

    #[test]
    fn test_minkowski_p3() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let d = distance(&GeometricSpace::Minkowski { p: 3.0 }, &a, &b).unwrap();
        let expected = (27.0_f64 + 64.0).powf(1.0 / 3.0);
        assert!(close(d, expected));
    }

    #[test]
    fn test_hyperbolic_inside_ball() {
        let a = array![0.1, 0.0];
        let b = array![0.2, 0.0];
        let d = distance(&GeometricSpace::Hyperbolic { curvature: 1.0 }, &a, &b).unwrap();
        assert!(d > 0.0);
        // Hyperbolic distance is larger than Euclidean for points away from origin
        let euclid = distance(&GeometricSpace::Euclidean, &a, &b).unwrap();
        assert!(d > euclid);
    }

    #[test]
    fn test_spherical_same_point() {
        let a = array![1.0, 0.0, 0.0];
        let d = distance(&GeometricSpace::Spherical { radius: 1.0 }, &a, &a).unwrap();
        assert!(close(d, 0.0));
    }

    #[test]
    fn test_spherical_orthogonal() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![0.0, 1.0, 0.0];
        let d = distance(&GeometricSpace::Spherical { radius: 1.0 }, &a, &b).unwrap();
        // 90 degrees = pi/2
        assert!(close(d, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn test_minkowski_spacetime_spacelike() {
        // Special relativity signature: [-1, 1, 1, 1]
        let a = array![0.0, 0.0, 0.0, 0.0];
        let b = array![0.0, 3.0, 4.0, 0.0];
        let space = GeometricSpace::MinkowskiSpacetime {
            signature: vec![-1.0, 1.0, 1.0, 1.0],
        };
        let d = distance(&space, &a, &b).unwrap();
        // |0 + 9 + 16 + 0|^(1/2) = 5
        assert!(close(d, 5.0));
    }

    #[test]
    fn test_mahalanobis_identity_matches_euclidean() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let inv_cov = Array2::eye(2);
        let d = distance(&GeometricSpace::Mahalanobis { inv_cov }, &a, &b).unwrap();
        assert!(close(d, 5.0));
    }

    #[test]
    fn test_wasserstein_1d() {
        // Two distributions with known EMD
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];
        let d = distance(&GeometricSpace::Wasserstein1D, &a, &b).unwrap();
        // Sorted: [1,2,3] vs [2,3,4] -> mean |diff| = (1+1+1)/3 = 1
        assert!(close(d, 1.0));
    }

    #[test]
    fn test_hamming() {
        let a = array![1.0, 0.0, 1.0, 0.0];
        let b = array![1.0, 1.0, 1.0, 1.0];
        let d = distance(&GeometricSpace::Hamming, &a, &b).unwrap();
        assert!(close(d, 2.0));
    }

    #[test]
    fn test_jaccard() {
        // A = {0, 2}, B = {0, 1}, intersection = {0}, union = {0, 1, 2}
        // distance = 1 - 1/3 = 2/3
        let a = array![1.0, 0.0, 1.0];
        let b = array![1.0, 1.0, 0.0];
        let d = distance(&GeometricSpace::Jaccard, &a, &b).unwrap();
        assert!(close(d, 2.0 / 3.0));
    }

    #[test]
    fn test_jaccard_identical() {
        let a = array![1.0, 0.0, 1.0];
        let d = distance(&GeometricSpace::Jaccard, &a, &a).unwrap();
        assert!(close(d, 0.0));
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = array![1.0, 2.0];
        let b = array![1.0, 2.0, 3.0];
        let err = distance(&GeometricSpace::Euclidean, &a, &b).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch { .. }));
    }
}
