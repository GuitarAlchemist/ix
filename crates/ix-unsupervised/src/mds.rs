//! Multi-Dimensional Scaling (Classical / Metric MDS).
//!
//! Given an `n x n` pairwise distance matrix, classical MDS finds an
//! embedding in `k` dimensions whose Euclidean distances best preserve
//! the original distances. It is the linear-algebra dual of PCA applied
//! to a double-centered squared-distance matrix.
//!
//! # Algorithm
//!
//! 1. Square the distance matrix element-wise: `D^2`.
//! 2. Double-center: `B = -0.5 * J D^2 J` where `J = I - (1/n) 1 1^T`.
//! 3. Eigendecompose `B`. The top `k` eigenvectors scaled by sqrt of
//!    their eigenvalues give the embedding coordinates.
//!
//! Unlike PCA, MDS does not need the original feature vectors — only the
//! pairwise distances. This makes it a perfect fit for:
//! - Non-metric similarity data
//! - Embedding graphs via shortest-path distances (Isomap uses this)
//! - Any space where we can compute distances but not features
//!
//! # Example
//!
//! ```
//! use ix_unsupervised::mds::classical_mds;
//! use ndarray::array;
//!
//! // Square with side length 1
//! let dists = array![
//!     [0.0, 1.0, 1.414, 1.0],
//!     [1.0, 0.0, 1.0, 1.414],
//!     [1.414, 1.0, 0.0, 1.0],
//!     [1.0, 1.414, 1.0, 0.0],
//! ];
//! let embedding = classical_mds(&dists, 2).unwrap();
//! assert_eq!(embedding.nrows(), 4);
//! assert_eq!(embedding.ncols(), 2);
//! ```

use ndarray::{Array1, Array2};

use ix_math::eigen::symmetric_eigen;
use ix_math::error::MathError;

/// Compute a classical MDS embedding from a pairwise distance matrix.
///
/// Returns an `(n, k)` embedding matrix whose rows are the coordinates
/// of each input point in the reduced space. Accepts both symmetric and
/// near-symmetric matrices (it symmetrizes defensively).
pub fn classical_mds(distances: &Array2<f64>, k: usize) -> Result<Array2<f64>, MathError> {
    let n = distances.nrows();
    if n == 0 {
        return Err(MathError::EmptyInput);
    }
    if distances.ncols() != n {
        return Err(MathError::NotSquare {
            rows: n,
            cols: distances.ncols(),
        });
    }
    if k == 0 || k >= n {
        return Err(MathError::InvalidParameter(format!(
            "k must be in 1..{}, got {}",
            n, k
        )));
    }

    // Square and symmetrize the distance matrix.
    let mut sq = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let d = 0.5 * (distances[[i, j]] + distances[[j, i]]);
            sq[[i, j]] = d * d;
        }
    }

    // Double-centering: B = -0.5 * J * D^2 * J where J = I - (1/n) 1 1^T.
    // Equivalent to B_ij = -0.5 * (D^2_ij - row_mean_i - col_mean_j + grand_mean)
    let row_means: Array1<f64> = sq.rows().into_iter().map(|r| r.sum() / n as f64).collect();
    let col_means: Array1<f64> = sq
        .columns()
        .into_iter()
        .map(|c| c.sum() / n as f64)
        .collect();
    let grand_mean = sq.iter().sum::<f64>() / (n * n) as f64;

    let mut b = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            b[[i, j]] = -0.5 * (sq[[i, j]] - row_means[i] - col_means[j] + grand_mean);
        }
    }

    // Full symmetric eigendecomposition via ix-math::eigen — returns pairs
    // already sorted in descending order, so we just take the top k.
    let (eigenvalues, eigenvectors) = symmetric_eigen(&b)?;

    let mut embedding = Array2::<f64>::zeros((n, k));
    for r in 0..k {
        let lambda = eigenvalues[r].max(0.0);
        let scale = lambda.sqrt();
        for i in 0..n {
            embedding[[i, r]] = eigenvectors[[i, r]] * scale;
        }
    }

    Ok(embedding)
}

/// Build a pairwise distance matrix from a set of row-vectors, using
/// Euclidean distance. Convenience helper for callers that want to go
/// straight from a feature matrix to an MDS embedding.
pub fn pairwise_euclidean(features: &Array2<f64>) -> Array2<f64> {
    let n = features.nrows();
    let mut dists = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut acc = 0.0;
            for k in 0..features.ncols() {
                let d = features[[i, k]] - features[[j, k]];
                acc += d * d;
            }
            let d = acc.sqrt();
            dists[[i, j]] = d;
            dists[[j, i]] = d;
        }
    }
    dists
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn pairwise_error(original: &Array2<f64>, embedding: &Array2<f64>) -> f64 {
        let n = original.nrows();
        let emb_d = pairwise_euclidean(embedding);
        let mut err = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                err += (original[[i, j]] - emb_d[[i, j]]).abs();
                count += 1;
            }
        }
        err / count as f64
    }

    #[test]
    fn test_mds_recovers_2d_square() {
        let s = std::f64::consts::SQRT_2;
        let dists = array![
            [0.0, 1.0, s, 1.0],
            [1.0, 0.0, 1.0, s],
            [s, 1.0, 0.0, 1.0],
            [1.0, s, 1.0, 0.0],
        ];
        let embedding = classical_mds(&dists, 2).unwrap();
        assert_eq!(embedding.nrows(), 4);
        assert_eq!(embedding.ncols(), 2);
        let err = pairwise_error(&dists, &embedding);
        assert!(err < 1e-6, "pairwise error too large: {}", err);
    }

    #[test]
    fn test_mds_from_features() {
        // Random-ish 3D points, embed into 2D
        let points = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0], // coplanar -> 2D is enough
        ];
        let dists = pairwise_euclidean(&points);
        let embedding = classical_mds(&dists, 2).unwrap();
        assert_eq!(embedding.nrows(), 5);
        let err = pairwise_error(&dists, &embedding);
        assert!(err < 1e-5, "coplanar points should embed exactly, err = {}", err);
    }

    #[test]
    fn test_mds_rejects_non_square() {
        let d = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]];
        assert!(classical_mds(&d, 1).is_err());
    }

    #[test]
    fn test_mds_rejects_bad_k() {
        let d = array![[0.0, 1.0], [1.0, 0.0]];
        assert!(classical_mds(&d, 0).is_err());
        assert!(classical_mds(&d, 5).is_err());
    }

    #[test]
    fn test_pairwise_euclidean_symmetric() {
        let points = array![[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]];
        let d = pairwise_euclidean(&points);
        assert!((d[[0, 1]] - 5.0).abs() < 1e-10);
        assert!((d[[1, 0]] - 5.0).abs() < 1e-10);
        assert!((d[[0, 2]] - 10.0).abs() < 1e-10);
        assert!((d[[0, 0]]).abs() < 1e-10);
    }
}
