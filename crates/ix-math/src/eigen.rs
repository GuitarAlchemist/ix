//! Eigenvalue decomposition for real matrices.
//!
//! Currently provides a robust symmetric eigendecomposition via cyclic
//! Jacobi rotations, which handles repeated eigenvalues correctly — unlike
//! power iteration + deflation, which can silently produce duplicated or
//! incorrect eigenvectors for matrices with near-degenerate spectra.
//!
//! This module is the canonical home for eigensolvers used across the
//! workspace. Downstream consumers (MDS, Kernel PCA, LDA, graph Laplacian
//! spectrum, persistent homology) should call into here rather than
//! duplicating the Jacobi routine.
//!
//! # Example
//!
//! ```
//! use ix_math::eigen::symmetric_eigen;
//! use ndarray::array;
//!
//! // Symmetric 2x2 matrix with eigenvalues 3 and 1
//! let a = array![[2.0, 1.0], [1.0, 2.0]];
//! let (values, vectors) = symmetric_eigen(&a).unwrap();
//! // Eigenvalues are returned sorted descending
//! assert!((values[0] - 3.0).abs() < 1e-9);
//! assert!((values[1] - 1.0).abs() < 1e-9);
//! ```

use ndarray::{Array1, Array2};

use crate::error::MathError;

/// Full eigendecomposition of a real symmetric matrix via cyclic Jacobi
/// rotations.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are sorted in
/// descending order and eigenvectors are the corresponding columns of the
/// returned matrix (so `eigenvectors.column(k)` is the eigenvector for
/// `eigenvalues[k]`).
///
/// The algorithm is numerically robust: it handles repeated eigenvalues,
/// near-zero eigenvalues, and moderately ill-conditioned matrices. It is
/// appropriate for small-to-medium matrices (up to a few hundred rows);
/// for larger problems use an iterative Krylov method or a LAPACK binding.
///
/// Returns `MathError::NotSquare` if the input is not square, and
/// `MathError::EmptyInput` if it is empty.
pub fn symmetric_eigen(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), MathError> {
    symmetric_eigen_with_opts(a, 100, 1e-12)
}

/// Symmetric eigendecomposition with explicit sweep and tolerance limits.
///
/// `max_sweeps` caps the outer loop (one sweep = touch every off-diagonal
/// pair once). `tol` is the Frobenius norm of the off-diagonal subject to
/// which the iteration terminates.
pub fn symmetric_eigen_with_opts(
    a: &Array2<f64>,
    max_sweeps: usize,
    tol: f64,
) -> Result<(Array1<f64>, Array2<f64>), MathError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MathError::EmptyInput);
    }
    if a.ncols() != n {
        return Err(MathError::NotSquare {
            rows: n,
            cols: a.ncols(),
        });
    }

    let mut m = a.clone();
    let mut v = Array2::<f64>::eye(n);

    for _sweep in 0..max_sweeps {
        // Compute off-diagonal Frobenius norm as a stopping criterion.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[[p, q]] * m[[p, q]];
            }
        }
        if off.sqrt() < tol {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = m[[p, p]];
                let aqq = m[[q, q]];
                // Compute Jacobi rotation angle that zeros the (p, q) entry
                // of the 2x2 submatrix [[app, apq], [apq, aqq]].
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal entries and zero the off-diagonal.
                m[[p, p]] = app - t * apq;
                m[[q, q]] = aqq + t * apq;
                m[[p, q]] = 0.0;
                m[[q, p]] = 0.0;

                // Update the other rows/columns touched by the rotation.
                for i in 0..n {
                    if i != p && i != q {
                        let aip = m[[i, p]];
                        let aiq = m[[i, q]];
                        m[[i, p]] = c * aip - s * aiq;
                        m[[p, i]] = m[[i, p]];
                        m[[i, q]] = s * aip + c * aiq;
                        m[[q, i]] = m[[i, q]];
                    }
                }
                // Accumulate the rotation into V.
                for i in 0..n {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }
    }

    // Extract eigenvalues from the now-diagonal matrix and sort descending.
    let raw_values: Vec<f64> = (0..n).map(|i| m[[i, i]]).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        raw_values[b]
            .partial_cmp(&raw_values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigenvalues = Array1::<f64>::zeros(n);
    let mut eigenvectors = Array2::<f64>::zeros((n, n));
    for (new_pos, &old_pos) in idx.iter().enumerate() {
        eigenvalues[new_pos] = raw_values[old_pos];
        for i in 0..n {
            eigenvectors[[i, new_pos]] = v[[i, old_pos]];
        }
    }

    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_diagonal_matrix() {
        let a = array![[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let (values, _) = symmetric_eigen(&a).unwrap();
        assert!(close(values[0], 3.0));
        assert!(close(values[1], 2.0));
        assert!(close(values[2], 1.0));
    }

    #[test]
    fn test_2x2_symmetric() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1 with eigenvectors
        // (1, 1)/sqrt(2) and (1, -1)/sqrt(2)
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let (values, vectors) = symmetric_eigen(&a).unwrap();
        assert!(close(values[0], 3.0));
        assert!(close(values[1], 1.0));
        // Verify first eigenvector: Av = 3v
        let v0 = vectors.column(0).to_owned();
        let av0 = a.dot(&v0);
        for i in 0..2 {
            assert!(close(av0[i], 3.0 * v0[i]));
        }
    }

    #[test]
    fn test_eigenvectors_are_orthonormal() {
        let a = array![[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]];
        let (_, vectors) = symmetric_eigen(&a).unwrap();
        // Columns should be orthonormal: V^T V = I
        let vtv = vectors.t().dot(&vectors);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vtv[[i, j]] - expected).abs() < 1e-9,
                    "V^T V[{},{}] = {}, expected {}",
                    i,
                    j,
                    vtv[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_reconstruction() {
        // A = V diag(lambda) V^T
        let a = array![[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]];
        let (values, vectors) = symmetric_eigen(&a).unwrap();
        let lambda = Array2::from_diag(&values);
        let reconstructed = vectors.dot(&lambda).dot(&vectors.t());
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-9,
                    "A[{},{}] mismatch: {} vs {}",
                    i,
                    j,
                    reconstructed[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_repeated_eigenvalues() {
        // Identity has all eigenvalues equal to 1
        let a = Array2::<f64>::eye(4);
        let (values, vectors) = symmetric_eigen(&a).unwrap();
        for v in values.iter() {
            assert!(close(*v, 1.0));
        }
        // Eigenvectors should still be orthonormal
        let vtv = vectors.t().dot(&vectors);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((vtv[[i, j]] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_eigenvalues_sorted_descending() {
        let a = array![[5.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 4.0]];
        let (values, _) = symmetric_eigen(&a).unwrap();
        for i in 1..values.len() {
            assert!(
                values[i - 1] >= values[i],
                "not sorted: {:?}",
                values
            );
        }
    }

    #[test]
    fn test_negative_eigenvalues() {
        // Matrix with both positive and negative eigenvalues
        let a = array![[1.0, 2.0], [2.0, -2.0]];
        let (values, _) = symmetric_eigen(&a).unwrap();
        assert!(values[0] > 0.0);
        assert!(values[1] < 0.0);
        // Trace = sum of eigenvalues = -1
        assert!((values[0] + values[1] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_rejects_non_square() {
        let a = Array2::<f64>::zeros((3, 4));
        assert!(symmetric_eigen(&a).is_err());
    }

    #[test]
    fn test_rejects_empty() {
        let a = Array2::<f64>::zeros((0, 0));
        assert!(symmetric_eigen(&a).is_err());
    }
}
