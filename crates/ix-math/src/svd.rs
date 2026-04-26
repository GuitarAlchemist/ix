//! Singular Value Decomposition (SVD) via one-sided Jacobi rotations.
//!
//! Computes A = U * S * V^T for an arbitrary real m-by-n matrix A.
//!
//! The one-sided Jacobi algorithm iteratively applies pairwise column
//! rotations to `V` (starting from the identity) until the columns of
//! `A * V` are mutually orthogonal. At convergence, the norms of those
//! columns are the singular values, the normalized columns form `U`, and
//! the accumulated rotations form `V^T`.
//!
//! This is a pure-Rust implementation with no LAPACK dependency, at the
//! cost of O(n^2) sweeps per iteration. Suitable for small-to-medium
//! matrices (up to a few hundred rows/columns); for larger matrices use
//! a bidiagonalization-based method or a LAPACK binding.
//!
//! # Example
//!
//! ```
//! use ix_math::svd::svd;
//! use ndarray::array;
//!
//! let a = array![[3.0, 0.0], [0.0, -2.0], [0.0, 0.0]];
//! let result = svd(&a).unwrap();
//! // Singular values are the absolute values of the entries, sorted desc.
//! assert!((result.singular_values[0] - 3.0).abs() < 1e-9);
//! assert!((result.singular_values[1] - 2.0).abs() < 1e-9);
//! ```

use ndarray::{s, Array1, Array2};

use crate::error::MathError;

/// Result of an SVD: `A = U * diag(s) * V^T`.
///
/// Singular values are returned in descending order. `u` has shape (m, k),
/// `v` has shape (n, k), and `singular_values` has length `k = min(m, n)`.
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors (columns of U), shape (m, k).
    pub u: Array2<f64>,
    /// Singular values in descending order, length k.
    pub singular_values: Array1<f64>,
    /// Right singular vectors (columns of V), shape (n, k).
    pub v: Array2<f64>,
}

impl SvdResult {
    /// Reconstruct the original matrix as `U * diag(s) * V^T`.
    pub fn reconstruct(&self) -> Array2<f64> {
        let m = self.u.nrows();
        let n = self.v.nrows();
        let k = self.singular_values.len();
        let mut result = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for r in 0..k {
                    acc += self.u[[i, r]] * self.singular_values[r] * self.v[[j, r]];
                }
                result[[i, j]] = acc;
            }
        }
        result
    }

    /// Rank estimate: count of singular values above `tol`.
    pub fn rank(&self, tol: f64) -> usize {
        self.singular_values.iter().filter(|&&s| s > tol).count()
    }

    /// Moore-Penrose pseudoinverse `V * diag(1/s) * U^T`, using `tol` to
    /// drop near-zero singular values.
    pub fn pseudo_inverse(&self, tol: f64) -> Array2<f64> {
        let m = self.u.nrows();
        let n = self.v.nrows();
        let k = self.singular_values.len();
        let mut result = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                let mut acc = 0.0;
                for r in 0..k {
                    let s = self.singular_values[r];
                    if s > tol {
                        acc += self.v[[i, r]] * (1.0 / s) * self.u[[j, r]];
                    }
                }
                result[[i, j]] = acc;
            }
        }
        result
    }
}

/// Compute the thin SVD of a real matrix via one-sided Jacobi rotations.
///
/// Returns `SvdResult { u, singular_values, v }` with singular values
/// sorted in descending order. Guaranteed to converge for any real matrix.
///
/// Uses default iteration limits: `max_sweeps = 50`, `tol = 1e-12`.
pub fn svd(a: &Array2<f64>) -> Result<SvdResult, MathError> {
    svd_with_opts(a, 50, 1e-12)
}

/// Compute SVD with explicit iteration limits.
pub fn svd_with_opts(a: &Array2<f64>, max_sweeps: usize, tol: f64) -> Result<SvdResult, MathError> {
    let m = a.nrows();
    let n = a.ncols();
    if m == 0 || n == 0 {
        return Err(MathError::EmptyInput);
    }

    // One-sided Jacobi works on the wider/equal dimension. Here we operate
    // on the columns of A directly: rotate pairs of columns until all pairs
    // are orthogonal. Accumulate rotations into V (initially the identity).
    let mut work = a.clone();
    let mut v = Array2::<f64>::eye(n);

    // Frobenius norm squared of A, used for scale-invariant convergence.
    // (||A||_F^2 = sum of squared singular values = sum of diagonal of A^T A.)
    let a_frob_sq: f64 = a.iter().map(|x| x * x).sum();
    let a_frob_sq = a_frob_sq.max(f64::MIN_POSITIVE);

    for _sweep in 0..max_sweeps {
        let mut off_diag_sum_sq = 0.0;
        for p in 0..(n.saturating_sub(1)) {
            for q in (p + 1)..n {
                // Compute inner products of columns p and q.
                let col_p = work.column(p).to_owned();
                let col_q = work.column(q).to_owned();
                let app = col_p.dot(&col_p);
                let aqq = col_q.dot(&col_q);
                let apq = col_p.dot(&col_q);

                off_diag_sum_sq += apq * apq;

                if apq.abs() < tol * (app * aqq).sqrt().max(f64::MIN_POSITIVE) {
                    continue;
                }

                // Compute Jacobi rotation angle that zeros the (p, q) inner
                // product in the 2x2 submatrix [[app, apq], [apq, aqq]].
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply rotation to columns p, q of `work`.
                for i in 0..m {
                    let wip = work[[i, p]];
                    let wiq = work[[i, q]];
                    work[[i, p]] = c * wip - s * wiq;
                    work[[i, q]] = s * wip + c * wiq;
                }
                // Apply the same rotation to V.
                for i in 0..n {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }

        // Scale-invariant stopping criterion: the off-diagonal Frobenius
        // mass should be a small fraction of the matrix's total Frobenius
        // mass. Using an absolute threshold (the old behavior) meant that
        // well-scaled matrices converged in 1-2 sweeps while any matrix
        // with entries much larger than `tol` stayed above the threshold
        // indefinitely.
        if off_diag_sum_sq < tol * tol * a_frob_sq {
            break;
        }
    }

    // Extract singular values and left singular vectors from `work`.
    let k = m.min(n);
    let mut singular_values = Array1::<f64>::zeros(k);
    let mut u = Array2::<f64>::zeros((m, k));

    // For each column of `work`, its norm is a singular value and the
    // normalized column is the corresponding left singular vector.
    let mut sigma_col: Vec<(f64, usize)> = Vec::with_capacity(n);
    for j in 0..n {
        let col = work.column(j);
        let norm = col.dot(&col).sqrt();
        sigma_col.push((norm, j));
    }
    // Sort descending by singular value.
    sigma_col.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take top-k columns.
    let mut v_sorted = Array2::<f64>::zeros((n, k));
    for (rank, (sig, orig)) in sigma_col.into_iter().take(k).enumerate() {
        singular_values[rank] = sig;
        if sig > tol {
            for i in 0..m {
                u[[i, rank]] = work[[i, orig]] / sig;
            }
        }
        for i in 0..n {
            v_sorted[[i, rank]] = v[[i, orig]];
        }
    }

    Ok(SvdResult {
        u,
        singular_values,
        v: v_sorted,
    })
}

/// Convenience: compute only the singular values of a matrix.
pub fn singular_values(a: &Array2<f64>) -> Result<Array1<f64>, MathError> {
    Ok(svd(a)?.singular_values)
}

/// Rank-k truncated SVD: keep only the top k singular components.
/// Useful for low-rank approximation and Latent Semantic Analysis.
pub fn truncated_svd(a: &Array2<f64>, k: usize) -> Result<SvdResult, MathError> {
    let mut full = svd(a)?;
    let k = k.min(full.singular_values.len());
    full.singular_values = full.singular_values.slice(s![..k]).to_owned();
    full.u = full.u.slice(s![.., ..k]).to_owned();
    full.v = full.v.slice(s![.., ..k]).to_owned();
    Ok(full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn frobenius(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let diff = a - b;
        diff.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn test_svd_diagonal_matrix() {
        let a = array![[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let res = svd(&a).unwrap();
        assert!((res.singular_values[0] - 3.0).abs() < 1e-9);
        assert!((res.singular_values[1] - 2.0).abs() < 1e-9);
        assert!((res.singular_values[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let res = svd(&a).unwrap();
        let recon = res.reconstruct();
        assert!(frobenius(&a, &recon) < 1e-9);
    }

    #[test]
    fn test_svd_singular_values_sorted_descending() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let res = svd(&a).unwrap();
        for i in 1..res.singular_values.len() {
            assert!(
                res.singular_values[i - 1] >= res.singular_values[i],
                "not sorted: {:?}",
                res.singular_values
            );
        }
    }

    #[test]
    fn test_svd_tall_matrix() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let res = svd(&a).unwrap();
        assert_eq!(res.singular_values.len(), 2);
        let recon = res.reconstruct();
        assert!(frobenius(&a, &recon) < 1e-8);
    }

    #[test]
    fn test_svd_rank_deficient() {
        // Rank-1 matrix: [1, 2; 2, 4]
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let res = svd(&a).unwrap();
        // First singular value should be sqrt(25) = 5
        assert!((res.singular_values[0] - 5.0).abs() < 1e-9);
        // Second should be effectively zero
        assert!(res.singular_values[1] < 1e-9);
        assert_eq!(res.rank(1e-6), 1);
    }

    #[test]
    fn test_pseudo_inverse_of_square_invertible() {
        let a = array![[2.0, 0.0], [0.0, 4.0]];
        let res = svd(&a).unwrap();
        let pinv = res.pseudo_inverse(1e-10);
        // pinv should be the regular inverse: [[0.5, 0], [0, 0.25]]
        assert!((pinv[[0, 0]] - 0.5).abs() < 1e-9);
        assert!((pinv[[1, 1]] - 0.25).abs() < 1e-9);
        assert!(pinv[[0, 1]].abs() < 1e-9);
    }

    #[test]
    fn test_truncated_svd_rank_1_approx() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let res = truncated_svd(&a, 1).unwrap();
        assert_eq!(res.singular_values.len(), 1);
        assert_eq!(res.u.ncols(), 1);
        assert_eq!(res.v.ncols(), 1);
        // Rank-1 approximation should still be close (matrix is rank-2
        // but highly correlated).
        let recon = res.reconstruct();
        let err = frobenius(&a, &recon) / frobenius(&a, &Array2::zeros(a.dim()));
        assert!(err < 0.10, "rank-1 relative error = {}", err);
    }

    #[test]
    fn test_singular_values_helper() {
        let a = array![[3.0, 0.0], [0.0, -2.0]];
        let svs = singular_values(&a).unwrap();
        assert!((svs[0] - 3.0).abs() < 1e-9);
        assert!((svs[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_svd_scale_invariance() {
        // Large-scale matrix: the absolute-tolerance convergence criterion
        // would have required many more sweeps to converge (or failed to
        // converge within the default 50). The relative criterion should
        // produce the same relative accuracy regardless of scale.
        let base = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let scale = 1e6_f64;
        let scaled = &base * scale;

        let res_small = svd(&base).unwrap();
        let res_large = svd(&scaled).unwrap();

        // Singular values should scale linearly with the input
        for i in 0..res_small.singular_values.len() {
            let ratio = res_large.singular_values[i] / res_small.singular_values[i];
            assert!(
                (ratio - scale).abs() / scale < 1e-9,
                "sv ratio {} differs from expected scale {}",
                ratio,
                scale
            );
        }

        // Both should reconstruct within machine precision relative to scale
        let recon_large = res_large.reconstruct();
        let rel_err: f64 = (&scaled - &recon_large)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
            / scale;
        assert!(
            rel_err < 1e-8,
            "relative reconstruction error on scaled matrix: {}",
            rel_err
        );
    }

    #[test]
    fn test_svd_empty_matrix_rejected() {
        let a = Array2::<f64>::zeros((0, 0));
        assert!(svd(&a).is_err());
    }
}
