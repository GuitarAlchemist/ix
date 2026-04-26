//! Non-negative Matrix Factorization (NMF).
//!
//! Given a non-negative matrix `V` of shape `(n_samples, n_features)`,
//! NMF finds non-negative matrices `W` of shape `(n_samples, n_components)`
//! and `H` of shape `(n_components, n_features)` such that `V ≈ W * H`.
//!
//! Unlike PCA (which allows negative loadings), NMF produces a parts-based
//! decomposition that is often more interpretable — useful for topic
//! modeling, image analysis, and any domain where negative values are
//! meaningless.
//!
//! # Algorithm
//!
//! Uses Lee & Seung's multiplicative update rules with the Frobenius norm
//! objective `||V - WH||_F^2`. At each iteration:
//!
//! ```text
//! H := H * (W^T V) / (W^T W H + eps)
//! W := W * (V H^T) / (W H H^T + eps)
//! ```
//!
//! All operations are element-wise. The updates preserve non-negativity
//! because W, H, and V are non-negative by construction. Convergence is
//! guaranteed (not necessarily to a global optimum, but to a local one).
//!
//! # Example
//!
//! ```
//! use ix_unsupervised::nmf::NonNegativeMatrixFactorization;
//! use ndarray::array;
//!
//! let v = array![
//!     [1.0, 2.0, 3.0],
//!     [2.0, 4.0, 6.0],
//!     [3.0, 6.0, 9.0],
//! ];
//! let mut nmf = NonNegativeMatrixFactorization::new(1).with_max_iter(200);
//! let w = nmf.fit_transform(&v).unwrap();
//! assert_eq!(w.ncols(), 1);
//! ```

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ix_math::error::MathError;

/// Non-negative Matrix Factorization model.
#[derive(Debug, Clone)]
pub struct NonNegativeMatrixFactorization {
    /// Number of components in the factorization.
    pub n_components: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the Frobenius reconstruction error.
    pub tol: f64,
    /// Small epsilon added to denominators to avoid division by zero.
    pub eps: f64,
    /// RNG seed for initialization.
    pub seed: u64,
    /// Component matrix `H` of shape `(n_components, n_features)`.
    pub components: Option<Array2<f64>>,
    /// Final reconstruction error after fit.
    pub reconstruction_err: Option<f64>,
    /// Number of iterations actually performed.
    pub n_iter: usize,
}

impl NonNegativeMatrixFactorization {
    /// Create an NMF model with the given number of components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            eps: 1e-10,
            seed: 42,
            components: None,
            reconstruction_err: None,
            n_iter: 0,
        }
    }

    /// Builder: set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Builder: set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Builder: set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Fit the model to `v` and return the loadings matrix `w` of shape
    /// `(n_samples, n_components)`. The factor matrix `h` is stored in
    /// `self.components`.
    pub fn fit_transform(&mut self, v: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        let n_samples = v.nrows();
        let n_features = v.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(MathError::EmptyInput);
        }
        if self.n_components == 0 || self.n_components > n_samples.min(n_features) {
            return Err(MathError::InvalidParameter(format!(
                "n_components must be in 1..={}, got {}",
                n_samples.min(n_features),
                self.n_components
            )));
        }
        if v.iter().any(|&x| x < 0.0) {
            return Err(MathError::InvalidParameter(
                "NMF requires non-negative input".into(),
            ));
        }

        // Initialize W and H with small positive random values scaled by
        // the mean of V so initial magnitudes are reasonable.
        let mean = v.mean().unwrap_or(1.0).abs().max(1e-3);
        let scale = (mean / self.n_components as f64).sqrt();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut w = Array2::<f64>::zeros((n_samples, self.n_components));
        for e in w.iter_mut() {
            *e = rng.random::<f64>() * scale + self.eps;
        }
        let mut h = Array2::<f64>::zeros((self.n_components, n_features));
        for e in h.iter_mut() {
            *e = rng.random::<f64>() * scale + self.eps;
        }

        let initial_err = frobenius_distance(v, &w.dot(&h));
        let mut prev_err = initial_err;
        let mut final_err = initial_err;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // H update: H = H * (W^T V) / (W^T W H + eps)
            let wt_v = w.t().dot(v);
            let wt_w_h = w.t().dot(&w).dot(&h);
            for ((i, j), h_ij) in h.indexed_iter_mut() {
                let denom = wt_w_h[[i, j]] + self.eps;
                *h_ij *= wt_v[[i, j]] / denom;
            }

            // W update: W = W * (V H^T) / (W H H^T + eps)
            let v_ht = v.dot(&h.t());
            let w_h_ht = w.dot(&h).dot(&h.t());
            for ((i, j), w_ij) in w.indexed_iter_mut() {
                let denom = w_h_ht[[i, j]] + self.eps;
                *w_ij *= v_ht[[i, j]] / denom;
            }

            if iter % 5 == 0 || iter == self.max_iter - 1 {
                let err = frobenius_distance(v, &w.dot(&h));
                final_err = err;
                if (prev_err - err).abs() < self.tol {
                    break;
                }
                prev_err = err;
            }
        }

        self.components = Some(h);
        self.reconstruction_err = Some(final_err);
        self.n_iter = n_iter;
        Ok(w)
    }

    /// Project new samples onto the learned component basis, keeping
    /// `H` fixed and solving for `W` via a few multiplicative updates.
    pub fn transform(&self, v: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        let h = self
            .components
            .as_ref()
            .ok_or_else(|| MathError::InvalidParameter("model not fitted".into()))?;
        let n_samples = v.nrows();

        if v.ncols() != h.ncols() {
            return Err(MathError::DimensionMismatch {
                expected: h.ncols(),
                got: v.ncols(),
            });
        }
        if v.iter().any(|&x| x < 0.0) {
            return Err(MathError::InvalidParameter(
                "NMF requires non-negative input".into(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mean = v.mean().unwrap_or(1.0).abs().max(1e-3);
        let scale = (mean / self.n_components as f64).sqrt();
        let mut w = Array2::<f64>::zeros((n_samples, self.n_components));
        for e in w.iter_mut() {
            *e = rng.random::<f64>() * scale + self.eps;
        }

        let h_ht = h.dot(&h.t());
        let v_ht = v.dot(&h.t());

        for _ in 0..50 {
            let w_h_ht = w.dot(&h_ht);
            for ((i, j), w_ij) in w.indexed_iter_mut() {
                let denom = w_h_ht[[i, j]] + self.eps;
                *w_ij *= v_ht[[i, j]] / denom;
            }
        }

        Ok(w)
    }

    /// Reconstruct `V` from the stored components and a provided loading
    /// matrix `w`.
    pub fn reconstruct(&self, w: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        let h = self
            .components
            .as_ref()
            .ok_or_else(|| MathError::InvalidParameter("model not fitted".into()))?;
        if w.ncols() != h.nrows() {
            return Err(MathError::DimensionMismatch {
                expected: h.nrows(),
                got: w.ncols(),
            });
        }
        Ok(w.dot(h))
    }
}

fn frobenius_distance(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let diff = a - b;
    diff.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Utility: total sum of entries, useful for testing reconstruction fidelity.
#[cfg(test)]
fn total(a: &Array2<f64>) -> f64 {
    a.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_nmf_rank_one_matrix() {
        // Outer product of two non-negative vectors is rank-1
        let v = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0],];
        let mut nmf = NonNegativeMatrixFactorization::new(1).with_max_iter(500);
        let w = nmf.fit_transform(&v).unwrap();
        assert_eq!(w.nrows(), 3);
        assert_eq!(w.ncols(), 1);
        let reconstructed = nmf.reconstruct(&w).unwrap();
        let err = frobenius_distance(&v, &reconstructed);
        assert!(err < 0.1, "reconstruction error too high: {}", err);
    }

    #[test]
    fn test_nmf_non_negative_output() {
        let v = array![[1.0, 0.0, 2.0], [0.0, 3.0, 1.0], [2.0, 1.0, 0.0],];
        let mut nmf = NonNegativeMatrixFactorization::new(2).with_max_iter(200);
        let w = nmf.fit_transform(&v).unwrap();
        // All entries should be non-negative
        assert!(w.iter().all(|&x| x >= 0.0));
        assert!(nmf.components.as_ref().unwrap().iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_nmf_rejects_negative_input() {
        let v = array![[1.0, -2.0], [3.0, 4.0]];
        let mut nmf = NonNegativeMatrixFactorization::new(1);
        assert!(nmf.fit_transform(&v).is_err());
    }

    #[test]
    fn test_nmf_rejects_zero_components() {
        let v = array![[1.0, 2.0], [3.0, 4.0]];
        let mut nmf = NonNegativeMatrixFactorization::new(0);
        assert!(nmf.fit_transform(&v).is_err());
    }

    #[test]
    fn test_nmf_rejects_too_many_components() {
        let v = array![[1.0, 2.0], [3.0, 4.0]];
        let mut nmf = NonNegativeMatrixFactorization::new(5);
        assert!(nmf.fit_transform(&v).is_err());
    }

    #[test]
    fn test_nmf_transform_matches_shape() {
        let v_train = array![
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 1.0, 1.0],
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 2.0],
        ];
        let mut nmf = NonNegativeMatrixFactorization::new(2).with_max_iter(200);
        nmf.fit_transform(&v_train).unwrap();

        let v_new = array![[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 2.0, 0.0]];
        let w_new = nmf.transform(&v_new).unwrap();
        assert_eq!(w_new.nrows(), 2);
        assert_eq!(w_new.ncols(), 2);
        assert!(w_new.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_nmf_error_decreases() {
        let v = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 1.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
        ];
        let mut nmf = NonNegativeMatrixFactorization::new(2).with_max_iter(300);
        let _w = nmf.fit_transform(&v).unwrap();
        let err = nmf.reconstruction_err.unwrap();
        let total_mag = total(&v);
        assert!(
            err < total_mag,
            "err {} should be < matrix magnitude {}",
            err,
            total_mag
        );
    }
}
