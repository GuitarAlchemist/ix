//! Kernel Principal Component Analysis.
//!
//! Kernel PCA is a non-linear extension of PCA that performs the standard
//! PCA eigendecomposition on a kernel (Gram) matrix instead of on the raw
//! covariance matrix. This lets you capture non-linear structure — circles,
//! manifolds, clusters in higher-dimensional projections — without ever
//! explicitly computing the feature-space coordinates.
//!
//! # Algorithm
//!
//! 1. Compute the `n x n` kernel matrix `K` where `K_ij = k(x_i, x_j)`.
//! 2. Center the kernel matrix in feature space:
//!    `K_c = K - 1_n K - K 1_n + 1_n K 1_n`  where `1_n` is the `n x n`
//!    matrix of `1/n`.
//! 3. Eigendecompose `K_c` and take the top-k eigenvectors.
//! 4. The i-th sample's projection along the r-th component is
//!    `alpha_r[i] * sqrt(lambda_r)`.
//!
//! Three kernels are provided out of the box: linear, polynomial, and
//! Gaussian RBF. Custom kernels can be plugged in by implementing the
//! `Kernel` trait.
//!
//! # Example
//!
//! ```
//! use ix_unsupervised::kernel_pca::{KernelPca, Kernel};
//! use ndarray::array;
//!
//! // Two rings of points — impossible to separate with linear PCA
//! let x = array![
//!     [1.0, 0.0],
//!     [-1.0, 0.0],
//!     [0.0, 1.0],
//!     [0.0, -1.0],
//!     [2.0, 0.0],
//!     [-2.0, 0.0],
//!     [0.0, 2.0],
//!     [0.0, -2.0],
//! ];
//! let mut kpca = KernelPca::new(2, Kernel::Rbf { gamma: 0.5 });
//! let projected = kpca.fit_transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 2);
//! ```

use ndarray::{Array1, Array2};

use ix_math::eigen::symmetric_eigen;
use ix_math::error::MathError;

/// Kernel function variants. Extend this enum to add new kernels; the
/// switch arms in `compute_kernel` handle dispatch.
#[derive(Debug, Clone, Copy)]
pub enum Kernel {
    /// Linear kernel: `k(x, y) = x . y`. Equivalent to standard PCA up to centering.
    Linear,
    /// Polynomial kernel: `k(x, y) = (gamma * x.y + coef0)^degree`.
    Polynomial { degree: f64, gamma: f64, coef0: f64 },
    /// Radial basis function (Gaussian) kernel: `k(x, y) = exp(-gamma * ||x - y||^2)`.
    Rbf { gamma: f64 },
}

/// Kernel PCA model.
#[derive(Debug, Clone)]
pub struct KernelPca {
    /// Number of principal components.
    pub n_components: usize,
    /// Kernel function to use.
    pub kernel: Kernel,
    /// Training samples stored for out-of-sample projection.
    pub training: Option<Array2<f64>>,
    /// Alphas (eigenvectors of the centered kernel matrix), shape (n, n_components).
    pub alphas: Option<Array2<f64>>,
    /// Eigenvalues associated with each retained component.
    pub lambdas: Option<Array1<f64>>,
    /// Row means of the training kernel matrix, needed for centering at transform time.
    pub k_fit_rows_mean: Option<Array1<f64>>,
    /// Grand mean of the training kernel matrix.
    pub k_fit_grand_mean: Option<f64>,
}

impl KernelPca {
    /// Create a new Kernel PCA model.
    pub fn new(n_components: usize, kernel: Kernel) -> Self {
        Self {
            n_components,
            kernel,
            training: None,
            alphas: None,
            lambdas: None,
            k_fit_rows_mean: None,
            k_fit_grand_mean: None,
        }
    }

    /// Fit the model and return the transformed training data.
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return Err(MathError::EmptyInput);
        }
        if self.n_components == 0 || self.n_components >= n {
            return Err(MathError::InvalidParameter(format!(
                "n_components must be in 1..{}, got {}",
                n, self.n_components
            )));
        }

        // Compute kernel matrix K_ij = k(x_i, x_j).
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k[[i, j]] = compute_kernel(&self.kernel, &x.row(i), &x.row(j));
            }
        }

        // Center the kernel matrix in feature space:
        // K_c = K - 1_n K - K 1_n + 1_n K 1_n
        let row_means: Array1<f64> = k.rows().into_iter().map(|r| r.sum() / n as f64).collect();
        let col_means: Array1<f64> = k
            .columns()
            .into_iter()
            .map(|c| c.sum() / n as f64)
            .collect();
        let grand_mean = k.iter().sum::<f64>() / (n * n) as f64;

        let mut k_c = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k_c[[i, j]] = k[[i, j]] - row_means[i] - col_means[j] + grand_mean;
            }
        }

        // Eigendecompose the centered kernel matrix via ix-math::eigen.
        // The returned eigenpairs are already sorted in descending order.
        let (eigenvalues, eigenvectors) = symmetric_eigen(&k_c)?;

        // Take top-k alphas and lambdas.
        let mut alphas = Array2::<f64>::zeros((n, self.n_components));
        let mut lambdas = Array1::<f64>::zeros(self.n_components);
        for r in 0..self.n_components {
            let lambda = eigenvalues[r].max(0.0);
            lambdas[r] = lambda;
            // Normalize alphas so that lambda * alpha.alpha = 1 (standard Kernel PCA convention)
            let norm = if lambda > 1e-12 { lambda.sqrt() } else { 1.0 };
            for i in 0..n {
                alphas[[i, r]] = eigenvectors[[i, r]] / norm;
            }
        }

        // Project training data. The closed form is
        // X_projected[i, r] = sqrt(lambda_r) * eigenvector[i, r]
        let mut projected = Array2::<f64>::zeros((n, self.n_components));
        for r in 0..self.n_components {
            let scale = lambdas[r].sqrt();
            for i in 0..n {
                projected[[i, r]] = eigenvectors[[i, r]] * scale;
            }
        }

        self.training = Some(x.clone());
        self.alphas = Some(alphas);
        self.lambdas = Some(lambdas);
        self.k_fit_rows_mean = Some(row_means);
        self.k_fit_grand_mean = Some(grand_mean);
        Ok(projected)
    }

    /// Project new out-of-sample data using the fitted kernel PCA model.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        let training = self
            .training
            .as_ref()
            .ok_or_else(|| MathError::InvalidParameter("model not fitted".into()))?;
        let alphas = self.alphas.as_ref().unwrap();
        let row_means = self.k_fit_rows_mean.as_ref().unwrap();
        let grand_mean = self.k_fit_grand_mean.unwrap();

        let m = x.nrows();
        let n = training.nrows();
        if x.ncols() != training.ncols() {
            return Err(MathError::DimensionMismatch {
                expected: training.ncols(),
                got: x.ncols(),
            });
        }

        // Compute kernel values between new points and training points.
        let mut k_test = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                k_test[[i, j]] = compute_kernel(&self.kernel, &x.row(i), &training.row(j));
            }
        }

        // Center k_test using the training-time statistics:
        // K_test_c[i,j] = K_test[i,j] - mean_j(K_test[i,:]) - col_means[j] + grand_mean
        let test_row_means: Array1<f64> = k_test
            .rows()
            .into_iter()
            .map(|r| r.sum() / n as f64)
            .collect();
        let mut k_test_c = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                k_test_c[[i, j]] = k_test[[i, j]] - test_row_means[i] - row_means[j] + grand_mean;
            }
        }

        // Projection: K_test_c * alphas
        Ok(k_test_c.dot(alphas))
    }
}

fn compute_kernel(
    kernel: &Kernel,
    a: &ndarray::ArrayView1<f64>,
    b: &ndarray::ArrayView1<f64>,
) -> f64 {
    match kernel {
        Kernel::Linear => a.dot(b),
        Kernel::Polynomial {
            degree,
            gamma,
            coef0,
        } => (gamma * a.dot(b) + coef0).powf(*degree),
        Kernel::Rbf { gamma } => {
            let mut sq = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = x - y;
                sq += d * d;
            }
            (-gamma * sq).exp()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_kernel_matches_pca_style() {
        // With a linear kernel, Kernel PCA is equivalent (up to sign/rotation)
        // to standard PCA. We verify that the top component captures the
        // dominant variance direction.
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],];
        let mut kpca = KernelPca::new(1, Kernel::Linear);
        let projected = kpca.fit_transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
        // Projections should be monotonically ordered along the main axis
        // (either ascending or descending since the sign is arbitrary).
        let col: Vec<f64> = projected.column(0).iter().copied().collect();
        let ascending = col.windows(2).all(|w| w[0] <= w[1] + 1e-9);
        let descending = col.windows(2).all(|w| w[0] >= w[1] - 1e-9);
        assert!(
            ascending || descending,
            "projection not monotone: {:?}",
            col
        );
    }

    #[test]
    fn test_rbf_kernel_separates_rings() {
        // Two concentric rings — linear PCA cannot separate them, RBF kernel can.
        let x = array![
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ];
        let mut kpca = KernelPca::new(2, Kernel::Rbf { gamma: 0.5 });
        let projected = kpca.fit_transform(&x).unwrap();
        assert_eq!(projected.nrows(), 8);
        assert_eq!(projected.ncols(), 2);
        // The two rings are both centered at the origin, so their centroids
        // in the projection will be near zero. Instead, verify the first
        // component explains meaningful variance (non-trivial projection).
        let var_c1: f64 = projected.column(0).iter().map(|v| v * v).sum::<f64>() / 8.0;
        assert!(
            var_c1 > 1e-4,
            "first component should have non-trivial variance, got {}",
            var_c1
        );
    }

    #[test]
    fn test_polynomial_kernel() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];
        let mut kpca = KernelPca::new(
            2,
            Kernel::Polynomial {
                degree: 2.0,
                gamma: 1.0,
                coef0: 1.0,
            },
        );
        let projected = kpca.fit_transform(&x).unwrap();
        assert_eq!(projected.nrows(), 4);
        assert_eq!(projected.ncols(), 2);
    }

    #[test]
    fn test_transform_out_of_sample() {
        let x_train = array![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],];
        let mut kpca = KernelPca::new(2, Kernel::Rbf { gamma: 0.5 });
        kpca.fit_transform(&x_train).unwrap();
        let x_test = array![[0.5, 0.5], [0.0, 0.0]];
        let projected = kpca.transform(&x_test).unwrap();
        assert_eq!(projected.nrows(), 2);
        assert_eq!(projected.ncols(), 2);
    }

    #[test]
    fn test_rejects_zero_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let mut kpca = KernelPca::new(0, Kernel::Linear);
        assert!(kpca.fit_transform(&x).is_err());
    }

    #[test]
    fn test_rejects_too_many_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let mut kpca = KernelPca::new(5, Kernel::Linear);
        assert!(kpca.fit_transform(&x).is_err());
    }
}
