//! Normalization layers: LayerNorm, RMSNorm.

use ndarray::{Array1, Array2, Axis};

/// Layer Normalization (Ba et al. 2016).
///
/// Normalizes each row (token) independently: `(x - mean) / sqrt(var + eps) * gamma + beta`.
pub struct LayerNorm {
    pub gamma: Array1<f64>,
    pub beta: Array1<f64>,
    pub eps: f64,
    /// Cached input from `forward_cache` for backward pass.
    input_cache: Option<Array2<f64>>,
    /// Cached normalized values (xhat) from `forward_cache`.
    normalized_cache: Option<Array2<f64>>,
    /// Cached per-row standard deviations from `forward_cache`.
    std_cache: Option<Array1<f64>>,
}

impl LayerNorm {
    /// Create a new LayerNorm for features of size `d`.
    pub fn new(d: usize) -> Self {
        Self {
            gamma: Array1::ones(d),
            beta: Array1::zeros(d),
            eps: 1e-5,
            input_cache: None,
            normalized_cache: None,
            std_cache: None,
        }
    }

    /// Forward pass: normalize each row.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());
        for (i, row) in x.rows().into_iter().enumerate() {
            let mean = row.mean().unwrap();
            let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            let std_inv = 1.0 / (var + self.eps).sqrt();
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = (val - mean) * std_inv * self.gamma[j] + self.beta[j];
            }
        }
        result
    }

    /// Forward pass that caches activations for backward.
    ///
    /// Same output as `forward`, but stores input, normalized values, and
    /// per-row standard deviations so that `backward` can compute gradients.
    pub fn forward_cache(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let nrows = x.nrows();
        let ncols = x.ncols();
        let mut normalized = Array2::zeros(x.raw_dim());
        let mut result = Array2::zeros(x.raw_dim());
        let mut stds = Array1::zeros(nrows);

        for (i, row) in x.rows().into_iter().enumerate() {
            let mean = row.mean().unwrap();
            let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            let std = (var + self.eps).sqrt();
            stds[i] = std;
            for j in 0..ncols {
                let xhat = (row[j] - mean) / std;
                normalized[[i, j]] = xhat;
                result[[i, j]] = xhat * self.gamma[j] + self.beta[j];
            }
        }

        self.input_cache = Some(x.clone());
        self.normalized_cache = Some(normalized);
        self.std_cache = Some(stds);
        result
    }

    /// Backward pass: computes gradient w.r.t. input and updates gamma/beta.
    ///
    /// `grad_output` has the same shape as the forward output `(rows, features)`.
    /// Returns gradient w.r.t. the input `x`.
    pub fn backward(&mut self, grad_output: &Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let xhat = self.normalized_cache.as_ref().expect("forward_cache() not called");
        let stds = self.std_cache.as_ref().expect("forward_cache() not called");

        let n = grad_output.ncols() as f64;

        // grad_gamma = sum over rows of (grad_output * xhat)
        let grad_gamma = (grad_output * xhat).sum_axis(Axis(0));
        // grad_beta = sum over rows of grad_output
        let grad_beta = grad_output.sum_axis(Axis(0));

        // grad_xhat = grad_output * gamma (broadcast gamma across rows)
        let grad_xhat = grad_output * &self.gamma;

        // grad_input for each row:
        // (1/N) * (1/std) * (N * grad_xhat - sum(grad_xhat) - xhat * sum(grad_xhat * xhat))
        let mut grad_input = Array2::zeros(grad_output.raw_dim());
        for i in 0..grad_output.nrows() {
            let gx_row = grad_xhat.row(i);
            let xh_row = xhat.row(i);
            let sum_gx: f64 = gx_row.sum();
            let sum_gx_xh: f64 = (&gx_row * &xh_row).sum();
            let inv_std = 1.0 / stds[i];
            for j in 0..grad_output.ncols() {
                grad_input[[i, j]] = inv_std / n
                    * (n * gx_row[j] - sum_gx - xh_row[j] * sum_gx_xh);
            }
        }

        // Update parameters
        self.gamma = &self.gamma - &(learning_rate * &grad_gamma);
        self.beta = &self.beta - &(learning_rate * &grad_beta);

        grad_input
    }
}

/// RMSNorm (Zhang & Sennrich 2019).
///
/// Simplified layer norm without centering: `x / sqrt(mean(x²) + eps) * gamma`.
/// Used in LLaMA, Gemma, and other modern architectures.
pub struct RMSNorm {
    pub gamma: Array1<f64>,
    pub eps: f64,
}

impl RMSNorm {
    pub fn new(d: usize) -> Self {
        Self {
            gamma: Array1::ones(d),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());
        for (i, row) in x.rows().into_iter().enumerate() {
            let rms = (row.mapv(|v| v * v).mean().unwrap() + self.eps).sqrt();
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = val / rms * self.gamma[j];
            }
        }
        result
    }
}

/// Batch Normalization (Ioffe & Szegedy 2015).
///
/// Normalizes each feature across the batch: `(x - batch_mean) / sqrt(batch_var + eps) * gamma + beta`.
/// Maintains running statistics for inference mode.
///
/// Unlike LayerNorm which normalizes across features per sample, BatchNorm
/// normalizes across the batch per feature. This makes it sensitive to batch
/// size but very effective for stabilizing training in feed-forward networks.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use ix_nn::norm::BatchNorm;
///
/// let mut bn = BatchNorm::new(3);
///
/// // Training mode: normalizes using batch statistics
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let out = bn.forward_train(&x);
///
/// // Each column (feature) should have mean ≈ 0
/// for col in 0..3 {
///     let col_mean: f64 = (0..3).map(|r| out[[r, col]]).sum::<f64>() / 3.0;
///     assert!(col_mean.abs() < 1e-10, "column {} mean should be ~0", col);
/// }
///
/// // Inference mode: uses running statistics
/// let single = array![[2.0, 3.0, 4.0]];
/// let out_inf = bn.forward_inference(&single);
/// assert_eq!(out_inf.shape(), &[1, 3]);
/// ```
pub struct BatchNorm {
    /// Learnable scale parameter per feature.
    pub gamma: Array1<f64>,
    /// Learnable shift parameter per feature.
    pub beta: Array1<f64>,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Momentum for running statistics (default 0.1).
    pub momentum: f64,
    /// Running mean for inference.
    pub running_mean: Array1<f64>,
    /// Running variance for inference.
    pub running_var: Array1<f64>,
    /// Cached normalized values for backward pass.
    normalized_cache: Option<Array2<f64>>,
    /// Cached batch standard deviations for backward pass.
    std_cache: Option<Array1<f64>>,
}

impl BatchNorm {
    /// Create a new BatchNorm layer for `d` features.
    pub fn new(d: usize) -> Self {
        Self {
            gamma: Array1::ones(d),
            beta: Array1::zeros(d),
            eps: 1e-5,
            momentum: 0.1,
            running_mean: Array1::zeros(d),
            running_var: Array1::ones(d),
            normalized_cache: None,
            std_cache: None,
        }
    }

    /// Forward pass in training mode.
    ///
    /// Uses batch statistics and updates running statistics.
    pub fn forward_train(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let d = x.ncols();

        // Compute batch mean and variance per feature (column)
        let batch_mean = x.mean_axis(Axis(0)).unwrap();
        let batch_var = {
            let centered = x - &batch_mean;
            centered.mapv(|v| v * v).mean_axis(Axis(0)).unwrap()
        };

        // Update running statistics
        let m = self.momentum;
        for j in 0..d {
            self.running_mean[j] = (1.0 - m) * self.running_mean[j] + m * batch_mean[j];
            self.running_var[j] = (1.0 - m) * self.running_var[j] + m * batch_var[j];
        }

        // Normalize
        let std: Array1<f64> = batch_var.mapv(|v| (v + self.eps).sqrt());
        let mut normalized = Array2::zeros(x.raw_dim());
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            for j in 0..d {
                let xhat = (x[[i, j]] - batch_mean[j]) / std[j];
                normalized[[i, j]] = xhat;
                result[[i, j]] = xhat * self.gamma[j] + self.beta[j];
            }
        }

        self.normalized_cache = Some(normalized);
        self.std_cache = Some(std);
        result
    }

    /// Forward pass in inference mode.
    ///
    /// Uses running statistics (no batch dependency).
    pub fn forward_inference(&self, x: &Array2<f64>) -> Array2<f64> {
        let d = x.ncols();
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            for j in 0..d {
                let xhat = (x[[i, j]] - self.running_mean[j])
                    / (self.running_var[j] + self.eps).sqrt();
                result[[i, j]] = xhat * self.gamma[j] + self.beta[j];
            }
        }

        result
    }

    /// Backward pass: computes gradient w.r.t. input and updates gamma/beta.
    ///
    /// Returns gradient w.r.t. the input `x`.
    pub fn backward(&mut self, grad_output: &Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let xhat = self.normalized_cache.as_ref().expect("forward_train() not called");
        let std = self.std_cache.as_ref().expect("forward_train() not called");
        let n = grad_output.nrows() as f64;
        let d = grad_output.ncols();

        // Gradients for gamma and beta
        let grad_gamma = (grad_output * xhat).sum_axis(Axis(0));
        let grad_beta = grad_output.sum_axis(Axis(0));

        // Gradient w.r.t. normalized input
        let grad_xhat = grad_output * &self.gamma;

        // Gradient w.r.t. input (BatchNorm gradient formula)
        let mut grad_input = Array2::zeros(grad_output.raw_dim());
        for j in 0..d {
            let inv_std = 1.0 / std[j];
            let sum_gx: f64 = (0..grad_output.nrows()).map(|i| grad_xhat[[i, j]]).sum();
            let sum_gx_xh: f64 = (0..grad_output.nrows())
                .map(|i| grad_xhat[[i, j]] * xhat[[i, j]])
                .sum();

            for i in 0..grad_output.nrows() {
                grad_input[[i, j]] = inv_std / n
                    * (n * grad_xhat[[i, j]] - sum_gx - xhat[[i, j]] * sum_gx_xh);
            }
        }

        // Update parameters
        self.gamma = &self.gamma - &(learning_rate * &grad_gamma);
        self.beta = &self.beta - &(learning_rate * &grad_beta);

        grad_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_layer_norm_zero_mean() {
        let ln = LayerNorm::new(4);
        let x = array![[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]];
        let out = ln.forward(&x);
        // Each row should have mean ≈ 0 (with gamma=1, beta=0)
        for row in out.rows() {
            assert!(row.mean().unwrap().abs() < 1e-10, "normalized mean should be ~0");
        }
    }

    #[test]
    fn test_layer_norm_unit_variance() {
        let ln = LayerNorm::new(4);
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let out = ln.forward(&x);
        let var = out.row(0).mapv(|v| v * v).mean().unwrap();
        // Variance should be ≈ 1 (population variance of normalized data)
        assert!((var - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_layer_norm_with_gamma_beta() {
        let mut ln = LayerNorm::new(2);
        ln.gamma = array![2.0, 2.0];
        ln.beta = array![1.0, 1.0];
        let x = array![[0.0, 0.0]];
        let out = ln.forward(&x);
        // All zeros normalized → all zeros × 2 + 1 = all ones
        for &v in out.iter() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_layer_norm_shape() {
        let ln = LayerNorm::new(8);
        let x = Array2::ones((3, 8));
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_layer_norm_constant_input() {
        // Constant input: all values equal → after norm, should be 0 (var → eps)
        let ln = LayerNorm::new(4);
        let x = array![[5.0, 5.0, 5.0, 5.0]];
        let out = ln.forward(&x);
        for &v in out.iter() {
            assert!(v.abs() < 1e-3, "constant input should normalize near 0");
        }
    }

    #[test]
    fn test_rms_norm_shape() {
        let rn = RMSNorm::new(6);
        let x = Array2::ones((4, 6));
        let out = rn.forward(&x);
        assert_eq!(out.shape(), &[4, 6]);
    }

    #[test]
    fn test_rms_norm_positive_output() {
        let rn = RMSNorm::new(3);
        let x = array![[1.0, 2.0, 3.0]];
        let out = rn.forward(&x);
        // All inputs positive, gamma=1 → all outputs positive
        for &v in out.iter() {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_rms_norm_scale_invariance() {
        // RMSNorm(alpha * x) ≈ RMSNorm(x) for scalar alpha
        let rn = RMSNorm::new(4);
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let out1 = rn.forward(&x);
        let scaled = &x * 10.0;
        let out2 = rn.forward(&scaled);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "RMSNorm should be scale-invariant: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rms_norm_unit_rms() {
        let rn = RMSNorm::new(4);
        let x = array![[3.0, 4.0, 5.0, 6.0]];
        let out = rn.forward(&x);
        let rms = (out.row(0).mapv(|v| v * v).mean().unwrap()).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "output RMS should be ~1");
    }

    // --- BatchNorm tests ---

    #[test]
    fn test_batch_norm_zero_mean_columns() {
        let mut bn = BatchNorm::new(3);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let out = bn.forward_train(&x);
        // Each column should have mean ≈ 0
        for j in 0..3 {
            let col_mean: f64 = (0..3).map(|i| out[[i, j]]).sum::<f64>() / 3.0;
            assert!(col_mean.abs() < 1e-10, "col {} mean should be ~0, got {}", j, col_mean);
        }
    }

    #[test]
    fn test_batch_norm_unit_variance_columns() {
        let mut bn = BatchNorm::new(3);
        let x = array![[1.0, 10.0, 100.0], [4.0, 40.0, 400.0], [7.0, 70.0, 700.0]];
        let out = bn.forward_train(&x);
        // Each column should have variance ≈ 1
        for j in 0..3 {
            let col_var: f64 = (0..3).map(|i| out[[i, j]].powi(2)).sum::<f64>() / 3.0;
            assert!((col_var - 1.0).abs() < 0.01, "col {} var should be ~1, got {}", j, col_var);
        }
    }

    #[test]
    fn test_batch_norm_running_stats_update() {
        let mut bn = BatchNorm::new(2);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let _out = bn.forward_train(&x);
        // Running mean should no longer be all zeros
        assert!(bn.running_mean.iter().any(|&v| v.abs() > 1e-10), "running mean should be updated");
    }

    #[test]
    fn test_batch_norm_inference_mode() {
        let mut bn = BatchNorm::new(2);
        // Train to build running stats
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let _out = bn.forward_train(&x);

        // Inference on a single sample should work
        let single = array![[2.0, 3.0]];
        let out_inf = bn.forward_inference(&single);
        assert_eq!(out_inf.shape(), &[1, 2]);
        // Should produce finite values
        assert!(out_inf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_batch_norm_backward_shape() {
        let mut bn = BatchNorm::new(4);
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let _out = bn.forward_train(&x);
        let grad_out = Array2::ones((2, 4));
        let grad_in = bn.backward(&grad_out, 0.01);
        assert_eq!(grad_in.shape(), &[2, 4]);
    }

    #[test]
    fn test_batch_norm_backward_updates_params() {
        let mut bn = BatchNorm::new(3);
        let gamma_before = bn.gamma.clone();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let _out = bn.forward_train(&x);
        let grad_out = array![[1.0, 0.5, -0.5], [-0.5, 1.0, 0.0], [0.2, -0.3, 0.8]];
        let _grad_in = bn.backward(&grad_out, 0.1);
        assert!(
            (&bn.gamma - &gamma_before).mapv(|v| v.abs()).sum() > 1e-10,
            "gamma should be updated after backward"
        );
    }

    // --- LayerNorm backward pass tests ---

    #[test]
    fn test_layer_norm_forward_cache_matches_forward() {
        let ln_ref = LayerNorm::new(4);
        let mut ln_cache = LayerNorm::new(4);
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let out1 = ln_ref.forward(&x);
        let out2 = ln_cache.forward_cache(&x);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-12, "forward_cache should match forward");
        }
    }

    #[test]
    fn test_layer_norm_backward_shape() {
        let mut ln = LayerNorm::new(4);
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let _out = ln.forward_cache(&x);
        let grad_out = Array2::ones((2, 4));
        let grad_in = ln.backward(&grad_out, 0.01);
        assert_eq!(grad_in.shape(), &[2, 4]);
    }

    #[test]
    fn test_layer_norm_backward_weights_update() {
        let mut ln = LayerNorm::new(4);
        let gamma_before = ln.gamma.clone();
        let beta_before = ln.beta.clone();
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let _out = ln.forward_cache(&x);
        let grad_out = array![[1.0, 0.5, -0.5, -1.0]];
        let _grad_in = ln.backward(&grad_out, 0.1);
        // Gamma and beta should have changed
        assert!(
            (&ln.gamma - &gamma_before).mapv(|v| v.abs()).sum() > 1e-10,
            "gamma should be updated"
        );
        assert!(
            (&ln.beta - &beta_before).mapv(|v| v.abs()).sum() > 1e-10,
            "beta should be updated"
        );
    }

    #[test]
    fn test_layer_norm_backward_numerical_gradient() {
        // Numerical gradient check for LayerNorm backward
        let eps = 1e-5;
        let x = array![[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]];
        let grad_out = array![[1.0, 0.5, -0.3, 0.8], [0.2, -0.4, 0.6, -0.1]];

        // Analytical gradient
        let mut ln = LayerNorm::new(4);
        let _out = ln.forward_cache(&x);
        let analytical = ln.backward(&grad_out, 0.0); // lr=0 so weights don't change

        // Numerical gradient: perturb each input element
        let ln_ref = LayerNorm::new(4);
        let mut numerical = Array2::zeros(x.raw_dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[i, j]] += eps;
                x_minus[[i, j]] -= eps;
                let out_plus = ln_ref.forward(&x_plus);
                let out_minus = ln_ref.forward(&x_minus);
                // Loss = sum(grad_out * output), so dL/dx[i,j] = sum(grad_out * d_output/dx[i,j])
                let diff = &out_plus - &out_minus;
                numerical[[i, j]] = (&grad_out * &diff).sum() / (2.0 * eps);
            }
        }

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let a = analytical[[i, j]];
                let n = numerical[[i, j]];
                let denom = a.abs().max(n.abs()).max(1e-8);
                let rel_err = (a - n).abs() / denom;
                assert!(
                    rel_err < 1e-4,
                    "LayerNorm grad mismatch at [{i},{j}]: analytical={a}, numerical={n}, rel_err={rel_err}"
                );
            }
        }
    }
}
