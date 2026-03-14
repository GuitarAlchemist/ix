//! Normalization layers: LayerNorm, RMSNorm.

use ndarray::{Array1, Array2};

/// Layer Normalization (Ba et al. 2016).
///
/// Normalizes each row (token) independently: `(x - mean) / sqrt(var + eps) * gamma + beta`.
pub struct LayerNorm {
    pub gamma: Array1<f64>,
    pub beta: Array1<f64>,
    pub eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm for features of size `d`.
    pub fn new(d: usize) -> Self {
        Self {
            gamma: Array1::ones(d),
            beta: Array1::zeros(d),
            eps: 1e-5,
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
        let out2 = rn.forward(&(x.clone() * 10.0));
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-8, "RMSNorm should be scale-invariant");
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
}
