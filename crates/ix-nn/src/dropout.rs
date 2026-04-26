//! Dropout regularization layer.
//!
//! During training, randomly zeroes elements with probability `p` and
//! scales surviving elements by `1/(1-p)` (inverted dropout).
//! During inference, acts as identity.

use ndarray::{Array2, Array3};
use rand::Rng;
use rand::SeedableRng;

/// Dropout layer with inverted scaling.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use ix_nn::dropout::Dropout;
///
/// let mut drop = Dropout::new(0.1, 42);
/// let x = Array2::ones((3, 4));
/// let out = drop.forward_train_2d(&x);
/// assert_eq!(out.shape(), &[3, 4]);
/// ```
pub struct Dropout {
    /// Probability of dropping each element (0.0 = no dropout).
    pub p: f64,
    rng: rand::rngs::StdRng,
    /// Binary mask from last forward pass (for backward).
    mask_2d: Option<Array2<f64>>,
    mask_3d: Option<Array3<f64>>,
}

impl Dropout {
    /// Create a new Dropout layer.
    ///
    /// `p` is the drop probability (0.0-1.0). `seed` for reproducibility.
    pub fn new(p: f64, seed: u64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            mask_2d: None,
            mask_3d: None,
        }
    }

    /// Forward pass during training for 2D input.
    ///
    /// Drops elements with probability `p`, scales survivors by `1/(1-p)`.
    pub fn forward_train_2d(&mut self, x: &Array2<f64>) -> Array2<f64> {
        if self.p == 0.0 {
            return x.clone();
        }
        let scale = 1.0 / (1.0 - self.p);
        let mask = Array2::from_shape_fn(x.raw_dim(), |_| {
            if self.rng.random::<f64>() >= self.p {
                scale
            } else {
                0.0
            }
        });
        let result = x * &mask;
        self.mask_2d = Some(mask);
        result
    }

    /// Forward pass during training for 3D input (batch, seq, features).
    ///
    /// Drops elements with probability `p`, scales survivors by `1/(1-p)`.
    pub fn forward_train_3d(&mut self, x: &Array3<f64>) -> Array3<f64> {
        if self.p == 0.0 {
            return x.clone();
        }
        let scale = 1.0 / (1.0 - self.p);
        let mask = Array3::from_shape_fn(x.raw_dim(), |_| {
            if self.rng.random::<f64>() >= self.p {
                scale
            } else {
                0.0
            }
        });
        let result = x * &mask;
        self.mask_3d = Some(mask);
        result
    }

    /// Backward pass for 2D: apply the same mask used during forward.
    pub fn backward_2d(&self, grad_output: &Array2<f64>) -> Array2<f64> {
        match &self.mask_2d {
            Some(mask) => grad_output * mask,
            None => grad_output.clone(),
        }
    }

    /// Backward pass for 3D: apply the same mask used during forward.
    pub fn backward_3d(&self, grad_output: &Array3<f64>) -> Array3<f64> {
        match &self.mask_3d {
            Some(mask) => grad_output * mask,
            None => grad_output.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_dropout_zero_rate() {
        let mut drop = Dropout::new(0.0, 42);
        let x = Array2::ones((3, 4));
        let out = drop.forward_train_2d(&x);
        assert_eq!(out, x);
    }

    #[test]
    fn test_dropout_shape_2d() {
        let mut drop = Dropout::new(0.5, 42);
        let x = Array2::ones((4, 8));
        let out = drop.forward_train_2d(&x);
        assert_eq!(out.shape(), &[4, 8]);
    }

    #[test]
    fn test_dropout_shape_3d() {
        let mut drop = Dropout::new(0.3, 42);
        let x = Array3::ones((2, 4, 8));
        let out = drop.forward_train_3d(&x);
        assert_eq!(out.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_dropout_zeros_some_elements() {
        let mut drop = Dropout::new(0.5, 42);
        let x = Array2::ones((100, 100));
        let out = drop.forward_train_2d(&x);
        let n_zeros = out.iter().filter(|&&v| v == 0.0).count();
        // Expect roughly 50% zeros, allow wide margin
        assert!(n_zeros > 3000, "expected many zeros, got {n_zeros}");
        assert!(
            n_zeros < 7000,
            "expected some non-zeros, got {n_zeros} zeros"
        );
    }

    #[test]
    fn test_dropout_inverted_scaling() {
        let mut drop = Dropout::new(0.5, 42);
        let x = Array2::ones((100, 100));
        let out = drop.forward_train_2d(&x);
        // Non-zero elements should be scaled by 1/(1-0.5) = 2.0
        for &v in out.iter() {
            assert!(v == 0.0 || (v - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dropout_preserves_expectation() {
        let mut drop = Dropout::new(0.3, 42);
        let x = Array2::ones((1000, 100));
        let out = drop.forward_train_2d(&x);
        let mean = out.mean().unwrap();
        // E[out] ≈ 1.0 (inverted dropout preserves expectation)
        assert!((mean - 1.0).abs() < 0.1, "mean should be ~1.0, got {mean}");
    }

    #[test]
    fn test_dropout_backward_matches_mask() {
        let mut drop = Dropout::new(0.5, 42);
        let x = Array2::ones((4, 4));
        let out = drop.forward_train_2d(&x);
        let grad = Array2::ones((4, 4));
        let grad_in = drop.backward_2d(&grad);
        // Backward should have same sparsity pattern as forward
        for (o, g) in out.iter().zip(grad_in.iter()) {
            if *o == 0.0 {
                assert_eq!(*g, 0.0, "gradient should be zero where output was zero");
            } else {
                assert!(
                    *g > 0.0,
                    "gradient should be non-zero where output was non-zero"
                );
            }
        }
    }

    #[test]
    fn test_dropout_backward_3d() {
        let mut drop = Dropout::new(0.3, 42);
        let x = Array3::ones((2, 4, 8));
        let _out = drop.forward_train_3d(&x);
        let grad = Array3::ones((2, 4, 8));
        let grad_in = drop.backward_3d(&grad);
        assert_eq!(grad_in.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_dropout_full_rate() {
        let mut drop = Dropout::new(1.0, 42);
        let x = Array2::ones((4, 4));
        let out = drop.forward_train_2d(&x);
        // All elements should be zero
        for &v in out.iter() {
            assert_eq!(v, 0.0);
        }
    }
}
