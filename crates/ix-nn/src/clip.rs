//! Gradient clipping utilities for training stability.
//!
//! Prevents exploding gradients during backpropagation by bounding
//! gradient magnitudes. Two strategies are provided:
//!
//! - **Clip by value**: clamp each element to `[-max_val, max_val]`
//! - **Clip by norm**: scale the entire gradient so its L2 norm ≤ `max_norm`
//!
//! # Example
//!
//! ```
//! use ndarray::array;
//! use ix_nn::clip::{clip_grad_value, clip_grad_norm};
//!
//! let grad = array![3.0, -5.0, 1.0, 10.0];
//!
//! // Clip by value: each element in [-2, 2]
//! let clipped = clip_grad_value(&grad, 2.0);
//! assert!(clipped.iter().all(|&v| v >= -2.0 && v <= 2.0));
//!
//! // Clip by norm: L2 norm ≤ 1.0
//! let clipped = clip_grad_norm(&grad, 1.0);
//! let norm: f64 = clipped.iter().map(|v| v * v).sum::<f64>().sqrt();
//! assert!(norm <= 1.0 + 1e-10);
//! ```

use ndarray::{Array1, Array2};

/// Clip each gradient element to `[-max_val, max_val]`.
///
/// # Panics
/// Panics if `max_val` is negative.
pub fn clip_grad_value(grad: &Array1<f64>, max_val: f64) -> Array1<f64> {
    assert!(max_val >= 0.0, "max_val must be non-negative");
    grad.mapv(|v| v.clamp(-max_val, max_val))
}

/// Clip a 2D gradient matrix element-wise to `[-max_val, max_val]`.
pub fn clip_grad_value_2d(grad: &Array2<f64>, max_val: f64) -> Array2<f64> {
    assert!(max_val >= 0.0, "max_val must be non-negative");
    grad.mapv(|v| v.clamp(-max_val, max_val))
}

/// Scale the gradient so its L2 norm is at most `max_norm`.
///
/// If the gradient norm is already ≤ `max_norm`, it is returned unchanged.
///
/// # Panics
/// Panics if `max_norm` is negative.
pub fn clip_grad_norm(grad: &Array1<f64>, max_norm: f64) -> Array1<f64> {
    assert!(max_norm >= 0.0, "max_norm must be non-negative");
    let norm: f64 = grad.dot(grad).sqrt();
    if norm <= max_norm {
        grad.clone()
    } else {
        grad * (max_norm / norm)
    }
}

/// Scale a 2D gradient matrix so its Frobenius norm is at most `max_norm`.
pub fn clip_grad_norm_2d(grad: &Array2<f64>, max_norm: f64) -> Array2<f64> {
    assert!(max_norm >= 0.0, "max_norm must be non-negative");
    let norm: f64 = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= max_norm {
        grad.clone()
    } else {
        grad * (max_norm / norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_clip_value_clamps() {
        let grad = array![3.0, -5.0, 1.0, 10.0, -0.5];
        let clipped = clip_grad_value(&grad, 2.0);
        assert_eq!(clipped, array![2.0, -2.0, 1.0, 2.0, -0.5]);
    }

    #[test]
    fn test_clip_value_no_op_when_within() {
        let grad = array![0.5, -0.3, 0.1];
        let clipped = clip_grad_value(&grad, 1.0);
        assert_eq!(clipped, grad);
    }

    #[test]
    fn test_clip_value_2d() {
        let grad = array![[10.0, -10.0], [0.5, 3.0]];
        let clipped = clip_grad_value_2d(&grad, 1.0);
        assert_eq!(clipped, array![[1.0, -1.0], [0.5, 1.0]]);
    }

    #[test]
    fn test_clip_norm_scales_down() {
        let grad = array![3.0, 4.0]; // norm = 5
        let clipped = clip_grad_norm(&grad, 1.0);
        let norm: f64 = clipped.dot(&clipped).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
        // Direction preserved
        assert!((clipped[0] / clipped[1] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_clip_norm_no_op_when_within() {
        let grad = array![0.3, 0.4]; // norm = 0.5
        let clipped = clip_grad_norm(&grad, 1.0);
        for (a, b) in clipped.iter().zip(grad.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_clip_norm_2d() {
        let grad = array![[3.0, 0.0], [0.0, 4.0]]; // frobenius = 5
        let clipped = clip_grad_norm_2d(&grad, 2.5);
        let norm: f64 = clipped.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_clip_norm_zero_gradient() {
        let grad = array![0.0, 0.0, 0.0];
        let clipped = clip_grad_norm(&grad, 1.0);
        assert_eq!(clipped, grad);
    }

    #[test]
    #[should_panic(expected = "max_val must be non-negative")]
    fn test_clip_value_negative_panics() {
        let grad = array![1.0];
        clip_grad_value(&grad, -1.0);
    }

    #[test]
    #[should_panic(expected = "max_norm must be non-negative")]
    fn test_clip_norm_negative_panics() {
        let grad = array![1.0];
        clip_grad_norm(&grad, -1.0);
    }
}
