//! Positional encodings: sinusoidal (Vaswani), rotary (RoPE), learned.
//!
//! # Examples
//!
//! ```
//! use ix_nn::positional::{sinusoidal_encoding, rope_rotate};
//! use ndarray::Array2;
//!
//! let pe = sinusoidal_encoding(10, 16);
//! assert_eq!(pe.shape(), &[10, 16]);
//!
//! // RoPE: rotate a (seq_len, d) query/key
//! let q = Array2::ones((4, 8));
//! let rotated = rope_rotate(&q, 10000.0);
//! assert_eq!(rotated.shape(), q.shape());
//! ```

use ndarray::Array2;

/// Sinusoidal positional encoding (Vaswani et al. 2017).
///
/// Returns `(max_len, d_model)` matrix. Even indices use sin, odd use cos.
/// `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`
/// `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
pub fn sinusoidal_encoding(max_len: usize, d_model: usize) -> Array2<f64> {
    Array2::from_shape_fn((max_len, d_model), |(pos, i)| {
        let angle = pos as f64 / 10000_f64.powf((i / 2 * 2) as f64 / d_model as f64);
        if i % 2 == 0 {
            angle.sin()
        } else {
            angle.cos()
        }
    })
}

/// Rotary Position Embedding (RoPE) — Su et al. 2021.
///
/// Rotates pairs of dimensions by position-dependent angles.
/// Input `x` has shape `(seq_len, d)` where `d` must be even.
/// `base` controls frequency scaling (typically 10000).
///
/// Returns rotated array with same shape.
pub fn rope_rotate(x: &Array2<f64>, base: f64) -> Array2<f64> {
    let seq_len = x.nrows();
    let d = x.ncols();
    assert!(d % 2 == 0, "RoPE requires even dimension");

    let mut result = x.clone();
    let half = d / 2;

    for pos in 0..seq_len {
        for i in 0..half {
            let theta = pos as f64 / base.powf(2.0 * i as f64 / d as f64);
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let x0 = x[[pos, 2 * i]];
            let x1 = x[[pos, 2 * i + 1]];

            result[[pos, 2 * i]] = x0 * cos_t - x1 * sin_t;
            result[[pos, 2 * i + 1]] = x0 * sin_t + x1 * cos_t;
        }
    }

    result
}

/// Learned positional embedding (lookup table).
///
/// Returns `(max_len, d_model)` matrix initialized with small random values.
/// In a real model, these would be trained via backprop.
pub fn learned_embedding(max_len: usize, d_model: usize, seed: u64) -> Array2<f64> {
    use rand::SeedableRng;
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let std = (1.0 / d_model as f64).sqrt();
    Array2::from_shape_fn((max_len, d_model), |_| rng.random_range(-std..std))
}

/// ALiBi (Attention with Linear Biases) — Press et al. 2022.
///
/// Returns a `(seq_len, seq_len)` bias matrix for a single head.
/// `slope` controls the decay rate (typically `2^{-8/n_heads * head_idx}`).
/// The bias is `slope * (j - i)` for `j <= i`, masked for future positions.
pub fn alibi_bias(seq_len: usize, slope: f64) -> Array2<f64> {
    Array2::from_shape_fn((seq_len, seq_len), |(i, j)| {
        if j > i {
            -1e9 // causal mask
        } else {
            slope * (j as f64 - i as f64) // negative bias for distant tokens
        }
    })
}

/// Compute ALiBi slopes for all heads.
///
/// For `n_heads` heads, slopes are geometric: `2^{-8/n_heads}, 2^{-16/n_heads}, ...`
pub fn alibi_slopes(n_heads: usize) -> Vec<f64> {
    (1..=n_heads)
        .map(|h| 2.0_f64.powf(-8.0 * h as f64 / n_heads as f64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_shape() {
        let pe = sinusoidal_encoding(100, 64);
        assert_eq!(pe.shape(), &[100, 64]);
    }

    #[test]
    fn test_sinusoidal_position_zero() {
        let pe = sinusoidal_encoding(10, 8);
        // pos=0: sin(0)=0 for even, cos(0)=1 for odd
        assert!(pe[[0, 0]].abs() < 1e-10); // sin(0)
        assert!((pe[[0, 1]] - 1.0).abs() < 1e-10); // cos(0)
    }

    #[test]
    fn test_sinusoidal_bounded() {
        let pe = sinusoidal_encoding(100, 32);
        for &v in pe.iter() {
            assert!(v >= -1.0 && v <= 1.0, "PE values must be in [-1, 1]");
        }
    }

    #[test]
    fn test_sinusoidal_different_positions() {
        let pe = sinusoidal_encoding(10, 16);
        // Different positions should give different encodings
        let row0 = pe.row(0).to_vec();
        let row5 = pe.row(5).to_vec();
        assert_ne!(row0, row5);
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation — should preserve vector magnitude per position
        let x = Array2::from_shape_fn((4, 6), |(i, j)| (i + j) as f64 * 0.3);
        let rotated = rope_rotate(&x, 10000.0);

        for pos in 0..4 {
            let orig_norm: f64 = x.row(pos).iter().map(|v| v * v).sum::<f64>().sqrt();
            let rot_norm: f64 = rotated.row(pos).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((orig_norm - rot_norm).abs() < 1e-10, "RoPE should preserve norm");
        }
    }

    #[test]
    fn test_rope_position_zero() {
        // At position 0, theta=0, so rotation is identity
        let x = Array2::from_shape_fn((1, 4), |(_, j)| (j + 1) as f64);
        let rotated = rope_rotate(&x, 10000.0);
        for j in 0..4 {
            assert!((x[[0, j]] - rotated[[0, j]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rope_even_dimension_required() {
        let x = Array2::ones((2, 4));
        let _ = rope_rotate(&x, 10000.0); // should not panic
    }

    #[test]
    #[should_panic(expected = "RoPE requires even dimension")]
    fn test_rope_odd_dimension_panics() {
        let x = Array2::ones((2, 3));
        let _ = rope_rotate(&x, 10000.0);
    }

    #[test]
    fn test_learned_embedding_shape() {
        let e = learned_embedding(50, 16, 42);
        assert_eq!(e.shape(), &[50, 16]);
    }

    #[test]
    fn test_learned_embedding_reproducible() {
        let e1 = learned_embedding(10, 8, 42);
        let e2 = learned_embedding(10, 8, 42);
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_learned_embedding_different_seeds() {
        let e1 = learned_embedding(10, 8, 42);
        let e2 = learned_embedding(10, 8, 99);
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_alibi_bias_shape() {
        let b = alibi_bias(5, 0.5);
        assert_eq!(b.shape(), &[5, 5]);
    }

    #[test]
    fn test_alibi_bias_causal() {
        let b = alibi_bias(4, 0.5);
        // Future positions should be masked
        assert!(b[[0, 1]] < -1e8);
        assert!(b[[1, 3]] < -1e8);
        // Past positions should have negative bias
        assert!(b[[3, 0]] < 0.0);
        // Diagonal should be 0
        assert_eq!(b[[2, 2]], 0.0);
    }

    #[test]
    fn test_alibi_bias_distance_scaling() {
        let slope = 0.25;
        let b = alibi_bias(5, slope);
        // b[4, 0] = slope * (0 - 4) = -1.0
        assert!((b[[4, 0]] - (-1.0)).abs() < 1e-10);
        // b[4, 3] = slope * (3 - 4) = -0.25
        assert!((b[[4, 3]] - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_alibi_slopes() {
        let slopes = alibi_slopes(4);
        assert_eq!(slopes.len(), 4);
        // Should be decreasing
        for i in 0..slopes.len() - 1 {
            assert!(slopes[i] > slopes[i + 1]);
        }
        // All positive
        for &s in &slopes {
            assert!(s > 0.0);
        }
    }
}
