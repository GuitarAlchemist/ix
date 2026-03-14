//! Attention mechanisms: scaled dot-product, multi-head, causal masking.
//!
//! Implements the core attention from "Attention Is All You Need" (Vaswani et al. 2017).
//!
//! # Examples
//!
//! ```
//! use ndarray::Array3;
//! use machin_nn::attention::{scaled_dot_product_attention, multi_head_attention};
//!
//! // batch=1, seq_len=4, d_model=8
//! let q = Array3::ones((1, 4, 8));
//! let k = Array3::ones((1, 4, 8));
//! let v = Array3::ones((1, 4, 8));
//!
//! let (output, weights) = scaled_dot_product_attention(&q, &k, &v, None);
//! assert_eq!(output.shape(), &[1, 4, 8]);
//! assert_eq!(weights.shape(), &[1, 4, 4]);
//! ```

use ndarray::{Array2, Array3, s};

/// Softmax along the last axis of a 2D array.
pub fn softmax_2d(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        if sum > 0.0 {
            row.mapv_inplace(|v| v / sum);
        }
    }
    result
}

/// Create a causal (lower-triangular) mask for autoregressive attention.
///
/// Returns a `seq_len × seq_len` matrix where `mask[i][j] = -1e9` if `j > i`,
/// else `0.0`. Add this to attention scores before softmax.
pub fn causal_mask(seq_len: usize) -> Array2<f64> {
    Array2::from_shape_fn((seq_len, seq_len), |(i, j)| {
        if j > i { -1e9 } else { 0.0 }
    })
}

/// Scaled dot-product attention.
///
/// `Q`, `K`, `V` have shape `(batch, seq_len, d_k)`.
/// Optional `mask` has shape `(seq_len, seq_len)` — added to scores before softmax.
///
/// Returns `(output, attention_weights)`:
/// - output: `(batch, seq_len, d_k)`
/// - weights: `(batch, seq_len, seq_len)`
pub fn scaled_dot_product_attention(
    q: &Array3<f64>,
    k: &Array3<f64>,
    v: &Array3<f64>,
    mask: Option<&Array2<f64>>,
) -> (Array3<f64>, Array3<f64>) {
    let batch = q.shape()[0];
    let seq_len = q.shape()[1];
    let d_k = q.shape()[2];
    let scale = (d_k as f64).sqrt();

    let kv_seq = k.shape()[1];
    let mut output = Array3::zeros((batch, seq_len, v.shape()[2]));
    let mut all_weights = Array3::zeros((batch, seq_len, kv_seq));

    for b in 0..batch {
        let q_b = q.slice(s![b, .., ..]).to_owned();
        let k_b = k.slice(s![b, .., ..]).to_owned();
        let v_b = v.slice(s![b, .., ..]).to_owned();

        // scores = Q @ K^T / sqrt(d_k)
        let mut scores = q_b.dot(&k_b.t()) / scale;

        // Apply mask
        if let Some(m) = mask {
            scores += m;
        }

        // Softmax
        let weights = softmax_2d(&scores);

        // output = weights @ V
        let out = weights.dot(&v_b);

        output.slice_mut(s![b, .., ..]).assign(&out);
        all_weights.slice_mut(s![b, .., ..]).assign(&weights);
    }

    (output, all_weights)
}

/// Multi-head attention.
///
/// Splits `d_model` into `n_heads` heads of size `d_k = d_model / n_heads`,
/// applies scaled dot-product attention independently, then concatenates.
///
/// `q`, `k`, `v` have shape `(batch, seq_len, d_model)`.
/// `w_q`, `w_k`, `w_v` are projection matrices `(d_model, d_model)`.
/// `w_o` is output projection `(d_model, d_model)`.
///
/// Returns `(output, attention_weights_per_head)`.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    q: &Array3<f64>,
    k: &Array3<f64>,
    v: &Array3<f64>,
    w_q: &Array2<f64>,
    w_k: &Array2<f64>,
    w_v: &Array2<f64>,
    w_o: &Array2<f64>,
    n_heads: usize,
    mask: Option<&Array2<f64>>,
) -> (Array3<f64>, Vec<Array3<f64>>) {
    let batch = q.shape()[0];
    let seq_len = q.shape()[1];
    let d_model = q.shape()[2];
    let d_k = d_model / n_heads;

    // Project Q, K, V
    let q_proj = matmul_3d_2d(q, w_q);
    let k_proj = matmul_3d_2d(k, w_k);
    let v_proj = matmul_3d_2d(v, w_v);

    // Split into heads and compute attention
    let mut head_outputs = Vec::with_capacity(n_heads);
    let mut head_weights = Vec::with_capacity(n_heads);

    for h in 0..n_heads {
        let start = h * d_k;
        let end = start + d_k;

        let q_h = q_proj.slice(s![.., .., start..end]).to_owned();
        let k_h = k_proj.slice(s![.., .., start..end]).to_owned();
        let v_h = v_proj.slice(s![.., .., start..end]).to_owned();

        let (out, w) = scaled_dot_product_attention(&q_h, &k_h, &v_h, mask);
        head_outputs.push(out);
        head_weights.push(w);
    }

    // Concatenate heads: (batch, seq_len, d_model)
    let mut concat = Array3::zeros((batch, seq_len, d_model));
    for (h, head_out) in head_outputs.iter().enumerate() {
        let start = h * d_k;
        concat.slice_mut(s![.., .., start..start + d_k]).assign(head_out);
    }

    // Output projection
    let output = matmul_3d_2d(&concat, w_o);

    (output, head_weights)
}

/// Helper: multiply (batch, seq, d_in) × (d_in, d_out) -> (batch, seq, d_out)
fn matmul_3d_2d(a: &Array3<f64>, b: &Array2<f64>) -> Array3<f64> {
    let batch = a.shape()[0];
    let seq = a.shape()[1];
    let d_out = b.shape()[1];
    let mut result = Array3::zeros((batch, seq, d_out));
    for i in 0..batch {
        let a_b = a.slice(s![i, .., ..]).to_owned();
        let out = a_b.dot(b);
        result.slice_mut(s![i, .., ..]).assign(&out);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_softmax_2d_uniform() {
        let x = Array2::zeros((2, 4));
        let s = softmax_2d(&x);
        // Uniform input → uniform output (0.25 each)
        for row in s.rows() {
            for &v in row.iter() {
                assert!((v - 0.25).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_softmax_2d_sums_to_one() {
        let x = Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f64 * 0.1);
        let s = softmax_2d(&x);
        for row in s.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_2d_large_values() {
        // Should not overflow thanks to max subtraction
        let x = Array2::from_shape_fn((1, 3), |(_, j)| [1000.0, 1001.0, 1000.0][j]);
        let s = softmax_2d(&x);
        assert!(s[[0, 1]] > s[[0, 0]]); // middle value is largest
        assert!((s.row(0).sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_causal_mask_shape() {
        let m = causal_mask(5);
        assert_eq!(m.shape(), &[5, 5]);
        assert_eq!(m[[0, 0]], 0.0);
        assert!(m[[0, 1]] < -1e8); // masked
        assert_eq!(m[[4, 3]], 0.0); // unmasked
        assert!(m[[3, 4]] < -1e8); // masked
    }

    #[test]
    fn test_causal_mask_diagonal() {
        let m = causal_mask(4);
        for i in 0..4 {
            assert_eq!(m[[i, i]], 0.0, "diagonal should be unmasked");
        }
    }

    #[test]
    fn test_scaled_attention_output_shape() {
        let q = Array3::ones((2, 4, 8));
        let k = Array3::ones((2, 4, 8));
        let v = Array3::ones((2, 4, 8));
        let (out, w) = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[2, 4, 8]);
        assert_eq!(w.shape(), &[2, 4, 4]);
    }

    #[test]
    fn test_scaled_attention_weights_sum_to_one() {
        let q = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| (i + j) as f64 * 0.1);
        let k = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| (i * 2 + j) as f64 * 0.1);
        let v = Array3::ones((1, 3, 4));
        let (_, w) = scaled_dot_product_attention(&q, &k, &v, None);
        for row in w.slice(s![0, .., ..]).rows() {
            assert!((row.sum() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scaled_attention_identity_qk() {
        // When Q=K, attention should be symmetric
        let qk = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| (i + j) as f64);
        let v = Array3::ones((1, 3, 4));
        let (out, _) = scaled_dot_product_attention(&qk, &qk, &v, None);
        // All outputs should be close to 1.0 since V is all ones
        for &val in out.iter() {
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scaled_attention_with_causal_mask() {
        let q = Array3::ones((1, 4, 2));
        let k = Array3::ones((1, 4, 2));
        let v = Array3::from_shape_fn((1, 4, 2), |(_, i, _)| i as f64);
        let mask = causal_mask(4);
        let (_, w) = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
        // First token can only attend to itself
        assert!((w[[0, 0, 0]] - 1.0).abs() < 1e-6);
        // Future positions should have ~0 weight
        assert!(w[[0, 0, 1]].abs() < 1e-6);
    }

    #[test]
    fn test_multi_head_attention_output_shape() {
        let d_model = 8;
        let n_heads = 2;
        let q = Array3::ones((1, 4, d_model));
        let k = Array3::ones((1, 4, d_model));
        let v = Array3::ones((1, 4, d_model));
        let w_q = Array2::from_diag(&ndarray::Array1::ones(d_model));
        let w_k = w_q.clone();
        let w_v = w_q.clone();
        let w_o = w_q.clone();

        let (out, head_weights) = multi_head_attention(
            &q, &k, &v, &w_q, &w_k, &w_v, &w_o, n_heads, None,
        );
        assert_eq!(out.shape(), &[1, 4, d_model]);
        assert_eq!(head_weights.len(), n_heads);
        assert_eq!(head_weights[0].shape(), &[1, 4, 4]);
    }

    #[test]
    fn test_multi_head_identity_projection() {
        // With identity projections, multi-head should equal single-head
        let d = 4;
        let q = Array3::from_shape_fn((1, 3, d), |(_, i, j)| (i + j) as f64 * 0.1);
        let k = q.clone();
        let v = Array3::ones((1, 3, d));
        let eye = Array2::from_diag(&ndarray::Array1::ones(d));

        let (mh_out, _) = multi_head_attention(
            &q, &k, &v, &eye, &eye, &eye, &eye, 1, None,
        );
        let (sh_out, _) = scaled_dot_product_attention(&q, &k, &v, None);

        for (a, b) in mh_out.iter().zip(sh_out.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_multi_head_with_mask() {
        let d = 4;
        let q = Array3::ones((1, 3, d));
        let k = Array3::ones((1, 3, d));
        let v = Array3::from_shape_fn((1, 3, d), |(_, i, _)| i as f64);
        let eye = Array2::from_diag(&ndarray::Array1::ones(d));
        let mask = causal_mask(3);

        let (out, _) = multi_head_attention(
            &q, &k, &v, &eye, &eye, &eye, &eye, 2, Some(&mask),
        );
        assert_eq!(out.shape(), &[1, 3, d]);
    }

    #[test]
    fn test_matmul_3d_2d() {
        let a = Array3::ones((2, 3, 4));
        let b = Array2::ones((4, 5));
        let c = matmul_3d_2d(&a, &b);
        assert_eq!(c.shape(), &[2, 3, 5]);
        // Each element = 4 (sum of 4 ones × 1)
        for &v in c.iter() {
            assert!((v - 4.0).abs() < 1e-10);
        }
    }
}
