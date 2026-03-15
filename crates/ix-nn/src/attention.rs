//! Attention mechanisms: scaled dot-product, multi-head, causal masking.
//!
//! Implements the core attention from "Attention Is All You Need" (Vaswani et al. 2017).
//!
//! # Examples
//!
//! ```
//! use ndarray::Array3;
//! use ix_nn::attention::{scaled_dot_product_attention, multi_head_attention};
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

/// Backward pass for scaled dot-product attention.
///
/// Given the gradient of the loss w.r.t. the attention output, computes
/// gradients w.r.t. Q, K, and V.
///
/// # Arguments
/// - `grad_output`: `(batch, seq_len, d_v)` — gradient from downstream
/// - `q`: cached Q from forward pass, `(batch, seq_len, d_k)`
/// - `k`: cached K, `(batch, kv_seq, d_k)`
/// - `v`: cached V, `(batch, kv_seq, d_v)`
/// - `attn_weights`: cached softmax(QK^T / sqrt(d_k)), `(batch, seq_len, kv_seq)`
///
/// # Returns
/// `(grad_q, grad_k, grad_v)` with matching shapes.
pub fn attention_backward(
    grad_output: &Array3<f64>,
    q: &Array3<f64>,
    k: &Array3<f64>,
    v: &Array3<f64>,
    attn_weights: &Array3<f64>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let batch = q.shape()[0];
    let d_k = q.shape()[2];
    let scale = (d_k as f64).sqrt();

    let mut grad_q = Array3::zeros(q.raw_dim());
    let mut grad_k = Array3::zeros(k.raw_dim());
    let mut grad_v = Array3::zeros(v.raw_dim());

    for b in 0..batch {
        let q_b = q.slice(s![b, .., ..]).to_owned();
        let k_b = k.slice(s![b, .., ..]).to_owned();
        let v_b = v.slice(s![b, .., ..]).to_owned();
        let w_b = attn_weights.slice(s![b, .., ..]).to_owned(); // (seq, kv_seq)
        let grad_out_b = grad_output.slice(s![b, .., ..]).to_owned(); // (seq, d_v)

        // grad_v = attn_weights^T @ grad_output
        let gv = w_b.t().dot(&grad_out_b);

        // grad_attn = grad_output @ v^T  — (seq, kv_seq)
        let grad_attn = grad_out_b.dot(&v_b.t());

        // Softmax backward: for each row i,
        // grad_scores[i] = w[i] * (grad_attn[i] - sum(grad_attn[i] * w[i]))
        let seq = w_b.nrows();
        let kv_seq = w_b.ncols();
        let mut grad_scores = Array2::zeros((seq, kv_seq));
        for i in 0..seq {
            let w_row = w_b.row(i);
            let ga_row = grad_attn.row(i);
            let dot_sum: f64 = (&ga_row * &w_row).sum();
            for j in 0..kv_seq {
                grad_scores[[i, j]] = w_row[j] * (ga_row[j] - dot_sum);
            }
        }

        // grad_q = grad_scores @ k / sqrt(d_k)
        let gq = grad_scores.dot(&k_b) / scale;
        // grad_k = grad_scores^T @ q / sqrt(d_k)
        let gk = grad_scores.t().dot(&q_b) / scale;

        grad_q.slice_mut(s![b, .., ..]).assign(&gq);
        grad_k.slice_mut(s![b, .., ..]).assign(&gk);
        grad_v.slice_mut(s![b, .., ..]).assign(&gv);
    }

    (grad_q, grad_k, grad_v)
}

/// Backward pass for multi-head attention.
///
/// Computes gradients w.r.t. the input and all projection weight matrices.
///
/// # Arguments
/// - `grad_output`: `(batch, seq_len, d_model)` — gradient from downstream
/// - `q_input`, `k_input`, `v_input`: original inputs before projection
/// - `w_q`, `w_k`, `w_v`, `w_o`: projection matrices `(d_model, d_model)`
/// - `attn_weights_per_head`: attention weights from each head
/// - `head_q`, `head_k`, `head_v`: per-head projected Q/K/V slices
/// - `concat`: concatenated head outputs before w_o projection `(batch, seq, d_model)`
/// - `n_heads`: number of attention heads
/// - `learning_rate`: step size for weight updates
///
/// # Returns
/// `(grad_input, updated_w_q, updated_w_k, updated_w_v, updated_w_o)`
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_backward(
    grad_output: &Array3<f64>,
    q_input: &Array3<f64>,
    k_input: &Array3<f64>,
    v_input: &Array3<f64>,
    w_q: &mut Array2<f64>,
    w_k: &mut Array2<f64>,
    w_v: &mut Array2<f64>,
    w_o: &mut Array2<f64>,
    attn_weights_per_head: &[Array3<f64>],
    head_q: &[Array3<f64>],
    head_k: &[Array3<f64>],
    head_v: &[Array3<f64>],
    concat: &Array3<f64>,
    n_heads: usize,
    learning_rate: f64,
) -> Array3<f64> {
    let batch = q_input.shape()[0];
    let seq = q_input.shape()[1];
    let d_model = q_input.shape()[2];
    let d_k = d_model / n_heads;

    // 1. Backward through output projection: output = concat @ w_o
    // grad_concat = grad_output @ w_o^T
    let grad_concat = matmul_3d_2d(grad_output, &w_o.t().to_owned());
    // grad_w_o = concat^T @ grad_output (summed over batch)
    let mut grad_w_o = Array2::zeros(w_o.raw_dim());
    for b in 0..batch {
        let c_b = concat.slice(s![b, .., ..]).to_owned();
        let g_b = grad_output.slice(s![b, .., ..]).to_owned();
        grad_w_o = grad_w_o + c_b.t().dot(&g_b);
    }
    grad_w_o /= (batch * seq) as f64;

    // 2. Split grad_concat into per-head gradients and backprop through attention
    let mut grad_q_proj = Array3::zeros((batch, seq, d_model));
    let mut grad_k_proj = Array3::zeros((batch, seq, d_model));
    let mut grad_v_proj = Array3::zeros((batch, seq, d_model));

    for h in 0..n_heads {
        let start = h * d_k;
        let end = start + d_k;
        let grad_head = grad_concat.slice(s![.., .., start..end]).to_owned();

        let (gq, gk, gv) = attention_backward(
            &grad_head,
            &head_q[h],
            &head_k[h],
            &head_v[h],
            &attn_weights_per_head[h],
        );

        grad_q_proj.slice_mut(s![.., .., start..end]).assign(&gq);
        grad_k_proj.slice_mut(s![.., .., start..end]).assign(&gk);
        grad_v_proj.slice_mut(s![.., .., start..end]).assign(&gv);
    }

    // 3. Backward through input projections: q_proj = q_input @ w_q, etc.
    // grad_w_q = q_input^T @ grad_q_proj (summed over batch)
    let mut grad_w_q = Array2::zeros(w_q.raw_dim());
    let mut grad_w_k = Array2::zeros(w_k.raw_dim());
    let mut grad_w_v = Array2::zeros(w_v.raw_dim());

    for b in 0..batch {
        let qi = q_input.slice(s![b, .., ..]).to_owned();
        let ki = k_input.slice(s![b, .., ..]).to_owned();
        let vi = v_input.slice(s![b, .., ..]).to_owned();
        let gqp = grad_q_proj.slice(s![b, .., ..]).to_owned();
        let gkp = grad_k_proj.slice(s![b, .., ..]).to_owned();
        let gvp = grad_v_proj.slice(s![b, .., ..]).to_owned();
        grad_w_q = grad_w_q + qi.t().dot(&gqp);
        grad_w_k = grad_w_k + ki.t().dot(&gkp);
        grad_w_v = grad_w_v + vi.t().dot(&gvp);
    }
    let scale = (batch * seq) as f64;
    grad_w_q /= scale;
    grad_w_k /= scale;
    grad_w_v /= scale;

    // grad_input = grad_q_proj @ w_q^T + grad_k_proj @ w_k^T + grad_v_proj @ w_v^T
    // (In self-attention, q_input = k_input = v_input, so all three contribute to the same input.)
    let grad_input_q = matmul_3d_2d(&grad_q_proj, &w_q.t().to_owned());
    let grad_input_k = matmul_3d_2d(&grad_k_proj, &w_k.t().to_owned());
    let grad_input_v = matmul_3d_2d(&grad_v_proj, &w_v.t().to_owned());
    let grad_input = &grad_input_q + &grad_input_k + &grad_input_v;

    // 4. Update weights
    *w_q = &*w_q - &(learning_rate * &grad_w_q);
    *w_k = &*w_k - &(learning_rate * &grad_w_k);
    *w_v = &*w_v - &(learning_rate * &grad_w_v);
    *w_o = &*w_o - &(learning_rate * &grad_w_o);

    grad_input
}

/// Multi-head attention forward pass that also returns cached values for backward.
///
/// Returns `(output, attn_weights_per_head, head_q, head_k, head_v, concat)`.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_forward_cache(
    q: &Array3<f64>,
    k: &Array3<f64>,
    v: &Array3<f64>,
    w_q: &Array2<f64>,
    w_k: &Array2<f64>,
    w_v: &Array2<f64>,
    w_o: &Array2<f64>,
    n_heads: usize,
    mask: Option<&Array2<f64>>,
) -> (
    Array3<f64>,
    Vec<Array3<f64>>,
    Vec<Array3<f64>>,
    Vec<Array3<f64>>,
    Vec<Array3<f64>>,
    Array3<f64>,
) {
    let batch = q.shape()[0];
    let seq_len = q.shape()[1];
    let d_model = q.shape()[2];
    let d_k = d_model / n_heads;

    // Project Q, K, V
    let q_proj = matmul_3d_2d(q, w_q);
    let k_proj = matmul_3d_2d(k, w_k);
    let v_proj = matmul_3d_2d(v, w_v);

    let mut head_outputs = Vec::with_capacity(n_heads);
    let mut head_weights = Vec::with_capacity(n_heads);
    let mut all_head_q = Vec::with_capacity(n_heads);
    let mut all_head_k = Vec::with_capacity(n_heads);
    let mut all_head_v = Vec::with_capacity(n_heads);

    for h in 0..n_heads {
        let start = h * d_k;
        let end = start + d_k;

        let q_h = q_proj.slice(s![.., .., start..end]).to_owned();
        let k_h = k_proj.slice(s![.., .., start..end]).to_owned();
        let v_h = v_proj.slice(s![.., .., start..end]).to_owned();

        let (out, w) = scaled_dot_product_attention(&q_h, &k_h, &v_h, mask);
        head_outputs.push(out);
        head_weights.push(w);
        all_head_q.push(q_h);
        all_head_k.push(k_h);
        all_head_v.push(v_h);
    }

    // Concatenate heads
    let mut concat = Array3::zeros((batch, seq_len, d_model));
    for (h, head_out) in head_outputs.iter().enumerate() {
        let start = h * d_k;
        concat.slice_mut(s![.., .., start..start + d_k]).assign(head_out);
    }

    // Output projection
    let output = matmul_3d_2d(&concat, w_o);

    (output, head_weights, all_head_q, all_head_k, all_head_v, concat)
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
pub(crate) fn matmul_3d_2d(a: &Array3<f64>, b: &Array2<f64>) -> Array3<f64> {
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

    // --- Backward pass tests ---

    #[test]
    fn test_attention_backward_shapes() {
        let q = Array3::from_shape_fn((2, 3, 4), |(b, i, j)| (b + i + j) as f64 * 0.1);
        let k = Array3::from_shape_fn((2, 3, 4), |(b, i, j)| (b * 2 + i + j) as f64 * 0.1);
        let v = Array3::from_shape_fn((2, 3, 4), |(b, i, j)| (b + i * 2 + j) as f64 * 0.1);
        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, None);
        let grad_out = Array3::ones((2, 3, 4));

        let (gq, gk, gv) = attention_backward(&grad_out, &q, &k, &v, &weights);
        assert_eq!(gq.shape(), &[2, 3, 4]);
        assert_eq!(gk.shape(), &[2, 3, 4]);
        assert_eq!(gv.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_attention_backward_numerical_gradient() {
        let eps = 1e-5;
        let q = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| (i + j) as f64 * 0.1 + 0.05);
        let k = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| (i * 2 + j) as f64 * 0.1 + 0.1);
        let v = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| (i + j * 2) as f64 * 0.1 + 0.02);
        let grad_out = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| {
            (i as f64 * 0.3) + (j as f64 * 0.2) - 0.5
        });

        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, None);
        let (gq_analytical, gk_analytical, gv_analytical) =
            attention_backward(&grad_out, &q, &k, &v, &weights);

        // Numerical gradient for Q
        for i in 0..3 {
            for j in 0..4 {
                let mut q_plus = q.clone();
                let mut q_minus = q.clone();
                q_plus[[0, i, j]] += eps;
                q_minus[[0, i, j]] -= eps;
                let (out_plus, _) = scaled_dot_product_attention(&q_plus, &k, &v, None);
                let (out_minus, _) = scaled_dot_product_attention(&q_minus, &k, &v, None);
                let numerical = (&grad_out * &(&out_plus - &out_minus)).sum() / (2.0 * eps);
                let analytical = gq_analytical[[0, i, j]];
                let denom = analytical.abs().max(numerical.abs()).max(1e-8);
                let rel_err = (analytical - numerical).abs() / denom;
                assert!(
                    rel_err < 1e-4,
                    "Q grad mismatch at [0,{i},{j}]: a={analytical:.6}, n={numerical:.6}, rel={rel_err:.6}"
                );
            }
        }

        // Numerical gradient for V
        for i in 0..3 {
            for j in 0..4 {
                let mut v_plus = v.clone();
                let mut v_minus = v.clone();
                v_plus[[0, i, j]] += eps;
                v_minus[[0, i, j]] -= eps;
                let (out_plus, _) = scaled_dot_product_attention(&q, &k, &v_plus, None);
                let (out_minus, _) = scaled_dot_product_attention(&q, &k, &v_minus, None);
                let numerical = (&grad_out * &(&out_plus - &out_minus)).sum() / (2.0 * eps);
                let analytical = gv_analytical[[0, i, j]];
                let denom = analytical.abs().max(numerical.abs()).max(1e-8);
                let rel_err = (analytical - numerical).abs() / denom;
                assert!(
                    rel_err < 1e-4,
                    "V grad mismatch at [0,{i},{j}]: a={analytical:.6}, n={numerical:.6}, rel={rel_err:.6}"
                );
            }
        }

        // Numerical gradient for K
        for i in 0..3 {
            for j in 0..4 {
                let mut k_plus = k.clone();
                let mut k_minus = k.clone();
                k_plus[[0, i, j]] += eps;
                k_minus[[0, i, j]] -= eps;
                let (out_plus, _) = scaled_dot_product_attention(&q, &k_plus, &v, None);
                let (out_minus, _) = scaled_dot_product_attention(&q, &k_minus, &v, None);
                let numerical = (&grad_out * &(&out_plus - &out_minus)).sum() / (2.0 * eps);
                let analytical = gk_analytical[[0, i, j]];
                let denom = analytical.abs().max(numerical.abs()).max(1e-8);
                let rel_err = (analytical - numerical).abs() / denom;
                assert!(
                    rel_err < 1e-4,
                    "K grad mismatch at [0,{i},{j}]: a={analytical:.6}, n={numerical:.6}, rel={rel_err:.6}"
                );
            }
        }
    }

    #[test]
    fn test_multi_head_attention_backward_shapes() {
        let d_model = 8;
        let n_heads = 2;
        let q = Array3::from_shape_fn((1, 3, d_model), |(_, i, j)| (i + j) as f64 * 0.1);
        let k = q.clone();
        let v = q.clone();
        let mut w_q = Array2::from_diag(&ndarray::Array1::from_elem(d_model, 0.5));
        let mut w_k = w_q.clone();
        let mut w_v = w_q.clone();
        let mut w_o = w_q.clone();

        let (_, weights, hq, hk, hv, concat) = multi_head_attention_forward_cache(
            &q, &k, &v, &w_q, &w_k, &w_v, &w_o, n_heads, None,
        );

        let grad_out = Array3::ones((1, 3, d_model));
        let grad_input = multi_head_attention_backward(
            &grad_out, &q, &k, &v,
            &mut w_q, &mut w_k, &mut w_v, &mut w_o,
            &weights, &hq, &hk, &hv, &concat,
            n_heads, 0.01,
        );
        assert_eq!(grad_input.shape(), &[1, 3, d_model]);
    }

    #[test]
    fn test_multi_head_attention_backward_weights_update() {
        let d_model = 4;
        let n_heads = 2;
        let q = Array3::from_shape_fn((1, 3, d_model), |(_, i, j)| (i + j) as f64 * 0.2);
        let mut w_q = Array2::from_diag(&ndarray::Array1::from_elem(d_model, 0.5));
        let mut w_k = w_q.clone();
        let mut w_v = w_q.clone();
        let mut w_o = w_q.clone();
        let w_q_before = w_q.clone();

        let (_, weights, hq, hk, hv, concat) = multi_head_attention_forward_cache(
            &q, &q, &q, &w_q, &w_k, &w_v, &w_o, n_heads, None,
        );

        let grad_out = Array3::from_shape_fn((1, 3, d_model), |(_, i, j)| {
            (i as f64 - 1.0) * (j as f64 - 1.5)
        });
        let _grad_input = multi_head_attention_backward(
            &grad_out, &q, &q, &q,
            &mut w_q, &mut w_k, &mut w_v, &mut w_o,
            &weights, &hq, &hk, &hv, &concat,
            n_heads, 0.1,
        );

        let diff: f64 = (&w_q - &w_q_before).mapv(|v| v.abs()).sum();
        assert!(diff > 1e-10, "w_q should have been updated");
    }
}
