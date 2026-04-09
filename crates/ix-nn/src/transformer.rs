//! Transformer building blocks: feed-forward network, transformer block, encoder stack.
//!
//! # Examples
//!
//! ```
//! use ndarray::Array3;
//! use ix_nn::transformer::{TransformerBlock, FeedForward};
//!
//! let d_model = 16;
//! let ff = FeedForward::new(d_model, 64, 42);
//! let block = TransformerBlock::new(d_model, 2, 64, 42);
//!
//! let x = Array3::ones((1, 4, d_model));
//! let out = block.forward(&x, None);
//! assert_eq!(out.shape(), &[1, 4, d_model]);
//! ```

use ndarray::{Array1, Array2, Array3, Axis, s};
use crate::attention::{
    multi_head_attention, multi_head_attention_forward_cache,
    multi_head_attention_backward,
};
use crate::norm::LayerNorm;
use crate::dropout::Dropout;

/// Position-wise feed-forward network: two linear layers with ReLU/GELU.
pub struct FeedForward {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    /// Cached input from `forward_cache` for backward pass.
    input_cache: Option<Array3<f64>>,
    /// Cached pre-GELU hidden values from `forward_cache`.
    hidden_cache: Option<Array3<f64>>,
}

impl FeedForward {
    /// Create FFN: `d_model -> d_ff -> d_model` with Xavier init.
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let std1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let std2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        Self {
            w1: Array2::from_shape_fn((d_model, d_ff), |_| rng.random_range(-std1..std1)),
            b1: Array1::zeros(d_ff),
            w2: Array2::from_shape_fn((d_ff, d_model), |_| rng.random_range(-std2..std2)),
            b2: Array1::zeros(d_model),
            input_cache: None,
            hidden_cache: None,
        }
    }

    /// Forward: GELU(x @ W1 + b1) @ W2 + b2
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let d_out = self.w2.shape()[1];
        let mut result = Array3::zeros((batch, seq, d_out));

        for b in 0..batch {
            let x_b = x.slice(s![b, .., ..]).to_owned();
            let hidden = (x_b.dot(&self.w1) + &self.b1).mapv(gelu);
            let out = hidden.dot(&self.w2) + &self.b2;
            result.slice_mut(s![b, .., ..]).assign(&out);
        }
        result
    }

    /// Forward pass that caches activations for backward.
    ///
    /// Same output as `forward`, but stores the input and pre-GELU hidden
    /// values so `backward` can compute gradients.
    pub fn forward_cache(&mut self, x: &Array3<f64>) -> Array3<f64> {
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let d_ff = self.w1.shape()[1];
        let d_out = self.w2.shape()[1];
        let mut result = Array3::zeros((batch, seq, d_out));
        let mut hidden_pre = Array3::zeros((batch, seq, d_ff));

        for b in 0..batch {
            let x_b = x.slice(s![b, .., ..]).to_owned();
            let pre_gelu = x_b.dot(&self.w1) + &self.b1;
            hidden_pre.slice_mut(s![b, .., ..]).assign(&pre_gelu);
            let hidden = pre_gelu.mapv(gelu);
            let out = hidden.dot(&self.w2) + &self.b2;
            result.slice_mut(s![b, .., ..]).assign(&out);
        }

        self.input_cache = Some(x.clone());
        self.hidden_cache = Some(hidden_pre);
        result
    }

    /// Backward pass: computes gradient w.r.t. input and updates weights.
    ///
    /// `grad_output` has shape `(batch, seq, d_model)`.
    /// Returns gradient w.r.t. the input.
    pub fn backward(&mut self, grad_output: &Array3<f64>, learning_rate: f64) -> Array3<f64> {
        let input = self.input_cache.as_ref().expect("forward_cache() not called").clone();
        let hidden_pre = self.hidden_cache.as_ref().expect("forward_cache() not called").clone();

        let batch = input.shape()[0];
        let seq = input.shape()[1];
        let d_model = self.w2.shape()[1];
        let d_ff = self.w1.shape()[1];

        let mut grad_w2 = Array2::zeros((d_ff, d_model));
        let mut grad_b2 = Array1::zeros(d_model);
        let mut grad_w1 = Array2::zeros((d_model, d_ff));
        let mut grad_b1 = Array1::zeros(d_ff);
        let mut grad_input = Array3::zeros(input.raw_dim());

        for b in 0..batch {
            let x_b = input.slice(s![b, .., ..]).to_owned();
            let h_pre = hidden_pre.slice(s![b, .., ..]).to_owned();
            let h_post = h_pre.mapv(gelu); // GELU(pre)
            let go_b = grad_output.slice(s![b, .., ..]).to_owned();

            // Backward through W2: output = h_post @ W2 + b2
            // grad_h_post = grad_out @ W2^T
            let grad_h_post = go_b.dot(&self.w2.t());
            // grad_w2 += h_post^T @ grad_out
            grad_w2 = grad_w2 + h_post.t().dot(&go_b);
            // grad_b2 += sum(grad_out, axis=0)
            grad_b2 = grad_b2 + go_b.sum_axis(Axis(0));

            // Backward through GELU
            let grad_h_pre = &grad_h_post * &h_pre.mapv(gelu_derivative);

            // Backward through W1: hidden = x @ W1 + b1
            // grad_x = grad_h_pre @ W1^T
            let gx = grad_h_pre.dot(&self.w1.t());
            grad_input.slice_mut(s![b, .., ..]).assign(&gx);
            // grad_w1 += x^T @ grad_h_pre
            grad_w1 = grad_w1 + x_b.t().dot(&grad_h_pre);
            // grad_b1 += sum(grad_h_pre, axis=0)
            grad_b1 = grad_b1 + grad_h_pre.sum_axis(Axis(0));
        }

        let scale = (batch * seq) as f64;
        self.w1 = &self.w1 - &(learning_rate * &grad_w1 / scale);
        self.b1 = &self.b1 - &(learning_rate * &grad_b1 / scale);
        self.w2 = &self.w2 - &(learning_rate * &grad_w2 / scale);
        self.b2 = &self.b2 - &(learning_rate * &grad_b2 / scale);

        grad_input
    }
}

/// GELU activation: `x * Phi(x)` approximated as `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Derivative of GELU activation.
///
/// `gelu'(x) = 0.5 * (1 + tanh(a)) + 0.5 * x * (1 - tanh^2(a)) * a'`
/// where `a = sqrt(2/pi) * (x + 0.044715 * x^3)` and `a' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)`.
pub fn gelu_derivative(x: f64) -> f64 {
    let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
    let a = sqrt_2_pi * (x + 0.044715 * x.powi(3));
    let tanh_a = a.tanh();
    let a_prime = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_a) + 0.5 * x * (1.0 - tanh_a * tanh_a) * a_prime
}

/// SwiGLU activation (Shazeer 2020) — used in LLaMA, PaLM.
///
/// Splits input into two halves, applies swish to first and gates with second.
pub fn swiglu(x: &Array1<f64>) -> Array1<f64> {
    let half = x.len() / 2;
    let gate = x.slice(s![..half]).mapv(|v| v / (1.0 + (-v).exp())); // swish = x * sigmoid(x)
    let value = x.slice(s![half..]).to_owned();
    &gate * &value
}

/// A single Transformer block: multi-head attention + FFN + residual + layer norm.
///
/// Uses pre-norm architecture (norm before attention/FFN), which is more stable
/// and used in GPT-2+, LLaMA, etc.
pub struct TransformerBlock {
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_o: Array2<f64>,
    pub ffn: FeedForward,
    pub n_heads: usize,
    /// Dropout after attention output.
    attn_dropout: Dropout,
    /// Dropout after FFN output.
    ffn_dropout: Dropout,
    // Caches for backward pass
    input_cache: Option<Array3<f64>>,
    normed1_cache: Option<Array3<f64>>,
    residual1_cache: Option<Array3<f64>>,
    normed2_cache: Option<Array3<f64>>,
    attn_weights_cache: Option<Vec<Array3<f64>>>,
    head_q_cache: Option<Vec<Array3<f64>>>,
    head_k_cache: Option<Vec<Array3<f64>>>,
    head_v_cache: Option<Vec<Array3<f64>>>,
    concat_cache: Option<Array3<f64>>,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, seed: u64) -> Self {
        Self::new_with_dropout(d_model, n_heads, d_ff, seed, 0.0)
    }

    /// Create a transformer block with dropout.
    pub fn new_with_dropout(d_model: usize, n_heads: usize, d_ff: usize, seed: u64, dropout: f64) -> Self {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let std = (1.0 / d_model as f64).sqrt();

        let random_matrix = |rng: &mut rand::rngs::StdRng| -> Array2<f64> {
            Array2::from_shape_fn((d_model, d_model), |_| rng.random_range(-std..std))
        };

        Self {
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            w_q: random_matrix(&mut rng),
            w_k: random_matrix(&mut rng),
            w_v: random_matrix(&mut rng),
            w_o: random_matrix(&mut rng),
            ffn: FeedForward::new(d_model, d_ff, seed + 1),
            n_heads,
            attn_dropout: Dropout::new(dropout, seed + 2),
            ffn_dropout: Dropout::new(dropout, seed + 3),
            input_cache: None,
            normed1_cache: None,
            residual1_cache: None,
            normed2_cache: None,
            attn_weights_cache: None,
            head_q_cache: None,
            head_k_cache: None,
            head_v_cache: None,
            concat_cache: None,
        }
    }

    /// Forward pass with pre-norm residual connections.
    ///
    /// `x` has shape `(batch, seq_len, d_model)`.
    /// Optional `mask` for causal attention.
    pub fn forward(&self, x: &Array3<f64>, mask: Option<&Array2<f64>>) -> Array3<f64> {
        // Pre-norm + multi-head attention + residual
        let normed1 = self.norm_3d(&self.norm1, x);
        let (attn_out, _) = multi_head_attention(
            &normed1, &normed1, &normed1,
            &self.w_q, &self.w_k, &self.w_v, &self.w_o,
            self.n_heads, mask,
        );
        let residual1 = x + &attn_out;

        // Pre-norm + FFN + residual
        let normed2 = self.norm_3d(&self.norm2, &residual1);
        let ffn_out = self.ffn.forward(&normed2);
        &residual1 + &ffn_out
    }

    /// Forward pass that caches all intermediate activations for backward.
    pub fn forward_cache(&mut self, x: &Array3<f64>, mask: Option<&Array2<f64>>) -> Array3<f64> {
        self.forward_cache_gpu(x, mask, None)
    }

    /// Forward pass with GPU acceleration and caching for backward.
    pub fn forward_cache_gpu(
        &mut self,
        x: &Array3<f64>,
        mask: Option<&Array2<f64>>,
        gpu_ctx: Option<&ix_gpu::context::GpuContext>,
    ) -> Array3<f64> {
        self.input_cache = Some(x.clone());

        // Pre-norm 1 (with cache for backward)
        let normed1 = Self::norm_3d_cache(&mut self.norm1, x);
        self.normed1_cache = Some(normed1.clone());

        // Multi-head attention with cache for backward pass
        let _ = gpu_ctx; // GPU forward path not yet differentiated; kept for API stability
        let (attn_out, weights, hq, hk, hv, concat) = multi_head_attention_forward_cache(
            &normed1, &normed1, &normed1,
            &self.w_q, &self.w_k, &self.w_v, &self.w_o,
            self.n_heads, mask,
        );

        // Apply dropout after attention
        let attn_out = self.attn_dropout.forward_train_3d(&attn_out);

        self.attn_weights_cache = Some(weights);
        self.head_q_cache = Some(hq);
        self.head_k_cache = Some(hk);
        self.head_v_cache = Some(hv);
        self.concat_cache = Some(concat);

        let residual1 = x + &attn_out;
        self.residual1_cache = Some(residual1.clone());

        // Pre-norm 2 (with cache for backward)
        let normed2 = Self::norm_3d_cache(&mut self.norm2, &residual1);
        self.normed2_cache = Some(normed2.clone());

        // FFN (with cache) + dropout
        let ffn_out = self.ffn.forward_cache(&normed2);
        let ffn_out = self.ffn_dropout.forward_train_3d(&ffn_out);

        &residual1 + &ffn_out
    }

    /// Backward pass through the transformer block.
    ///
    /// `grad_output` has shape `(batch, seq_len, d_model)`.
    /// Returns gradient w.r.t. the block input.
    pub fn backward(&mut self, grad_output: &Array3<f64>, learning_rate: f64) -> Array3<f64> {
        let input = self.input_cache.as_ref().expect("forward_cache() not called").clone();
        let normed1 = self.normed1_cache.as_ref().expect("forward_cache() not called").clone();
        let residual1 = self.residual1_cache.as_ref().expect("forward_cache() not called").clone();
        let attn_weights = self.attn_weights_cache.take().expect("forward_cache() not called");
        let head_q = self.head_q_cache.take().expect("forward_cache() not called");
        let head_k = self.head_k_cache.take().expect("forward_cache() not called");
        let head_v = self.head_v_cache.take().expect("forward_cache() not called");
        let concat = self.concat_cache.take().expect("forward_cache() not called");

        // --- Backward through residual2: output = residual1 + ffn_out ---
        let grad_ffn_out = grad_output.clone();

        // --- Backward through FFN dropout ---
        let grad_ffn_out = self.ffn_dropout.backward_3d(&grad_ffn_out);

        // --- Backward through FFN ---
        let grad_normed2 = self.ffn.backward(&grad_ffn_out, learning_rate);

        // --- Backward through norm2 ---
        let grad_residual1_from_norm2 = Self::norm_3d_backward(&mut self.norm2, &grad_normed2, learning_rate);

        // Total gradient on residual1 = skip connection + norm2 path
        let grad_residual1 = grad_output + &grad_residual1_from_norm2;

        // --- Backward through residual1: residual1 = input + attn_out ---
        let grad_attn_out = grad_residual1.clone();

        // --- Backward through attention dropout ---
        let grad_attn_out = self.attn_dropout.backward_3d(&grad_attn_out);

        // --- Backward through multi-head attention ---
        let grad_normed1 = multi_head_attention_backward(
            &grad_attn_out,
            &normed1, &normed1, &normed1,
            &mut self.w_q, &mut self.w_k, &mut self.w_v, &mut self.w_o,
            &attn_weights, &head_q, &head_k, &head_v, &concat,
            self.n_heads, learning_rate,
        );

        // --- Backward through norm1 ---
        let grad_input_from_norm1 = Self::norm_3d_backward(&mut self.norm1, &grad_normed1, learning_rate);

        // Total gradient on input = skip connection + norm1 path
        let grad_input = &grad_residual1 + &grad_input_from_norm1;

        // Restore caches for potential re-use (put back non-taken ones)
        self.input_cache = Some(input);
        self.normed1_cache = Some(normed1);
        self.residual1_cache = Some(residual1);

        grad_input
    }

    /// Apply LayerNorm to each batch element of a 3D tensor.
    fn norm_3d(&self, norm: &LayerNorm, x: &Array3<f64>) -> Array3<f64> {
        let batch = x.shape()[0];
        let mut result = x.clone();
        for b in 0..batch {
            let slice = x.slice(s![b, .., ..]).to_owned();
            let normed = norm.forward(&slice);
            result.slice_mut(s![b, .., ..]).assign(&normed);
        }
        result
    }

    /// Apply LayerNorm with caching to each batch element of a 3D tensor.
    fn norm_3d_cache(norm: &mut LayerNorm, x: &Array3<f64>) -> Array3<f64> {
        let batch = x.shape()[0];
        // We need to cache per the full flattened (batch*seq, d_model) view
        let seq = x.shape()[1];
        let d = x.shape()[2];
        let flat = x.clone().into_shape_with_order((batch * seq, d)).unwrap();
        let normed_flat = norm.forward_cache(&flat);
        normed_flat.into_shape_with_order((batch, seq, d)).unwrap()
    }

    /// Backward through norm_3d: reshape to 2D, call LayerNorm backward, reshape back.
    fn norm_3d_backward(norm: &mut LayerNorm, grad: &Array3<f64>, learning_rate: f64) -> Array3<f64> {
        let batch = grad.shape()[0];
        let seq = grad.shape()[1];
        let d = grad.shape()[2];
        let grad_flat = grad.clone().into_shape_with_order((batch * seq, d)).unwrap();
        let grad_input_flat = norm.backward(&grad_flat, learning_rate);
        grad_input_flat.into_shape_with_order((batch, seq, d)).unwrap()
    }
}

/// Stack of transformer blocks (encoder or decoder).
pub struct TransformerStack {
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: LayerNorm,
}

impl TransformerStack {
    /// Create a stack of `n_layers` identical transformer blocks.
    pub fn new(n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize, seed: u64) -> Self {
        Self::new_with_dropout(n_layers, d_model, n_heads, d_ff, seed, 0.0)
    }

    /// Create a stack of `n_layers` transformer blocks with dropout.
    pub fn new_with_dropout(
        n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize,
        seed: u64, dropout: f64,
    ) -> Self {
        let blocks = (0..n_layers)
            .map(|i| TransformerBlock::new_with_dropout(
                d_model, n_heads, d_ff, seed + i as u64 * 100, dropout,
            ))
            .collect();
        Self {
            blocks,
            final_norm: LayerNorm::new(d_model),
        }
    }

    /// Forward pass through all blocks with optional causal mask.
    pub fn forward(&self, x: &Array3<f64>, mask: Option<&Array2<f64>>) -> Array3<f64> {
        let mut hidden = x.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, mask);
        }
        // Final layer norm
        let batch = hidden.shape()[0];
        for b in 0..batch {
            let slice = hidden.slice(s![b, .., ..]).to_owned();
            let normed = self.final_norm.forward(&slice);
            hidden.slice_mut(s![b, .., ..]).assign(&normed);
        }
        hidden
    }

    /// Forward pass with caching for backward.
    pub fn forward_cache(&mut self, x: &Array3<f64>, mask: Option<&Array2<f64>>) -> Array3<f64> {
        self.forward_cache_gpu(x, mask, None)
    }

    /// Forward pass with GPU acceleration and caching for backward.
    pub fn forward_cache_gpu(
        &mut self,
        x: &Array3<f64>,
        mask: Option<&Array2<f64>>,
        gpu_ctx: Option<&ix_gpu::context::GpuContext>,
    ) -> Array3<f64> {
        let mut hidden = x.clone();
        for block in &mut self.blocks {
            hidden = block.forward_cache_gpu(&hidden, mask, gpu_ctx);
        }
        // Final layer norm with cache
        let batch = hidden.shape()[0];
        let seq = hidden.shape()[1];
        let d = hidden.shape()[2];
        let flat = hidden.into_shape_with_order((batch * seq, d)).unwrap();
        let normed_flat = self.final_norm.forward_cache(&flat);
        normed_flat.into_shape_with_order((batch, seq, d)).unwrap()
    }

    /// Backward pass through all blocks in reverse order.
    ///
    /// `grad_output` has shape `(batch, seq_len, d_model)`.
    /// Returns gradient w.r.t. the stack input.
    pub fn backward(&mut self, grad_output: &Array3<f64>, learning_rate: f64) -> Array3<f64> {
        // Backward through final norm
        let batch = grad_output.shape()[0];
        let seq = grad_output.shape()[1];
        let d = grad_output.shape()[2];
        let grad_flat = grad_output.clone().into_shape_with_order((batch * seq, d)).unwrap();
        let grad_after_norm = self.final_norm.backward(&grad_flat, learning_rate);
        let mut grad = grad_after_norm.into_shape_with_order((batch, seq, d)).unwrap();

        // Backward through blocks in reverse
        for block in self.blocks.iter_mut().rev() {
            grad = block.backward(&grad, learning_rate);
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_gelu_zero() {
        assert!(gelu(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gelu_positive() {
        // GELU(x) ≈ x for large positive x
        assert!((gelu(3.0) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_negative() {
        // GELU(x) ≈ 0 for large negative x
        assert!(gelu(-3.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_monotonic() {
        let vals: Vec<f64> = (-20..=20).map(|i| gelu(i as f64 * 0.5)).collect();
        // GELU is approximately monotonic (not strictly for x < -0.5)
        // but should be increasing for x > 0
        for i in 21..vals.len() - 1 {
            assert!(vals[i + 1] >= vals[i] - 1e-10);
        }
    }

    #[test]
    fn test_gelu_derivative_at_zero() {
        // gelu'(0) = 0.5
        let d = gelu_derivative(0.0);
        assert!((d - 0.5).abs() < 1e-6, "gelu'(0) should be 0.5, got {d}");
    }

    #[test]
    fn test_gelu_derivative_numerical() {
        // Check gelu_derivative against numerical differentiation
        let eps = 1e-6;
        for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let numerical = (gelu(x + eps) - gelu(x - eps)) / (2.0 * eps);
            let analytical = gelu_derivative(x);
            let err = (analytical - numerical).abs();
            assert!(
                err < 1e-5,
                "gelu' mismatch at x={x}: analytical={analytical}, numerical={numerical}"
            );
        }
    }

    #[test]
    fn test_swiglu_shape() {
        let x = Array1::ones(8);
        let out = swiglu(&x);
        assert_eq!(out.len(), 4); // half the input
    }

    #[test]
    fn test_swiglu_zero_gate() {
        let mut x = Array1::zeros(4);
        x[2] = 5.0; // value half
        x[3] = 3.0;
        let out = swiglu(&x);
        // Gate half is 0 → swish(0) = 0 → output ≈ 0
        assert!(out[0].abs() < 1e-10);
    }

    #[test]
    fn test_ffn_output_shape() {
        let ffn = FeedForward::new(8, 32, 42);
        let x = Array3::ones((2, 4, 8));
        let out = ffn.forward(&x);
        assert_eq!(out.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_ffn_forward_cache_matches_forward() {
        let ffn_ref = FeedForward::new(8, 16, 42);
        let mut ffn_cache = FeedForward::new(8, 16, 42);
        let x = Array3::from_shape_fn((1, 3, 8), |(_, i, j)| (i + j) as f64 * 0.1);
        let out1 = ffn_ref.forward(&x);
        let out2 = ffn_cache.forward_cache(&x);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-12, "forward_cache should match forward");
        }
    }

    #[test]
    fn test_ffn_backward_shape() {
        let mut ffn = FeedForward::new(8, 16, 42);
        let x = Array3::from_shape_fn((2, 3, 8), |(b, i, j)| (b + i + j) as f64 * 0.1);
        let _out = ffn.forward_cache(&x);
        let grad_out = Array3::ones((2, 3, 8));
        let grad_in = ffn.backward(&grad_out, 0.01);
        assert_eq!(grad_in.shape(), &[2, 3, 8]);
    }

    #[test]
    fn test_ffn_backward_weights_update() {
        let mut ffn = FeedForward::new(4, 8, 42);
        let w1_before = ffn.w1.clone();
        let w2_before = ffn.w2.clone();
        let x = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| (i + j) as f64 * 0.2);
        let _out = ffn.forward_cache(&x);
        let grad_out = Array3::from_shape_fn((1, 3, 4), |(_, i, j)| {
            (i as f64 - 1.0) * (j as f64 - 1.5)
        });
        let _grad_in = ffn.backward(&grad_out, 0.1);
        let diff_w1: f64 = (&ffn.w1 - &w1_before).mapv(|v| v.abs()).sum();
        let diff_w2: f64 = (&ffn.w2 - &w2_before).mapv(|v| v.abs()).sum();
        assert!(diff_w1 > 1e-10, "w1 should have been updated");
        assert!(diff_w2 > 1e-10, "w2 should have been updated");
    }

    #[test]
    fn test_ffn_backward_numerical_gradient() {
        let eps = 1e-5;
        let x = Array3::from_shape_fn((1, 2, 4), |(_, i, j)| (i + j) as f64 * 0.15 + 0.1);
        let grad_out = Array3::from_shape_fn((1, 2, 4), |(_, i, j)| {
            (i as f64 * 0.3) - (j as f64 * 0.2) + 0.1
        });

        // Analytical gradient
        let mut ffn = FeedForward::new(4, 8, 42);
        let _out = ffn.forward_cache(&x);
        let analytical = ffn.backward(&grad_out, 0.0);

        // Numerical gradient
        let ffn_ref = FeedForward::new(4, 8, 42);
        for i in 0..2 {
            for j in 0..4 {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[0, i, j]] += eps;
                x_minus[[0, i, j]] -= eps;
                let out_plus = ffn_ref.forward(&x_plus);
                let out_minus = ffn_ref.forward(&x_minus);
                let numerical = (&grad_out * &(&out_plus - &out_minus)).sum() / (2.0 * eps);
                let a = analytical[[0, i, j]];
                let denom = a.abs().max(numerical.abs()).max(1e-8);
                let rel_err = (a - numerical).abs() / denom;
                assert!(
                    rel_err < 1e-4,
                    "FFN grad mismatch at [0,{i},{j}]: a={a:.6}, n={numerical:.6}, rel={rel_err:.6}"
                );
            }
        }
    }

    #[test]
    fn test_transformer_block_output_shape() {
        let block = TransformerBlock::new(16, 4, 64, 42);
        let x = Array3::ones((1, 6, 16));
        let out = block.forward(&x, None);
        assert_eq!(out.shape(), &[1, 6, 16]);
    }

    #[test]
    fn test_transformer_block_residual() {
        // Output should differ from input (weights + FFN modify)
        let block = TransformerBlock::new(8, 2, 32, 42);
        let x = Array3::from_shape_fn((1, 3, 8), |(_, i, j)| (i + j) as f64 * 0.1);
        let out = block.forward(&x, None);
        let diff: f64 = (&out - &x).mapv(|v| v.abs()).sum();
        assert!(diff > 0.0, "output should differ from input");
    }

    #[test]
    fn test_transformer_block_with_mask() {
        let block = TransformerBlock::new(8, 2, 32, 42);
        let x = Array3::ones((1, 4, 8));
        let mask = crate::attention::causal_mask(4);
        let out = block.forward(&x, Some(&mask));
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_transformer_block_batch() {
        let block = TransformerBlock::new(8, 2, 16, 42);
        let x = Array3::ones((3, 5, 8));
        let out = block.forward(&x, None);
        assert_eq!(out.shape(), &[3, 5, 8]);
    }

    #[test]
    fn test_transformer_block_forward_cache_matches_forward() {
        let block_ref = TransformerBlock::new(8, 2, 16, 42);
        let mut block_cache = TransformerBlock::new(8, 2, 16, 42);
        let x = Array3::from_shape_fn((1, 3, 8), |(_, i, j)| (i + j) as f64 * 0.1);
        let out1 = block_ref.forward(&x, None);
        let out2 = block_cache.forward_cache(&x, None);
        // With dropout=0.0 (default), forward_cache should match forward
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "forward_cache should match forward: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_transformer_block_backward_shape() {
        let mut block = TransformerBlock::new(8, 2, 16, 42);
        let x = Array3::from_shape_fn((1, 3, 8), |(_, i, j)| (i + j) as f64 * 0.1);
        let _out = block.forward_cache(&x, None);
        let grad_out = Array3::ones((1, 3, 8));
        let grad_in = block.backward(&grad_out, 0.01);
        assert_eq!(grad_in.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_transformer_block_backward_weights_update() {
        let mut block = TransformerBlock::new(8, 2, 16, 42);
        let w_o_before = block.w_o.clone();
        let ffn_w1_before = block.ffn.w1.clone();
        // Use varied inputs to avoid degenerate gradient patterns
        let x = Array3::from_shape_fn((2, 4, 8), |(b, i, j)| {
            ((b * 7 + i * 3 + j) as f64 * 0.17).sin()
        });
        let _out = block.forward_cache(&x, None);
        let grad_out = Array3::from_shape_fn((2, 4, 8), |(b, i, j)| {
            ((b * 5 + i * 2 + j) as f64 * 0.23).cos()
        });
        let _grad_in = block.backward(&grad_out, 0.1);
        // Check output projection weights changed (most directly affected)
        let diff_wo: f64 = (&block.w_o - &w_o_before).mapv(|v| v.abs()).sum();
        assert!(diff_wo > 1e-10, "w_o should have been updated after backward");
        // Check FFN weights changed
        let diff_ffn: f64 = (&block.ffn.w1 - &ffn_w1_before).mapv(|v| v.abs()).sum();
        assert!(diff_ffn > 1e-10, "ffn.w1 should have been updated after backward");
    }

    #[test]
    fn test_transformer_stack_output_shape() {
        let stack = TransformerStack::new(3, 16, 4, 64, 42);
        let x = Array3::ones((1, 4, 16));
        let out = stack.forward(&x, None);
        assert_eq!(out.shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_transformer_stack_with_mask() {
        let stack = TransformerStack::new(2, 8, 2, 32, 42);
        let x = Array3::from_shape_fn((1, 5, 8), |(_, i, j)| (i + j) as f64 * 0.01);
        let mask = crate::attention::causal_mask(5);
        let out = stack.forward(&x, Some(&mask));
        assert_eq!(out.shape(), &[1, 5, 8]);
    }

    #[test]
    fn test_transformer_stack_depth_matters() {
        // Deeper stack should produce different output than shallow
        let shallow = TransformerStack::new(1, 8, 2, 16, 42);
        let deep = TransformerStack::new(4, 8, 2, 16, 42);
        let x = Array3::from_shape_fn((1, 3, 8), |(_, i, j)| (i + j) as f64 * 0.1);
        let out1 = shallow.forward(&x, None);
        let out2 = deep.forward(&x, None);
        let diff: f64 = (&out1 - &out2).mapv(|v| v.abs()).sum();
        assert!(diff > 0.01, "deeper stack should produce different output");
    }

    #[test]
    fn test_transformer_stack_forward_cache_matches_forward() {
        let stack_ref = TransformerStack::new(2, 8, 2, 16, 42);
        let mut stack_cache = TransformerStack::new(2, 8, 2, 16, 42);
        let x = Array3::from_shape_fn((1, 3, 8), |(_, i, j)| (i + j) as f64 * 0.1);
        let out1 = stack_ref.forward(&x, None);
        let out2 = stack_cache.forward_cache(&x, None);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "forward_cache should match forward: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_transformer_stack_backward_shape() {
        let mut stack = TransformerStack::new(2, 8, 2, 16, 42);
        let x = Array3::from_shape_fn((1, 3, 8), |(_, i, j)| (i + j) as f64 * 0.1);
        let _out = stack.forward_cache(&x, None);
        let grad_out = Array3::ones((1, 3, 8));
        let grad_in = stack.backward(&grad_out, 0.01);
        assert_eq!(grad_in.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_transformer_stack_backward_weights_update() {
        let mut stack = TransformerStack::new(2, 8, 2, 16, 42);
        let wo_before = stack.blocks[0].w_o.clone();
        let ffn_w1_before = stack.blocks[0].ffn.w1.clone();
        let x = Array3::from_shape_fn((2, 4, 8), |(b, i, j)| {
            ((b * 7 + i * 3 + j) as f64 * 0.17).sin()
        });
        let _out = stack.forward_cache(&x, None);
        let grad_out = Array3::from_shape_fn((2, 4, 8), |(b, i, j)| {
            ((b * 5 + i * 2 + j) as f64 * 0.23).cos()
        });
        let _grad_in = stack.backward(&grad_out, 0.1);
        let diff_wo: f64 = (&stack.blocks[0].w_o - &wo_before).mapv(|v| v.abs()).sum();
        assert!(diff_wo > 1e-10, "first block w_o should have been updated");
        let diff_ffn: f64 = (&stack.blocks[0].ffn.w1 - &ffn_w1_before).mapv(|v| v.abs()).sum();
        assert!(diff_ffn > 1e-10, "first block ffn.w1 should have been updated");
    }

    // --- Dropout integration tests ---

    #[test]
    fn test_transformer_block_with_dropout() {
        let mut block = TransformerBlock::new_with_dropout(8, 2, 16, 42, 0.1);
        let x = Array3::from_shape_fn((2, 3, 8), |(b, i, j)| (b + i + j) as f64 * 0.1);
        let out = block.forward_cache(&x, None);
        assert_eq!(out.shape(), &[2, 3, 8]);
        // Should be able to backward
        let grad = Array3::ones((2, 3, 8));
        let grad_in = block.backward(&grad, 0.01);
        assert_eq!(grad_in.shape(), &[2, 3, 8]);
    }

    #[test]
    fn test_transformer_stack_with_dropout() {
        let mut stack = TransformerStack::new_with_dropout(2, 8, 2, 16, 42, 0.1);
        let x = Array3::from_shape_fn((1, 3, 8), |(_, i, j)| (i + j) as f64 * 0.1);
        let out = stack.forward_cache(&x, None);
        assert_eq!(out.shape(), &[1, 3, 8]);
        let grad = Array3::ones((1, 3, 8));
        let grad_in = stack.backward(&grad, 0.01);
        assert_eq!(grad_in.shape(), &[1, 3, 8]);
    }
}
