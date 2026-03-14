//! Transformer building blocks: feed-forward network, transformer block, encoder stack.
//!
//! # Examples
//!
//! ```
//! use ndarray::Array3;
//! use machin_nn::transformer::{TransformerBlock, FeedForward};
//!
//! let d_model = 16;
//! let ff = FeedForward::new(d_model, 64, 42);
//! let block = TransformerBlock::new(d_model, 2, 64, 42);
//!
//! let x = Array3::ones((1, 4, d_model));
//! let out = block.forward(&x, None);
//! assert_eq!(out.shape(), &[1, 4, d_model]);
//! ```

use ndarray::{Array1, Array2, Array3, s};
use crate::attention::multi_head_attention;
use crate::norm::LayerNorm;

/// Position-wise feed-forward network: two linear layers with ReLU/GELU.
pub struct FeedForward {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
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
}

/// GELU activation: `x * Φ(x)` approximated as `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
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
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, seed: u64) -> Self {
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
}

/// Stack of transformer blocks (encoder or decoder).
pub struct TransformerStack {
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: LayerNorm,
}

impl TransformerStack {
    /// Create a stack of `n_layers` identical transformer blocks.
    pub fn new(n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize, seed: u64) -> Self {
        let blocks = (0..n_layers)
            .map(|i| TransformerBlock::new(d_model, n_heads, d_ff, seed + i as u64 * 100))
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
}
