---
name: ix-nn
description: Neural networks — trainable transformers with GPU attention, dense layers, backprop, loss functions, positional encodings
---

# Neural Networks

Trainable neural network components including end-to-end transformer models with GPU-accelerated attention.

## When to Use
When the user needs neural network operations — from simple dense layers to full transformer training. For high-level ML pipelines, use `/ix-ml-builder` instead.

## Capabilities

### Trainable Models
- **TransformerClassifier** — Full transformer with backprop, implements `Classifier` trait
- **TransformerRegressor** — Same for regression, implements `Regressor` trait
- **Sequential** — Stack of Dense layers with forward/backward/fit

### Building Blocks (all with backward passes)
- **TransformerStack** — N transformer blocks with final LayerNorm
- **TransformerBlock** — Pre-norm: LayerNorm → MultiHeadAttention → Residual → LayerNorm → FFN → Residual
- **MultiHeadAttention** — Scaled dot-product attention with Q/K/V projections
- **FeedForward** — Two-layer MLP with GELU activation
- **LayerNorm** / **RMSNorm** — Normalization layers
- **Dense** — Fully-connected layer with Xavier initialization

### GPU Acceleration
- **scaled_dot_product_attention_gpu** — GPU matmul for Q·K^T and attn·V via WGPU
- **multi_head_attention_gpu** — GPU for all 4 projection matrices
- Auto-fallback to CPU when no GPU available

### Loss Functions
- **MSE** — Mean squared error + gradient
- **BCE** — Binary cross-entropy + gradient

### Positional Encodings
- **Sinusoidal** (Vaswani), **RoPE** (Su), **ALiBi** (Press), **Learned**

## Programmatic Usage
```rust
use ix_nn::classifier::{TransformerClassifier, TransformerConfig};
use ix_supervised::traits::Classifier;

let config = TransformerConfig {
    d_model: 64, n_heads: 4, n_layers: 2, d_ff: 128,
    seq_len: None, epochs: 50, learning_rate: 0.001, seed: 42,
};
let mut model = TransformerClassifier::new(config);
model.fit(&x_train, &y_train);
let predictions = model.predict(&x_test);
```

## MCP Tools
- `ix_nn_forward` — Dense forward, loss functions, sinusoidal encoding
- `ix_ml_pipeline` with `"model": "transformer"` — Full transformer training pipeline
