---
date: 2026-03-15
topic: transformer-training
---

# Trainable Transformers for ix-nn + ix_ml_pipeline

## What We're Building

End-to-end trainable transformer models in ix-nn that implement the `Classifier` and `Regressor` traits from ix-supervised, enabling `"model": "transformer"` in `ix_ml_pipeline`. Supports both tabular and sequence data.

Two deliverables:
1. **Backward passes** for all transformer components (attention, LayerNorm, FeedForward) so the full transformer stack is trainable via gradient descent
2. **TransformerClassifier / TransformerRegressor** structs that wrap embedding → positional encoding → transformer blocks → classification/regression head, implementing fit/predict

## Why This Approach

We chose full backpropagation over simpler alternatives (numerical gradients, feature extraction) because:
- ix is a real ML framework, not a toy — training should work properly
- Numerical gradients are O(params) slower — unusable for real models
- Feature extraction (frozen weights) limits what the transformer can learn
- The Dense layer already has a working backward pass — the pattern exists, we extend it

## Key Decisions

- **Full backprop**: Implement backward for MultiHeadAttention, LayerNorm, FeedForward, TransformerBlock
- **Layer trait integration**: Make transformer components implement the existing `Layer` trait (forward/backward with &mut self)
- **2D interface for traits**: `Classifier`/`Regressor` traits use `Array2<f64>`. For sequence data, the transformer internally reshapes `(n_samples, seq_len * d_model)` → `(n_samples, seq_len, d_model)` → processes → pools → `(n_samples, n_classes)`
- **Tabular mode**: Each row treated as a length-1 sequence (or features split into patches)
- **Sequence mode**: Each row is a flattened sequence — user specifies `seq_len` and the pipeline reshapes
- **Pooling**: Use CLS token pooling (prepend a learnable CLS token, use its output) or mean pooling over sequence positions
- **Model serialization**: All weights serializable via the existing ModelEnvelope pattern

## Architecture

```
Input: Array2<f64> (n_samples, n_features)
  ↓
Reshape: (n_samples, seq_len, d_model)  where n_features = seq_len * d_model
  ↓
Add positional encoding (sinusoidal)
  ↓
TransformerBlock × N layers (with backward pass)
  ↓
Pool: mean over seq_len → (n_samples, d_model)
  ↓
Dense head: (n_samples, d_model) → (n_samples, n_classes) or (n_samples, 1)
  ↓
Softmax (classification) or identity (regression)
```

## Backward Passes Needed

| Component | Forward | Backward | Complexity |
|-----------|---------|----------|------------|
| LayerNorm | `(x-μ)/σ * γ + β` | Chain rule through norm, grad for γ, β | Medium |
| FeedForward | `GELU(x·W1+b1)·W2+b2` | Standard dense backprop + GELU derivative | Medium |
| ScaledDotProductAttention | `softmax(QK^T/√d)V` | Grad through softmax, matmuls for Q,K,V grads | Hard |
| MultiHeadAttention | Split→attend→concat→project | Aggregate head grads, project back | Hard |
| TransformerBlock | Norm→Attn→Residual→Norm→FFN→Residual | Chain all components, residual passthrough | Medium (composition) |

## Parameters for ix_ml_pipeline

```json
{
  "model": "transformer",
  "model_params": {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 128,
    "seq_len": null,
    "epochs": 50,
    "learning_rate": 0.001,
    "pooling": "mean"
  }
}
```

When `seq_len` is null: auto-detect. For tabular data, treat each feature as a position (seq_len = n_features, d_model = 1) or use patch-based (split features into groups).

## Governance (Article 4: Proportionality)

Transformers are expensive. The skill should warn when simpler models would suffice:
- < 1000 rows → "Consider LinearRegression or DecisionTree instead"
- < 10 features → "Transformer adds overhead for low-dimensional data"
- Tabular data with no sequential structure → "DecisionTree or RandomForest may perform better"

## Open Questions

1. **Batch training**: Current Sequential does full-batch gradient descent. Should we add mini-batch SGD? (Leaning yes for transformers — full batch is memory-intensive)
2. **Learning rate schedule**: Transformers typically need warmup. Add a simple linear warmup? (Leaning defer to v2)
3. **Dropout**: Not implemented in ix-nn. Add it? (Leaning defer — regularization via early stopping for now)

## Next Steps

→ `/ce:plan` for implementation phases:
1. Backward passes for LayerNorm, FeedForward, attention
2. TransformerClassifier/TransformerRegressor structs
3. Integration with ix_ml_pipeline
4. Serialization + skill update
