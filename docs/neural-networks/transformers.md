# Transformer Neural Networks in ix

## The Problem

Traditional neural networks process sequences element by element. Recurrent
networks (RNNs, LSTMs) pass hidden state forward one step at a time, which
means tokens at the start of a long sequence can be "forgotten" by the time
the network reaches the end. Convolution-based models can only see a fixed
local window.

Many real tasks require *global context*. A word at position 2 may depend on
a word at position 50. A sensor reading at time step 10 may only make sense
in light of a spike at time step 200. We need a mechanism that lets every
position attend to every other position in a single step.

Transformers solve this with **self-attention**.

## The Intuition

Imagine reading a sentence: "The cat sat on the mat because **it** was tired."
To understand what "it" refers to, you look back at every previous word and
decide which ones matter most. That is exactly what self-attention does --
every token produces a *query* ("what am I looking for?"), a *key* ("what do
I contain?"), and a *value* ("what information do I carry?"). Attention
scores are computed as the dot product of queries and keys, scaled and
softmaxed, then used to weight the values.

Multi-head attention runs several independent attention operations in
parallel, each learning to focus on a different aspect of the input --
syntax, semantics, position, and so on.

## How It Works

A single transformer block performs four steps:

1. **Multi-Head Attention** -- each head projects Q, K, V into a subspace,
   computes scaled dot-product attention, then concatenates the results.
2. **Add & LayerNorm** -- a residual connection adds the attention output
   back to the input, followed by layer normalization.
3. **Feed-Forward Network** -- two linear layers with a GELU activation
   in between (`d_model -> d_ff -> d_model`).
4. **Add & LayerNorm** -- another residual connection and normalization.

Before the first block, **positional encoding** injects sequence-order
information using sinusoidal functions (Vaswani et al. 2017). Without it,
the model would be permutation-invariant and unable to distinguish
word order.

```
Input
  |
  +-- Positional Encoding (sinusoidal)
  |
  v
[ Multi-Head Attention ] --+-- Add & LayerNorm
  |                         |
  v                         |
[ Feed-Forward (GELU) ]  --+-- Add & LayerNorm
  |
  v
Output (same shape as input)
```

Multiple blocks are stacked to form a **transformer encoder**. The ix
implementation supports arbitrary depth via `n_layers`.

## In Rust: Building Blocks

### FeedForward and TransformerBlock

The lowest-level API works directly with `ndarray::Array3<f64>` tensors
shaped `(batch, seq_len, d_model)`.

```rust
use ndarray::Array3;
use ix_nn::transformer::{TransformerBlock, FeedForward};

// A feed-forward network: 16-dim input, 64-dim hidden
let ff = FeedForward::new(16, 64, /*seed=*/42);

// A full transformer block: d_model=16, 2 heads, d_ff=64
let block = TransformerBlock::new(16, 2, 64, /*seed=*/42);

// Forward pass: batch=1, seq_len=4, d_model=16
let x = Array3::ones((1, 4, 16));
let out = block.forward(&x, None); // None = no dropout
assert_eq!(out.shape(), &[1, 4, 16]);
```

### Positional Encoding

```rust
use ix_nn::positional::sinusoidal_encoding;

// Generate a (seq_len=10, d_model=16) encoding matrix
let pe = sinusoidal_encoding(10, 16);
assert_eq!(pe.shape(), &[10, 16]);
```

### Multi-Head Attention

```rust
use ndarray::Array3;
use ix_nn::attention::{scaled_dot_product_attention, multi_head_attention};

let q = Array3::ones((1, 4, 8));
let k = Array3::ones((1, 4, 8));
let v = Array3::ones((1, 4, 8));

let (output, weights) = scaled_dot_product_attention(&q, &k, &v, None);
assert_eq!(output.shape(), &[1, 4, 8]);
assert_eq!(weights.shape(), &[1, 4, 4]); // seq_len x seq_len
```

## Training a Transformer Classifier

`TransformerConfig` wraps all hyperparameters into a single struct.
`TransformerClassifier` implements the `Classifier` trait from
`ix-supervised`, so you call `fit` and `predict` just like any other model.

```rust
use ix_nn::classifier::{TransformerConfig, TransformerClassifier, LrSchedule};
use ix_supervised::traits::Classifier;
use ndarray::Array2;

// Configure the model
let config = TransformerConfig {
    d_model: 32,
    n_heads: 4,
    n_layers: 2,
    d_ff: 128,
    seq_len: None,        // inferred: n_features / d_model
    epochs: 100,
    learning_rate: 0.001,
    seed: 42,
    dropout: 0.1,         // 10% dropout after attention and FFN
    batch_size: Some(16), // mini-batch gradient descent
    lr_schedule: LrSchedule::WarmupCosine {
        warmup_steps: 10, // linear warmup for 10 steps
        min_lr: 1e-5,     // cosine decay floor
    },
    use_gpu: false,
};

let mut model = TransformerClassifier::new(config);

// Training data: 100 samples, 64 features each
let x_train = Array2::ones((100, 64));
let y_train = vec![0usize; 50].into_iter()
    .chain(vec![1usize; 50])
    .collect::<Vec<_>>();

model.fit(&x_train, &y_train);

// Predict on new data
let x_test = Array2::ones((10, 64));
let predictions = model.predict(&x_test);
assert_eq!(predictions.len(), 10);
```

### How `seq_len` Is Inferred

When `seq_len` is `None`, the model computes it as `n_features / d_model`.
For the example above: 64 features / 32 d_model = 2 sequence positions.
Each row of `x_train` is reshaped into a `(2, 32)` matrix, then positional
encoding is added.

### Learning Rate Schedule

The `WarmupCosine` schedule avoids two common problems:

- **Cold start** -- large gradients in early training destabilize attention
  weights. Linear warmup ramps the learning rate from 0 to `learning_rate`
  over `warmup_steps` steps.
- **Overtraining** -- cosine decay smoothly reduces the learning rate toward
  `min_lr`, preventing the model from overshooting once it is near a
  minimum.

## When to Use Transformers

**Good candidates:**

- Sequence data with long-range dependencies (text, time series, genomics)
- Tabular data with many features that interact in non-obvious ways
- Problems where you need interpretable attention weights

**Consider simpler models first when:**

- You have fewer than a few hundred training samples
- Features are independent or only locally correlated
- Training time and memory are constrained (attention is O(n^2) in seq_len)

A random forest or linear regression will often outperform a transformer on
small tabular datasets. Start simple, then upgrade if the data justifies it.

## Key Parameters Reference

| Parameter       | Type              | Default   | Description                                        |
|-----------------|-------------------|-----------|----------------------------------------------------|
| `d_model`       | `usize`           | 32        | Embedding dimension. Must be divisible by `n_heads` |
| `n_heads`       | `usize`           | 4         | Number of parallel attention heads                  |
| `n_layers`      | `usize`           | 2         | Number of stacked transformer blocks                |
| `d_ff`          | `usize`           | 128       | Hidden dimension in the feed-forward network        |
| `seq_len`       | `Option<usize>`   | `None`    | Sequence length (inferred from data if `None`)      |
| `epochs`        | `usize`           | 50        | Number of full passes over the training set         |
| `learning_rate` | `f64`             | 0.001     | Base learning rate for gradient descent              |
| `seed`          | `u64`             | 42        | Random seed for weight initialization               |
| `dropout`       | `f64`             | 0.0       | Dropout probability (0.0 to 1.0)                    |
| `batch_size`    | `Option<usize>`   | `None`    | Mini-batch size (`None` = full-batch)               |
| `lr_schedule`   | `LrSchedule`      | `Constant`| Learning rate schedule (Constant or WarmupCosine)   |
| `use_gpu`       | `bool`            | `false`   | Use WGPU-accelerated attention if available         |

### Rules of Thumb

- Set `d_ff` to 4 * `d_model` as a starting point.
- Use `dropout` between 0.1 and 0.3 for regularization.
- `warmup_steps` of 5--10% of total steps works well in practice.
- `batch_size` of 16--64 balances gradient noise and computation.
- Enable `use_gpu` for large models or long sequences; for small problems
  the CPU path is often faster due to GPU kernel launch overhead.

## Further Reading

- Vaswani et al., "Attention Is All You Need" (2017) -- the original paper.
- The ix source: `crates/ix-nn/src/attention.rs`, `transformer.rs`,
  `classifier.rs`, `positional.rs`.
- The `ix-demo` crate includes an interactive transformer demo tab.
