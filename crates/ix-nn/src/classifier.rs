//! Trainable transformer-based classifier and regressor.
//!
//! Wraps TransformerStack + Dense head into a model implementing
//! the Classifier/Regressor traits from ix-supervised.

use ndarray::{Array1, Array2, Array3, Axis, s};
use rand::SeedableRng;
use serde::{Serialize, Deserialize};

use crate::transformer::TransformerStack;
use crate::positional::sinusoidal_encoding;
use ix_supervised::traits::{Classifier, Regressor};

/// Learning rate schedule.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum LrSchedule {
    /// Constant learning rate (no scheduling).
    #[default]
    Constant,
    /// Linear warmup for `warmup_steps` steps, then cosine decay to `min_lr`.
    WarmupCosine {
        warmup_steps: usize,
        min_lr: f64,
    },
}

/// Compute effective learning rate for a given step.
fn scheduled_lr(base_lr: f64, schedule: &LrSchedule, step: usize, total_steps: usize) -> f64 {
    match schedule {
        LrSchedule::Constant => base_lr,
        LrSchedule::WarmupCosine { warmup_steps, min_lr } => {
            if step < *warmup_steps {
                // Linear warmup: 0 → base_lr
                base_lr * (step + 1) as f64 / *warmup_steps as f64
            } else {
                // Cosine decay: base_lr → min_lr
                let decay_steps = total_steps.saturating_sub(*warmup_steps).max(1);
                let progress = (step - warmup_steps) as f64 / decay_steps as f64;
                let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                min_lr + (base_lr - min_lr) * cosine
            }
        }
    }
}

/// Configuration for a transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Model embedding dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Feed-forward hidden dimension (typically 4 * d_model).
    pub d_ff: usize,
    /// Sequence length (if None, inferred from data: seq_len = n_features / d_model).
    pub seq_len: Option<usize>,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Dropout probability (0.0 = no dropout). Applied after attention and FFN.
    pub dropout: f64,
    /// Mini-batch size. If None or >= n_samples, uses full-batch.
    pub batch_size: Option<usize>,
    /// Learning rate schedule.
    pub lr_schedule: LrSchedule,
    /// Whether to use GPU-accelerated attention when available.
    pub use_gpu: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            seq_len: None,
            epochs: 50,
            learning_rate: 0.001,
            seed: 42,
            dropout: 0.0,
            batch_size: None,
            lr_schedule: LrSchedule::Constant,
            use_gpu: false,
        }
    }
}

/// A transformer-based classifier.
///
/// Architecture: reshape -> positional encoding -> transformer stack -> mean pool -> dense -> softmax
pub struct TransformerClassifier {
    config: TransformerConfig,
    stack: Option<TransformerStack>,
    head_weights: Option<Array2<f64>>,  // (d_model, n_classes)
    head_bias: Option<Array1<f64>>,     // (n_classes,)
    pos_encoding: Option<Array2<f64>>,  // (max_seq_len, d_model)
    n_classes: usize,
    seq_len: usize,
    training_losses: Vec<f64>,
}

impl TransformerClassifier {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            config,
            stack: None,
            head_weights: None,
            head_bias: None,
            pos_encoding: None,
            n_classes: 0,
            seq_len: 0,
            training_losses: Vec::new(),
        }
    }

    /// Get training loss history.
    pub fn losses(&self) -> &[f64] {
        &self.training_losses
    }

    /// Determine sequence length from data dimensions.
    fn resolve_seq_len(&self, n_features: usize) -> usize {
        if let Some(sl) = self.config.seq_len {
            sl
        } else {
            // Auto: if n_features divisible by d_model, use that
            // Otherwise treat each feature as a position (d_model=1 effective)
            if n_features % self.config.d_model == 0 {
                n_features / self.config.d_model
            } else {
                n_features
            }
        }
    }

    /// Reshape 2D input (n_samples, n_features) to 3D (n_samples, seq_len, d_model).
    fn reshape_input(&self, x: &Array2<f64>) -> Array3<f64> {
        let n = x.nrows();
        let n_features = x.ncols();
        let seq_len = self.seq_len;
        let d_model = self.config.d_model;

        let needed = seq_len * d_model;
        let mut padded = Array2::zeros((n, needed));
        let copy_cols = n_features.min(needed);
        padded.slice_mut(s![.., ..copy_cols]).assign(&x.slice(s![.., ..copy_cols]));

        padded.into_shape_with_order((n, seq_len, d_model))
            .expect("reshape failed")
    }

    /// Mean pooling over sequence dimension: (batch, seq, d_model) -> (batch, d_model).
    fn mean_pool(x: &Array3<f64>) -> Array2<f64> {
        x.mean_axis(Axis(1)).expect("mean pool failed")
    }

    /// Softmax over last axis of 2D array.
    fn softmax_2d(x: &Array2<f64>) -> Array2<f64> {
        let mut result = x.clone();
        for mut row in result.rows_mut() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - max_val).exp());
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                row.mapv_inplace(|v| v / sum);
            }
        }
        result
    }

    /// Initialize weights using Xavier initialization.
    fn xavier_init(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let std = (2.0 / (rows + cols) as f64).sqrt();
        let dist = Normal::new(0.0, std).unwrap();
        Array2::from_shape_fn((rows, cols), |_| dist.sample(&mut rng))
    }
}

impl Classifier for TransformerClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Determine classes
        self.n_classes = *y.iter().max().unwrap_or(&0) + 1;
        self.seq_len = self.resolve_seq_len(n_features);

        let d_model = self.config.d_model;
        let seed = self.config.seed;
        let dropout_p = self.config.dropout;

        // Initialize transformer stack
        self.stack = Some(TransformerStack::new_with_dropout(
            self.config.n_layers, d_model, self.config.n_heads, self.config.d_ff, seed, dropout_p,
        ));

        // Initialize classification head
        self.head_weights = Some(Self::xavier_init(d_model, self.n_classes, seed + 999));
        self.head_bias = Some(Array1::zeros(self.n_classes));

        // Positional encoding
        self.pos_encoding = Some(sinusoidal_encoding(self.seq_len, d_model));

        // One-hot encode targets
        let mut target_onehot = Array2::zeros((n_samples, self.n_classes));
        for (i, &cls) in y.iter().enumerate() {
            if cls < self.n_classes {
                target_onehot[[i, cls]] = 1.0;
            }
        }

        // Resolve batch size
        let batch_size = self.config.batch_size
            .map(|bs| bs.min(n_samples).max(1))
            .unwrap_or(n_samples);

        // Total training steps for LR scheduling
        let steps_per_epoch = n_samples.div_ceil(batch_size);
        let total_steps = self.config.epochs * steps_per_epoch;

        // Try GPU context if requested
        let gpu_ctx = if self.config.use_gpu {
            crate::attention::try_gpu()
        } else {
            None
        };

        // Training loop
        self.training_losses.clear();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 777);
        let mut global_step = 0usize;

        for _epoch in 0..self.config.epochs {
            // Shuffle indices for mini-batch
            let mut indices: Vec<usize> = (0..n_samples).collect();
            shuffle(&mut indices, &mut rng);

            let mut epoch_loss = 0.0;
            let mut epoch_count = 0;

            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_idx = &indices[batch_start..batch_end];
                let bs = batch_idx.len();

                // Extract mini-batch
                let x_batch = select_rows(x, batch_idx);
                let target_batch = select_rows(&target_onehot, batch_idx);

                // Compute LR for this step
                let lr = scheduled_lr(
                    self.config.learning_rate,
                    &self.config.lr_schedule,
                    global_step,
                    total_steps,
                );

                // Forward: reshape -> add pos encoding -> transformer -> pool -> head -> softmax
                let x3d = self.reshape_input(&x_batch);
                let pos = self.pos_encoding.as_ref().unwrap();
                let mut encoded = x3d;
                for mut sample in encoded.outer_iter_mut() {
                    sample += pos;
                }

                // Transformer forward (with cache for backward)
                let stack = self.stack.as_mut().unwrap();
                let transformed = stack.forward_cache_gpu(&encoded, None, gpu_ctx.as_ref());

                // Mean pool -> (batch, d_model)
                let pooled = Self::mean_pool(&transformed);

                // Classification head: pooled @ W + b
                let hw = self.head_weights.as_ref().unwrap();
                let hb = self.head_bias.as_ref().unwrap();
                let logits = pooled.dot(hw) + hb;

                // Softmax
                let probs = Self::softmax_2d(&logits);

                // Cross-entropy loss
                let eps = 1e-12;
                let loss: f64 = -(0..bs)
                    .map(|i| {
                        (0..self.n_classes)
                            .map(|c| target_batch[[i, c]] * (probs[[i, c]] + eps).ln())
                            .sum::<f64>()
                    })
                    .sum::<f64>() / bs as f64;

                epoch_loss += loss * bs as f64;
                epoch_count += bs;

                // Backward: gradient of cross-entropy + softmax = probs - targets
                let grad_logits = (&probs - &target_batch) / bs as f64;

                // Head backward
                let grad_hw = pooled.t().dot(&grad_logits);
                let grad_hb = grad_logits.sum_axis(Axis(0));
                let grad_pooled = grad_logits.dot(&hw.t());

                // Update head weights
                let hw_mut = self.head_weights.as_mut().unwrap();
                *hw_mut = &*hw_mut - &(&grad_hw * lr);
                let hb_mut = self.head_bias.as_mut().unwrap();
                *hb_mut = &*hb_mut - &(&grad_hb * lr);

                // Backprop through mean pool: distribute gradient equally across sequence positions
                let seq_len = self.seq_len;
                let d_model = self.config.d_model;
                let mut grad_transformed = Array3::zeros((bs, seq_len, d_model));
                for i in 0..bs {
                    for s in 0..seq_len {
                        for d in 0..d_model {
                            grad_transformed[[i, s, d]] = grad_pooled[[i, d]] / seq_len as f64;
                        }
                    }
                }

                // Full transformer stack backward
                let stack = self.stack.as_mut().unwrap();
                stack.backward(&grad_transformed, lr);

                global_step += 1;
            }

            self.training_losses.push(epoch_loss / epoch_count as f64);
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let probs = self.predict_proba(x);
        let mut preds = Array1::zeros(x.nrows());
        for (i, row) in probs.rows().into_iter().enumerate() {
            let (max_idx, _) = row.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            preds[i] = max_idx;
        }
        preds
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let x3d = self.reshape_input(x);
        let pos = self.pos_encoding.as_ref().expect("model not fitted");
        let mut encoded = x3d;
        for mut sample in encoded.outer_iter_mut() {
            sample += pos;
        }

        let stack = self.stack.as_ref().expect("model not fitted");
        let transformed = stack.forward(&encoded, None);
        let pooled = Self::mean_pool(&transformed);

        let hw = self.head_weights.as_ref().unwrap();
        let hb = self.head_bias.as_ref().unwrap();
        let logits = pooled.dot(hw) + hb;
        Self::softmax_2d(&logits)
    }
}

/// A transformer-based regressor.
///
/// Architecture: reshape -> positional encoding -> transformer stack -> mean pool -> dense -> output
pub struct TransformerRegressor {
    config: TransformerConfig,
    stack: Option<TransformerStack>,
    head_weights: Option<Array2<f64>>,  // (d_model, 1)
    head_bias: Option<f64>,
    pos_encoding: Option<Array2<f64>>,
    seq_len: usize,
    training_losses: Vec<f64>,
}

impl TransformerRegressor {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            config,
            stack: None,
            head_weights: None,
            head_bias: None,
            pos_encoding: None,
            seq_len: 0,
            training_losses: Vec::new(),
        }
    }

    pub fn losses(&self) -> &[f64] {
        &self.training_losses
    }

    fn resolve_seq_len(&self, n_features: usize) -> usize {
        if let Some(sl) = self.config.seq_len {
            sl
        } else if n_features % self.config.d_model == 0 {
            n_features / self.config.d_model
        } else {
            n_features
        }
    }

    fn reshape_input(&self, x: &Array2<f64>) -> Array3<f64> {
        let n = x.nrows();
        let n_features = x.ncols();
        let needed = self.seq_len * self.config.d_model;
        let mut padded = Array2::zeros((n, needed));
        let copy_cols = n_features.min(needed);
        padded.slice_mut(s![.., ..copy_cols]).assign(&x.slice(s![.., ..copy_cols]));
        padded.into_shape_with_order((n, self.seq_len, self.config.d_model))
            .expect("reshape failed")
    }

    fn mean_pool(x: &Array3<f64>) -> Array2<f64> {
        x.mean_axis(Axis(1)).expect("mean pool failed")
    }

    fn xavier_init(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let std = (2.0 / (rows + cols) as f64).sqrt();
        let dist = Normal::new(0.0, std).unwrap();
        Array2::from_shape_fn((rows, cols), |_| dist.sample(&mut rng))
    }
}

impl Regressor for TransformerRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.seq_len = self.resolve_seq_len(n_features);

        let d_model = self.config.d_model;
        let seed = self.config.seed;
        let dropout_p = self.config.dropout;

        self.stack = Some(TransformerStack::new_with_dropout(
            self.config.n_layers, d_model, self.config.n_heads, self.config.d_ff, seed, dropout_p,
        ));
        self.head_weights = Some(Self::xavier_init(d_model, 1, seed + 999));
        self.head_bias = Some(0.0);
        self.pos_encoding = Some(sinusoidal_encoding(self.seq_len, d_model));

        let target = y.clone().into_shape_with_order((n_samples, 1)).unwrap();

        // Resolve batch size
        let batch_size = self.config.batch_size
            .map(|bs| bs.min(n_samples).max(1))
            .unwrap_or(n_samples);

        let steps_per_epoch = n_samples.div_ceil(batch_size);
        let total_steps = self.config.epochs * steps_per_epoch;

        let gpu_ctx = if self.config.use_gpu {
            crate::attention::try_gpu()
        } else {
            None
        };

        self.training_losses.clear();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 777);
        let mut global_step = 0usize;

        for _epoch in 0..self.config.epochs {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            shuffle(&mut indices, &mut rng);

            let mut epoch_loss = 0.0;
            let mut epoch_count = 0;

            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_idx = &indices[batch_start..batch_end];
                let bs = batch_idx.len();

                let x_batch = select_rows(x, batch_idx);
                let target_batch = select_rows(&target, batch_idx);

                let lr = scheduled_lr(
                    self.config.learning_rate,
                    &self.config.lr_schedule,
                    global_step,
                    total_steps,
                );

                let x3d = self.reshape_input(&x_batch);
                let pos = self.pos_encoding.as_ref().unwrap();
                let mut encoded = x3d;
                for mut sample in encoded.outer_iter_mut() {
                    sample += pos;
                }

                let stack = self.stack.as_mut().unwrap();
                let transformed = stack.forward_cache_gpu(&encoded, None, gpu_ctx.as_ref());
                let pooled = Self::mean_pool(&transformed);

                let hw = self.head_weights.as_ref().unwrap();
                let bias = self.head_bias.unwrap();
                let predictions = pooled.dot(hw) + bias;

                // MSE loss
                let diff = &predictions - &target_batch;
                let loss = diff.mapv(|v| v * v).mean().unwrap();
                epoch_loss += loss * bs as f64;
                epoch_count += bs;

                // MSE gradient: 2 * (pred - target) / n
                let grad_pred = &diff * (2.0 / bs as f64);

                // Head backward
                let grad_hw = pooled.t().dot(&grad_pred);
                let grad_hb: f64 = grad_pred.sum();
                let grad_pooled = grad_pred.dot(&hw.t());

                let hw_mut = self.head_weights.as_mut().unwrap();
                *hw_mut = &*hw_mut - &(&grad_hw * lr);
                *self.head_bias.as_mut().unwrap() -= lr * grad_hb;

                // Backprop through mean pool -> transformer stack
                let seq_len = self.seq_len;
                let d_model = self.config.d_model;
                let mut grad_transformed = Array3::zeros((bs, seq_len, d_model));
                for i in 0..bs {
                    for s in 0..seq_len {
                        for d in 0..d_model {
                            grad_transformed[[i, s, d]] = grad_pooled[[i, d]] / seq_len as f64;
                        }
                    }
                }
                let stack = self.stack.as_mut().unwrap();
                stack.backward(&grad_transformed, lr);

                global_step += 1;
            }

            self.training_losses.push(epoch_loss / epoch_count as f64);
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x3d = self.reshape_input(x);
        let pos = self.pos_encoding.as_ref().expect("model not fitted");
        let mut encoded = x3d;
        for mut sample in encoded.outer_iter_mut() {
            sample += pos;
        }

        let stack = self.stack.as_ref().expect("model not fitted");
        let transformed = stack.forward(&encoded, None);
        let pooled = Self::mean_pool(&transformed);

        let hw = self.head_weights.as_ref().unwrap();
        let bias = self.head_bias.unwrap();
        let predictions = pooled.dot(hw) + bias;

        predictions.column(0).to_owned()
    }
}

/// Fisher-Yates shuffle.
fn shuffle(indices: &mut [usize], rng: &mut rand::rngs::StdRng) {
    use rand::Rng;
    let n = indices.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }
}

/// Extract rows by index from a 2D array.
fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let ncols = x.ncols();
    let mut result = Array2::zeros((indices.len(), ncols));
    for (out_i, &src_i) in indices.iter().enumerate() {
        result.row_mut(out_i).assign(&x.row(src_i));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_classifier_fit_predict() {
        let config = TransformerConfig {
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            d_ff: 8,
            seq_len: Some(2),
            epochs: 10,
            learning_rate: 0.01,
            seed: 42,
            ..Default::default()
        };

        let x = Array2::from_shape_fn((20, 8), |(i, j)| {
            if i < 10 { (j as f64) * 0.1 } else { (j as f64) * -0.1 }
        });
        let y = Array1::from_vec(
            (0..10).map(|_| 0usize).chain((0..10).map(|_| 1usize)).collect()
        );

        let mut model = TransformerClassifier::new(config);
        model.fit(&x, &y);

        assert!(!model.losses().is_empty());
        // Loss should decrease
        let first = model.losses()[0];
        let last = *model.losses().last().unwrap();
        assert!(last <= first, "loss should decrease: {} -> {}", first, last);

        let preds = model.predict(&x);
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_classifier_predict_proba_sums_to_one() {
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(2), epochs: 5, learning_rate: 0.01, seed: 42,
            ..Default::default()
        };
        let x = Array2::from_shape_fn((10, 8), |(i, j)| (i * j) as f64 * 0.01);
        let y = Array1::from_vec(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);

        let mut model = TransformerClassifier::new(config);
        model.fit(&x, &y);

        let probs = model.predict_proba(&x);
        for row in probs.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "probs should sum to 1, got {}", sum);
        }
    }

    #[test]
    fn test_regressor_fit_predict() {
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(2), epochs: 20, learning_rate: 0.01, seed: 42,
            ..Default::default()
        };
        let x = Array2::from_shape_fn((20, 8), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let y = Array1::from_vec((0..20).map(|i| i as f64 * 0.5).collect());

        let mut model = TransformerRegressor::new(config);
        model.fit(&x, &y);

        assert!(!model.losses().is_empty());
        let preds = model.predict(&x);
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_regressor_loss_decreases() {
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(1), epochs: 30, learning_rate: 0.001, seed: 42,
            ..Default::default()
        };
        let x = Array2::from_shape_fn((10, 4), |(i, j)| (i + j) as f64);
        let y = Array1::from_vec((0..10).map(|i| i as f64 * 2.0).collect());

        let mut model = TransformerRegressor::new(config);
        model.fit(&x, &y);

        let losses = model.losses();
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(last <= first, "loss should decrease: {} -> {}", first, last);
    }

    #[test]
    fn test_auto_seq_len() {
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: None, epochs: 1, learning_rate: 0.01, seed: 42,
            ..Default::default()
        };

        let model = TransformerClassifier::new(config);
        assert_eq!(model.resolve_seq_len(12), 3);  // 12 / 4 = 3
        assert_eq!(model.resolve_seq_len(7), 7);   // not divisible, use n_features
    }

    // --- New tests for mini-batch, dropout, LR schedule, GPU ---

    #[test]
    fn test_mini_batch_training() {
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(2), epochs: 10, learning_rate: 0.01, seed: 42,
            batch_size: Some(5),
            ..Default::default()
        };
        let x = Array2::from_shape_fn((20, 8), |(i, j)| {
            if i < 10 { (j as f64) * 0.1 } else { (j as f64) * -0.1 }
        });
        let y = Array1::from_vec(
            (0..10).map(|_| 0usize).chain((0..10).map(|_| 1usize)).collect()
        );

        let mut model = TransformerClassifier::new(config);
        model.fit(&x, &y);

        assert!(!model.losses().is_empty());
        let preds = model.predict(&x);
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_dropout_training() {
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(2), epochs: 10, learning_rate: 0.01, seed: 42,
            dropout: 0.1,
            ..Default::default()
        };
        let x = Array2::from_shape_fn((20, 8), |(i, j)| {
            if i < 10 { (j as f64) * 0.1 } else { (j as f64) * -0.1 }
        });
        let y = Array1::from_vec(
            (0..10).map(|_| 0usize).chain((0..10).map(|_| 1usize)).collect()
        );

        let mut model = TransformerClassifier::new(config);
        model.fit(&x, &y);

        assert!(!model.losses().is_empty());
        let preds = model.predict(&x);
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_warmup_cosine_schedule() {
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(2), epochs: 20, learning_rate: 0.01, seed: 42,
            lr_schedule: LrSchedule::WarmupCosine { warmup_steps: 5, min_lr: 0.0001 },
            ..Default::default()
        };
        let x = Array2::from_shape_fn((20, 8), |(i, j)| {
            if i < 10 { (j as f64) * 0.1 } else { (j as f64) * -0.1 }
        });
        let y = Array1::from_vec(
            (0..10).map(|_| 0usize).chain((0..10).map(|_| 1usize)).collect()
        );

        let mut model = TransformerClassifier::new(config);
        model.fit(&x, &y);
        assert!(!model.losses().is_empty());
    }

    #[test]
    fn test_scheduled_lr_warmup() {
        // During warmup, LR should increase linearly
        let schedule = LrSchedule::WarmupCosine { warmup_steps: 10, min_lr: 0.0 };
        let lr0 = scheduled_lr(0.1, &schedule, 0, 100);
        let lr5 = scheduled_lr(0.1, &schedule, 4, 100);
        let lr9 = scheduled_lr(0.1, &schedule, 9, 100);
        assert!(lr0 < lr5, "LR should increase during warmup: {lr0} < {lr5}");
        assert!(lr5 < lr9, "LR should increase during warmup: {lr5} < {lr9}");
        assert!((lr9 - 0.1).abs() < 1e-10, "LR at end of warmup should equal base_lr");
    }

    #[test]
    fn test_scheduled_lr_cosine_decay() {
        let schedule = LrSchedule::WarmupCosine { warmup_steps: 0, min_lr: 0.0 };
        let lr_start = scheduled_lr(0.1, &schedule, 0, 100);
        let lr_mid = scheduled_lr(0.1, &schedule, 50, 100);
        let lr_end = scheduled_lr(0.1, &schedule, 99, 100);
        assert!(lr_start > lr_mid, "LR should decay: {lr_start} > {lr_mid}");
        assert!(lr_mid > lr_end, "LR should decay: {lr_mid} > {lr_end}");
    }

    #[test]
    fn test_scheduled_lr_constant() {
        let schedule = LrSchedule::Constant;
        assert_eq!(scheduled_lr(0.1, &schedule, 0, 100), 0.1);
        assert_eq!(scheduled_lr(0.1, &schedule, 50, 100), 0.1);
        assert_eq!(scheduled_lr(0.1, &schedule, 99, 100), 0.1);
    }

    #[test]
    fn test_gpu_training_fallback() {
        // With use_gpu=true but no GPU available, should fall back to CPU
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(2), epochs: 5, learning_rate: 0.01, seed: 42,
            use_gpu: true,
            ..Default::default()
        };
        let x = Array2::from_shape_fn((10, 8), |(i, j)| (i + j) as f64 * 0.1);
        let y = Array1::from_vec(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);

        let mut model = TransformerClassifier::new(config);
        model.fit(&x, &y);
        assert!(!model.losses().is_empty());
    }

    #[test]
    fn test_mini_batch_regressor() {
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(2), epochs: 10, learning_rate: 0.01, seed: 42,
            batch_size: Some(4),
            ..Default::default()
        };
        let x = Array2::from_shape_fn((20, 8), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let y = Array1::from_vec((0..20).map(|i| i as f64 * 0.5).collect());

        let mut model = TransformerRegressor::new(config);
        model.fit(&x, &y);
        assert!(!model.losses().is_empty());
        let preds = model.predict(&x);
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_all_features_combined() {
        // Mini-batch + dropout + warmup cosine + gpu fallback
        let config = TransformerConfig {
            d_model: 4, n_heads: 2, n_layers: 1, d_ff: 8,
            seq_len: Some(2), epochs: 10, learning_rate: 0.01, seed: 42,
            dropout: 0.1,
            batch_size: Some(5),
            lr_schedule: LrSchedule::WarmupCosine { warmup_steps: 3, min_lr: 0.0001 },
            use_gpu: true,
        };
        let x = Array2::from_shape_fn((20, 8), |(i, j)| {
            if i < 10 { (j as f64) * 0.1 } else { (j as f64) * -0.1 }
        });
        let y = Array1::from_vec(
            (0..10).map(|_| 0usize).chain((0..10).map(|_| 1usize)).collect()
        );

        let mut model = TransformerClassifier::new(config);
        model.fit(&x, &y);
        assert!(!model.losses().is_empty());
        let preds = model.predict(&x);
        assert_eq!(preds.len(), 20);
    }
}
