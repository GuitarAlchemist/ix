# Gradient Boosting

## The Problem

Random forests average many independent trees to reduce variance. But what if instead of training trees in parallel on random subsets, you trained them one after another, with each new tree specifically targeting the mistakes of all previous trees?

That's gradient boosting. Where random forests fight *variance* (noisy, overfitting predictions), gradient boosting fights *bias* (underfitting, systematically wrong predictions). Each tree is small and weak on its own, but the sequence of trees compounds into a strong learner.

Gradient boosting consistently tops leaderboards on tabular data -- from Kaggle competitions to real-world fraud detection and click prediction systems. XGBoost, LightGBM, and CatBoost are all implementations of this core idea.

## The Intuition

Imagine learning to throw darts. On your first throw, you miss the bullseye by a lot -- you hit the upper-left. On your second throw, you *correct* for that error by aiming lower-right. On your third throw, you correct for the remaining error. Each throw isn't aimed at the bullseye -- it's aimed at fixing the accumulated error from all previous throws.

Gradient boosting works the same way:
1. Start with a simple prediction (e.g., predict the most common class for everyone)
2. Compute the errors (residuals) -- where is the model wrong?
3. Train a small tree to predict those errors
4. Add the tree's predictions (scaled by a learning rate) to the running total
5. Repeat: compute new residuals, train another tree, add it in

After 50-100 rounds, you have a model that started simple and gradually corrected all its mistakes.

## How It Works

### The Algorithm (for classification)

1. **Initialize** with class prior probabilities (log-odds):
$$
F_0(x) = \log\left(\frac{\text{count}(c) + 1}{n + K}\right) \quad \text{for each class } c
$$

2. **For each round** t = 1, ..., T:
   - Compute probabilities via softmax:
   $$
   p_c(x_i) = \frac{e^{F_c(x_i)}}{\sum_j e^{F_j(x_i)}}
   $$
   - Compute pseudo-residuals (negative gradient of log-loss):
   $$
   r_{ic} = y_{ic} - p_c(x_i)
   $$
   where $y_{ic}$ is 1 if sample i belongs to class c, else 0.
   - Fit a regression stump $h_{tc}$ to the residuals $r_{ic}$ for each class
   - Update:
   $$
   F_c(x) \leftarrow F_c(x) + \eta \cdot h_{tc}(x)
   $$
   where $\eta$ is the learning rate

3. **Predict** by taking argmax of softmax(F(x))

### Why "Gradient" Boosting?

The pseudo-residuals are the *negative gradient* of the loss function. Each tree performs one step of gradient descent in function space. Instead of updating model parameters (like in neural networks), we add a new function (tree) that points in the steepest descent direction.

### The Learning Rate

The learning rate $\eta$ (typically 0.01-0.3) controls how much each tree contributes. Lower learning rates require more trees but produce smoother, more generalizable models. This is sometimes called "shrinkage."

$$
\text{Model} = F_0 + \eta \cdot h_1 + \eta \cdot h_2 + \cdots + \eta \cdot h_T
$$

## In Rust

### Binary Classification

```rust
use ndarray::{array, Array2};
use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
use ix_ensemble::traits::EnsembleClassifier;
use ix_supervised::metrics::{accuracy, precision, recall, f1_score};

fn main() {
    // Fraud detection features: [amount, hour, distance_km, card_present]
    let x = Array2::from_shape_vec((8, 4), vec![
        25.0,  12.0,  2.0,  1.0,    // legitimate
        15.0,  14.0,  1.0,  1.0,    // legitimate
        120.0, 10.0,  5.0,  1.0,    // legitimate
        45.0,  18.0,  0.5,  1.0,    // legitimate
        4500.0, 3.0, 800.0, 0.0,    // fraud
        2200.0, 2.0, 500.0, 0.0,    // fraud
        3100.0, 4.0, 650.0, 0.0,    // fraud
        5000.0, 1.0, 900.0, 0.0,    // fraud
    ]).unwrap();
    let y = array![0, 0, 0, 0, 1, 1, 1, 1];

    let mut gbc = GradientBoostedClassifier::new(50, 0.1, 3);
    gbc.fit(&x, &y);

    let preds = gbc.predict(&x);
    let proba = gbc.predict_proba(&x);

    println!("Predictions: {}", preds);
    println!("Accuracy: {:.4}", accuracy(&y, &preds));
    println!("Precision (fraud): {:.4}", precision(&y, &preds, 1));
    println!("Recall (fraud): {:.4}", recall(&y, &preds, 1));

    // Probability estimates
    for i in 0..x.nrows() {
        println!("  Sample {}: {:.1}% fraud", i, proba[[i, 1]] * 100.0);
    }
}
```

### Comparing with Random Forest

```rust
use ndarray::{array, Array2};
use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
use ix_ensemble::random_forest::RandomForest;
use ix_ensemble::traits::EnsembleClassifier;
use ix_supervised::validation::cross_val_score;
use ix_supervised::decision_tree::DecisionTree;

fn main() {
    let x = Array2::from_shape_vec((12, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,  0.8, 0.1,  0.2, 0.7,
        5.0, 5.0,  5.5, 5.5,  6.0, 5.0,  5.3, 5.2,  5.8, 5.1,  5.2, 5.7,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

    // Cross-validate a decision tree
    let dt_scores = cross_val_score(&x, &y, || DecisionTree::new(5), 4, 42);
    let dt_mean = dt_scores.iter().sum::<f64>() / dt_scores.len() as f64;

    println!("Decision Tree:     {:.4} (+/- {:.4})",
        dt_mean, std_dev(&dt_scores, dt_mean));

    // Note: cross_val_score works with any Classifier trait implementor.
    // For ensembles, you'd typically use the full train/test approach:
    let mut rf = RandomForest::new(50, 5).with_seed(42);
    rf.fit(&x, &y);
    println!("Random Forest acc: {:.4}",
        ix_supervised::metrics::accuracy(&y, &rf.predict(&x)));

    let mut gbc = GradientBoostedClassifier::new(50, 0.1, 3);
    gbc.fit(&x, &y);
    println!("Gradient Boost acc: {:.4}",
        ix_supervised::metrics::accuracy(&y, &gbc.predict(&x)));
}

fn std_dev(scores: &[f64], mean: f64) -> f64 {
    (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64).sqrt()
}
```

## When To Use This

| Situation | Gradient Boosting | Alternative | Why |
|-----------|-------------------|-------------|-----|
| Maximum accuracy on tabular data | Yes | -- | Consistently best performer |
| Quick baseline, minimal tuning | No | Random Forest | GBM needs LR tuning |
| Very small dataset (< 50 samples) | No | KNN, Logistic Reg | Risk of overfitting |
| Need probability calibration | Decent | Logistic Regression | GBM softmax outputs are reasonably calibrated |
| Real-time low-latency prediction | Maybe | Linear model | Prediction is sequential through trees |
| Interpretability required | No | Decision Tree | Ensemble of trees is a black box |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | (required) | Number of boosting rounds. 50-200 typical. |
| `learning_rate` | (required) | Step size shrinkage. Lower = more stable but needs more rounds. |
| `max_depth` | (required) | Depth of each weak learner. 1-3 typical for boosting. |
| `min_samples_leaf` | `1` | Minimum samples required in each leaf node. |

### Tuning Guide

| learning_rate | n_estimators | Behavior |
|---------------|-------------|----------|
| 0.3-0.5 | 20-50 | Fast training, risk of overfitting |
| 0.1 | 50-200 | Good default |
| 0.01-0.05 | 200-1000 | Best generalization, slow training |

**Rule of thumb:** Lower the learning rate and increase the number of estimators together. `lr=0.1, n=100` and `lr=0.01, n=1000` often give similar results, but the latter generalizes better.

## Pitfalls

**Overfitting with too many rounds.** Unlike random forests where more trees never hurts, gradient boosting can overfit if you add too many rounds. Use cross-validation to find the optimal number.

**Learning rate too high.** A high learning rate makes each tree too influential. The model oscillates around the optimum instead of converging smoothly. Start with 0.1 and decrease if needed.

**Deep trees defeat the purpose.** The power of boosting comes from combining many *weak* learners. Using deep trees (max_depth > 5) makes each learner too strong, defeating the ensemble effect and risking overfitting. Depth 1-3 is typical.

**Sensitive to outliers.** The squared-error loss amplifies outliers. A single extreme residual can dominate a tree's split. Robust losses (Huber, quantile) help but aren't implemented yet.

**Sequential, not parallel.** Each tree depends on the previous ones, so training is inherently sequential. Random forests can train all trees in parallel.

## Going Further

- **Early stopping:** Monitor validation loss and stop when it starts increasing. This automatically finds the optimal number of rounds.
- **Feature importance:** Like random forests, count how often each feature is used for splitting, weighted by the impurity decrease.
- **Regularization:** Besides learning rate, you can regularize by limiting max_depth, min_samples_leaf, or adding L2 penalty on leaf values.
- **Regression support:** The current implementation is classification-only. Gradient boosting for regression uses squared-error loss instead of log-loss, with the same sequential tree-fitting framework.
- **Histogram-based splitting:** LightGBM bins continuous features into histograms for O(n) instead of O(n log n) split finding. Much faster on large datasets.
