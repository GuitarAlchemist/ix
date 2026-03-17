---
name: ix-random-forest
description: Random forest and gradient boosted trees — ensemble classifiers for tabular data
---

# Ensemble Methods

Ensemble classifiers using decision tree base learners.

## When to Use
When the user needs classification with probability estimates, wants a robust baseline classifier, has tabular data, or needs gradient boosting for high accuracy.

## Algorithms

### Random Forest (Bagging)
- Each tree trained on bootstrap sample with random feature subsets
- Predictions via majority vote — reduces variance
- More trees → better performance (diminishing returns)
- Best for: stable baseline, feature importance, parallel training

### Gradient Boosted Trees (Boosting)
- Sequentially fits trees to pseudo-residuals of log-loss
- Learning rate controls step size — lower = smoother but more rounds needed
- Best for: maximum accuracy on tabular data, tunable bias-variance tradeoff
- Supports binary and multiclass classification

## Choosing Between Them

| Criterion | Random Forest | Gradient Boosting |
|-----------|--------------|-------------------|
| Tuning effort | Low (just n_trees) | Medium (LR + n_estimators) |
| Overfitting risk | Low | Higher (use low LR) |
| Accuracy ceiling | Good | Often better |
| Training speed | Fast (parallelizable) | Sequential rounds |

## Programmatic Usage
```rust
use ix_ensemble::random_forest::RandomForest;
use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
use ix_ensemble::traits::EnsembleClassifier;

// Random Forest
let mut rf = RandomForest::new(100, 10).with_seed(42);
rf.fit(&x_train, &y_train);
let preds = rf.predict(&x_test);

// Gradient Boosting
let mut gbc = GradientBoostedClassifier::new(50, 0.1);
gbc.fit(&x_train, &y_train);
let preds = gbc.predict(&x_test);
let probas = gbc.predict_proba(&x_test);
```

## MCP Tools
- `ix_random_forest` — Parameters: `x_train`, `y_train`, `x_test`, `n_trees`, `max_depth`
- `ix_gradient_boosting` — Parameters: `x_train`, `y_train`, `x_test`, `n_estimators`, `learning_rate`, `max_depth`
