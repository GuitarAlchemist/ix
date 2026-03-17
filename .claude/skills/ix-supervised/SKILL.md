---
name: ix-supervised
description: Supervised learning — regression, classification, evaluation metrics, cross-validation, confusion matrix, ROC/AUC, SMOTE resampling
---

# Supervised Learning

Train and evaluate classification and regression models with full evaluation toolkit.

## When to Use
When the user asks to classify data, predict values, train a model, evaluate accuracy/precision/recall, compute confusion matrices, ROC/AUC curves, cross-validate, handle imbalanced data, or compare ML algorithms.

## Algorithm Selection
- **Linear Regression** — Continuous target, linear relationship, OLS normal equation
- **Logistic Regression** — Binary/multiclass classification, gradient descent
- **SVM** — Binary classification with margin maximization, hinge loss
- **KNN** — Instance-based, no training phase, works with any distribution
- **Naive Bayes** — Fast classification assuming feature independence, good for text
- **Decision Tree** — Non-linear boundaries, interpretable rules, Gini impurity splits

## Evaluation Toolkit

### Metrics
- **Regression**: MSE, RMSE, MAE, R²
- **Classification**: Accuracy, precision, recall, F1 (per-class, macro, weighted)
- **Probabilistic**: Log loss (binary cross-entropy)

### Confusion Matrix
Full confusion matrix with TP/FP/FN/TN per class, classification report (precision/recall/F1/support), and display formatting.

### ROC / AUC
ROC curve (FPR vs TPR at varying thresholds) and Area Under Curve for binary classification. Use `auc_score()` for one-liner evaluation.

### Cross-Validation
- **KFold** — Standard k-fold splits with optional shuffle
- **StratifiedKFold** — Preserves class distribution (essential for imbalanced data)
- **cross_val_score()** — One-liner: returns per-fold accuracy for any Classifier

## Programmatic Usage
```rust
use ix_supervised::linear_regression::LinearRegression;
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::svm::LinearSVM;
use ix_supervised::knn::KNN;
use ix_supervised::naive_bayes::GaussianNaiveBayes;
use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::{Regressor, Classifier};
use ix_supervised::metrics;
use ix_supervised::metrics::{ConfusionMatrix, roc_curve, roc_auc, auc_score, Average};
use ix_supervised::validation::{KFold, StratifiedKFold, cross_val_score};
use ix_supervised::resampling::{Smote, random_undersample, class_distribution};
```

## Examples

### Confusion Matrix
```rust
use ndarray::array;
use ix_supervised::metrics::ConfusionMatrix;

let y_true = array![0, 0, 1, 1, 2, 2];
let y_pred = array![0, 1, 1, 1, 2, 0];
let cm = ConfusionMatrix::from_labels(&y_true, &y_pred, 3);

println!("{}", cm.display());
let (prec, rec, f1, support) = cm.classification_report();
```

### ROC/AUC
```rust
use ndarray::array;
use ix_supervised::metrics::auc_score;

let y_true = array![0, 0, 1, 1];
let y_scores = array![0.1, 0.2, 0.8, 0.9];
let auc = auc_score(&y_true, &y_scores);
// auc = 1.0 (perfect separation)
```

### Cross-Validation
```rust
use ndarray::{array, Array2};
use ix_supervised::validation::cross_val_score;
use ix_supervised::decision_tree::DecisionTree;

let x = Array2::from_shape_vec((8, 2), vec![
    0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.3, 0.2,
    5.0, 5.0, 5.5, 5.5, 6.0, 5.0, 5.3, 5.2,
]).unwrap();
let y = array![0, 0, 0, 0, 1, 1, 1, 1];

let scores = cross_val_score(&x, &y, || DecisionTree::new(5), 4, 42);
let mean = scores.iter().sum::<f64>() / scores.len() as f64;
```

### Resampling (SMOTE)
- **Smote** — Synthetic Minority Over-sampling Technique (interpolates between minority neighbors)
- **random_undersample()** — Reduce majority class to match minority
- **class_distribution()** — Diagnose class imbalance

### SMOTE Example
```rust
use ndarray::{array, Array2};
use ix_supervised::resampling::{Smote, class_distribution};

// 8 legitimate (class 0), 2 fraud (class 1) — severe imbalance
let x = Array2::from_shape_vec((10, 2), vec![
    0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.3, 0.2,
    0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4, 0.8,
    5.0, 5.0, 5.5, 5.5,
]).unwrap();
let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

// Before: class 0 = 80%, class 1 = 20%
let dist = class_distribution(&y);

let smote = Smote::new(5, 42);
let (x_balanced, y_balanced) = smote.fit_resample(&x, &y);
// After: class 0 = 50%, class 1 = 50%
```

## MCP Tool Reference
Tool: `ix_supervised`
Operations: `linear_regression`, `logistic_regression`, `svm`, `knn`, `naive_bayes`, `decision_tree`, `metrics`, `cross_validate`, `confusion_matrix`, `roc_auc`
