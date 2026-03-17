# Cross-Validation

## The Problem

You've built a classifier and it scores 95% accuracy on your test set. Ship it? Not so fast. What if your random train/test split happened to be lucky -- the easy examples ended up in the test set, and the hard ones in the training set? Or what if the test set accidentally contained no examples of the minority class?

A single train/test split gives you one number. That number could be high or low depending on the random split. You don't know if 95% is stable or if it would drop to 80% with a different split. You need multiple evaluations to get a reliable estimate.

Cross-validation solves this by systematically rotating which data is used for training and testing, giving you not one accuracy number but *k* of them. The average tells you how well the model generalizes. The variance tells you how stable it is.

## The Intuition

Imagine a teacher who wants to fairly test a student. Instead of one exam, they give five exams, each covering different material. The student's average score across all five exams is a much better estimate of their knowledge than any single exam score.

K-fold cross-validation does the same thing. It splits the data into k equal parts ("folds"), trains the model k times, and each time uses a different fold as the test set:

```
Fold 1: [TEST] [train] [train] [train] [train]
Fold 2: [train] [TEST] [train] [train] [train]
Fold 3: [train] [train] [TEST] [train] [train]
Fold 4: [train] [train] [train] [TEST] [train]
Fold 5: [train] [train] [train] [train] [TEST]
```

Every sample gets tested exactly once. No sample is ever in the training and test set at the same time.

## How It Works

### K-Fold

1. Shuffle the dataset (optional but recommended)
2. Split into k equally-sized folds
3. For each fold i:
   - Train on all folds except fold i
   - Evaluate on fold i
   - Record the score
4. Report the k scores, their mean, and standard deviation

### Stratified K-Fold

Standard k-fold can create problems with imbalanced data. If you have 100 samples (90 class A, 10 class B) and split into 5 folds of 20, some folds might get 0 class B samples by chance.

Stratified k-fold ensures each fold has the same class distribution as the full dataset. Each fold would get ~18 class A and ~2 class B samples.

$$
\text{Class ratio in each fold} \approx \text{Class ratio in full dataset}
$$

This is essential for:
- Imbalanced datasets (fraud detection, medical diagnosis)
- Small datasets where random variation matters more
- Any classification problem where you want reliable per-fold scores

## In Rust

### Basic K-Fold

```rust
use ix_supervised::validation::KFold;

fn main() {
    // 5-fold CV on 100 samples
    let kf = KFold::new(5).with_seed(42);
    let folds = kf.split(100);

    for (i, (train, test)) in folds.iter().enumerate() {
        println!("Fold {}: {} train, {} test", i + 1, train.len(), test.len());
    }
    // Fold 1: 80 train, 20 test
    // Fold 2: 80 train, 20 test
    // ...
}
```

### Stratified K-Fold

```rust
use ndarray::array;
use ix_supervised::validation::StratifiedKFold;

fn main() {
    // Imbalanced data: 80% class 0, 20% class 1
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    let skf = StratifiedKFold::new(5).with_seed(42);
    let folds = skf.split(&y);

    for (i, (train, test)) in folds.iter().enumerate() {
        let test_class_1 = test.iter().filter(|&&idx| y[idx] == 1).count();
        println!("Fold {}: {} train, {} test ({} class-1 in test)",
            i + 1, train.len(), test.len(), test_class_1);
    }
}
```

### One-Liner Cross-Validation Score

```rust
use ndarray::{array, Array2};
use ix_supervised::validation::cross_val_score;
use ix_supervised::decision_tree::DecisionTree;

fn main() {
    let x = Array2::from_shape_vec((12, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,  0.8, 0.1,  0.2, 0.7,
        5.0, 5.0,  5.5, 5.5,  6.0, 5.0,  5.3, 5.2,  5.8, 5.1,  5.2, 5.7,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

    // 4-fold stratified cross-validation with a decision tree
    let scores = cross_val_score(&x, &y, || DecisionTree::new(5), 4, 42);

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let std = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64).sqrt();

    println!("Fold scores: {:?}", scores);
    println!("Mean accuracy: {:.4} (+/- {:.4})", mean, std);
}
```

## When To Use This

| Situation | Use | Why |
|-----------|-----|-----|
| Model selection (comparing algorithms) | Always | Fair comparison requires averaging over multiple splits |
| Small dataset (< 1,000 samples) | Always | Single split is too noisy |
| Imbalanced classes | Stratified K-Fold | Ensures minority class appears in every fold |
| Large dataset (> 100,000 samples) | Optional | Single 80/20 split is usually stable enough |
| Hyperparameter tuning | Yes | Avoid overfitting to one particular test set |
| Final model evaluation | Yes, then retrain on all data | Get reliable estimate, then use all data for final model |

### Choosing k

| k | Behavior | When |
|---|----------|------|
| 2 | 50/50 split, high variance | Never recommended |
| 5 | Good balance of bias and variance | Default for most problems |
| 10 | Low bias, more compute | Standard in academic papers |
| n (LOO) | Leave-one-out, lowest bias | Very small datasets only |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | (required) | Number of folds. 5 or 10 is typical. |
| `shuffle` | `true` | Whether to shuffle data before splitting. |
| `seed` | `42` | Random seed for reproducibility. |

## Pitfalls

**Data leakage across folds.** Preprocessing (normalization, PCA, feature selection) must be done *inside* each fold, not before splitting. If you normalize the entire dataset first, information from the test fold leaks into training.

**Time series data breaks i.i.d. assumption.** If data has temporal order (stock prices, sensor readings), random shuffling creates leakage -- the model sees future data during training. Use time-based splits instead.

**Stratification matters more than you think.** With 5% positive class and 5-fold CV, some folds may have 0% or 10% positives by chance. Always use StratifiedKFold for classification.

**k models, not one.** Cross-validation trains k separate models. The scores estimate generalization, but you don't keep the models. After cross-validation, retrain once on *all* data for deployment.

## Going Further

- **Nested cross-validation:** Use an outer CV loop for evaluation and an inner CV loop for hyperparameter tuning. Prevents optimistic bias from tuning on the same data you evaluate on.
- **Repeated k-fold:** Run k-fold CV multiple times with different random seeds and average all scores. Reduces variance of the estimate at the cost of k * n_repeats model trainings.
- **Group k-fold:** When samples are grouped (e.g., multiple measurements per patient), ensure all samples from one group stay in the same fold.
