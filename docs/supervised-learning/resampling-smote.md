# Resampling & SMOTE

## The Problem

You're building a credit card fraud detection system. Out of 100,000 transactions in your training data, only 500 are fraudulent — a 0.5% positive rate. You train a classifier and it reports 99.5% accuracy. Impressive? No. A model that blindly predicts "not fraud" for every transaction achieves the exact same accuracy. Your model hasn't learned anything about fraud — it's learned to ignore it.

This is the class imbalance problem. When one class vastly outnumbers another, classifiers take the path of least resistance: predict the majority class for everything. The loss function sees no reason to correctly classify the 0.5% minority when getting the 99.5% majority right is so much easier.

There are two strategies: make the minority class bigger (oversampling) or make the majority class smaller (undersampling). SMOTE — Synthetic Minority Over-sampling Technique — is the most widely used oversampling method. Instead of duplicating existing minority samples (which causes overfitting to exact copies), it generates *new* synthetic samples by interpolating between existing ones.

## The Intuition

Imagine you have a map with 500 red pins (fraud) clustered in a few neighborhoods, and 99,500 blue pins (legitimate) spread everywhere. A classifier looking at this map will draw boundaries that ignore the red clusters because they're so small compared to the blue sea.

SMOTE's solution: for each red pin, find its nearest red neighbors and place new red pins along the lines connecting them. If two fraud cases are at coordinates (3, 5) and (5, 7), SMOTE might place a synthetic sample at (4, 6) — halfway between them. This fills in the "fraud neighborhood" with plausible synthetic cases, making it large enough that the classifier can't ignore it.

The key insight is that interpolating between real minority samples produces *plausible* new samples. A point between two real fraud cases likely looks like a fraud case too. This is much better than random oversampling (duplicating), which just makes the classifier memorize specific examples.

## How It Works

### SMOTE Algorithm

For each minority sample x_i:
1. Find its k nearest neighbors within the same class
2. Randomly select one neighbor x_nn
3. Generate a synthetic sample along the line segment:

$$
x_{new} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim \text{Uniform}(0, 1)
$$

4. Repeat until the minority class reaches the target count

### Random Undersampling

The simpler alternative: randomly remove majority samples until classes are balanced. Fast and easy, but throws away potentially useful data.

### Choosing a Strategy

| Strategy | Pros | Cons |
|----------|------|------|
| SMOTE (oversample minority) | No data lost, creates plausible samples | Can create noisy samples near class boundaries |
| Random undersample (reduce majority) | Fast, simple, reduces training time | Throws away potentially useful data |
| Combine both | Best of both worlds | More complex to tune |

## In Rust

### Diagnose Imbalance

```rust
use ndarray::array;
use ix_supervised::resampling::class_distribution;

fn main() {
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
    let dist = class_distribution(&y);

    for (class, count, pct) in &dist {
        println!("Class {}: {} samples ({:.1}%)", class, count, pct);
    }
    // Class 0: 8 samples (80.0%)
    // Class 1: 2 samples (20.0%)
}
```

### SMOTE Oversampling

```rust
use ndarray::{array, Array2};
use ix_supervised::resampling::{Smote, class_distribution};

fn main() {
    // Imbalanced: 8 legitimate, 2 fraud
    let x = Array2::from_shape_vec((10, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
        0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
        5.0, 5.0,  5.5, 5.5,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    println!("Before SMOTE:");
    for (c, n, pct) in class_distribution(&y) {
        println!("  Class {}: {} ({:.0}%)", c, n, pct);
    }

    let smote = Smote::new(5, 42);
    let (x_balanced, y_balanced) = smote.fit_resample(&x, &y);

    println!("After SMOTE:");
    for (c, n, pct) in class_distribution(&y_balanced) {
        println!("  Class {}: {} ({:.0}%)", c, n, pct);
    }
    // Class 0: 8 (50%), Class 1: 8 (50%)
}
```

### Random Undersampling

```rust
use ndarray::{array, Array2};
use ix_supervised::resampling::random_undersample;

fn main() {
    let x = Array2::from_shape_vec((10, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
        0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
        5.0, 5.0,  5.5, 5.5,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    let (x_under, y_under) = random_undersample(&x, &y, 42);
    // 2 class-0, 2 class-1 — balanced by removing majority samples
    println!("Undersampled: {} samples", y_under.len()); // 4
}
```

### Full Pipeline: SMOTE + Cross-Validation

```rust
use ndarray::{array, Array2};
use ix_supervised::resampling::Smote;
use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{accuracy, recall, ConfusionMatrix};

fn main() {
    // Imbalanced dataset
    let x = Array2::from_shape_vec((12, 2), vec![
        0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
        0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
        0.9, 0.4,  0.1, 0.6,
        5.0, 5.0,  5.5, 5.5,
    ]).unwrap();
    let y = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    // Step 1: Balance with SMOTE
    let smote = Smote::new(1, 42);
    let (x_bal, y_bal) = smote.fit_resample(&x, &y);

    // Step 2: Train on balanced data
    let mut tree = DecisionTree::new(5);
    tree.fit(&x_bal, &y_bal);

    // Step 3: Evaluate on original data
    let preds = tree.predict(&x);
    let cm = ConfusionMatrix::from_labels(&y, &preds, 2);

    println!("Recall (fraud): {:.4}", recall(&y, &preds, 1));
    println!("{}", cm.display());
}
```

## When To Use This

| Situation | Strategy | Why |
|-----------|----------|-----|
| Minority < 10% of data | SMOTE | Classifier will ignore minority without help |
| Minority 10-30% of data | Maybe SMOTE | Try without first; add if recall is low |
| Balanced data (40-60%) | Neither | No imbalance to fix |
| Very small minority (< 10 samples) | Undersample or collect more data | SMOTE can't interpolate well with too few seeds |
| Large dataset (100K+ majority) | Undersample + SMOTE | Reduce majority first, then SMOTE minority |
| Time series / ordered data | Be careful | SMOTE assumes i.i.d.; synthetic samples may violate temporal order |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | (required) | Number of nearest neighbors for interpolation. 5 is standard. |
| `seed` | (required) | Random seed for reproducibility. |
| `target_ratio` | `1.0` | Ratio of minority to majority. 1.0 = full balance. 0.5 = oversample to half. |

## Pitfalls

**Apply SMOTE only to training data.** Never resample the test set. Synthetic samples in the test set make evaluation meaningless — you'd be testing on fabricated data. Split first, then SMOTE the training fold.

**SMOTE near class boundaries creates noise.** If a minority sample sits right next to majority samples, SMOTE interpolates toward the boundary, creating ambiguous synthetic samples. Variants like Borderline-SMOTE and SMOTE-Tomek address this.

**Single-sample classes can't be SMOTEd.** You need at least 2 samples in a class to interpolate between them. Classes with only 1 sample are skipped.

**SMOTE doesn't help if classes aren't geometrically clustered.** The interpolation assumption is that the line between two minority samples stays in minority territory. If minority samples are scattered among majority samples, synthetic points may land in majority regions.

**Combine with appropriate metrics.** SMOTE fixes the training data, but you still need to evaluate with recall, precision, F1, and AUC — not accuracy. See [evaluation-metrics.md](./evaluation-metrics.md).

## Going Further

- **Borderline-SMOTE:** Only oversample minority samples that are near the decision boundary (the hardest cases), not samples deep inside the minority cluster.
- **SMOTE + Tomek Links:** After SMOTE, remove Tomek links (pairs of nearest neighbors from different classes) to clean up the boundary region.
- **ADASYN:** Adaptive Synthetic Sampling — generates more synthetic samples for minority examples that are harder to learn (more majority neighbors).
- **Cost-sensitive learning:** Instead of resampling, assign higher misclassification costs to the minority class. The model learns to pay more attention to rare cases without changing the data.
