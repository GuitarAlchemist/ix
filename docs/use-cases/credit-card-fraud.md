# Use Case: Credit Card Fraud Detection with SMOTE

> Tackling extreme class imbalance with synthetic oversampling and gradient boosting.

## The Problem

You receive 100,000 credit card transactions per day. Only 1% are fraudulent
-- that is 1,000 fraudulent transactions hidden among 99,000 legitimate ones.
A naive model that always predicts "legitimate" achieves 99% accuracy and
catches zero fraud. **Accuracy is a lie when classes are imbalanced.**

The key insight: we need to teach the model what fraud *looks like* by giving
it enough examples. SMOTE (Synthetic Minority Over-sampling Technique)
generates realistic synthetic fraud samples so the classifier can learn
meaningful decision boundaries.

## The Data

Each transaction has four features:

| Feature       | Description                        | Range       |
|---------------|------------------------------------|-------------|
| `amount`      | Transaction amount in dollars      | 0.5 -- 5000 |
| `hour`        | Hour of day (0--23)                | 0 -- 23     |
| `distance_km` | Distance from cardholder home      | 0 -- 500    |
| `card_present`| Physical card used (1) or not (0)  | 0 or 1      |

Labels: `0` = legitimate, `1` = fraud.

## Step 1: Diagnose the Imbalance

Before doing anything, measure the class distribution:

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::resampling::class_distribution;

// Simulated dataset: 20 legitimate, 2 fraudulent (10:1 imbalance)
let y = array![
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1,
];

let dist = class_distribution(&y);
for (class, count, pct) in &dist {
    let label = if *class == 0 { "Legitimate" } else { "Fraud" };
    println!("{}: {} samples ({:.1}%)", label, count, pct);
}
// Legitimate: 20 samples (90.9%)
// Fraud:       2 samples  (9.1%)
```

The classifier will see 10 legitimate transactions for every fraud -- it will
learn to ignore the minority class entirely.

## Step 2: Apply SMOTE to Balance Training Data

SMOTE generates synthetic fraud samples by interpolating between existing
fraud examples and their k-nearest neighbors. The key: **apply SMOTE only
to the training set**, never to the test set.

```rust
use ix_supervised::resampling::{Smote, class_distribution};

// Features: [amount, hour, distance_km, card_present]
// 20 legitimate + 2 fraudulent transactions
let x = Array2::from_shape_vec((22, 4), vec![
    // -- Legitimate transactions (class 0) --
    45.0, 10.0,  2.0, 1.0,   120.0, 14.0,  5.0, 1.0,
    30.0, 9.0,   1.0, 1.0,    85.0, 16.0,  3.0, 1.0,
    15.0, 11.0,  0.5, 1.0,   200.0, 13.0,  8.0, 1.0,
    55.0, 15.0,  4.0, 1.0,    70.0, 17.0,  2.5, 1.0,
    25.0, 12.0,  1.5, 1.0,    90.0, 10.0,  3.5, 1.0,
    40.0, 8.0,   2.0, 1.0,   110.0, 14.0,  6.0, 1.0,
    35.0, 9.0,   1.0, 1.0,    60.0, 16.0,  2.0, 1.0,
    20.0, 11.0,  0.8, 1.0,   150.0, 13.0,  7.0, 1.0,
    50.0, 15.0,  3.0, 1.0,    75.0, 17.0,  4.0, 1.0,
    28.0, 12.0,  1.2, 1.0,    95.0, 10.0,  5.0, 1.0,
    // -- Fraudulent transactions (class 1) --
    980.0, 3.0, 350.0, 0.0,  1500.0, 2.0, 420.0, 0.0,
]).unwrap();

let y = array![
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1,
];

// Apply SMOTE: k=1 (only 2 minority samples), seed=42
let smote = Smote::new(1, 42);
let (x_bal, y_bal) = smote.fit_resample(&x, &y);

// Compare before/after
println!("Before SMOTE:");
for (cls, count, pct) in class_distribution(&y) {
    println!("  Class {}: {} ({:.1}%)", cls, count, pct);
}

println!("After SMOTE:");
for (cls, count, pct) in class_distribution(&y_bal) {
    println!("  Class {}: {} ({:.1}%)", cls, count, pct);
}
// Before: Class 0: 20 (90.9%), Class 1: 2 (9.1%)
// After:  Class 0: 20 (50.0%), Class 1: 20 (50.0%)
```

SMOTE created 18 synthetic fraud samples by interpolating between the two
original fraud transactions. The classifier now sees an equal number of both
classes.

## Step 3: Train a Gradient Boosted Classifier

Gradient boosting builds an ensemble of weak learners (decision stumps) that
sequentially correct each other's mistakes -- ideal for complex fraud patterns.

```rust
use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
use ix_ensemble::traits::EnsembleClassifier;

// Train on the SMOTE-balanced data
let mut gbc = GradientBoostedClassifier::new(
    100,  // 100 boosting rounds
    0.1,  // learning rate
    3,    // max depth
);
gbc.fit(&x_bal, &y_bal);

// Predict on original (unbalanced) data as a smoke test
let predictions = gbc.predict(&x);
let probabilities = gbc.predict_proba(&x);

// Check fraud probabilities for the known fraud samples
println!("Fraud probability for sample 20: {:.3}", probabilities[[20, 1]]);
println!("Fraud probability for sample 21: {:.3}", probabilities[[21, 1]]);
```

## Step 4: Evaluate with Confusion Matrix, Precision, Recall, F1, and AUC

Never trust accuracy alone. Use metrics that reveal how well the model
detects the minority class.

```rust
use ix_supervised::metrics::{
    ConfusionMatrix, precision, recall, f1_score, auc_score,
};

let y_pred = gbc.predict(&x);

// Confusion matrix
let cm = ConfusionMatrix::from_labels(&y, &y_pred, 2);
println!("{}", cm.display());
// pred ->  0    1
// true 0: 20    0   (all legitimate correctly classified)
// true 1:  0    2   (both frauds caught)

let (prec_vec, rec_vec, f1_vec, support) = cm.classification_report();
println!("Fraud precision: {:.3}", prec_vec[1]);
println!("Fraud recall:    {:.3}", rec_vec[1]);
println!("Fraud F1:        {:.3}", f1_vec[1]);

// AUC using predicted probabilities
let y_scores = Array1::from_iter(
    (0..x.nrows()).map(|i| probabilities[[i, 1]])
);
let auc = auc_score(&y, &y_scores);
println!("AUC: {:.3}", auc);
```

## Step 5: Compare With and Without SMOTE

The real payoff -- showing that SMOTE dramatically improves fraud recall.

```rust
// --- WITHOUT SMOTE: train on imbalanced data ---
let mut gbc_no_smote = GradientBoostedClassifier::new(100, 0.1);
gbc_no_smote.fit(&x, &y);  // original imbalanced data
let pred_no_smote = gbc_no_smote.predict(&x);
let recall_no_smote = recall(&y, &pred_no_smote, 1);

// --- WITH SMOTE: train on balanced data ---
let mut gbc_smote = GradientBoostedClassifier::new(100, 0.1);
gbc_smote.fit(&x_bal, &y_bal);  // SMOTE-balanced data
let pred_smote = gbc_smote.predict(&x);
let recall_smote = recall(&y, &pred_smote, 1);

println!("Recall WITHOUT SMOTE: {:.3}", recall_no_smote);
println!("Recall WITH SMOTE:    {:.3}", recall_smote);
// Without SMOTE the model often misses fraud entirely (recall ~ 0.0-0.5)
// With SMOTE the model catches most or all fraud (recall ~ 0.5-1.0)
```

## Key Takeaways

1. **Accuracy lies with imbalanced data.** A 99% accuracy score means nothing
   when 99% of transactions are legitimate. Always use precision, recall, F1,
   and AUC for imbalanced problems.

2. **SMOTE + gradient boosting is a powerful combination.** SMOTE gives the
   classifier enough minority examples to learn from. Gradient boosting
   sequentially focuses on hard-to-classify samples -- exactly what fraud
   transactions are.

3. **Apply SMOTE only to training data.** Never resample the test set. The
   test set must reflect real-world class proportions to give honest metrics.

4. **Recall is king for fraud detection.** Missing a fraud costs far more than
   a false alarm. Optimize for recall first, then tune precision to reduce
   false positives to an acceptable level.

5. **Monitor class distribution before and after resampling.** The
   `class_distribution` function makes it easy to verify that SMOTE achieved
   the desired balance.

## Algorithms Used

| Algorithm | Crate | Role |
|-----------|-------|------|
| SMOTE | `ix_supervised::resampling` | Balance training data |
| Gradient Boosted Classifier | `ix_ensemble::gradient_boosting` | Classification |
| Confusion Matrix | `ix_supervised::metrics` | Error analysis |
| Precision / Recall / F1 | `ix_supervised::metrics` | Class-specific evaluation |
| AUC | `ix_supervised::metrics` | Threshold-independent evaluation |
| `class_distribution` | `ix_supervised::resampling` | Imbalance diagnosis |

## Related Docs

- [SMOTE and Resampling](../supervised-learning/resampling-smote.md)
- [Gradient Boosting](../supervised-learning/gradient-boosting.md)
- [Evaluation Metrics](../supervised-learning/evaluation-metrics.md)
- [Cross-Validation](../supervised-learning/cross-validation.md)
- [Fraud Detection (PCA + Random Forest)](fraud-detection.md)
