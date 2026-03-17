# Customer Churn Prediction with ix_supervised

An end-to-end binary classification workflow: from raw data inspection
through cross-validated training to ROC/AUC evaluation.

## Problem

A telecom company wants to predict which subscribers will cancel their
plan next quarter. The business cost of a missed churner (false negative)
far outweighs the cost of a retention offer sent to a loyal customer
(false positive), so **recall on the churn class matters more than raw
accuracy**.

## The Data

Each subscriber is described by four features:

| Feature          | Type  | Description                              |
|------------------|-------|------------------------------------------|
| tenure_months    | f64   | Months since sign-up                     |
| monthly_charges  | f64   | Average monthly bill (USD)               |
| support_calls    | f64   | Support tickets opened in last 90 days   |
| contract_type    | f64   | 0 = month-to-month, 1 = annual contract  |

Label: `0` = stayed, `1` = churned.

Churn datasets are almost always imbalanced -- typically 20-30 % positive
class. The synthetic dataset below mirrors that ratio.

---

## Step 1 -- Inspect Class Distribution

Before training anything, check the label balance.

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::resampling::class_distribution;

// 20 subscribers: 14 stayed (0), 6 churned (1)
let y = array![0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1];
let dist = class_distribution(&y);

for (class, count, pct) in &dist {
    println!("Class {}: {} samples ({:.1}%)", class, count, pct);
}
// Class 0: 14 samples (70.0%)
// Class 1:  6 samples (30.0%)
```

A 70/30 split is moderate imbalance. A naive classifier that always
predicts "stayed" would reach 70 % accuracy while catching zero churners.
This is why accuracy alone is misleading here.

---

## Step 2 -- Cross-Validate with StratifiedKFold

Stratified splitting guarantees that each fold preserves the 70/30 ratio,
preventing folds where the minority class is absent.

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::validation::{StratifiedKFold, cross_val_score};

// Features: tenure, charges, support_calls, contract_type
let x = Array2::from_shape_vec((20, 4), vec![
    24.0, 65.0, 1.0, 1.0,
    48.0, 55.0, 0.0, 1.0,
    12.0, 70.0, 2.0, 0.0,
    36.0, 50.0, 0.0, 1.0,
    60.0, 45.0, 1.0, 1.0,
    30.0, 60.0, 0.0, 1.0,
    42.0, 52.0, 1.0, 1.0,
    18.0, 58.0, 0.0, 0.0,
    54.0, 48.0, 0.0, 1.0,
    36.0, 55.0, 1.0, 1.0,
    15.0, 62.0, 0.0, 0.0,
    45.0, 50.0, 0.0, 1.0,
    28.0, 57.0, 1.0, 1.0,
    33.0, 53.0, 0.0, 1.0,
     3.0, 80.0, 4.0, 0.0,
     6.0, 75.0, 3.0, 0.0,
     2.0, 85.0, 5.0, 0.0,
     5.0, 78.0, 3.0, 0.0,
     4.0, 82.0, 4.0, 0.0,
     7.0, 73.0, 2.0, 0.0,
]).unwrap();

let y = array![0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1];

let scores = cross_val_score(
    &x, &y,
    || LogisticRegression::new()
        .with_learning_rate(0.001)
        .with_max_iterations(2000),
    5,   // 5-fold
    42,  // seed
);

let mean = scores.iter().sum::<f64>() / scores.len() as f64;
println!("Per-fold accuracy: {:?}", scores);
println!("Mean CV accuracy:  {:.3}", mean);
```

`cross_val_score` internally uses `StratifiedKFold`, so every fold
reflects the original class proportions.

---

## Step 3 -- Train Final Model and Evaluate with ConfusionMatrix

After cross-validation confirms the model generalizes, retrain on the
full dataset and examine the confusion matrix.

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::ConfusionMatrix;

let x = Array2::from_shape_vec((20, 4), vec![
    24.0, 65.0, 1.0, 1.0,
    48.0, 55.0, 0.0, 1.0,
    12.0, 70.0, 2.0, 0.0,
    36.0, 50.0, 0.0, 1.0,
    60.0, 45.0, 1.0, 1.0,
    30.0, 60.0, 0.0, 1.0,
    42.0, 52.0, 1.0, 1.0,
    18.0, 58.0, 0.0, 0.0,
    54.0, 48.0, 0.0, 1.0,
    36.0, 55.0, 1.0, 1.0,
    15.0, 62.0, 0.0, 0.0,
    45.0, 50.0, 0.0, 1.0,
    28.0, 57.0, 1.0, 1.0,
    33.0, 53.0, 0.0, 1.0,
     3.0, 80.0, 4.0, 0.0,
     6.0, 75.0, 3.0, 0.0,
     2.0, 85.0, 5.0, 0.0,
     5.0, 78.0, 3.0, 0.0,
     4.0, 82.0, 4.0, 0.0,
     7.0, 73.0, 2.0, 0.0,
]).unwrap();

let y = array![0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1];

let mut model = LogisticRegression::new()
    .with_learning_rate(0.001)
    .with_max_iterations(2000);
model.fit(&x, &y);

let preds = model.predict(&x);
let cm = ConfusionMatrix::from_labels(&y, &preds, 2);

println!("{}", cm.display());

let (prec, rec, f1, support) = cm.classification_report();
println!("Class | Precision | Recall | F1   | Support");
for c in 0..2 {
    println!("  {}   |   {:.3}   | {:.3} | {:.3} |   {}",
        c, prec[c], rec[c], f1[c], support[c]);
}
```

Pay special attention to recall on class 1 (churn). A model with 95 %
accuracy but 40 % churn recall misses more than half the at-risk
subscribers.

---

## Step 4 -- ROC/AUC Analysis

ROC curves evaluate the classifier across all possible decision
thresholds, making them insensitive to class imbalance.

```rust
use ndarray::{array, Array1, Array2};
use ix_supervised::logistic_regression::LogisticRegression;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics::{roc_curve, roc_auc};

let x = Array2::from_shape_vec((20, 4), vec![
    24.0, 65.0, 1.0, 1.0,
    48.0, 55.0, 0.0, 1.0,
    12.0, 70.0, 2.0, 0.0,
    36.0, 50.0, 0.0, 1.0,
    60.0, 45.0, 1.0, 1.0,
    30.0, 60.0, 0.0, 1.0,
    42.0, 52.0, 1.0, 1.0,
    18.0, 58.0, 0.0, 0.0,
    54.0, 48.0, 0.0, 1.0,
    36.0, 55.0, 1.0, 1.0,
    15.0, 62.0, 0.0, 0.0,
    45.0, 50.0, 0.0, 1.0,
    28.0, 57.0, 1.0, 1.0,
    33.0, 53.0, 0.0, 1.0,
     3.0, 80.0, 4.0, 0.0,
     6.0, 75.0, 3.0, 0.0,
     2.0, 85.0, 5.0, 0.0,
     5.0, 78.0, 3.0, 0.0,
     4.0, 82.0, 4.0, 0.0,
     7.0, 73.0, 2.0, 0.0,
]).unwrap();

let y = array![0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1];

let mut model = LogisticRegression::new()
    .with_learning_rate(0.001)
    .with_max_iterations(2000);
model.fit(&x, &y);

let probas = model.predict_proba(&x);
let scores = probas.column(1).to_owned();

let (fpr, tpr, thresholds) = roc_curve(&y, &scores);
let auc = roc_auc(&fpr, &tpr);

println!("AUC = {:.3}", auc);
println!("\nSample ROC points (FPR, TPR, Threshold):");
for i in (0..fpr.len()).step_by(fpr.len() / 5) {
    println!("  ({:.3}, {:.3}, {:.3})", fpr[i], tpr[i], thresholds[i]);
}
```

An AUC of 0.5 means random guessing; 1.0 means perfect separation.
For churn prediction, an AUC above 0.80 is typically considered
production-ready.

---

## Key Takeaways

1. **Always inspect class distribution first.** `class_distribution`
   reveals whether accuracy is a trustworthy metric or a trap.
2. **Use stratified cross-validation.** `StratifiedKFold` prevents folds
   where the minority class is under-represented or absent.
3. **Read the confusion matrix, not just accuracy.** Recall on class 1
   answers: "Of all actual churners, how many did we catch?"
4. **ROC/AUC gives a threshold-free view.** ROC curves let you pick the
   right precision/recall trade-off after training.
5. **Consider resampling for severe imbalance.** When the minority class
   drops below 10-15 %, `Smote` can synthesize samples before training.

*Crates used: `ix_supervised` (logistic regression, validation, metrics, resampling).*
