# Use Case: Fraud Detection

> Combining PCA, Random Forest, and evaluation metrics to detect fraudulent credit card transactions.

## The Scenario

You work at a payment processor handling 500,000 transactions per day. About 0.1% are fraudulent — that's 500 fraudulent transactions hidden among 499,500 legitimate ones. Each transaction has 30 features: amount, time, merchant category, card age, distance from home, velocity (transactions per hour), and 24 anonymized features from a partner risk model.

Your job: flag fraudulent transactions in real-time while minimizing false positives (blocking legitimate customers is expensive).

## Why This Is Hard

**Class imbalance**: 99.9% of transactions are legitimate. A model that always predicts "not fraud" achieves 99.9% accuracy — and catches zero fraudsters. Accuracy is useless here.

**High dimensionality**: 30 features, some correlated, some noisy. The fraud signal might live in a low-dimensional subspace.

**Cost asymmetry**: Missing fraud costs ~$500 per transaction. A false positive costs ~$5 in customer friction. You need to heavily penalize missed fraud.

## The Pipeline

We'll combine three techniques:

```
Raw features (30D)
    │
    ▼
PCA (reduce to 10D)  ← Remove noise, decorrelate features
    │
    ▼
Random Forest         ← Classify fraud vs legitimate
    │
    ▼
Evaluation Metrics    ← Precision, Recall, F1 (not accuracy!)
```

## Step 1: Dimensionality Reduction with PCA

Some features are redundant (e.g., "transaction amount" and "log amount" contain the same information). PCA removes this redundancy and can help the classifier focus on the signal.

```rust
use ndarray::{Array2, Array1};
use ix_math::linalg;
use ix_unsupervised::pca::PCA;
use ix_unsupervised::traits::DimensionReducer;

// Load and standardize (critical for PCA)
let (standardized, means, stds) = linalg::standardize(&raw_features);

// Reduce 30 features to 10 principal components
let mut pca = PCA::new(10);
let reduced = pca.fit_transform(&standardized);

// Check: how much variance did we keep?
if let Some(ratios) = pca.explained_variance_ratio() {
    let total: f64 = ratios.sum();
    println!("Variance retained: {:.1}%", total * 100.0);
    // Aim for 85-95%. If too low, increase n_components.
}
```

**Why PCA here?** Fraud patterns often exist in specific combinations of features, not individual ones. PCA finds these combinations. Also, fewer features = faster training and less overfitting.

## Step 2: Classification with Random Forest

Random Forest is ideal for fraud detection:
- Handles class imbalance well (trees can still split on minority class)
- Resistant to overfitting
- Provides feature importance
- Fast inference for real-time scoring

```rust
use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::Classifier;
use ix_supervised::metrics;
use ndarray::Array1;

// Build individual trees (Random Forest = many trees voting)
// ix's ensemble crate builds on DecisionTree
let mut tree = DecisionTree::new(10)     // max depth 10
    .with_min_samples_split(5);

tree.fit(&train_features, &train_labels);
let predictions = tree.predict(&test_features);

// For a full Random Forest, train multiple trees on bootstrapped samples
// and average their predictions (majority vote for classification)
let n_trees = 100;
let mut all_predictions: Vec<Array1<usize>> = Vec::new();

for i in 0..n_trees {
    let mut tree = DecisionTree::new(8)
        .with_min_samples_split(10);

    // Bootstrap sample (sample with replacement)
    // ... create bootstrap_x, bootstrap_y ...

    tree.fit(&bootstrap_x, &bootstrap_y);
    all_predictions.push(tree.predict(&test_features));
}

// Majority vote across all trees
let final_predictions = majority_vote(&all_predictions);
```

## Step 3: Evaluation — Precision, Recall, F1

For fraud detection, the metrics that matter are:

- **Recall** (sensitivity): Of all actual frauds, what fraction did we catch?
  - Recall = 0.95 means we catch 95% of fraud. The other 5% slip through.

- **Precision**: Of all transactions we flagged as fraud, what fraction actually were?
  - Precision = 0.80 means 20% of our flags are false alarms.

- **F1 Score**: Harmonic mean of precision and recall. Balances both.

```rust
use ix_supervised::metrics;
use ndarray::array;

// Example: model predictions vs actual labels
// 0 = legitimate, 1 = fraud
let y_true = array![0, 0, 1, 0, 1, 1, 0, 0, 1, 0];
let y_pred = array![0, 0, 1, 0, 0, 1, 0, 1, 1, 0];
//                              ↑miss    ↑false alarm

let accuracy = metrics::accuracy(&y_true, &y_pred);
println!("Accuracy: {:.2}", accuracy);       // 0.80 — misleading!

let precision = metrics::precision(&y_true, &y_pred, 1);
println!("Precision: {:.2}", precision);     // 0.75 — 3 correct flags out of 4

let recall = metrics::recall(&y_true, &y_pred, 1);
println!("Recall: {:.2}", recall);           // 0.75 — caught 3 of 4 frauds

let f1 = metrics::f1_score(&y_true, &y_pred, 1);
println!("F1: {:.2}", f1);                  // 0.75
```

### The Precision-Recall Trade-off

You can't maximize both. Lowering the classification threshold catches more fraud (higher recall) but creates more false alarms (lower precision).

**For fraud detection, prioritize recall.** Missing a fraud ($500 cost) is worse than a false alarm ($5 cost). A recall of 0.98 with precision of 0.50 means you catch 98% of fraud but half your alerts are false positives — still worth it when the cost ratio is 100:1.

## Putting It All Together

```rust
// Full pipeline: load → standardize → PCA → train → evaluate

// 1. Standardize
let (std_train, means, stds) = linalg::standardize(&train_features);
// Apply same transform to test (don't fit on test!)
let std_test = (&test_features - &means) / &stds;

// 2. PCA
let mut pca = PCA::new(10);
let train_reduced = pca.fit_transform(&std_train);
let test_reduced = pca.transform(&std_test);  // transform only, don't re-fit

// 3. Train ensemble of trees
let predictions = train_random_forest(&train_reduced, &train_labels, &test_reduced, 100);

// 4. Evaluate
let recall = metrics::recall(&test_labels, &predictions, 1);
let precision = metrics::precision(&test_labels, &predictions, 1);
let f1 = metrics::f1_score(&test_labels, &predictions, 1);

println!("Fraud Detection Results:");
println!("  Recall:    {:.2} (fraction of frauds caught)", recall);
println!("  Precision: {:.2} (fraction of alerts that are real)", precision);
println!("  F1 Score:  {:.2} (balanced measure)", f1);
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Standardize before PCA | Always | PCA is scale-sensitive — large features dominate |
| PCA components | 10 out of 30 | Keep 85-95% variance, reduce noise |
| Random Forest over Logistic Regression | More accurate on nonlinear patterns | Fraud patterns are rarely linear |
| Recall over Precision | Prioritize catching fraud | Cost of missed fraud >> cost of false alarm |
| Don't fit PCA/scaler on test data | Prevent data leakage | Test set must be truly unseen |

## Pitfalls

- **Data leakage**: Never fit PCA, standardization, or any transform on the test set. Fit on training data, then apply to test data. Leaking test information into training gives unrealistically good results.
- **Accuracy is misleading**: With 99.9% legitimate transactions, a "predict all legit" model gets 99.9% accuracy. Always use precision, recall, and F1 for imbalanced problems.
- **Temporal ordering matters**: In real fraud detection, always split train/test by time (train on older data, test on newer). Random splits can leak future patterns into training.
- **Feature engineering matters more than algorithm choice**: The raw features (amount, time, velocity) often matter more than which classifier you use. Spend time on features.

## Algorithms Used

| Algorithm | Doc | Role in Pipeline |
|-----------|-----|-----------------|
| PCA | [Unsupervised: PCA](../unsupervised-learning/pca.md) | Dimension reduction, noise removal |
| Decision Tree | [Supervised: Decision Trees](../supervised-learning/decision-trees.md) | Base classifier |
| Random Forest | [Supervised: Random Forest](../supervised-learning/random-forest.md) | Ensemble classifier |
| Precision/Recall/F1 | [Supervised: Evaluation Metrics](../supervised-learning/evaluation-metrics.md) | Model evaluation |
| Standardization | [Foundations: Probability & Statistics](../foundations/probability-and-statistics.md) | Feature scaling |
