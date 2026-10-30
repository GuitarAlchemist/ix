# PCA (Principal Component Analysis)

> Reduce the number of features while keeping the most important information — see high-dimensional data in 2D or 3D.

## The Problem

You're analyzing a dataset with 50 features per sample — income, age, purchase history, browsing patterns, location data, and more. You want to:

1. **Visualize** the data (you can't plot 50 dimensions)
2. **Speed up** downstream algorithms (KNN with 50 features is slow)
3. **Remove noise** (some features are redundant or noisy)

You need to reduce 50 features to a handful — say 2 or 3 — while losing as little information as possible. PCA finds the best low-dimensional summary of your data.

## The Intuition

Imagine photographing a 3D object. One photo from the front captures most of the interesting detail. A photo from the side adds a bit more. A photo from the top might add very little because the object is flat. PCA finds the "best camera angles" for your data.

More precisely: your 50 features contain redundancy. If income and spending are highly correlated, you can almost capture both with one number (a "wealth" axis). PCA finds these natural axes — the directions in which your data varies the most — and ranks them by importance.

**Principal Component 1**: The direction with the most variance (most spread). This single axis captures the most information possible.

**Principal Component 2**: The direction with the most remaining variance, perpendicular to PC1.

And so on. Typically the first 2-3 components capture 80-95% of the total variance, and you can throw away the rest.

## How It Works

**Input**: Dataset X (n samples, d features), target number of components k.

**Steps:**

1. **Center the data**: Subtract the mean of each feature so the data is centered at zero.

   `X_centered = X - mean(X)`

   In plain English: shift the data so its center is at the origin. This doesn't lose information — it just makes the math cleaner.

2. **Compute the covariance matrix**: A d×d matrix that captures how features relate to each other.

   `C = (1/n) × X_centered^T × X_centered`

   In plain English: entry C[i][j] tells you how much feature i and feature j vary together. The diagonal entries are the variance of each feature.

3. **Find eigenvectors and eigenvalues**: The eigenvectors of C are the principal component directions. The eigenvalues tell you how much variance each direction captures.

   In plain English: eigenvectors are the "natural axes" of your data's spread. The eigenvalue is the "importance" of each axis.

4. **Sort by eigenvalue** (descending) and keep the top k eigenvectors.

5. **Project**: Multiply the centered data by the top k eigenvectors to get the reduced representation.

   `X_reduced = X_centered × W_k`

   where W_k is a d×k matrix of the top k eigenvectors.

**Explained variance ratio**: What fraction of the total variance each component captures. If PC1 has eigenvalue 50 and the total is 100, PC1 explains 50% of the variance.

## In Rust

```rust
use ndarray::array;
use ix_unsupervised::pca::PCA;
use ix_unsupervised::traits::DimensionReducer;

// 6 customers with 4 features
let data = array![
    [65000.0, 35.0, 12.0, 500.0],   // income, age, purchases, avg_spent
    [72000.0, 42.0, 15.0, 620.0],
    [45000.0, 28.0, 8.0, 300.0],
    [120000.0, 55.0, 25.0, 950.0],
    [38000.0, 23.0, 5.0, 200.0],
    [95000.0, 48.0, 20.0, 800.0],
];

// Reduce to 2 components for visualization
let mut pca = PCA::new(2);
let reduced = pca.fit_transform(&data);

println!("Original shape: {:?}", data.dim());    // (6, 4)
println!("Reduced shape: {:?}", reduced.dim());   // (6, 2)

// How much information did we keep?
if let Some(ratios) = pca.explained_variance_ratio() {
    println!("Variance explained: {:?}", ratios);
    let total: f64 = ratios.sum();
    println!("Total: {:.1}%", total * 100.0);
}

// The principal component directions
if let Some(components) = pca.components() {
    println!("PC1 direction: {:?}", components.row(0));
    println!("PC2 direction: {:?}", components.row(1));
}
```

### PCA + Clustering Pipeline

A common pattern — reduce dimensions first, then cluster:

```rust
use ix_unsupervised::kmeans::KMeans;
use ix_unsupervised::pca::PCA;
use ix_unsupervised::traits::{Clusterer, DimensionReducer};

// High-dimensional data
let data = /* ... 50-feature dataset ... */;

// Step 1: Reduce to 5 components
let mut pca = PCA::new(5);
let reduced = pca.fit_transform(&data);

// Step 2: Cluster in the reduced space (much faster)
let mut kmeans = KMeans::new(4).with_seed(42);
let labels = kmeans.fit_predict(&reduced);
```

## When To Use This

| Situation | Use PCA? |
|-----------|----------|
| Too many features, need to reduce | Yes — that's its primary use |
| Visualizing high-dimensional data in 2D/3D | Yes — plot first 2-3 components |
| Speeding up distance-based algorithms (KNN, K-Means) | Yes — fewer dimensions = faster |
| Removing multicollinearity (correlated features) | Yes — PCA components are uncorrelated |
| Features have nonlinear relationships | No — PCA only captures linear structure |
| You need to interpret which original features matter | Tricky — PCA components are mixes of features |

## Key Parameters

| Parameter | What It Controls | Guidance |
|-----------|-----------------|----------|
| `n_components` | Dimensions in the output | Choose so explained variance ratio > 0.8-0.95. For visualization: 2 or 3. |

**How many components?** Plot cumulative explained variance ratio vs. number of components. Pick the "knee" — where adding more components gives diminishing returns.

## Pitfalls

- **Standardize first!** PCA finds directions of maximum variance. If one feature ranges 0-100,000 and another 0-1, the first dominates purely due to scale. Use `linalg::standardize()` before PCA.
- **PCA components are hard to interpret.** PC1 might be "0.5 × income + 0.3 × age + 0.4 × purchases" — it's a blend, not a single feature. If interpretability matters, consider feature selection instead.
- **PCA is linear.** It can't capture nonlinear structure. If your data lies on a curve or manifold, PCA gives a poor reduction.
- **Don't use PCA on categorical data.** PCA assumes continuous features. For categorical data, consider Multiple Correspondence Analysis (not in ix).
- **Information loss is inevitable.** You're trading some accuracy for simplicity. Always check the explained variance ratio to ensure you're keeping enough.
- **ix uses power iteration** for eigendecomposition, not LAPACK. This is accurate for the top few components but less precise for many components on large matrices.

## Going Further

- **Before PCA**: [Vectors & Matrices](../foundations/vectors-and-matrices.md) — understand the matrix operations behind PCA
- **Before PCA**: [Probability & Statistics](../foundations/probability-and-statistics.md) — covariance matrices
- **Combine with**: [K-Means](kmeans.md) or [DBSCAN](dbscan.md) — cluster in the reduced space
- **Use case**: [Fraud Detection](../use-cases/fraud-detection.md) — PCA for dimension reduction before classification
