# K-Means Clustering

> Partition data into K groups by iteratively assigning points to the nearest centroid and updating centroids.

## The Problem

You run an e-commerce platform with 50,000 customers. Each customer has features: average order value, purchase frequency, time since last purchase, and number of product categories browsed. You want to segment these customers into distinct groups for targeted marketing — but nobody told you what the groups *are*. You need the algorithm to discover them.

This is unsupervised learning. There are no labels. You just have data and a question: "What natural groupings exist?"

## The Intuition

Imagine you're organizing a party and need to place 3 food stations in a large room full of people. You want each person to be close to at least one station.

1. **Start**: Place 3 stations randomly.
2. **Assign**: Each person walks to the nearest station.
3. **Update**: Move each station to the center of the people standing around it.
4. **Repeat**: People reassign to the now-closest station, stations move again.
5. **Stop**: When nobody changes stations.

That's K-Means. The "stations" are **centroids** (cluster centers), and the algorithm alternates between assigning points and updating centroids until convergence.

### K-Means++ Initialization

Random initial centroids can give bad results. K-Means++ is smarter: it picks the first centroid randomly, then picks each subsequent centroid with probability proportional to its distance from existing centroids. This spreads initial centroids apart, leading to better and more consistent clusters.

ix uses K-Means++ by default.

## How It Works

**Input**: Dataset X (n points, d features), number of clusters K.

**Algorithm**:

1. **Initialize** K centroids using K-Means++ (or randomly)
2. **Assign** each point to the nearest centroid:

   `cluster(xᵢ) = argmin_k ||xᵢ - μₖ||²`

   In plain English: for each data point, find which centroid is closest (using Euclidean distance) and assign it to that cluster.

3. **Update** each centroid to the mean of its assigned points:

   `μₖ = (1/|Cₖ|) × Σᵢ∈Cₖ xᵢ`

   In plain English: move each centroid to the average position of all points currently in its cluster.

4. **Repeat** steps 2-3 until assignments stop changing or max iterations reached.

**Inertia** (within-cluster sum of squares):

`inertia = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²`

In plain English: the total distance of all points to their assigned centroids. Lower is better. Use this to compare different values of K (the "elbow method" — plot inertia vs K and look for the bend).

## In Rust

```rust
use ndarray::array;
use ix_unsupervised::kmeans::KMeans;
use ix_unsupervised::traits::Clusterer;

// Customer data: [avg_order, frequency, recency, categories]
let customers = array![
    [50.0, 12.0, 5.0, 3.0],    // Frequent, moderate spender
    [45.0, 10.0, 7.0, 2.0],    // Similar to above
    [200.0, 2.0, 30.0, 8.0],   // Rare, high-value buyer
    [180.0, 3.0, 25.0, 7.0],   // Similar to above
    [15.0, 1.0, 90.0, 1.0],    // Churned, low-value
    [10.0, 1.0, 120.0, 1.0],   // Similar to above
];

// Cluster into 3 segments
let mut kmeans = KMeans::new(3).with_seed(42);
let labels = kmeans.fit_predict(&customers);

println!("Cluster assignments: {:?}", labels);
// e.g., [0, 0, 1, 1, 2, 2] — three distinct customer segments

// Check cluster quality
if let Some(centroids) = &kmeans.centroids {
    let score = ix_unsupervised::kmeans::inertia(&customers, &labels, centroids);
    println!("Inertia: {:.2}", score);
}
```

> See [`examples/unsupervised/kmeans_clustering.rs`](../../examples/unsupervised/kmeans_clustering.rs) for the full runnable version.

### Elbow Method: Choosing K

```rust
use ix_unsupervised::kmeans::{KMeans, inertia};
use ix_unsupervised::traits::Clusterer;

for k in 2..=8 {
    let mut kmeans = KMeans::new(k).with_seed(42);
    let labels = kmeans.fit_predict(&data);
    let score = inertia(&data, &labels, kmeans.centroids.as_ref().unwrap());
    println!("K={}: inertia={:.2}", k, score);
}
// Look for the "elbow" — where adding more clusters stops helping much
```

## When To Use This

| Algorithm | Best For | Limitations |
|-----------|----------|-------------|
| **K-Means** | Roughly spherical clusters of similar size | Must specify K; assumes equal-size clusters |
| **DBSCAN** | Irregular shapes, finding outliers | Sensitive to eps/min_points; struggles with varying density |
| **PCA** | Reducing dimensions before clustering | Not a clustering algorithm itself |

Choose K-Means when:
- You have a rough idea of how many clusters you want
- Clusters are roughly spherical (not elongated or irregular)
- You want fast, scalable results

## Key Parameters

| Parameter | What It Controls | Guidance |
|-----------|-----------------|----------|
| `k` | Number of clusters | Use elbow method or domain knowledge |
| `max_iterations` | When to stop if not converged | Default is usually fine (100-300) |
| `seed` | Reproducibility of K-Means++ init | Set for consistent results |

## Pitfalls

- **K-Means always finds K clusters**, even if the data has fewer natural groups. Run with different K values and validate.
- **Sensitive to scale.** Always standardize your features first — a feature ranging 0-1000 will dominate one ranging 0-1. Use `linalg::standardize()`.
- **Sensitive to initialization.** Even with K-Means++, results can vary. Run multiple times with different seeds and pick the best inertia.
- **Can't find non-spherical clusters.** If your clusters are elongated, ring-shaped, or irregular, use DBSCAN instead.
- **Outliers distort centroids.** A single extreme point pulls its cluster's centroid far from the true center. Consider removing outliers first, or using a robust alternative.

## Going Further

- **Alternative**: [DBSCAN](dbscan.md) — density-based clustering that finds irregular shapes
- **Preprocessing**: [PCA](pca.md) — reduce dimensions before clustering for better results
- **Foundation**: [Distance & Similarity](../foundations/distance-and-similarity.md) — the distance metrics K-Means uses
- **Use case**: [Fraud Detection](../use-cases/fraud-detection.md) — combining clustering with classification
