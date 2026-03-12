---
name: machin-cluster
description: Cluster data using K-Means or DBSCAN
---

# Cluster

Group data points into clusters using unsupervised learning.

## When to Use
When the user has unlabeled data and wants to find natural groupings, segments, or patterns.

## Algorithms
- **K-Means** — Fast, works well when clusters are spherical and k is known
- **DBSCAN** — Density-based, finds arbitrary-shaped clusters, handles noise/outliers

## Choosing k
If the user doesn't specify k, suggest the elbow method:
1. Run K-Means for k=2..10
2. Plot inertia (sum of squared distances to centroids)
3. Pick the "elbow" where improvement plateaus

## Programmatic Usage
```rust
use machin_unsupervised::kmeans::KMeans;
use machin_unsupervised::dbscan::DBSCAN;

// K-Means
let model = KMeans::new(k, max_iter, seed);
let assignments = model.fit(&data);

// DBSCAN
let model = DBSCAN::new(eps, min_points);
let labels = model.fit(&data); // -1 = noise
```

## Interpretation
- Report cluster sizes and centroids
- Flag if any cluster is disproportionately large (may need more k)
- For DBSCAN, report number of noise points
