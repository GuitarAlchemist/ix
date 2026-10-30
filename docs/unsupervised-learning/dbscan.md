# DBSCAN

> Density-Based Spatial Clustering — finds clusters of any shape and automatically detects outliers.

## The Problem

You're analyzing GPS data from a fleet of delivery trucks. You want to find areas where trucks spend a lot of time (delivery hotspots, warehouses, traffic bottlenecks). But these areas aren't neat circles — they follow roads, cluster around intersections, and have irregular shapes. You also need to identify GPS points that don't belong to any cluster (noise — perhaps signal errors or one-off stops).

K-Means can't do this. It only finds spherical clusters and forces every point into a cluster. You need something that understands *density*.

## The Intuition

Think of a crowd at a concert. People naturally form dense groups — near the stage, at the bar, by the exits. Between these groups, there are sparse areas with few people.

DBSCAN works like this:
1. Pick any person. Look around them within arm's reach (distance ε).
2. If there are at least `min_points` people within arm's reach, this person is in a **dense area** — start a cluster.
3. Each of those neighbors also looks around. If they also have enough nearby people, the cluster grows.
4. Keep expanding until no more dense connections are found.
5. People not reachable from any dense area are **noise** (outliers).

The key insight: clusters are connected regions of high density, separated by regions of low density. No need to specify how many clusters — DBSCAN finds them automatically.

## How It Works

**Input**: Dataset X, neighborhood radius ε (eps), minimum neighbors `min_points`.

**Three types of points:**
- **Core point**: Has at least `min_points` neighbors within distance ε
- **Border point**: Within ε of a core point, but doesn't have enough neighbors itself
- **Noise point**: Neither core nor border — an outlier

**Algorithm:**
1. For each unvisited point p:
   - Find all points within distance ε of p
   - If fewer than `min_points` neighbors → mark as noise (for now)
   - If enough neighbors → p is a core point. Start a new cluster:
     - Add p and all its ε-neighbors to the cluster
     - For each neighbor that's also a core point, recursively add *their* neighbors
     - Continue until no more density-reachable points

In plain English: start at a dense point, grow the cluster outward through density-connected points, stop when you hit sparse regions.

**Key property**: A noise point near no core points stays as noise. A noise point later found to be within ε of a core point in another cluster gets reclassified as a border point.

## In Rust

```rust
use ndarray::array;
use ix_unsupervised::dbscan::DBSCAN;
use ix_unsupervised::traits::Clusterer;

// GPS coordinates of truck stops [latitude, longitude]
let stops = array![
    // Cluster 1: Warehouse area
    [40.712, -74.006],
    [40.713, -74.005],
    [40.711, -74.007],
    [40.714, -74.004],
    // Cluster 2: Downtown delivery zone
    [40.758, -73.985],
    [40.757, -73.986],
    [40.759, -73.984],
    [40.756, -73.987],
    // Noise: random one-off stops
    [40.800, -73.950],
    [40.650, -74.100],
];

let mut dbscan = DBSCAN::new(
    0.005,  // eps: ~500m radius at NYC latitude
    3,      // min_points: need at least 3 neighbors
);

let labels = dbscan.fit_predict(&stops);
println!("Labels: {:?}", labels);
// Cluster 1: labeled 1, Cluster 2: labeled 2, Noise: labeled 0

// Count clusters and noise
let n_clusters = *labels.iter().max().unwrap_or(&0);
let n_noise = labels.iter().filter(|&&l| l == 0).count();
println!("{} clusters found, {} noise points", n_clusters, n_noise);
```

> See [`examples/unsupervised/dbscan_anomaly.rs`](../../examples/unsupervised/dbscan_anomaly.rs) for the full runnable version.

## When To Use This

| Situation | Use DBSCAN | Use K-Means |
|-----------|------------|-------------|
| Unknown number of clusters | Yes — finds K automatically | No — must specify K |
| Irregular cluster shapes | Yes — follows density | No — assumes spherical |
| Need to detect outliers | Yes — labels noise as 0 | No — forces all points into clusters |
| Clusters of different sizes | Partially — works if density is similar | Yes — handles different sizes |
| Very large datasets | Slower (O(n²) without spatial index) | Fast (O(n×k×iterations)) |
| Clusters with varying density | Poor — single eps can't capture both | Also poor — different problem |

## Key Parameters

| Parameter | What It Controls | How to Choose |
|-----------|-----------------|---------------|
| `eps` | Neighborhood radius | Plot k-distance graph (sort distances to k-th nearest neighbor). The "elbow" suggests a good eps. |
| `min_points` | Minimum neighbors to be a core point | Rule of thumb: `min_points ≥ dimensions + 1`. For noisy data, increase it. Common: 5-10. |

**Choosing eps is the hard part.** Too small → everything is noise. Too large → everything is one cluster. The k-distance plot helps: compute the distance from each point to its `min_points`-th nearest neighbor, sort these distances, and look for a sharp bend.

## Pitfalls

- **Sensitive to eps and min_points.** Unlike K-Means where K is intuitive, tuning eps requires understanding your data's scale. Standardize features first.
- **Can't handle varying density.** If one cluster is dense (points 1cm apart) and another is sparse (points 1m apart), no single eps works for both. Consider HDBSCAN (not yet in ix) or running DBSCAN at multiple scales.
- **Noise label is 0.** In ix, cluster labels start at 1. Points labeled 0 are noise/outliers. Don't confuse 0 with "cluster 0."
- **O(n²) without spatial indexing.** For large datasets, the pairwise distance computation is expensive. ix uses brute-force, which is fine for up to ~10K points.
- **Border points are non-deterministic.** A border point reachable from two clusters is assigned to whichever cluster discovers it first. This rarely matters in practice.

## Going Further

- **Alternative**: [K-Means](kmeans.md) — when you know K and clusters are spherical
- **Preprocessing**: [PCA](pca.md) — reduce dimensions before running DBSCAN
- **Foundation**: [Distance & Similarity](../foundations/distance-and-similarity.md) — DBSCAN uses Euclidean distance
- **Use case**: [Anomaly Detection](../use-cases/anomaly-detection.md) — using DBSCAN's noise detection for anomalies
