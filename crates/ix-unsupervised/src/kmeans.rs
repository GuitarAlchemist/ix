//! K-Means clustering with K-Means++ initialization.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::traits::Clusterer;
use ix_math::distance::euclidean_squared;

/// Serializable state for a fitted [`KMeans`] model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansState {
    /// Centroid coordinates: `k` rows, each a feature vector.
    pub centroids: Vec<Vec<f64>>,
    /// Number of clusters.
    pub k: usize,
}

/// K-Means clustering.
pub struct KMeans {
    pub k: usize,
    pub max_iterations: usize,
    pub seed: u64,
    pub centroids: Option<Array2<f64>>,
}

impl KMeans {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 300,
            seed: 42,
            centroids: None,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Save the trained model state. Returns `None` if the model has not been fitted.
    pub fn save_state(&self) -> Option<KMeansState> {
        self.centroids.as_ref().map(|c| {
            let rows = c.nrows();
            let centroids = (0..rows)
                .map(|r| c.row(r).to_vec())
                .collect();
            KMeansState {
                centroids,
                k: self.k,
            }
        })
    }

    /// Reconstruct a fitted model from a previously saved state.
    pub fn load_state(state: &KMeansState) -> Self {
        let k = state.k;
        let cols = state.centroids.first().map_or(0, |r| r.len());
        let flat: Vec<f64> = state.centroids.iter().flat_map(|r| r.iter().copied()).collect();
        let centroids = Array2::from_shape_vec((k, cols), flat)
            .expect("KMeansState centroids dimensions mismatch");
        Self {
            k,
            max_iterations: 300,
            seed: 42,
            centroids: Some(centroids),
        }
    }

    /// K-Means++ initialization.
    fn init_centroids(&self, x: &Array2<f64>, rng: &mut StdRng) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut centroids = Array2::zeros((self.k, p));

        // First centroid: random
        let idx = rng.random_range(0..n);
        centroids.row_mut(0).assign(&x.row(idx));

        for c in 1..self.k {
            // Compute distances to nearest centroid
            let mut distances = Array1::from_elem(n, f64::INFINITY);
            for i in 0..n {
                let point = x.row(i).to_owned();
                for j in 0..c {
                    let centroid = centroids.row(j).to_owned();
                    let d = euclidean_squared(&point, &centroid).unwrap();
                    if d < distances[i] {
                        distances[i] = d;
                    }
                }
            }

            // Weighted random selection
            let total: f64 = distances.sum();
            let mut r = rng.random::<f64>() * total;
            for i in 0..n {
                r -= distances[i];
                if r <= 0.0 {
                    centroids.row_mut(c).assign(&x.row(i));
                    break;
                }
            }
        }

        centroids
    }
}

impl Clusterer for KMeans {
    fn fit(&mut self, x: &Array2<f64>) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n = x.nrows();
        let mut centroids = self.init_centroids(x, &mut rng);

        for _ in 0..self.max_iterations {
            // Assign clusters
            let labels = assign_clusters(x, &centroids);

            // Update centroids
            let mut new_centroids = Array2::zeros(centroids.dim());
            let mut counts = vec![0usize; self.k];

            for i in 0..n {
                let c = labels[i];
                counts[c] += 1;
                for j in 0..x.ncols() {
                    new_centroids[[c, j]] += x[[i, j]];
                }
            }

            for (c, &count) in counts.iter().enumerate().take(self.k) {
                if count > 0 {
                    new_centroids
                        .row_mut(c)
                        .mapv_inplace(|v| v / count as f64);
                }
            }

            // Check convergence
            let diff = (&new_centroids - &centroids).mapv(|v| v * v).sum();
            centroids = new_centroids;
            if diff < 1e-10 {
                break;
            }
        }

        self.centroids = Some(centroids);
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let centroids = self.centroids.as_ref().expect("Model not fitted");
        assign_clusters(x, centroids)
    }
}

fn assign_clusters(x: &Array2<f64>, centroids: &Array2<f64>) -> Array1<usize> {
    Array1::from_iter((0..x.nrows()).map(|i| {
        let point = x.row(i).to_owned();
        let mut best = 0;
        let mut best_dist = f64::INFINITY;
        for c in 0..centroids.nrows() {
            let centroid = centroids.row(c).to_owned();
            let d = euclidean_squared(&point, &centroid).unwrap();
            if d < best_dist {
                best_dist = d;
                best = c;
            }
        }
        best
    }))
}

/// Inertia: sum of squared distances to nearest centroid.
pub fn inertia(x: &Array2<f64>, labels: &Array1<usize>, centroids: &Array2<f64>) -> f64 {
    (0..x.nrows())
        .map(|i| {
            let point = x.row(i).to_owned();
            let centroid = centroids.row(labels[i]).to_owned();
            euclidean_squared(&point, &centroid).unwrap()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kmeans_two_clusters() {
        let x = array![
            [0.0, 0.0], [0.5, 0.5], [1.0, 0.0],
            [10.0, 10.0], [10.5, 10.5], [11.0, 10.0]
        ];

        let mut km = KMeans::new(2).with_seed(42);
        let labels = km.fit_predict(&x);

        // First 3 should be same cluster, last 3 same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_kmeans_save_load_roundtrip() {
        let x = array![
            [0.0, 0.0], [0.5, 0.5], [1.0, 0.0],
            [10.0, 10.0], [10.5, 10.5], [11.0, 10.0]
        ];

        let mut km = KMeans::new(2).with_seed(42);
        km.fit(&x);
        let orig_labels = km.predict(&x);

        let state = km.save_state().expect("fitted model should produce state");

        // Roundtrip through JSON
        let json = serde_json::to_string(&state).unwrap();
        let restored_state: KMeansState = serde_json::from_str(&json).unwrap();
        let restored = KMeans::load_state(&restored_state);

        let rest_labels = restored.predict(&x);
        assert_eq!(orig_labels, rest_labels, "predictions must match after roundtrip");
    }

    #[test]
    fn test_kmeans_save_state_unfitted() {
        let km = KMeans::new(3);
        assert!(km.save_state().is_none(), "unfitted model should return None");
    }
}
