//! DBSCAN density-based clustering.
//!
//! Implements DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
//! Points not assigned to any cluster are given label 0 (noise), and actual
//! clusters are labeled starting from 1.

use crate::traits::Clusterer;
use ix_math::distance::euclidean;
use ndarray::{Array1, Array2};

/// DBSCAN clustering algorithm.
///
/// Clusters are labeled 1, 2, ... and noise points are labeled 0.
pub struct DBSCAN {
    pub eps: f64,
    pub min_points: usize,
    labels: Option<Array1<usize>>,
    fitted_data: Option<Array2<f64>>,
}

impl DBSCAN {
    pub fn new(eps: f64, min_points: usize) -> Self {
        Self {
            eps,
            min_points,
            labels: None,
            fitted_data: None,
        }
    }
}

/// Find all points within eps distance of point i.
fn region_query(x: &Array2<f64>, i: usize, eps: f64) -> Vec<usize> {
    let n = x.nrows();
    let pi = x.row(i).to_owned();
    let mut neighbors = Vec::new();
    for j in 0..n {
        let pj = x.row(j).to_owned();
        if let Ok(d) = euclidean(&pi, &pj) {
            if d <= eps {
                neighbors.push(j);
            }
        }
    }
    neighbors
}

// Noise label constant
const NOISE: usize = 0;
const UNDEFINED: usize = usize::MAX;

impl Clusterer for DBSCAN {
    fn fit(&mut self, x: &Array2<f64>) {
        let n = x.nrows();
        let mut labels = vec![UNDEFINED; n];
        let mut cluster_id = 0usize; // 0 is noise, clusters start at 1

        for i in 0..n {
            if labels[i] != UNDEFINED {
                continue;
            }

            let neighbors = region_query(x, i, self.eps);

            if neighbors.len() < self.min_points {
                labels[i] = NOISE;
                continue;
            }

            // Start a new cluster
            cluster_id += 1;
            labels[i] = cluster_id;

            // Seed set: neighbors minus point i
            let mut seed_set: Vec<usize> = neighbors.into_iter().filter(|&j| j != i).collect();
            let mut idx = 0;

            while idx < seed_set.len() {
                let q = seed_set[idx];
                idx += 1;

                if labels[q] == NOISE {
                    // Change noise to border point of this cluster
                    labels[q] = cluster_id;
                }

                if labels[q] != UNDEFINED {
                    continue;
                }

                labels[q] = cluster_id;

                let q_neighbors = region_query(x, q, self.eps);
                if q_neighbors.len() >= self.min_points {
                    // q is a core point; add its neighbors to the seed set
                    for &neighbor in &q_neighbors {
                        if (labels[neighbor] == UNDEFINED || labels[neighbor] == NOISE)
                            && !seed_set.contains(&neighbor)
                        {
                            seed_set.push(neighbor);
                        }
                    }
                }
            }
        }

        self.labels = Some(Array1::from_vec(labels));
        self.fitted_data = Some(x.to_owned());
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        // For DBSCAN, predict assigns new points to the nearest core point's cluster,
        // or NOISE if no core point is within eps.
        let fitted = self.fitted_data.as_ref().expect("Model not fitted");
        let labels = self.labels.as_ref().expect("Model not fitted");
        let n = x.nrows();
        let n_train = fitted.nrows();

        let mut result = Array1::from_elem(n, NOISE);

        for i in 0..n {
            let pi = x.row(i).to_owned();
            let mut best_dist = f64::INFINITY;
            let mut best_label = NOISE;

            for j in 0..n_train {
                let pj = fitted.row(j).to_owned();
                if let Ok(d) = euclidean(&pi, &pj) {
                    if d <= self.eps && d < best_dist && labels[j] != NOISE {
                        best_dist = d;
                        best_label = labels[j];
                    }
                }
            }

            result[i] = best_label;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dbscan_two_clusters() {
        // Two well-separated clusters
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [0.0, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
            [5.0, 5.2]
        ];

        let mut db = DBSCAN::new(0.5, 2);
        let labels = db.fit_predict(&x);

        // First 4 should be in one cluster, last 4 in another
        assert!(labels[0] > 0, "Point 0 should not be noise");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[2], labels[3]);

        assert!(labels[4] > 0, "Point 4 should not be noise");
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[6], labels[7]);

        assert_ne!(
            labels[0], labels[4],
            "Two groups should be different clusters"
        );
    }

    #[test]
    fn test_dbscan_noise() {
        // Cluster + outlier
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [100.0, 100.0] // outlier
        ];

        let mut db = DBSCAN::new(0.5, 2);
        let labels = db.fit_predict(&x);

        assert!(labels[0] > 0, "Dense points should be clustered");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], 0, "Outlier should be noise");
    }

    #[test]
    fn test_dbscan_all_noise() {
        // Points too far apart for any cluster
        let x = array![[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]];

        let mut db = DBSCAN::new(0.5, 2);
        let labels = db.fit_predict(&x);

        for i in 0..3 {
            assert_eq!(labels[i], 0, "All points should be noise");
        }
    }

    #[test]
    fn test_dbscan_predict_new_data() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0]
        ];

        let mut db = DBSCAN::new(0.5, 2);
        db.fit(&x);

        let new_data = array![
            [0.05, 0.05], // should be cluster 1
            [5.05, 5.05], // should be cluster 2
            [50.0, 50.0], // should be noise
        ];

        let pred = db.predict(&new_data);
        assert!(pred[0] > 0, "Close to cluster 1, should not be noise");
        assert!(pred[1] > 0, "Close to cluster 2, should not be noise");
        assert_ne!(pred[0], pred[1], "Should be different clusters");
        assert_eq!(pred[2], 0, "Far away point should be noise");
    }

    #[test]
    fn test_dbscan_single_dense_cluster() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.2, 0.1],
            [0.3, 0.1]
        ];

        let mut db = DBSCAN::new(0.5, 3);
        let labels = db.fit_predict(&x);

        // All points should be in the same cluster
        let cluster = labels[0];
        assert!(cluster > 0);
        for i in 1..8 {
            assert_eq!(
                labels[i], cluster,
                "All dense points should share a cluster"
            );
        }
    }
}
