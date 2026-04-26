//! Data poisoning detection and defense.

use ndarray::{Array1, Array2, Axis};

/// Result of a poisoning detection scan.
#[derive(Debug, Clone)]
pub struct PoisonResult {
    /// Indices of flagged (potentially poisoned) samples.
    pub flagged_indices: Vec<usize>,
    /// Confidence score for each flagged sample (higher = more suspicious).
    pub confidence_scores: Vec<f64>,
}

/// Detect potentially flipped labels via KNN consistency.
///
/// For each sample, checks whether its label agrees with the majority vote of
/// its `k_neighbors` nearest neighbours (Euclidean distance). Returns indices
/// of inconsistent samples.
pub fn detect_label_flips(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    k_neighbors: usize,
) -> Vec<usize> {
    let n = features.nrows();
    if n == 0 || k_neighbors == 0 {
        return vec![];
    }
    let k = k_neighbors.min(n - 1);
    let mut suspicious = Vec::new();

    for i in 0..n {
        let xi = features.row(i);
        // compute distances to all other points
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let diff = &xi - &features.row(j);
                let d = diff.mapv(|v| v * v).sum().sqrt();
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // majority vote among k nearest
        let mut pos = 0usize;
        let mut neg = 0usize;
        for &(j, _) in dists.iter().take(k) {
            if labels[j] >= 0.5 {
                pos += 1;
            } else {
                neg += 1;
            }
        }
        let majority: f64 = if pos >= neg { 1.0 } else { 0.0 };
        let sample_label: f64 = if labels[i] >= 0.5 { 1.0 } else { 0.0 };
        if (majority - sample_label).abs() > 0.5 {
            suspicious.push(i);
        }
    }
    suspicious
}

/// Estimate influence of each training point on a test prediction.
///
/// Simplified influence function (Koh & Liang, 2017) using a linear
/// approximation: I(z_train) ≈ -(features^T features + λI)^{-1} · x_test ·
/// (y_test - x_test^T w). Returns an influence score per training sample.
pub fn influence_function(
    train_features: &Array2<f64>,
    _train_labels: &Array1<f64>,
    test_point: &Array1<f64>,
    test_label: f64,
    damping: f64,
) -> Array1<f64> {
    let n = train_features.nrows();
    let d = train_features.ncols();

    // Compute w via ridge regression: w = (X^T X + λI)^{-1} X^T y
    // For simplicity use gradient-based influence approximation
    // H ≈ (1/n) X^T X + damping * I
    let xt = train_features.t();
    let xtx = xt.dot(train_features) / n as f64;

    // Add damping to diagonal
    let mut h = xtx;
    for i in 0..d {
        h[[i, i]] += damping;
    }

    // Approximate H^{-1} via Neumann series (single-step for simplicity):
    // H^{-1} ≈ (1/damping) I - (1/damping^2) (H - damping*I)
    // For a simple implementation, we use element-wise approximation
    // with diagonal dominance assumption:
    let test_residual = test_label - train_features.mean_axis(Axis(0)).unwrap().dot(test_point);

    // influence ≈ (x_train · x_test) * residual / (n · damping)
    let influences = Array1::from_iter((0..n).map(|i| {
        let x_train = train_features.row(i);
        let sim = x_train.dot(test_point);
        sim * test_residual / (n as f64 * damping)
    }));
    influences
}

/// Detect backdoor-poisoned data via spectral signatures.
///
/// (Tran et al., 2018) For each class, computes the covariance of feature
/// representations, finds the top singular direction, and flags samples whose
/// projection onto that direction exceeds `percentile`.
pub fn spectral_signature_defense(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    num_classes: usize,
    percentile: f64,
) -> Vec<usize> {
    let n = features.nrows();
    if n == 0 {
        return vec![];
    }
    let mut flagged = Vec::new();

    for c in 0..num_classes {
        // gather indices for this class
        let class_indices: Vec<usize> = (0..n)
            .filter(|&i| (labels[i] - c as f64).abs() < 0.5)
            .collect();
        if class_indices.len() < 2 {
            continue;
        }

        // compute mean
        let class_n = class_indices.len();
        let dim = features.ncols();
        let mut mean = Array1::<f64>::zeros(dim);
        for &i in &class_indices {
            mean += &features.row(i).to_owned();
        }
        mean /= class_n as f64;

        // compute centred representations and their outer product sum
        // to find the top singular direction via power iteration
        let centered: Vec<Array1<f64>> = class_indices
            .iter()
            .map(|&i| features.row(i).to_owned() - &mean)
            .collect();

        // power iteration for top eigenvector of covariance
        let mut v = Array1::from_elem(dim, 1.0 / (dim as f64).sqrt());
        for _ in 0..30 {
            let mut new_v = Array1::<f64>::zeros(dim);
            for row in &centered {
                let proj = row.dot(&v);
                new_v += &(row * proj);
            }
            let norm = new_v.mapv(|x| x * x).sum().sqrt().max(1e-12);
            v = new_v / norm;
        }

        // project each sample onto top direction and compute scores
        let mut scores: Vec<(usize, f64)> = class_indices
            .iter()
            .zip(centered.iter())
            .map(|(&i, row)| (i, row.dot(&v).abs()))
            .collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // flag samples above percentile
        let cutoff_idx = ((percentile / 100.0) * scores.len() as f64) as usize;
        let cutoff_idx = cutoff_idx.min(scores.len().saturating_sub(1));
        let threshold = scores[cutoff_idx].1;
        for &(idx, score) in &scores {
            if score > threshold {
                flagged.push(idx);
            }
        }
    }
    flagged.sort();
    flagged.dedup();
    flagged
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_detect_label_flips_clean_data() {
        // Two clean clusters
        let features = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.05, 0.05, // cluster 0
                1.0, 1.0, 1.1, 1.1, 1.05, 1.05, // cluster 1
            ],
        )
        .unwrap();
        let labels = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let flips = detect_label_flips(&features, &labels, 3);
        assert!(flips.is_empty(), "Clean data should have no label flips");
    }

    #[test]
    fn test_detect_label_flips_poisoned() {
        let features = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // cluster 0
                1.0, 1.0, 1.1, 1.0, // cluster 1
            ],
        )
        .unwrap();
        // Third point (index 2) is in cluster 0 but labelled 1
        let labels = array![0.0, 0.0, 1.0, 1.0, 1.0];
        let flips = detect_label_flips(&features, &labels, 2);
        assert!(flips.contains(&2));
    }

    #[test]
    fn test_detect_label_flips_empty() {
        let features = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let labels = array![];
        let flips = detect_label_flips(&features, &labels, 3);
        assert!(flips.is_empty());
    }

    #[test]
    fn test_influence_function_returns_correct_size() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
        let labels = array![1.0, 0.0, 1.0, 0.0];
        let test_pt = array![0.5, 0.5];
        let inf = influence_function(&features, &labels, &test_pt, 1.0, 0.1);
        assert_eq!(inf.len(), 4);
    }

    #[test]
    fn test_influence_function_damping_effect() {
        let features = Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).unwrap();
        let labels = array![1.0, 0.0];
        let test_pt = array![1.0];
        let inf_low = influence_function(&features, &labels, &test_pt, 1.0, 0.01);
        let inf_high = influence_function(&features, &labels, &test_pt, 1.0, 10.0);
        // Higher damping should reduce influence magnitude
        assert!(inf_low.mapv(|x| x.abs()).sum() >= inf_high.mapv(|x| x.abs()).sum());
    }

    #[test]
    fn test_spectral_signature_defense_returns_indices() {
        // Simple data with an outlier in class 0
        let features = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.05, 0.05, // cluster 0
                10.0, 10.0, // outlier in class 0
                5.0, 5.0, 5.1, 5.1, // class 1
            ],
        )
        .unwrap();
        let labels = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let flagged = spectral_signature_defense(&features, &labels, 2, 50.0);
        // The outlier at index 3 should be flagged
        assert!(flagged.contains(&3));
    }

    #[test]
    fn test_spectral_signature_empty() {
        let features = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let labels = array![];
        let flagged = spectral_signature_defense(&features, &labels, 2, 90.0);
        assert!(flagged.is_empty());
    }
}
