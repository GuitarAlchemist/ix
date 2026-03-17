//! Resampling strategies for imbalanced datasets.
//!
//! Imbalanced datasets — where one class vastly outnumbers others — cause
//! classifiers to ignore the minority class. SMOTE (Synthetic Minority
//! Over-sampling Technique) generates synthetic samples along line segments
//! between existing minority samples, balancing the dataset without
//! simply duplicating existing points.
//!
//! # Example: Balance a fraud detection dataset
//!
//! ```
//! use ndarray::{array, Array2};
//! use ix_supervised::resampling::Smote;
//!
//! // 8 legitimate (class 0), 2 fraudulent (class 1) — 4:1 imbalance
//! let x = Array2::from_shape_vec((10, 2), vec![
//!     0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
//!     0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
//!     5.0, 5.0,  5.5, 5.5,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
//!
//! let smote = Smote::new(5, 42);
//! let (x_new, y_new) = smote.fit_resample(&x, &y);
//!
//! // Minority class has been augmented
//! let class_1_count = y_new.iter().filter(|&&c| c == 1).count();
//! assert!(class_1_count > 2, "SMOTE should generate synthetic minority samples");
//! ```
//!
//! # Example: Custom target ratio
//!
//! ```
//! use ndarray::{array, Array2};
//! use ix_supervised::resampling::Smote;
//!
//! let x = Array2::from_shape_vec((12, 2), vec![
//!     0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
//!     0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
//!     0.9, 0.4,  0.1, 0.6,
//!     5.0, 5.0,  5.5, 5.5,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
//!
//! // Only oversample to 50% of majority (not full balance)
//! let smote = Smote::new(5, 42).with_target_ratio(0.5);
//! let (x_new, y_new) = smote.fit_resample(&x, &y);
//!
//! let class_0 = y_new.iter().filter(|&&c| c == 0).count();
//! let class_1 = y_new.iter().filter(|&&c| c == 1).count();
//! let ratio = class_1 as f64 / class_0 as f64;
//! assert!(ratio >= 0.4 && ratio <= 0.6,
//!     "Target ratio ~0.5 expected, got {:.2}", ratio);
//! ```

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;

/// SMOTE — Synthetic Minority Over-sampling Technique.
///
/// Generates synthetic samples for minority classes by interpolating
/// between existing minority samples and their k-nearest neighbors.
///
/// # Algorithm
///
/// For each minority sample:
/// 1. Find its `k` nearest neighbors (within the same class)
/// 2. Pick one neighbor at random
/// 3. Create a synthetic sample at a random point along the line segment
///    between the original and the neighbor:
///    `x_new = x_orig + rand(0,1) * (x_neighbor - x_orig)`
/// 4. Repeat until the minority class reaches the target count
///
/// # References
///
/// Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling
/// Technique." JAIR 16 (2002): 321-357.
pub struct Smote {
    /// Number of nearest neighbors to consider.
    pub k: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Target ratio of minority to majority class (1.0 = full balance).
    pub target_ratio: f64,
}

impl Smote {
    /// Create a new SMOTE resampler.
    ///
    /// # Arguments
    /// - `k` — number of nearest neighbors (typically 5)
    /// - `seed` — random seed for reproducibility
    pub fn new(k: usize, seed: u64) -> Self {
        assert!(k >= 1, "k must be at least 1");
        Self { k, seed, target_ratio: 1.0 }
    }

    /// Set the target ratio of minority to majority samples.
    ///
    /// - `1.0` (default) — fully balance classes
    /// - `0.5` — oversample minority to 50% of majority count
    pub fn with_target_ratio(mut self, ratio: f64) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0, "target_ratio must be in (0, 1]");
        self.target_ratio = ratio;
        self
    }

    /// Resample the dataset, returning `(x_resampled, y_resampled)`.
    ///
    /// The original data is preserved; only synthetic minority samples
    /// are appended. If there are multiple minority classes, each is
    /// oversampled independently.
    pub fn fit_resample(
        &self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> (Array2<f64>, Array1<usize>) {
        let n = x.nrows();
        let p = x.ncols();
        let n_classes = *y.iter().max().unwrap() + 1;

        // Count per class
        let mut class_counts = vec![0usize; n_classes];
        for &label in y.iter() {
            class_counts[label] += 1;
        }

        let majority_count = *class_counts.iter().max().unwrap();
        let target_count = (majority_count as f64 * self.target_ratio).ceil() as usize;

        // Collect synthetic samples
        let mut synthetic_x: Vec<Vec<f64>> = Vec::new();
        let mut synthetic_y: Vec<usize> = Vec::new();
        let mut rng = StdRng::seed_from_u64(self.seed);

        for (class, &count) in class_counts.iter().enumerate() {
            if count >= target_count || count <= 1 {
                continue; // skip majority class, empty, or single-sample classes
            }

            let n_to_generate = target_count - count;

            // Indices of this class
            let class_indices: Vec<usize> = (0..n)
                .filter(|&i| y[i] == class)
                .collect();

            // Effective k (can't exceed number of same-class neighbors - 1)
            let effective_k = self.k.min(class_indices.len() - 1).max(1);

            // For each sample in the minority class, find k nearest neighbors
            // within the same class
            let neighbors = find_knn_within_class(x, &class_indices, effective_k);

            // Generate synthetic samples
            for _ in 0..n_to_generate {
                // Pick a random minority sample
                let idx_in_class = rng.random_range(0..class_indices.len());
                let orig_idx = class_indices[idx_in_class];

                // Pick a random neighbor
                let neighbor_pos = rng.random_range(0..effective_k);
                let neighbor_idx = neighbors[idx_in_class][neighbor_pos];

                // Interpolate
                let gap: f64 = rng.random();
                let mut new_sample = Vec::with_capacity(p);
                for j in 0..p {
                    let val = x[[orig_idx, j]] + gap * (x[[neighbor_idx, j]] - x[[orig_idx, j]]);
                    new_sample.push(val);
                }

                synthetic_x.push(new_sample);
                synthetic_y.push(class);
            }
        }

        // Combine original + synthetic
        let n_synthetic = synthetic_x.len();
        let total = n + n_synthetic;

        let mut result_x = Array2::zeros((total, p));
        // Copy original
        for i in 0..n {
            for j in 0..p {
                result_x[[i, j]] = x[[i, j]];
            }
        }
        // Copy synthetic
        for (si, row) in synthetic_x.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result_x[[n + si, j]] = val;
            }
        }

        let mut result_y_vec: Vec<usize> = y.to_vec();
        result_y_vec.extend_from_slice(&synthetic_y);
        let result_y = Array1::from_vec(result_y_vec);

        (result_x, result_y)
    }
}

/// Random undersampling — reduce majority class to match minority.
///
/// # Example
///
/// ```
/// use ndarray::{array, Array2};
/// use ix_supervised::resampling::random_undersample;
///
/// let x = Array2::from_shape_vec((10, 2), vec![
///     0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
///     0.8, 0.1,  0.2, 0.7,  0.6, 0.3,  0.4, 0.8,
///     5.0, 5.0,  5.5, 5.5,
/// ]).unwrap();
/// let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
///
/// let (x_new, y_new) = random_undersample(&x, &y, 42);
///
/// let class_0 = y_new.iter().filter(|&&c| c == 0).count();
/// let class_1 = y_new.iter().filter(|&&c| c == 1).count();
/// assert_eq!(class_0, class_1, "Classes should be balanced");
/// assert_eq!(class_1, 2, "Minority count preserved");
/// ```
pub fn random_undersample(
    x: &Array2<f64>,
    y: &Array1<usize>,
    seed: u64,
) -> (Array2<f64>, Array1<usize>) {
    let n_classes = *y.iter().max().unwrap() + 1;

    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &label) in y.iter().enumerate() {
        class_indices[label].push(i);
    }

    let min_count = class_indices.iter()
        .filter(|v| !v.is_empty())
        .map(|v| v.len())
        .min()
        .unwrap_or(0);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut selected: Vec<usize> = Vec::new();

    for indices in &mut class_indices {
        if indices.is_empty() {
            continue;
        }
        // Shuffle and take min_count
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);
        selected.extend_from_slice(&indices[..min_count]);
    }

    selected.sort();

    let p = x.ncols();
    let result_x = Array2::from_shape_fn((selected.len(), p), |(r, c)| x[[selected[r], c]]);
    let result_y = Array1::from_iter(selected.iter().map(|&i| y[i]));

    (result_x, result_y)
}

/// Compute class distribution as a Vec of (class, count, percentage).
///
/// Useful for diagnosing imbalance before and after resampling.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use ix_supervised::resampling::class_distribution;
///
/// let y = array![0, 0, 0, 0, 0, 1, 1, 2];
/// let dist = class_distribution(&y);
/// assert_eq!(dist[0], (0, 5, 62.5));
/// assert_eq!(dist[1], (1, 2, 25.0));
/// assert_eq!(dist[2], (2, 1, 12.5));
/// ```
pub fn class_distribution(y: &Array1<usize>) -> Vec<(usize, usize, f64)> {
    let n = y.len();
    let n_classes = *y.iter().max().unwrap() + 1;
    let mut counts = vec![0usize; n_classes];
    for &label in y.iter() {
        counts[label] += 1;
    }
    counts.iter().enumerate()
        .map(|(c, &count)| (c, count, count as f64 / n as f64 * 100.0))
        .collect()
}

/// Find k nearest neighbors for each sample within a class subset.
/// Returns neighbors[i] = indices into the original x matrix.
fn find_knn_within_class(
    x: &Array2<f64>,
    class_indices: &[usize],
    k: usize,
) -> Vec<Vec<usize>> {
    let n = class_indices.len();
    let mut neighbors = Vec::with_capacity(n);

    for &i in class_indices {
        // Compute distances to all other samples in the same class
        let mut dists: Vec<(usize, f64)> = class_indices.iter()
            .filter(|&&j| j != i)
            .map(|&j| {
                let dist: f64 = x.row(i).iter().zip(x.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (j, dist)
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let knn: Vec<usize> = dists.iter().take(k).map(|&(idx, _)| idx).collect();
        neighbors.push(knn);
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_smote_balances_binary() {
        // 8 class-0, 2 class-1
        let x = Array2::from_shape_vec((10, 2), vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.3, 0.2,
            0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4, 0.8,
            5.0, 5.0, 5.5, 5.5,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

        let smote = Smote::new(1, 42);
        let (x_new, y_new) = smote.fit_resample(&x, &y);

        let class_0 = y_new.iter().filter(|&&c| c == 0).count();
        let class_1 = y_new.iter().filter(|&&c| c == 1).count();

        assert_eq!(class_0, 8, "Majority class should be unchanged");
        assert_eq!(class_1, 8, "Minority class should match majority");
        assert_eq!(x_new.nrows(), 16);
        assert_eq!(x_new.ncols(), 2);
    }

    #[test]
    fn test_smote_preserves_original_data() {
        let x = Array2::from_shape_vec((6, 2), vec![
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0,
            5.0, 5.0, 6.0, 6.0, 7.0, 7.0,
        ]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let smote = Smote::new(2, 42);
        let (x_new, y_new) = smote.fit_resample(&x, &y);

        // Original data should be first 6 rows
        for i in 0..6 {
            assert_eq!(x_new[[i, 0]], x[[i, 0]]);
            assert_eq!(x_new[[i, 1]], x[[i, 1]]);
            assert_eq!(y_new[i], y[i]);
        }
    }

    #[test]
    fn test_smote_synthetic_in_range() {
        // Minority class at (5,5) and (6,6)
        let x = Array2::from_shape_vec((6, 2), vec![
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
            5.0, 5.0, 6.0, 6.0,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1];

        let smote = Smote::new(1, 42);
        let (x_new, y_new) = smote.fit_resample(&x, &y);

        // Synthetic class-1 samples should be between (5,5) and (6,6)
        for i in 6..x_new.nrows() {
            assert_eq!(y_new[i], 1);
            assert!(x_new[[i, 0]] >= 5.0 - 1e-10 && x_new[[i, 0]] <= 6.0 + 1e-10,
                "Synthetic x[0]={} should be in [5, 6]", x_new[[i, 0]]);
            assert!(x_new[[i, 1]] >= 5.0 - 1e-10 && x_new[[i, 1]] <= 6.0 + 1e-10,
                "Synthetic x[1]={} should be in [5, 6]", x_new[[i, 1]]);
        }
    }

    #[test]
    fn test_smote_target_ratio() {
        // 10 class-0, 2 class-1
        let x = Array2::from_shape_vec((12, 2), vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.3, 0.2,
            0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4, 0.8,
            0.9, 0.4, 0.1, 0.6,
            5.0, 5.0, 5.5, 5.5,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

        let smote = Smote::new(1, 42).with_target_ratio(0.5);
        let (_, y_new) = smote.fit_resample(&x, &y);

        let class_0 = y_new.iter().filter(|&&c| c == 0).count();
        let class_1 = y_new.iter().filter(|&&c| c == 1).count();

        assert_eq!(class_0, 10);
        assert_eq!(class_1, 5, "Should oversample to 50% of majority");
    }

    #[test]
    fn test_smote_multiclass() {
        // 6 class-0, 3 class-1, 2 class-2
        let x = Array2::from_shape_vec((11, 2), vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.3, 0.2, 0.8, 0.1, 0.2, 0.7,
            5.0, 5.0, 5.5, 5.5, 5.2, 5.3,
            9.0, 9.0, 9.5, 9.5,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2];

        let smote = Smote::new(2, 42);
        let (_, y_new) = smote.fit_resample(&x, &y);

        let class_0 = y_new.iter().filter(|&&c| c == 0).count();
        let class_1 = y_new.iter().filter(|&&c| c == 1).count();
        let class_2 = y_new.iter().filter(|&&c| c == 2).count();

        assert_eq!(class_0, 6, "Majority unchanged");
        assert_eq!(class_1, 6, "Class 1 oversampled to 6");
        assert_eq!(class_2, 6, "Class 2 oversampled to 6");
    }

    #[test]
    fn test_smote_already_balanced() {
        let x = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0, 1.0, 1.0,
            5.0, 5.0, 6.0, 6.0,
        ]).unwrap();
        let y = array![0, 0, 1, 1];

        let smote = Smote::new(1, 42);
        let (x_new, y_new) = smote.fit_resample(&x, &y);

        assert_eq!(x_new.nrows(), 4, "Already balanced, no change");
        assert_eq!(y_new.len(), 4);
    }

    #[test]
    fn test_random_undersample() {
        let x = Array2::from_shape_vec((10, 2), vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.3, 0.2,
            0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4, 0.8,
            5.0, 5.0, 5.5, 5.5,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

        let (x_new, y_new) = random_undersample(&x, &y, 42);

        let class_0 = y_new.iter().filter(|&&c| c == 0).count();
        let class_1 = y_new.iter().filter(|&&c| c == 1).count();

        assert_eq!(class_0, 2);
        assert_eq!(class_1, 2);
        assert_eq!(x_new.nrows(), 4);
    }

    #[test]
    fn test_class_distribution() {
        let y = array![0, 0, 0, 0, 0, 1, 1, 2];
        let dist = class_distribution(&y);

        assert_eq!(dist.len(), 3);
        assert_eq!(dist[0], (0, 5, 62.5));
        assert_eq!(dist[1], (1, 2, 25.0));
        assert_eq!(dist[2], (2, 1, 12.5));
    }

    #[test]
    fn test_smote_reproducible() {
        let x = Array2::from_shape_vec((6, 2), vec![
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
            5.0, 5.0, 6.0, 6.0,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1];

        let smote = Smote::new(1, 99);
        let (x1, y1) = smote.fit_resample(&x, &y);
        let (x2, y2) = smote.fit_resample(&x, &y);

        assert_eq!(x1, x2, "Same seed should produce same results");
        assert_eq!(y1, y2);
    }
}
