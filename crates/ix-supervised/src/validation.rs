//! Cross-validation utilities for model evaluation.
//!
//! Provides k-fold splitting and cross-validated scoring to estimate
//! how well a model generalizes to unseen data.
//!
//! # Example: K-Fold Cross-Validation
//!
//! ```
//! use ndarray::{array, Array2};
//! use ix_supervised::validation::{KFold, StratifiedKFold};
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
//!     5.0, 5.0,  5.5, 5.5,  6.0, 5.0,  5.3, 5.2,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! // Simple k-fold: 4 folds
//! let kf = KFold::new(4);
//! let folds: Vec<_> = kf.split(x.nrows());
//! assert_eq!(folds.len(), 4);
//! for (train, test) in &folds {
//!     assert_eq!(train.len() + test.len(), 8);
//! }
//!
//! // Stratified k-fold: preserves class proportions
//! let skf = StratifiedKFold::new(4);
//! let folds: Vec<_> = skf.split(&y);
//! assert_eq!(folds.len(), 4);
//! ```

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::metrics::accuracy;
use crate::traits::Classifier;

/// K-Fold cross-validation splitter.
///
/// Divides the dataset into `k` consecutive folds. Each fold is used
/// exactly once as a validation set while the remaining `k-1` folds
/// form the training set.
///
/// # Example
///
/// ```
/// use ix_supervised::validation::KFold;
///
/// let kf = KFold::new(5).with_seed(42);
/// let folds = kf.split(100);
/// assert_eq!(folds.len(), 5);
/// // Each sample appears in exactly one test fold
/// let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.clone()).collect();
/// all_test.sort();
/// assert_eq!(all_test, (0..100).collect::<Vec<_>>());
/// ```
pub struct KFold {
    pub k: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl KFold {
    pub fn new(k: usize) -> Self {
        assert!(k >= 2, "k must be at least 2");
        Self { k, shuffle: true, seed: 42 }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Generate `k` train/test index splits for `n` samples.
    pub fn split(&self, n: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert!(n >= self.k, "n ({}) must be >= k ({})", n, self.k);

        let mut indices: Vec<usize> = (0..n).collect();
        if self.shuffle {
            let mut rng = StdRng::seed_from_u64(self.seed);
            indices.shuffle(&mut rng);
        }

        let fold_size = n / self.k;
        let remainder = n % self.k;

        let mut folds = Vec::with_capacity(self.k);
        let mut start = 0;

        for i in 0..self.k {
            // Distribute remainder across first folds
            let size = fold_size + if i < remainder { 1 } else { 0 };
            let end = start + size;
            let test: Vec<usize> = indices[start..end].to_vec();
            let train: Vec<usize> = indices[..start].iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();
            folds.push((train, test));
            start = end;
        }

        folds
    }
}

/// Stratified K-Fold cross-validation splitter.
///
/// Like [`KFold`], but preserves the class distribution in each fold.
/// Essential for imbalanced datasets where random splitting might
/// leave some folds without minority-class samples.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use ix_supervised::validation::StratifiedKFold;
///
/// // 90% class 0, 10% class 1 — stratified split preserves this ratio
/// let y = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
/// let skf = StratifiedKFold::new(2);
/// let folds = skf.split(&y);
/// assert_eq!(folds.len(), 2);
/// ```
pub struct StratifiedKFold {
    pub k: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl StratifiedKFold {
    pub fn new(k: usize) -> Self {
        assert!(k >= 2, "k must be at least 2");
        Self { k, shuffle: true, seed: 42 }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Generate `k` stratified train/test index splits.
    pub fn split(&self, y: &Array1<usize>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n = y.len();
        assert!(n >= self.k, "n ({}) must be >= k ({})", n, self.k);

        let n_classes = *y.iter().max().unwrap() + 1;

        // Group indices by class
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (i, &label) in y.iter().enumerate() {
            class_indices[label].push(i);
        }

        // Shuffle within each class
        if self.shuffle {
            let mut rng = StdRng::seed_from_u64(self.seed);
            for indices in &mut class_indices {
                indices.shuffle(&mut rng);
            }
        }

        // Assign each class's samples round-robin to folds
        let mut fold_indices: Vec<Vec<usize>> = vec![Vec::new(); self.k];
        for class_idx in &class_indices {
            for (i, &idx) in class_idx.iter().enumerate() {
                fold_indices[i % self.k].push(idx);
            }
        }

        // Build train/test pairs
        let mut folds = Vec::with_capacity(self.k);
        for fold_i in 0..self.k {
            let test = fold_indices[fold_i].clone();
            let train: Vec<usize> = fold_indices.iter()
                .enumerate()
                .filter(|(j, _)| *j != fold_i)
                .flat_map(|(_, v)| v.iter().copied())
                .collect();
            folds.push((train, test));
        }

        folds
    }
}

/// Cross-validate a classifier, returning per-fold accuracy scores.
///
/// Trains the classifier on the training fold and evaluates on the test fold
/// for each of `k` splits. The `make_model` closure creates a fresh model
/// for each fold.
///
/// # Example
///
/// ```
/// use ndarray::{array, Array2};
/// use ix_supervised::validation::cross_val_score;
/// use ix_supervised::knn::KNN;
///
/// let x = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
///     5.0, 5.0,  5.5, 5.5,  6.0, 5.0,  5.3, 5.2,
/// ]).unwrap();
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
///
/// let scores = cross_val_score(
///     &x, &y,
///     || KNN::new(3),
///     4,  // 4-fold
///     42, // seed
/// );
/// assert_eq!(scores.len(), 4);
/// let mean = scores.iter().sum::<f64>() / scores.len() as f64;
/// assert!(mean > 0.5, "KNN should do better than random on separable data");
/// ```
pub fn cross_val_score<C, F>(
    x: &Array2<f64>,
    y: &Array1<usize>,
    make_model: F,
    k: usize,
    seed: u64,
) -> Vec<f64>
where
    C: Classifier,
    F: Fn() -> C,
{
    let skf = StratifiedKFold::new(k).with_seed(seed);
    let folds = skf.split(y);

    folds.iter().map(|(train_idx, test_idx)| {
        let x_train = select_rows(x, train_idx);
        let y_train = select_elements(y, train_idx);
        let x_test = select_rows(x, test_idx);
        let y_test = select_elements(y, test_idx);

        let mut model = make_model();
        model.fit(&x_train, &y_train);
        let preds = model.predict(&x_test);
        accuracy(&y_test, &preds)
    }).collect()
}

/// Select rows from a 2D array by index.
fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((indices.len(), x.ncols()), |(r, c)| x[[indices[r], c]])
}

/// Select elements from a 1D array by index.
fn select_elements(y: &Array1<usize>, indices: &[usize]) -> Array1<usize> {
    Array1::from_iter(indices.iter().map(|&i| y[i]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kfold_basic() {
        let kf = KFold::new(5).with_seed(42);
        let folds = kf.split(100);

        assert_eq!(folds.len(), 5);
        // Each fold test set has 20 elements
        for (train, test) in &folds {
            assert_eq!(test.len(), 20);
            assert_eq!(train.len(), 80);
        }
        // All samples appear exactly once in test sets
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.clone()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_kfold_uneven() {
        let kf = KFold::new(3).with_seed(0);
        let folds = kf.split(10);

        assert_eq!(folds.len(), 3);
        let total_test: usize = folds.iter().map(|(_, t)| t.len()).sum();
        assert_eq!(total_test, 10);
    }

    #[test]
    fn test_kfold_no_shuffle() {
        let kf = KFold::new(3).with_shuffle(false);
        let folds = kf.split(9);

        // Without shuffle, first fold test set is [0, 1, 2]
        assert_eq!(folds[0].1, vec![0, 1, 2]);
        assert_eq!(folds[1].1, vec![3, 4, 5]);
        assert_eq!(folds[2].1, vec![6, 7, 8]);
    }

    #[test]
    fn test_stratified_kfold_preserves_ratio() {
        // 8 class-0, 2 class-1
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
        let skf = StratifiedKFold::new(2).with_seed(42);
        let folds = skf.split(&y);

        assert_eq!(folds.len(), 2);
        for (_, test_idx) in &folds {
            let class_1_count = test_idx.iter().filter(|&&i| y[i] == 1).count();
            assert_eq!(class_1_count, 1, "Each fold should have 1 class-1 sample");
        }
    }

    #[test]
    fn test_stratified_kfold_all_samples_covered() {
        let y = array![0, 0, 1, 1, 2, 2, 0, 1];
        let skf = StratifiedKFold::new(2).with_seed(0);
        let folds = skf.split(&y);

        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.clone()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_cross_val_score_separable() {
        use crate::knn::KNN;

        let x = Array2::from_shape_vec((8, 2), vec![
            0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
            5.0, 5.0,  5.5, 5.5,  6.0, 5.0,  5.3, 5.2,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let scores = cross_val_score(
            &x, &y,
            || KNN::new(1),
            2, 42,
        );
        assert_eq!(scores.len(), 2);
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        assert!(mean >= 0.75, "KNN-1 should classify well-separated data, mean_acc={}", mean);
    }

    #[test]
    fn test_cross_val_score_decision_tree() {
        use crate::decision_tree::DecisionTree;

        let x = Array2::from_shape_vec((12, 2), vec![
            0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,  0.8, 0.1,  0.2, 0.7,
            5.0, 5.0,  5.5, 5.5,  6.0, 5.0,  5.3, 5.2,  5.8, 5.1,  5.2, 5.7,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        let scores = cross_val_score(
            &x, &y,
            || DecisionTree::new(5),
            3, 99,
        );
        assert_eq!(scores.len(), 3);
        for &s in &scores {
            assert!(s >= 0.5, "Each fold should be better than random, got {}", s);
        }
    }
}
