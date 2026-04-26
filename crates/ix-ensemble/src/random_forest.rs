//! Random Forest classifier.
//!
//! Ensemble of decision trees trained on bootstrap samples with random
//! feature subsets (sqrt(n_features) features per split). Predictions
//! are made by majority vote across all trees.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use ix_supervised::decision_tree::DecisionTree;
use ix_supervised::traits::Classifier;

use crate::traits::EnsembleClassifier;

/// Random Forest classifier.
pub struct RandomForest {
    pub n_trees: usize,
    pub max_depth: usize,
    pub max_features: Option<usize>,
    pub seed: u64,
    trees: Vec<(DecisionTree, Vec<usize>)>, // (tree, feature_indices)
    n_classes: usize,
}

impl RandomForest {
    pub fn new(n_trees: usize, max_depth: usize) -> Self {
        Self {
            n_trees,
            max_depth,
            max_features: None,
            seed: 42,
            trees: Vec::new(),
            n_classes: 0,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }
}

impl EnsembleClassifier for RandomForest {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        let (n, p) = x.dim();
        self.n_classes = *y.iter().max().unwrap() + 1;

        // Default: sqrt(n_features)
        let max_features = self
            .max_features
            .unwrap_or((p as f64).sqrt().ceil() as usize)
            .min(p);

        let mut rng = StdRng::seed_from_u64(self.seed);
        self.trees.clear();

        for _ in 0..self.n_trees {
            // Bootstrap sample: sample n indices with replacement
            let sample_indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();

            // Random feature subset
            let mut all_features: Vec<usize> = (0..p).collect();
            // Fisher-Yates partial shuffle for first max_features elements
            for i in 0..max_features {
                let j = rng.random_range(i..p);
                all_features.swap(i, j);
            }
            let feature_indices: Vec<usize> = all_features[..max_features].to_vec();

            // Build sub-dataset with selected features
            let sub_x = Array2::from_shape_fn((n, max_features), |(r, c)| {
                x[[sample_indices[r], feature_indices[c]]]
            });
            let sub_y = Array1::from_iter(sample_indices.iter().map(|&i| y[i]));

            let mut tree = DecisionTree::new(self.max_depth);
            tree.fit(&sub_x, &sub_y);

            self.trees.push((tree, feature_indices));
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let proba = self.predict_proba(x);
        Array1::from_iter((0..x.nrows()).map(|i| {
            proba
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        }))
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut proba = Array2::zeros((n, self.n_classes));

        for (tree, feature_indices) in &self.trees {
            // Select the same feature subset used during training
            let sub_x = Array2::from_shape_fn((n, feature_indices.len()), |(r, c)| {
                x[[r, feature_indices[c]]]
            });
            let tree_proba = tree.predict_proba(&sub_x);

            // Accumulate (columns may differ if tree saw fewer classes, handle gracefully)
            let tree_cols = tree_proba.ncols().min(self.n_classes);
            for i in 0..n {
                for j in 0..tree_cols {
                    proba[[i, j]] += tree_proba[[i, j]];
                }
            }
        }

        // Average
        let n_trees = self.trees.len() as f64;
        proba.mapv_inplace(|v| v / n_trees);

        proba
    }

    fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ix_supervised::metrics::accuracy;
    use ndarray::array;

    #[test]
    fn test_random_forest_basic() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.3, 0.2],
            [5.0, 5.0],
            [5.5, 5.5],
            [6.0, 5.0],
            [5.3, 5.2]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mut rf = RandomForest::new(10, 5).with_seed(42);
        rf.fit(&x, &y);

        let pred = rf.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(
            acc >= 0.9,
            "Random forest should classify separable data well, got acc={}",
            acc
        );
    }

    #[test]
    fn test_random_forest_predict_proba() {
        let x = array![[0.0, 0.0], [0.5, 0.5], [5.0, 5.0], [5.5, 5.5]];
        let y = array![0, 0, 1, 1];

        let mut rf = RandomForest::new(10, 5).with_seed(42);
        rf.fit(&x, &y);

        let proba = rf.predict_proba(&x);
        for i in 0..proba.nrows() {
            let sum: f64 = proba.row(i).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities should sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_random_forest_n_estimators() {
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![0, 1];

        let mut rf = RandomForest::new(7, 3).with_seed(123);
        rf.fit(&x, &y);

        assert_eq!(rf.n_estimators(), 7);
    }

    #[test]
    fn test_random_forest_multiclass() {
        let x = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [5.0, 0.0],
            [5.5, 0.0],
            [0.0, 5.0],
            [0.5, 5.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let mut rf = RandomForest::new(20, 5).with_seed(42);
        rf.fit(&x, &y);

        let pred = rf.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(
            acc >= 0.8,
            "Random forest should handle 3 classes, got acc={}",
            acc
        );
    }
}
