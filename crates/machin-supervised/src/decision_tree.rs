//! Decision Tree (CART) for classification.
//!
//! Implements CART with Gini impurity splitting, supporting configurable
//! max_depth and min_samples_split.

use ndarray::{Array1, Array2};
use crate::traits::Classifier;

/// A node in the decision tree.
#[derive(Debug, Clone)]
enum Node {
    /// Internal split node: feature index, threshold, left child, right child.
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
    /// Leaf node: class distribution (counts per class) and predicted class.
    Leaf {
        class: usize,
        class_counts: Vec<f64>,
    },
}

/// CART Decision Tree classifier.
pub struct DecisionTree {
    pub max_depth: usize,
    pub min_samples_split: usize,
    root: Option<Node>,
    n_classes: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            min_samples_split: 2,
            root: None,
            n_classes: 0,
        }
    }

    pub fn with_min_samples_split(mut self, min_samples: usize) -> Self {
        self.min_samples_split = min_samples;
        self
    }
}

/// Compute Gini impurity for a label array.
fn gini_impurity(labels: &[usize], n_classes: usize) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let n = labels.len() as f64;
    let mut counts = vec![0usize; n_classes];
    for &l in labels {
        counts[l] += 1;
    }
    1.0 - counts.iter().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
}

/// Find the best split across all features.
fn best_split(
    x: &Array2<f64>,
    y: &[usize],
    n_classes: usize,
) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
    let n = y.len();
    let n_features = x.ncols();
    let parent_gini = gini_impurity(y, n_classes);

    let mut best_gain = 0.0;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_left_idx = Vec::new();
    let mut best_right_idx = Vec::new();
    let mut found = false;

    // Collect row indices for sorting per feature
    let indices: Vec<usize> = (0..n).collect();

    for feat in 0..n_features {
        // Sort indices by feature value
        let mut sorted = indices.clone();
        sorted.sort_by(|&a, &b| x[[a, feat]].partial_cmp(&x[[b, feat]]).unwrap());

        // Sweep through possible thresholds (midpoints between consecutive distinct values)
        for split_pos in 1..n {
            let left_val = x[[sorted[split_pos - 1], feat]];
            let right_val = x[[sorted[split_pos], feat]];
            if (left_val - right_val).abs() < 1e-12 {
                continue; // same value, skip
            }

            let threshold = (left_val + right_val) / 2.0;
            let left_idx: Vec<usize> = sorted[..split_pos].to_vec();
            let right_idx: Vec<usize> = sorted[split_pos..].to_vec();

            let left_labels: Vec<usize> = left_idx.iter().map(|&i| y[i]).collect();
            let right_labels: Vec<usize> = right_idx.iter().map(|&i| y[i]).collect();

            let left_gini = gini_impurity(&left_labels, n_classes);
            let right_gini = gini_impurity(&right_labels, n_classes);

            let n_left = left_idx.len() as f64;
            let n_right = right_idx.len() as f64;
            let weighted_gini = (n_left * left_gini + n_right * right_gini) / n as f64;
            let gain = parent_gini - weighted_gini;

            if gain > best_gain {
                best_gain = gain;
                best_feature = feat;
                best_threshold = threshold;
                best_left_idx = left_idx;
                best_right_idx = right_idx;
                found = true;
            }
        }
    }

    if found {
        Some((best_feature, best_threshold, best_left_idx, best_right_idx))
    } else {
        None
    }
}

/// Recursively build the tree.
fn build_tree(
    x: &Array2<f64>,
    y: &[usize],
    indices: &[usize],
    n_classes: usize,
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
) -> Node {
    let labels: Vec<usize> = indices.iter().map(|&i| y[i]).collect();
    let n = indices.len();

    // Build class counts for this node
    let mut counts = vec![0.0_f64; n_classes];
    for &l in &labels {
        counts[l] += 1.0;
    }
    let predicted_class = counts
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    // Stopping conditions
    if depth >= max_depth || n < min_samples_split || gini_impurity(&labels, n_classes) < 1e-12 {
        return Node::Leaf {
            class: predicted_class,
            class_counts: counts,
        };
    }

    // Build a sub-matrix with only current indices
    let sub_x = Array2::from_shape_fn((n, x.ncols()), |(r, c)| x[[indices[r], c]]);

    if let Some((feature, threshold, left_local, right_local)) =
        best_split(&sub_x, &labels, n_classes)
    {
        // Map local indices back to global indices
        let left_global: Vec<usize> = left_local.iter().map(|&i| indices[i]).collect();
        let right_global: Vec<usize> = right_local.iter().map(|&i| indices[i]).collect();

        if left_global.is_empty() || right_global.is_empty() {
            return Node::Leaf {
                class: predicted_class,
                class_counts: counts,
            };
        }

        let left = build_tree(x, y, &left_global, n_classes, depth + 1, max_depth, min_samples_split);
        let right = build_tree(x, y, &right_global, n_classes, depth + 1, max_depth, min_samples_split);

        Node::Split {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    } else {
        Node::Leaf {
            class: predicted_class,
            class_counts: counts,
        }
    }
}

/// Predict a single sample through the tree.
fn predict_one<'a>(node: &'a Node, sample: &ndarray::ArrayView1<f64>) -> (usize, &'a [f64]) {
    match node {
        Node::Leaf { class, class_counts } => (*class, class_counts),
        Node::Split { feature, threshold, left, right } => {
            if sample[*feature] <= *threshold {
                predict_one(left, sample)
            } else {
                predict_one(right, sample)
            }
        }
    }
}

impl Classifier for DecisionTree {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        let n_classes = *y.iter().max().unwrap() + 1;
        self.n_classes = n_classes;

        let y_slice: Vec<usize> = y.to_vec();
        let indices: Vec<usize> = (0..x.nrows()).collect();

        self.root = Some(build_tree(
            x,
            &y_slice,
            &indices,
            n_classes,
            0,
            self.max_depth,
            self.min_samples_split,
        ));
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let root = self.root.as_ref().expect("Model not fitted");
        Array1::from_iter((0..x.nrows()).map(|i| {
            let (class, _) = predict_one(root, &x.row(i));
            class
        }))
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let root = self.root.as_ref().expect("Model not fitted");
        let n = x.nrows();
        let mut proba = Array2::zeros((n, self.n_classes));

        for i in 0..n {
            let (_, counts) = predict_one(root, &x.row(i));
            let total: f64 = counts.iter().sum();
            if total > 0.0 {
                for (j, &c) in counts.iter().enumerate() {
                    proba[[i, j]] = c / total;
                }
            }
        }

        proba
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::accuracy;
    use ndarray::array;

    #[test]
    fn test_decision_tree_simple() {
        // Simple dataset: class 0 in bottom-left, class 1 in top-right
        let x = array![
            [0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.2, 0.3],
            [3.0, 3.0], [3.5, 3.5], [4.0, 3.0], [3.2, 3.3]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mut tree = DecisionTree::new(5);
        tree.fit(&x, &y);

        let pred = tree.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(acc >= 1.0, "Should perfectly classify linearly separable data, got acc={}", acc);
    }

    #[test]
    fn test_decision_tree_three_classes() {
        let x = array![
            [0.0, 0.0], [0.5, 0.0],
            [5.0, 0.0], [5.5, 0.0],
            [0.0, 5.0], [0.5, 5.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let mut tree = DecisionTree::new(10);
        tree.fit(&x, &y);

        let pred = tree.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(acc >= 1.0, "Should perfectly fit training data with 3 classes, got acc={}", acc);
    }

    #[test]
    fn test_decision_tree_predict_proba() {
        let x = array![
            [0.0, 0.0], [0.5, 0.5],
            [3.0, 3.0], [3.5, 3.5]
        ];
        let y = array![0, 0, 1, 1];

        let mut tree = DecisionTree::new(5);
        tree.fit(&x, &y);

        let proba = tree.predict_proba(&x);
        // Each row should sum to ~1.0
        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1, got {}", row_sum);
        }
    }

    #[test]
    fn test_gini_impurity() {
        // Pure node
        assert!(gini_impurity(&[0, 0, 0], 2) < 1e-10);
        // Maximally impure binary
        let g = gini_impurity(&[0, 1], 2);
        assert!((g - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decision_tree_max_depth_1() {
        // With max_depth=1, should create a single stump
        let x = array![
            [0.0], [1.0], [2.0], [3.0],
            [10.0], [11.0], [12.0], [13.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mut tree = DecisionTree::new(1);
        tree.fit(&x, &y);

        let pred = tree.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(acc >= 1.0, "Stump should separate well-separated data, got acc={}", acc);
    }
}
