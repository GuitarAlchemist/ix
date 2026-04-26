//! k-Nearest Neighbors classifier.

use ndarray::{Array1, Array2};

use crate::traits::Classifier;
use ix_math::distance::euclidean;

/// k-Nearest Neighbors classifier (brute-force).
pub struct KNN {
    pub k: usize,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<usize>>,
    n_classes: usize,
}

impl KNN {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            x_train: None,
            y_train: None,
            n_classes: 0,
        }
    }
}

impl Classifier for KNN {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        self.n_classes = *y.iter().max().unwrap() + 1;
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");

        Array1::from_iter((0..x.nrows()).map(|i| {
            let point = x.row(i).to_owned();
            let mut distances: Vec<(f64, usize)> = (0..x_train.nrows())
                .map(|j| {
                    let train_point = x_train.row(j).to_owned();
                    (euclidean(&point, &train_point).unwrap(), y_train[j])
                })
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Vote among k nearest
            let mut votes = vec![0usize; self.n_classes];
            for &(_, label) in distances.iter().take(self.k) {
                votes[label] += 1;
            }
            votes.iter().enumerate().max_by_key(|(_, &v)| v).unwrap().0
        }))
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");
        let n = x.nrows();
        let mut proba = Array2::zeros((n, self.n_classes));

        for i in 0..n {
            let point = x.row(i).to_owned();
            let mut distances: Vec<(f64, usize)> = (0..x_train.nrows())
                .map(|j| {
                    let train_point = x_train.row(j).to_owned();
                    (euclidean(&point, &train_point).unwrap(), y_train[j])
                })
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for &(_, label) in distances.iter().take(self.k) {
                proba[[i, label]] += 1.0 / self.k as f64;
            }
        }

        proba
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_knn_simple() {
        let x_train = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0]
        ];
        let y_train = array![0, 0, 0, 1, 1, 1];

        let mut knn = KNN::new(3);
        knn.fit(&x_train, &y_train);

        let x_test = array![[0.5, 0.5], [11.5, 11.5]];
        let pred = knn.predict(&x_test);
        assert_eq!(pred[0], 0);
        assert_eq!(pred[1], 1);
    }
}
