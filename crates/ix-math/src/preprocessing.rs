//! Preprocessing helpers: scaling, splitting, NaN handling, task inference.

use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashSet;

use crate::error::MathError;

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

/// Z-score normalisation: `(x - mean) / std` per column.
#[derive(Debug, Clone)]
pub struct StandardScaler {
    /// Per-column means.
    pub means: Array1<f64>,
    /// Per-column standard deviations (zero-variance columns use 1.0).
    pub stds: Array1<f64>,
}

impl StandardScaler {
    /// Compute column means and standard deviations from `x`.
    pub fn fit(x: &Array2<f64>) -> Result<Self, MathError> {
        let (n, _p) = x.dim();
        if n == 0 {
            return Err(MathError::EmptyInput);
        }
        let means = x.mean_axis(Axis(0)).unwrap();

        let stds = if n == 1 {
            // Single row → no variance; use 1.0 everywhere.
            Array1::ones(means.len())
        } else {
            let centered = x - &means;
            let var = centered.mapv(|v| v * v).mean_axis(Axis(0)).unwrap();
            var.mapv(|v| {
                let s = v.sqrt();
                if s == 0.0 { 1.0 } else { s }
            })
        };

        Ok(Self { means, stds })
    }

    /// Transform `x` using the fitted parameters.
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        (x - &self.means) / &self.stds
    }

    /// Fit and transform in one step.
    pub fn fit_transform(x: &Array2<f64>) -> Result<(Self, Array2<f64>), MathError> {
        let scaler = Self::fit(x)?;
        let transformed = scaler.transform(x);
        Ok((scaler, transformed))
    }

    /// Undo the transformation: `x * std + mean`.
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Array2<f64> {
        x * &self.stds + &self.means
    }
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------

/// Scale features to a given range (default `[0, 1]`).
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    /// Per-column minimums (set after [`fit`]).
    pub mins: Array1<f64>,
    /// Per-column ranges `max - min` (constant columns use 1.0).
    pub ranges: Array1<f64>,
    /// Target output range.
    pub feature_range: (f64, f64),
}

impl MinMaxScaler {
    /// Create an unfitted scaler with the desired output range.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self {
            mins: Array1::zeros(0),
            ranges: Array1::zeros(0),
            feature_range: (lo, hi),
        }
    }

    /// Compute per-column min and range from `x`.
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), MathError> {
        let (n, _p) = x.dim();
        if n == 0 {
            return Err(MathError::EmptyInput);
        }
        let col_min = x
            .axis_iter(Axis(0))
            .fold(x.row(0).to_owned(), |acc, row| {
                ndarray::Zip::from(&acc)
                    .and(&row)
                    .map_collect(|&a, &b| a.min(b))
            });
        let col_max = x
            .axis_iter(Axis(0))
            .fold(x.row(0).to_owned(), |acc, row| {
                ndarray::Zip::from(&acc)
                    .and(&row)
                    .map_collect(|&a, &b| a.max(b))
            });
        let ranges = (&col_max - &col_min).mapv(|r| if r == 0.0 { 1.0 } else { r });
        self.mins = col_min;
        self.ranges = ranges;
        Ok(())
    }

    /// Transform `x` into `[lo, hi]` using fitted parameters.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, MathError> {
        if self.mins.is_empty() {
            return Err(MathError::InvalidParameter(
                "scaler has not been fitted".into(),
            ));
        }
        let (lo, hi) = self.feature_range;
        let scaled_01 = (x - &self.mins) / &self.ranges;
        Ok(scaled_01 * (hi - lo) + lo)
    }

    /// Convenience: fit then transform.
    pub fn fit_transform(
        x: &Array2<f64>,
        lo: f64,
        hi: f64,
    ) -> Result<(Self, Array2<f64>), MathError> {
        let mut scaler = Self::new(lo, hi);
        scaler.fit(x)?;
        let transformed = scaler.transform(x)?;
        Ok((scaler, transformed))
    }
}

// ---------------------------------------------------------------------------
// Train / test split
// ---------------------------------------------------------------------------

/// Result of [`train_test_split`].
pub struct SplitResult {
    pub x_train: Array2<f64>,
    pub x_test: Array2<f64>,
    pub y_train: Array1<f64>,
    pub y_test: Array1<f64>,
}

/// Randomly split `(x, y)` into train and test sets.
///
/// `test_ratio` must be in `(0.0, 1.0)`. Uses a deterministic RNG seeded
/// with `seed` for reproducibility.
pub fn train_test_split(
    x: &Array2<f64>,
    y: &Array1<f64>,
    test_ratio: f64,
    seed: u64,
) -> Result<SplitResult, MathError> {
    let n = x.nrows();
    if n == 0 {
        return Err(MathError::EmptyInput);
    }
    if x.nrows() != y.len() {
        return Err(MathError::DimensionMismatch {
            expected: x.nrows(),
            got: y.len(),
        });
    }
    if test_ratio <= 0.0 || test_ratio >= 1.0 {
        return Err(MathError::InvalidParameter(
            "test_ratio must be in (0.0, 1.0)".into(),
        ));
    }

    let n_test = (n as f64 * test_ratio).round().max(1.0).min((n - 1) as f64) as usize;
    let n_train = n - n_test;

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let train_idx = &indices[..n_train];
    let test_idx = &indices[n_train..];

    let x_train = x.select(Axis(0), train_idx);
    let x_test = x.select(Axis(0), test_idx);
    let y_train = y.select(Axis(0), train_idx);
    let y_test = y.select(Axis(0), test_idx);

    Ok(SplitResult {
        x_train,
        x_test,
        y_train,
        y_test,
    })
}

// ---------------------------------------------------------------------------
// NaN handling
// ---------------------------------------------------------------------------

/// Drop any row that contains at least one NaN.
pub fn drop_nan_rows(x: &Array2<f64>) -> Array2<f64> {
    let keep: Vec<usize> = (0..x.nrows())
        .filter(|&i| !x.row(i).iter().any(|v| v.is_nan()))
        .collect();

    if keep.is_empty() {
        Array2::zeros((0, x.ncols()))
    } else {
        x.select(Axis(0), &keep)
    }
}

// ---------------------------------------------------------------------------
// Task-type inference
// ---------------------------------------------------------------------------

/// Automatically inferred ML task type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferredTask {
    BinaryClassification,
    MulticlassClassification { n_classes: usize },
    Regression,
}

/// Heuristic: if every value in `y` is a non-negative integer and the number
/// of distinct values is at most `cardinality_threshold`, treat it as
/// classification; otherwise regression.
pub fn infer_task_type(y: &[f64], cardinality_threshold: usize) -> InferredTask {
    let all_integer = y.iter().all(|&v| v.fract() == 0.0 && v >= 0.0 && v.is_finite());
    if !all_integer {
        return InferredTask::Regression;
    }

    let unique: HashSet<u64> = y.iter().map(|&v| v as u64).collect();
    let n_classes = unique.len();

    if n_classes > cardinality_threshold {
        return InferredTask::Regression;
    }

    if n_classes == 2 {
        InferredTask::BinaryClassification
    } else {
        InferredTask::MulticlassClassification { n_classes }
    }
}

// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    // -- StandardScaler -------------------------------------------------

    #[test]
    fn standard_scaler_roundtrip() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (scaler, transformed) = StandardScaler::fit_transform(&x).unwrap();
        let recovered = scaler.inverse_transform(&transformed);
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-12, "roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn standard_scaler_zero_variance() {
        let x = array![[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]];
        let scaler = StandardScaler::fit(&x).unwrap();
        assert_eq!(scaler.stds[0], 1.0);
        assert_eq!(scaler.stds[1], 1.0);
        let t = scaler.transform(&x);
        // All values should be 0 because (x - mean) == 0
        assert!(t.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn standard_scaler_single_row() {
        let x = array![[3.0, 7.0]];
        let scaler = StandardScaler::fit(&x).unwrap();
        assert_eq!(scaler.stds, array![1.0, 1.0]);
    }

    #[test]
    fn standard_scaler_empty() {
        let x = Array2::<f64>::zeros((0, 3));
        assert!(StandardScaler::fit(&x).is_err());
    }

    // -- MinMaxScaler ---------------------------------------------------

    #[test]
    fn minmax_scaler_01() {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let (_, t) = MinMaxScaler::fit_transform(&x, 0.0, 1.0).unwrap();
        for &v in t.iter() {
            assert!(v >= -1e-12 && v <= 1.0 + 1e-12, "out of range: {v}");
        }
        // min row should be 0, max row should be 1
        assert!((t[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((t[[2, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn minmax_scaler_custom_range() {
        let x = array![[0.0], [5.0], [10.0]];
        let (_, t) = MinMaxScaler::fit_transform(&x, -1.0, 1.0).unwrap();
        assert!((t[[0, 0]] - (-1.0)).abs() < 1e-12);
        assert!((t[[1, 0]] - 0.0).abs() < 1e-12);
        assert!((t[[2, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn minmax_scaler_constant_column() {
        let x = array![[5.0], [5.0], [5.0]];
        let (scaler, _) = MinMaxScaler::fit_transform(&x, 0.0, 1.0).unwrap();
        assert_eq!(scaler.ranges[0], 1.0);
    }

    #[test]
    fn minmax_scaler_empty() {
        let x = Array2::<f64>::zeros((0, 2));
        assert!(MinMaxScaler::fit_transform(&x, 0.0, 1.0).is_err());
    }

    // -- train_test_split -----------------------------------------------

    #[test]
    fn split_reproducibility() {
        let x = Array2::from_shape_fn((100, 3), |(i, j)| (i * 3 + j) as f64);
        let y = Array1::from_shape_fn(100, |i| i as f64);
        let s1 = train_test_split(&x, &y, 0.2, 42).unwrap();
        let s2 = train_test_split(&x, &y, 0.2, 42).unwrap();
        assert_eq!(s1.x_train, s2.x_train);
        assert_eq!(s1.y_test, s2.y_test);
    }

    #[test]
    fn split_ratio() {
        let x = Array2::from_shape_fn((100, 2), |(i, j)| (i + j) as f64);
        let y = Array1::from_shape_fn(100, |i| i as f64);
        let s = train_test_split(&x, &y, 0.3, 0).unwrap();
        // 30% of 100 = 30, tolerance ±1
        assert!((s.x_test.nrows() as i64 - 30).unsigned_abs() <= 1);
        assert_eq!(s.x_train.nrows() + s.x_test.nrows(), 100);
    }

    #[test]
    fn split_invalid_ratio() {
        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];
        assert!(train_test_split(&x, &y, 0.0, 0).is_err());
        assert!(train_test_split(&x, &y, 1.0, 0).is_err());
        assert!(train_test_split(&x, &y, -0.1, 0).is_err());
    }

    #[test]
    fn split_empty() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        assert!(train_test_split(&x, &y, 0.5, 0).is_err());
    }

    // -- drop_nan_rows --------------------------------------------------

    #[test]
    fn drop_nan_basic() {
        let x = array![
            [1.0, 2.0],
            [f64::NAN, 3.0],
            [4.0, 5.0],
            [6.0, f64::NAN]
        ];
        let clean = drop_nan_rows(&x);
        assert_eq!(clean.nrows(), 2);
        assert_eq!(clean, array![[1.0, 2.0], [4.0, 5.0]]);
    }

    #[test]
    fn drop_nan_all_nan() {
        let x = array![[f64::NAN, 1.0], [2.0, f64::NAN]];
        let clean = drop_nan_rows(&x);
        assert_eq!(clean.nrows(), 0);
    }

    // -- infer_task_type ------------------------------------------------

    #[test]
    fn infer_binary() {
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        assert_eq!(infer_task_type(&y, 10), InferredTask::BinaryClassification);
    }

    #[test]
    fn infer_multiclass() {
        let y: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        assert_eq!(
            infer_task_type(&y, 20),
            InferredTask::MulticlassClassification { n_classes: 10 }
        );
    }

    #[test]
    fn infer_regression() {
        let y = vec![0.1, 0.5, 1.3, 2.7, 3.14];
        assert_eq!(infer_task_type(&y, 10), InferredTask::Regression);
    }

    #[test]
    fn infer_high_cardinality_integers() {
        // Many distinct integers beyond threshold → regression
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(infer_task_type(&y, 10), InferredTask::Regression);
    }
}
