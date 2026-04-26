//! Time series utilities: rolling windows, lag features, differencing,
//! and temporal train/test splitting.
//!
//! These utilities prepare time-ordered data for ML models by creating
//! features from historical observations and respecting temporal ordering
//! in train/test splits.
//!
//! # Example: Rolling statistics
//!
//! ```
//! use ix_signal::timeseries::{rolling_mean, rolling_std};
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//! let means = rolling_mean(&data, 3);
//! // [NaN, NaN, 2.0, 3.0, 4.0, 5.0, 6.0]
//! assert!((means[2] - 2.0).abs() < 1e-10);
//! assert!((means[6] - 6.0).abs() < 1e-10);
//! ```
//!
//! # Example: Lag features for ML
//!
//! ```
//! use ndarray::Array2;
//! use ix_signal::timeseries::lag_features;
//!
//! let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
//! let (x, y) = lag_features(&data, 3);
//! // x[0] = [10, 20, 30], y[0] = 40
//! // x[1] = [20, 30, 40], y[1] = 50
//! assert_eq!(x.nrows(), 2);
//! assert_eq!(x.ncols(), 3);
//! ```

use ndarray::{Array1, Array2};

/// Streaming drift state for online monitoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriftState {
    Stable,
    Warning,
    Drift,
}

/// Configuration for the Drift Detection Method (DDM).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DdmConfig {
    /// Minimum number of observations before warning/drift states can trigger.
    pub min_samples: usize,
    /// Warning threshold multiplier.
    pub warning_level: f64,
    /// Drift threshold multiplier.
    pub drift_level: f64,
}

impl Default for DdmConfig {
    fn default() -> Self {
        Self {
            min_samples: 30,
            warning_level: 2.0,
            drift_level: 3.0,
        }
    }
}

/// Snapshot of DDM detector state after processing one sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DdmSnapshot {
    pub samples: usize,
    pub error_rate: f64,
    pub std_dev: f64,
    pub min_error_rate: f64,
    pub min_std_dev: f64,
    pub state: DriftState,
}

/// Online Drift Detection Method (DDM) detector.
#[derive(Clone, Debug, PartialEq)]
pub struct DdmDetector {
    config: DdmConfig,
    samples: usize,
    errors: usize,
    min_error_rate: f64,
    min_std_dev: f64,
}

impl DdmDetector {
    pub fn new(config: DdmConfig) -> Self {
        Self {
            config,
            samples: 0,
            errors: 0,
            min_error_rate: f64::INFINITY,
            min_std_dev: f64::INFINITY,
        }
    }

    pub fn update(&mut self, is_error: bool) -> DdmSnapshot {
        self.samples += 1;
        self.errors += usize::from(is_error);

        let samples = self.samples as f64;
        let error_rate = self.errors as f64 / samples;
        let std_dev = (error_rate * (1.0 - error_rate) / samples).sqrt();

        if self.samples >= self.config.min_samples
            && error_rate + std_dev < self.min_error_rate + self.min_std_dev
        {
            self.min_error_rate = error_rate;
            self.min_std_dev = std_dev;
        }

        let state = if self.samples < self.config.min_samples
            || !self.min_error_rate.is_finite()
            || !self.min_std_dev.is_finite()
        {
            DriftState::Stable
        } else {
            let current = error_rate + std_dev;
            let baseline = self.min_error_rate + self.min_std_dev;

            if current > baseline + self.config.drift_level * self.min_std_dev {
                DriftState::Drift
            } else if current > baseline + self.config.warning_level * self.min_std_dev {
                DriftState::Warning
            } else {
                DriftState::Stable
            }
        };

        DdmSnapshot {
            samples: self.samples,
            error_rate,
            std_dev,
            min_error_rate: self.min_error_rate,
            min_std_dev: self.min_std_dev,
            state,
        }
    }
}

/// Run DDM over a sequence of boolean error indicators.
pub fn ddm_detect(errors: &[bool], config: DdmConfig) -> Vec<DdmSnapshot> {
    let mut detector = DdmDetector::new(config);
    errors
        .iter()
        .map(|&is_error| detector.update(is_error))
        .collect()
}

/// Configuration for the Page-Hinkley mean shift detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageHinkleyConfig {
    /// Minimum number of observations before drift can trigger.
    pub min_samples: usize,
    /// Small tolerance subtracted from each centered sample.
    pub delta: f64,
    /// Drift threshold on cumulative deviation.
    pub lambda: f64,
    /// Update weight for the running mean. Use 1.0 for standard averaging.
    pub alpha: f64,
}

impl Default for PageHinkleyConfig {
    fn default() -> Self {
        Self {
            min_samples: 30,
            delta: 0.005,
            lambda: 50.0,
            alpha: 1.0,
        }
    }
}

/// Snapshot of Page-Hinkley state after processing one sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageHinkleySnapshot {
    pub samples: usize,
    pub mean: f64,
    pub cumulative_sum: f64,
    pub min_cumulative_sum: f64,
    pub state: DriftState,
}

/// Online Page-Hinkley detector for abrupt mean shifts.
#[derive(Clone, Debug, PartialEq)]
pub struct PageHinkleyDetector {
    config: PageHinkleyConfig,
    samples: usize,
    mean: f64,
    cumulative_sum: f64,
    min_cumulative_sum: f64,
}

impl PageHinkleyDetector {
    pub fn new(config: PageHinkleyConfig) -> Self {
        Self {
            config,
            samples: 0,
            mean: 0.0,
            cumulative_sum: 0.0,
            min_cumulative_sum: 0.0,
        }
    }

    pub fn update(&mut self, value: f64) -> PageHinkleySnapshot {
        self.samples += 1;

        let weight = (self.config.alpha / self.samples as f64).clamp(0.0, 1.0);
        self.mean += weight * (value - self.mean);

        self.cumulative_sum += value - self.mean - self.config.delta;
        self.min_cumulative_sum = self.min_cumulative_sum.min(self.cumulative_sum);

        let state = if self.samples >= self.config.min_samples
            && self.cumulative_sum - self.min_cumulative_sum > self.config.lambda
        {
            DriftState::Drift
        } else {
            DriftState::Stable
        };

        PageHinkleySnapshot {
            samples: self.samples,
            mean: self.mean,
            cumulative_sum: self.cumulative_sum,
            min_cumulative_sum: self.min_cumulative_sum,
            state,
        }
    }
}

/// Run Page-Hinkley over a sequence of scalar observations.
pub fn page_hinkley_detect(data: &[f64], config: PageHinkleyConfig) -> Vec<PageHinkleySnapshot> {
    let mut detector = PageHinkleyDetector::new(config);
    data.iter().map(|&value| detector.update(value)).collect()
}

/// Compute rolling (moving) mean with a given window size.
///
/// Returns a Vec of the same length as `data`. The first `window - 1`
/// values are `f64::NAN` (incomplete window).
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::rolling_mean;
///
/// let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
/// let rm = rolling_mean(&data, 3);
/// assert!(rm[0].is_nan());
/// assert!(rm[1].is_nan());
/// assert!((rm[2] - 3.0).abs() < 1e-10); // mean(1,3,5) = 3
/// assert!((rm[3] - 5.0).abs() < 1e-10); // mean(3,5,7) = 5
/// assert!((rm[4] - 7.0).abs() < 1e-10); // mean(5,7,9) = 7
/// ```
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    assert!(window >= 1, "window must be >= 1");
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if n < window {
        return result;
    }

    // Initialize first window sum
    let mut sum: f64 = data[..window].iter().sum();
    result[window - 1] = sum / window as f64;

    // Slide the window
    for i in window..n {
        sum += data[i] - data[i - window];
        result[i] = sum / window as f64;
    }

    result
}

/// Compute rolling (moving) standard deviation with a given window size.
///
/// Returns a Vec of the same length as `data`. The first `window - 1`
/// values are `f64::NAN`.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::rolling_std;
///
/// let data = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
/// let rs = rolling_std(&data, 3);
/// assert!((rs[2]).abs() < 1e-10); // std(1,1,1) = 0
/// assert!(rs[5].abs() < 1e-10);   // std(5,5,5) = 0
/// assert!(rs[3] > 0.0);           // std(1,1,5) > 0
/// ```
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    assert!(window >= 1, "window must be >= 1");
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &data[i + 1 - window..=i];
        let mean = slice.iter().sum::<f64>() / window as f64;
        let var = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / window as f64;
        result[i] = var.sqrt();
    }

    result
}

/// Compute rolling min over a window.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::rolling_min;
///
/// let data = vec![5.0, 3.0, 8.0, 1.0, 4.0];
/// let rm = rolling_min(&data, 3);
/// assert!((rm[2] - 3.0).abs() < 1e-10); // min(5,3,8) = 3
/// assert!((rm[3] - 1.0).abs() < 1e-10); // min(3,8,1) = 1
/// ```
pub fn rolling_min(data: &[f64], window: usize) -> Vec<f64> {
    rolling_apply(data, window, |slice| {
        slice.iter().cloned().fold(f64::INFINITY, f64::min)
    })
}

/// Compute rolling max over a window.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::rolling_max;
///
/// let data = vec![5.0, 3.0, 8.0, 1.0, 4.0];
/// let rm = rolling_max(&data, 3);
/// assert!((rm[2] - 8.0).abs() < 1e-10); // max(5,3,8) = 8
/// assert!((rm[3] - 8.0).abs() < 1e-10); // max(3,8,1) = 8
/// ```
pub fn rolling_max(data: &[f64], window: usize) -> Vec<f64> {
    rolling_apply(data, window, |slice| {
        slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    })
}

/// Generic rolling window apply.
fn rolling_apply<F>(data: &[f64], window: usize, f: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    assert!(window >= 1, "window must be >= 1");
    let n = data.len();
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        result[i] = f(&data[i + 1 - window..=i]);
    }
    result
}

/// Compute first-order differencing: `diff[i] = data[i] - data[i-1]`.
///
/// Returns a Vec of length `data.len() - 1`.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::difference;
///
/// let data = vec![10.0, 12.0, 15.0, 13.0];
/// let diff = difference(&data, 1);
/// assert_eq!(diff, vec![2.0, 3.0, -2.0]);
/// ```
pub fn difference(data: &[f64], order: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    for _ in 0..order {
        if result.len() <= 1 {
            return vec![];
        }
        result = result.windows(2).map(|w| w[1] - w[0]).collect();
    }
    result
}

/// Compute percentage change: `pct[i] = (data[i] - data[i-1]) / data[i-1]`.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::pct_change;
///
/// let data = vec![100.0, 110.0, 99.0];
/// let pct = pct_change(&data);
/// assert!((pct[0] - 0.1).abs() < 1e-10);   // +10%
/// assert!((pct[1] - (-0.1)).abs() < 1e-10); // -10%
/// ```
pub fn pct_change(data: &[f64]) -> Vec<f64> {
    data.windows(2)
        .map(|w| {
            if w[0].abs() > 1e-15 {
                (w[1] - w[0]) / w[0]
            } else {
                0.0
            }
        })
        .collect()
}

/// Create lag features for supervised learning from a time series.
///
/// Given a series and `n_lags`, produces:
/// - `x`: matrix where each row is `[data[i-n_lags], ..., data[i-1]]`
/// - `y`: target vector `data[i]`
///
/// The output has `data.len() - n_lags` samples.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::lag_features;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let (x, y) = lag_features(&data, 2);
/// // x = [[1, 2], [2, 3], [3, 4]]
/// // y = [3, 4, 5]
/// assert_eq!(x.nrows(), 3);
/// assert_eq!(x.ncols(), 2);
/// assert_eq!(y.len(), 3);
/// assert!((x[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((y[2] - 5.0).abs() < 1e-10);
/// ```
pub fn lag_features(data: &[f64], n_lags: usize) -> (Array2<f64>, Array1<f64>) {
    assert!(data.len() > n_lags, "data length must exceed n_lags");
    let n_samples = data.len() - n_lags;

    let x = Array2::from_shape_fn((n_samples, n_lags), |(i, j)| data[i + j]);
    let y = Array1::from_iter((n_lags..data.len()).map(|i| data[i]));

    (x, y)
}

/// Create lag features with rolling statistics appended.
///
/// For each sample, appends rolling mean and rolling std (computed from
/// the lag window) as additional features. Output has `n_lags + 2` columns.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::lag_features_with_stats;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let (x, y) = lag_features_with_stats(&data, 3);
/// assert_eq!(x.ncols(), 5); // 3 lags + rolling_mean + rolling_std
/// assert_eq!(x.nrows(), 4); // 7 - 3 = 4 samples
/// ```
pub fn lag_features_with_stats(data: &[f64], n_lags: usize) -> (Array2<f64>, Array1<f64>) {
    assert!(data.len() > n_lags, "data length must exceed n_lags");
    let n_samples = data.len() - n_lags;

    let x = Array2::from_shape_fn((n_samples, n_lags + 2), |(i, j)| {
        if j < n_lags {
            data[i + j]
        } else {
            let window = &data[i..i + n_lags];
            let mean = window.iter().sum::<f64>() / n_lags as f64;
            if j == n_lags {
                mean
            } else {
                let var = window.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_lags as f64;
                var.sqrt()
            }
        }
    });
    let y = Array1::from_iter((n_lags..data.len()).map(|i| data[i]));

    (x, y)
}

/// Temporal train/test split that respects time ordering.
///
/// Unlike random splits, this ensures all training data comes before
/// test data — preventing data leakage from the future.
///
/// Returns `(train_indices, test_indices)`.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::temporal_split;
///
/// let (train, test) = temporal_split(100, 0.8);
/// assert_eq!(train.len(), 80);
/// assert_eq!(test.len(), 20);
/// assert_eq!(*train.last().unwrap(), 79);
/// assert_eq!(test[0], 80);
/// ```
pub fn temporal_split(n: usize, train_ratio: f64) -> (Vec<usize>, Vec<usize>) {
    assert!(
        train_ratio > 0.0 && train_ratio < 1.0,
        "train_ratio must be in (0, 1)"
    );
    let split = (n as f64 * train_ratio).floor() as usize;
    let train: Vec<usize> = (0..split).collect();
    let test: Vec<usize> = (split..n).collect();
    (train, test)
}

/// Expanding (cumulative) mean: mean of all values up to index i.
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::expanding_mean;
///
/// let data = vec![2.0, 4.0, 6.0, 8.0];
/// let em = expanding_mean(&data);
/// assert!((em[0] - 2.0).abs() < 1e-10);
/// assert!((em[1] - 3.0).abs() < 1e-10); // (2+4)/2
/// assert!((em[2] - 4.0).abs() < 1e-10); // (2+4+6)/3
/// assert!((em[3] - 5.0).abs() < 1e-10); // (2+4+6+8)/4
/// ```
pub fn expanding_mean(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for (i, &v) in data.iter().enumerate() {
        sum += v;
        result.push(sum / (i + 1) as f64);
    }
    result
}

/// Exponentially weighted moving average (EWMA).
///
/// `ewma[0] = data[0]`, `ewma[i] = alpha * data[i] + (1 - alpha) * ewma[i-1]`
///
/// # Example
///
/// ```
/// use ix_signal::timeseries::ewma;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = ewma(&data, 0.5);
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// // result[1] = 0.5 * 2 + 0.5 * 1 = 1.5
/// assert!((result[1] - 1.5).abs() < 1e-10);
/// ```
pub fn ewma(data: &[f64], alpha: f64) -> Vec<f64> {
    assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
    let mut result = Vec::with_capacity(data.len());
    if data.is_empty() {
        return result;
    }
    result.push(data[0]);
    for i in 1..data.len() {
        let prev = result[i - 1];
        result.push(alpha * data[i] + (1.0 - alpha) * prev);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rm = rolling_mean(&data, 3);
        assert!(rm[0].is_nan());
        assert!(rm[1].is_nan());
        assert!((rm[2] - 2.0).abs() < 1e-10);
        assert!((rm[3] - 3.0).abs() < 1e-10);
        assert!((rm[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean_window_1() {
        let data = vec![3.0, 5.0, 7.0];
        let rm = rolling_mean(&data, 1);
        assert!((rm[0] - 3.0).abs() < 1e-10);
        assert!((rm[1] - 5.0).abs() < 1e-10);
        assert!((rm[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_std_constant() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let rs = rolling_std(&data, 3);
        assert!(rs[0].is_nan());
        assert!(rs[1].is_nan());
        assert!(rs[2].abs() < 1e-10); // zero variance
        assert!(rs[3].abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min_max() {
        let data = vec![5.0, 3.0, 8.0, 1.0, 4.0];
        let rmin = rolling_min(&data, 3);
        let rmax = rolling_max(&data, 3);
        assert!((rmin[2] - 3.0).abs() < 1e-10);
        assert!((rmax[2] - 8.0).abs() < 1e-10);
        assert!((rmin[3] - 1.0).abs() < 1e-10);
        assert!((rmax[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_difference_order_1() {
        let data = vec![10.0, 12.0, 15.0, 13.0];
        let diff = difference(&data, 1);
        assert_eq!(diff, vec![2.0, 3.0, -2.0]);
    }

    #[test]
    fn test_difference_order_2() {
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let diff1 = difference(&data, 1); // [2, 3, 4, 5]
        let diff2 = difference(&data, 2); // [1, 1, 1]
        assert_eq!(diff1, vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(diff2, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_pct_change() {
        let data = vec![100.0, 110.0, 99.0];
        let pct = pct_change(&data);
        assert!((pct[0] - 0.1).abs() < 1e-10);
        assert!((pct[1] - (-0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_lag_features() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (x, y) = lag_features(&data, 2);
        assert_eq!(x.nrows(), 3);
        assert_eq!(x.ncols(), 2);
        assert!((x[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((x[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_lag_features_with_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (x, y) = lag_features_with_stats(&data, 3);
        assert_eq!(x.ncols(), 5); // 3 lags + mean + std
        assert_eq!(x.nrows(), 3);
        // First row: lags=[1,2,3], mean=2, std=sqrt(2/3)
        assert!((x[[0, 3]] - 2.0).abs() < 1e-10); // rolling mean
        assert!((y[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_split() {
        let (train, test) = temporal_split(100, 0.8);
        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);
        assert_eq!(train[0], 0);
        assert_eq!(*train.last().unwrap(), 79);
        assert_eq!(test[0], 80);
        assert_eq!(*test.last().unwrap(), 99);
    }

    #[test]
    fn test_expanding_mean() {
        let data = vec![2.0, 4.0, 6.0];
        let em = expanding_mean(&data);
        assert!((em[0] - 2.0).abs() < 1e-10);
        assert!((em[1] - 3.0).abs() < 1e-10);
        assert!((em[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ewma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewma(&data, 0.5);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.5).abs() < 1e-10);
        // result[2] = 0.5 * 3 + 0.5 * 1.5 = 2.25
        assert!((result[2] - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_ewma_alpha_1() {
        // alpha=1 means no smoothing
        let data = vec![1.0, 5.0, 3.0];
        let result = ewma(&data, 1.0);
        assert_eq!(result, data);
    }

    #[test]
    fn test_ddm_stays_stable_on_low_error_stream() {
        let errors = vec![false; 80];
        let snapshots = ddm_detect(&errors, DdmConfig::default());
        assert_eq!(snapshots.len(), errors.len());
        assert!(snapshots
            .iter()
            .all(|snapshot| snapshot.state == DriftState::Stable));
    }

    #[test]
    fn test_ddm_detects_error_rate_jump() {
        let mut errors = vec![false; 60];
        errors.extend(vec![true; 40]);
        let snapshots = ddm_detect(&errors, DdmConfig::default());
        assert!(snapshots
            .iter()
            .skip(60)
            .any(|snapshot| snapshot.state != DriftState::Stable));
        assert!(snapshots
            .iter()
            .any(|snapshot| snapshot.state == DriftState::Drift));
    }

    #[test]
    fn test_page_hinkley_stays_stable_on_constant_series() {
        let data = vec![1.0; 120];
        let snapshots = page_hinkley_detect(
            &data,
            PageHinkleyConfig {
                min_samples: 20,
                delta: 0.01,
                lambda: 5.0,
                alpha: 1.0,
            },
        );
        assert!(snapshots
            .iter()
            .all(|snapshot| snapshot.state == DriftState::Stable));
    }

    #[test]
    fn test_page_hinkley_detects_mean_shift() {
        let mut data = vec![1.0; 80];
        data.extend(vec![4.0; 40]);
        let snapshots = page_hinkley_detect(
            &data,
            PageHinkleyConfig {
                min_samples: 20,
                delta: 0.05,
                lambda: 10.0,
                alpha: 1.0,
            },
        );
        assert!(snapshots
            .iter()
            .any(|snapshot| snapshot.state == DriftState::Drift));
    }
}
