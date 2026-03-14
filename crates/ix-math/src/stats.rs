//! Statistical functions: mean, variance, covariance, correlation.

use ndarray::{Array1, Array2, Axis};

use crate::error::MathError;

/// Arithmetic mean of a 1D array.
pub fn mean(x: &Array1<f64>) -> Result<f64, MathError> {
    if x.is_empty() {
        return Err(MathError::EmptyInput);
    }
    Ok(x.mean().unwrap())
}

/// Variance (population) of a 1D array.
pub fn variance(x: &Array1<f64>) -> Result<f64, MathError> {
    if x.is_empty() {
        return Err(MathError::EmptyInput);
    }
    let m = x.mean().unwrap();
    Ok(x.mapv(|v| (v - m).powi(2)).mean().unwrap())
}

/// Sample variance (Bessel's correction: N-1).
pub fn sample_variance(x: &Array1<f64>) -> Result<f64, MathError> {
    let n = x.len();
    if n < 2 {
        return Err(MathError::InvalidParameter("need at least 2 samples".into()));
    }
    let m = x.mean().unwrap();
    let sum_sq: f64 = x.mapv(|v| (v - m).powi(2)).sum();
    Ok(sum_sq / (n as f64 - 1.0))
}

/// Standard deviation (population).
pub fn std_dev(x: &Array1<f64>) -> Result<f64, MathError> {
    variance(x).map(|v| v.sqrt())
}

/// Covariance matrix of column vectors in a matrix.
/// Each row is an observation, each column a variable.
pub fn covariance_matrix(x: &Array2<f64>) -> Result<Array2<f64>, MathError> {
    let n = x.nrows();
    if n < 2 {
        return Err(MathError::InvalidParameter("need at least 2 observations".into()));
    }
    let means = x.mean_axis(Axis(0)).unwrap();
    let centered = x - &means;
    let cov = centered.t().dot(&centered) / (n as f64 - 1.0);
    Ok(cov)
}

/// Pearson correlation matrix.
pub fn correlation_matrix(x: &Array2<f64>) -> Result<Array2<f64>, MathError> {
    let cov = covariance_matrix(x)?;
    let p = cov.nrows();
    let mut corr = Array2::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let denom = (cov[[i, i]] * cov[[j, j]]).sqrt();
            corr[[i, j]] = if denom > 1e-12 {
                cov[[i, j]] / denom
            } else {
                0.0
            };
        }
    }
    Ok(corr)
}

/// Median of a 1D array.
pub fn median(x: &Array1<f64>) -> Result<f64, MathError> {
    if x.is_empty() {
        return Err(MathError::EmptyInput);
    }
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        Ok((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    } else {
        Ok(sorted[n / 2])
    }
}

/// Min and max of a 1D array.
pub fn min_max(x: &Array1<f64>) -> Result<(f64, f64), MathError> {
    if x.is_empty() {
        return Err(MathError::EmptyInput);
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in x.iter() {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&x).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((variance(&x).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_variance() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sample_variance(&x).unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_median_odd() {
        let x = array![3.0, 1.0, 2.0];
        assert!((median(&x).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        let x = array![4.0, 1.0, 3.0, 2.0];
        assert!((median(&x).unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_matrix() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let cov = covariance_matrix(&x).unwrap();
        // Var of col0 = 4, var of col1 = 4, cov = 4
        assert!((cov[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((cov[[1, 1]] - 4.0).abs() < 1e-10);
        assert!((cov[[0, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_perfect() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let corr = correlation_matrix(&x).unwrap();
        assert!((corr[[0, 1]] - 1.0).abs() < 1e-10); // Perfect positive correlation
    }
}
