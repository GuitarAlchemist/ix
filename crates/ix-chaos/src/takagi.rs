//! Takagi (Blancmange) curve — continuous but nowhere differentiable.
//!
//! `T(t) = Σ_{k=0}^{N} dist(2^k·t, nearest_int) / 2^k`
//!
//! The standard Blancmange curve has fractal dimension 1.5.
//! Uses periodic extension: `t` is mapped to `[0, 1]` via `t - floor(t)`.
//!
//! # Examples
//!
//! ```
//! use ix_chaos::takagi;
//!
//! // Known values of the Blancmange function
//! assert!((takagi::takagi(0.0, 20) - 0.0).abs() < 1e-10);
//! assert!((takagi::takagi(0.5, 20) - 0.5).abs() < 1e-10);
//! assert!((takagi::takagi(1.0, 20) - 0.0).abs() < 1e-10); // periodic
//!
//! // Symmetry: T(t) = T(1 - t)
//! let t = 0.37;
//! assert!((takagi::takagi(t, 20) - takagi::takagi(1.0 - t, 20)).abs() < 1e-10);
//!
//! // Sample 101 points for plotting
//! let curve = takagi::takagi_series(101, 20);
//! assert_eq!(curve.len(), 101);
//! ```

use ndarray::Array1;

/// Maximum number of terms (beyond k=53, 2^k is not exactly representable as f64).
const MAX_TERMS: usize = 53;

/// Evaluate the Takagi function at point `t`.
///
/// Uses periodic extension: `t` is mapped to `[0, 1]` via `t - floor(t)`.
/// `terms` is silently capped at 53.
pub fn takagi(t: f64, terms: usize) -> f64 {
    let terms = terms.min(MAX_TERMS);
    let t = t - t.floor(); // periodic extension to [0, 1]

    let mut sum = 0.0;
    let mut scale = 1.0; // 2^k
    let mut weight = 1.0; // 1/2^k

    for _ in 0..terms {
        let st = (scale * t).fract();
        // dist(x, nearest_int) = min(frac, 1 - frac)
        sum += st.min(1.0 - st) * weight;
        scale *= 2.0;
        weight *= 0.5;
    }

    sum
}

/// Sample the Takagi curve at `n_points` evenly spaced in `[0, 1]`.
///
/// Returns `Array1` of length `n_points`. For `n_points = 0`, returns an empty array.
/// `terms` is silently capped at 53.
///
/// Structured as outer loop over terms, inner loop over points for SIMD auto-vectorization.
pub fn takagi_series(n_points: usize, terms: usize) -> Array1<f64> {
    if n_points == 0 {
        return Array1::zeros(0);
    }
    let terms = terms.min(MAX_TERMS);

    let mut result = Array1::zeros(n_points);
    let step = if n_points == 1 {
        0.0
    } else {
        1.0 / (n_points - 1) as f64
    };

    // Outer loop over terms (k), inner loop over points — SIMD-friendly
    let mut scale = 1.0;
    let mut weight = 1.0;

    for _ in 0..terms {
        for (i, val) in result.iter_mut().enumerate() {
            let t = i as f64 * step;
            let st = (scale * t).fract();
            *val += st.min(1.0 - st) * weight;
        }
        scale *= 2.0;
        weight *= 0.5;
    }

    result
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_takagi_at_zero() {
        assert!((takagi(0.0, 20) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_takagi_at_one() {
        // Periodic: T(1) = T(0) = 0
        assert!((takagi(1.0, 20) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_takagi_at_half() {
        // T(0.5) = 0.5 (maximum of the Blancmange function)
        assert!((takagi(0.5, 20) - 0.5).abs() < TOL);
    }

    #[test]
    fn test_takagi_symmetry() {
        // T(t) = T(1-t) — symmetric about t=0.5
        let t = 0.3;
        assert!((takagi(t, 20) - takagi(1.0 - t, 20)).abs() < TOL);
    }

    #[test]
    fn test_takagi_periodic() {
        // T(t + 1) = T(t)
        let t = 0.37;
        assert!((takagi(t, 20) - takagi(t + 1.0, 20)).abs() < TOL);
        assert!((takagi(t, 20) - takagi(t - 3.0, 20)).abs() < TOL);
    }

    #[test]
    fn test_takagi_non_negative() {
        for i in 0..100 {
            let t = i as f64 / 99.0;
            assert!(takagi(t, 20) >= 0.0);
        }
    }

    #[test]
    fn test_takagi_bounded() {
        // Blancmange maximum is 2/3 (attained at t=1/3)
        for i in 0..100 {
            let t = i as f64 / 99.0;
            let v = takagi(t, 20);
            assert!((0.0..=2.0 / 3.0 + TOL).contains(&v));
        }
    }

    #[test]
    fn test_takagi_terms_capped() {
        // 100 terms should produce same result as 53
        let a = takagi(0.3, 53);
        let b = takagi(0.3, 100);
        assert!((a - b).abs() < TOL);
    }

    #[test]
    fn test_takagi_zero_terms() {
        assert!((takagi(0.5, 0) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_takagi_series_length() {
        let s = takagi_series(50, 20);
        assert_eq!(s.len(), 50);
    }

    #[test]
    fn test_takagi_series_empty() {
        let s = takagi_series(0, 20);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_takagi_series_consistency() {
        // Series should match point-by-point evaluation
        let n = 51;
        let series = takagi_series(n, 20);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            assert!(
                (series[i] - takagi(t, 20)).abs() < TOL,
                "mismatch at i={}, t={}",
                i,
                t
            );
        }
    }

    #[test]
    fn test_takagi_series_endpoints() {
        let s = takagi_series(101, 20);
        assert!((s[0] - 0.0).abs() < TOL);
        assert!((s[100] - 0.0).abs() < TOL);
        assert!((s[50] - 0.5).abs() < TOL);
    }
}
