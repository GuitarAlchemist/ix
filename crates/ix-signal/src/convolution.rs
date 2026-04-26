//! Linear and circular convolution.

use crate::fft::{self, Complex};

/// Linear convolution of two signals (direct method).
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] += a[i] * b[j];
        }
    }
    result
}

/// Fast convolution using FFT (for large signals).
pub fn convolve_fft(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = (a.len() + b.len() - 1).next_power_of_two();

    let mut a_padded: Vec<Complex> = a.iter().map(|&x| Complex::from_real(x)).collect();
    a_padded.resize(n, Complex::zero());

    let mut b_padded: Vec<Complex> = b.iter().map(|&x| Complex::from_real(x)).collect();
    b_padded.resize(n, Complex::zero());

    let fa = fft::fft(&a_padded);
    let fb = fft::fft(&b_padded);

    // Pointwise multiply
    let fc: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&a, &b)| a * b).collect();

    let result = fft::ifft(&fc);
    result
        .iter()
        .take(a.len() + b.len() - 1)
        .map(|c| c.re)
        .collect()
}

/// Circular convolution of two equal-length signals.
pub fn circular_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "Circular convolution requires equal lengths"
    );
    let n = a.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += a[j] * b[(i + n - j) % n];
        }
    }
    result
}

/// Cross-correlation (convolution with reversed b).
pub fn cross_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();
    convolve(a, &b_rev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let result = convolve(&a, &b);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve_fft_matches_direct() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.0, 0.5];
        let direct = convolve(&a, &b);
        let fast = convolve_fft(&a, &b);

        for (d, f) in direct.iter().zip(fast.iter()) {
            assert!((d - f).abs() < 1e-8, "{} != {}", d, f);
        }
    }
}
