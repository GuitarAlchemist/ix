//! Fast Fourier Transform (Cooley-Tukey radix-2 DIT).
//!
//! Pure Rust implementation — no FFTW dependency.
//! Input length must be a power of 2 (or will be zero-padded).

use std::f64::consts::PI;

/// A complex number (re, im).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

/// Forward FFT (Cooley-Tukey radix-2 DIT).
/// Input is zero-padded to next power of 2 if needed.
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len().next_power_of_two();
    let mut data: Vec<Complex> = input.to_vec();
    data.resize(n, Complex::zero());

    fft_in_place(&mut data, false);
    data
}

/// Inverse FFT. Returns time-domain signal.
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len().next_power_of_two();
    let mut data: Vec<Complex> = input.to_vec();
    data.resize(n, Complex::zero());

    fft_in_place(&mut data, true);

    let scale = 1.0 / n as f64;
    data.iter_mut().for_each(|c| {
        c.re *= scale;
        c.im *= scale;
    });
    data
}

/// FFT of real-valued signal.
pub fn rfft(signal: &[f64]) -> Vec<Complex> {
    let input: Vec<Complex> = signal.iter().map(|&x| Complex::from_real(x)).collect();
    fft(&input)
}

/// Inverse FFT returning real values (discards imaginary parts).
pub fn irfft(spectrum: &[Complex]) -> Vec<f64> {
    ifft(spectrum).iter().map(|c| c.re).collect()
}

/// Magnitude spectrum (|X[k]|).
pub fn magnitude_spectrum(spectrum: &[Complex]) -> Vec<f64> {
    spectrum.iter().map(|c| c.magnitude()).collect()
}

/// Power spectrum (|X[k]|²).
pub fn power_spectrum(spectrum: &[Complex]) -> Vec<f64> {
    spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect()
}

/// Phase spectrum (angle of X[k]).
pub fn phase_spectrum(spectrum: &[Complex]) -> Vec<f64> {
    spectrum.iter().map(|c| c.phase()).collect()
}

/// In-place Cooley-Tukey radix-2 FFT.
fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "FFT length must be power of 2");

    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterfly operations
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };

        let w_base = Complex::new(angle.cos(), angle.sin());

        for start in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let even = data[start + k];
                let odd = data[start + k + half] * w;
                data[start + k] = even + odd;
                data[start + k + half] = even - odd;
                w = w * w_base;
            }
        }

        len <<= 1;
    }
}

/// Frequency bins for a given FFT size and sample rate.
pub fn frequency_bins(fft_size: usize, sample_rate: f64) -> Vec<f64> {
    (0..fft_size)
        .map(|k| k as f64 * sample_rate / fft_size as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_dc() {
        // Constant signal -> all energy at DC
        let signal: Vec<Complex> = vec![Complex::from_real(1.0); 8];
        let spectrum = fft(&signal);
        assert!((spectrum[0].magnitude() - 8.0).abs() < 1e-10);
        for value in spectrum.iter().take(8).skip(1) {
            assert!(value.magnitude() < 1e-10);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let spectrum = rfft(&signal);
        let recovered = irfft(&spectrum);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_fft_pure_sine() {
        // 8-point signal with frequency at bin 1
        let n = 8;
        let signal: Vec<Complex> = (0..n)
            .map(|k| Complex::from_real((2.0 * PI * k as f64 / n as f64).sin()))
            .collect();
        let spectrum = fft(&signal);
        let mags = magnitude_spectrum(&spectrum);

        // Energy should be concentrated at bin 1 and bin 7 (conjugate)
        assert!(mags[1] > 3.0);
        assert!(mags[7] > 3.0);
        assert!(mags[0] < 0.1);
        assert!(mags[4] < 0.1);
    }

    #[test]
    fn test_parsevals_theorem() {
        // Energy in time domain should equal energy in frequency domain / N
        let signal = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let time_energy: f64 = signal.iter().map(|x| x * x).sum();
        let spectrum = rfft(&signal);
        let freq_energy: f64 = power_spectrum(&spectrum).iter().sum::<f64>() / signal.len() as f64;
        assert!((time_energy - freq_energy).abs() < 1e-6);
    }
}
