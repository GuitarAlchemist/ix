//! Digital filters: FIR and IIR.

use std::f64::consts::PI;

use crate::window;

/// FIR filter coefficients.
#[derive(Debug, Clone)]
pub struct FirFilter {
    pub coefficients: Vec<f64>,
}

impl FirFilter {
    /// Design a low-pass FIR filter using the windowed sinc method.
    /// `cutoff`: normalized cutoff frequency (0 to 0.5, where 0.5 = Nyquist).
    /// `order`: filter order (number of taps - 1). Must be even.
    pub fn lowpass(cutoff: f64, order: usize) -> Self {
        let n = order + 1;
        let mid = order as f64 / 2.0;
        let win = window::hamming(n);

        let coeffs: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 - mid;
                let sinc = if x.abs() < 1e-10 {
                    2.0 * cutoff
                } else {
                    (2.0 * PI * cutoff * x).sin() / (PI * x)
                };
                sinc * win[i]
            })
            .collect();

        Self {
            coefficients: coeffs,
        }
    }

    /// Design a high-pass FIR filter (spectral inversion of lowpass).
    pub fn highpass(cutoff: f64, order: usize) -> Self {
        let mut lp = Self::lowpass(cutoff, order);
        let mid = order / 2;
        for (i, c) in lp.coefficients.iter_mut().enumerate() {
            *c = -*c;
            if i == mid {
                *c += 1.0;
            }
        }
        lp
    }

    /// Design a bandpass FIR filter.
    pub fn bandpass(low_cutoff: f64, high_cutoff: f64, order: usize) -> Self {
        let lp = Self::lowpass(high_cutoff, order);
        let hp = Self::highpass(low_cutoff, order);
        let _coeffs: Vec<f64> = lp
            .coefficients
            .iter()
            .zip(hp.coefficients.iter())
            .map(|(l, h)| {
                // Convolve the two filters (simplified: element-wise for same-length)
                // Actually, for a proper bandpass we subtract:
                // bandpass = lowpass(high) convolved with highpass(low)
                // Simpler approach: lowpass(high) - lowpass(low)
                l + h - if l == h { 1.0 } else { 0.0 }
            })
            .collect();

        // Better approach: difference of two lowpass filters
        let lp_high = Self::lowpass(high_cutoff, order);
        let lp_low = Self::lowpass(low_cutoff, order);
        let bp_coeffs: Vec<f64> = lp_high
            .coefficients
            .iter()
            .zip(lp_low.coefficients.iter())
            .map(|(h, l)| h - l)
            .collect();

        Self {
            coefficients: bp_coeffs,
        }
    }

    /// Apply the FIR filter to a signal.
    pub fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let m = self.coefficients.len();
        let mut output = vec![0.0; n];

        for i in 0..n {
            for j in 0..m {
                if i >= j {
                    output[i] += self.coefficients[j] * signal[i - j];
                }
            }
        }
        output
    }
}

/// IIR filter (Direct Form II).
#[derive(Debug, Clone)]
pub struct IirFilter {
    /// Feedforward (numerator) coefficients b[].
    pub b: Vec<f64>,
    /// Feedback (denominator) coefficients a[]. a[0] is assumed 1.0.
    pub a: Vec<f64>,
}

impl IirFilter {
    /// Create a first-order IIR low-pass filter (exponential moving average).
    /// `alpha`: smoothing factor (0 < alpha < 1). Smaller = more smoothing.
    pub fn first_order_lowpass(alpha: f64) -> Self {
        Self {
            b: vec![alpha],
            a: vec![1.0, -(1.0 - alpha)],
        }
    }

    /// Apply the IIR filter to a signal.
    pub fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let mut output = vec![0.0; n];

        for i in 0..n {
            // Feedforward
            let mut y = 0.0;
            for (j, &bj) in self.b.iter().enumerate() {
                if i >= j {
                    y += bj * signal[i - j];
                }
            }
            // Feedback (skip a[0] which is 1.0)
            for (j, &aj) in self.a.iter().enumerate().skip(1) {
                if i >= j {
                    y -= aj * output[i - j];
                }
            }
            output[i] = y;
        }
        output
    }
}

/// Design a 2nd-order Butterworth low-pass filter (IIR).
/// `cutoff`: normalized frequency (0 to 0.5).
pub fn butterworth_lowpass_2nd(cutoff: f64) -> IirFilter {
    let wc = (PI * cutoff).tan();
    let wc2 = wc * wc;
    let sqrt2 = std::f64::consts::SQRT_2;

    let k = 1.0 + sqrt2 * wc + wc2;

    IirFilter {
        b: vec![wc2 / k, 2.0 * wc2 / k, wc2 / k],
        a: vec![1.0, 2.0 * (wc2 - 1.0) / k, (1.0 - sqrt2 * wc + wc2) / k],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fir_lowpass_passes_dc() {
        let fir = FirFilter::lowpass(0.25, 32);
        // DC signal should pass through
        let dc = vec![1.0; 100];
        let out = fir.apply(&dc);
        // After transient, should be ~1.0
        assert!((out[90] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_iir_smoothing() {
        let iir = IirFilter::first_order_lowpass(0.1);
        let noisy: Vec<f64> = (0..100).map(|_| 1.0).collect();
        let out = iir.apply(&noisy);
        // Should converge to ~1.0
        assert!((out[99] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_butterworth_attenuates_high_freq() {
        let bw = butterworth_lowpass_2nd(0.1); // Low cutoff
        let n = 256;
        let fs = 1.0;

        // High frequency signal
        let high: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 0.4 * i as f64 / fs).sin())
            .collect();
        let filtered = bw.apply(&high);

        // After transient, high freq should be attenuated
        let late_power: f64 = filtered[100..].iter().map(|x| x * x).sum::<f64>() / (n - 100) as f64;
        let input_power: f64 = high[100..].iter().map(|x| x * x).sum::<f64>() / (n - 100) as f64;
        assert!(late_power < input_power * 0.1, "Should attenuate high freq");
    }
}
