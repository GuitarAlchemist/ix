//! Spectral analysis: STFT, spectrograms, power spectral density.

use crate::fft;
use crate::window;

/// Short-Time Fourier Transform (STFT).
/// Returns a matrix of complex spectra: rows = time frames, cols = frequency bins.
pub fn stft(
    signal: &[f64],
    window_size: usize,
    hop_size: usize,
    window_fn: &[f64],
) -> Vec<Vec<fft::Complex>> {
    let mut frames = Vec::new();
    let mut start = 0;

    while start + window_size <= signal.len() {
        let frame: Vec<f64> = signal[start..start + window_size]
            .iter()
            .zip(window_fn.iter())
            .map(|(s, w)| s * w)
            .collect();

        let spectrum = fft::rfft(&frame);
        frames.push(spectrum);
        start += hop_size;
    }

    frames
}

/// Spectrogram: magnitude of STFT (dB scale optional).
pub fn spectrogram(signal: &[f64], window_size: usize, hop_size: usize, db: bool) -> Vec<Vec<f64>> {
    let win = window::hanning(window_size);
    let frames = stft(signal, window_size, hop_size, &win);

    frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .take(window_size / 2 + 1)
                .map(|c| {
                    let mag = c.magnitude();
                    if db {
                        20.0 * (mag.max(1e-10)).log10()
                    } else {
                        mag
                    }
                })
                .collect()
        })
        .collect()
}

/// Welch's method for Power Spectral Density estimation.
pub fn welch_psd(
    signal: &[f64],
    window_size: usize,
    overlap: usize,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    let hop = window_size - overlap;
    let win = window::hanning(window_size);
    let win_power: f64 = win.iter().map(|w| w * w).sum::<f64>();

    let mut psd = vec![0.0; window_size / 2 + 1];
    let mut n_segments = 0;

    let mut start = 0;
    while start + window_size <= signal.len() {
        let frame: Vec<f64> = signal[start..start + window_size]
            .iter()
            .zip(win.iter())
            .map(|(s, w)| s * w)
            .collect();

        let spectrum = fft::rfft(&frame);
        for (i, c) in spectrum.iter().take(window_size / 2 + 1).enumerate() {
            psd[i] += (c.re * c.re + c.im * c.im) / win_power;
        }

        n_segments += 1;
        start += hop;
    }

    if n_segments > 0 {
        let scale = 1.0 / (n_segments as f64 * sample_rate);
        for p in psd.iter_mut() {
            *p *= scale;
        }
    }

    let freqs: Vec<f64> = (0..=window_size / 2)
        .map(|k| k as f64 * sample_rate / window_size as f64)
        .collect();

    (freqs, psd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_stft_frame_count() {
        let signal = vec![0.0; 1024];
        let win = window::hanning(256);
        let frames = stft(&signal, 256, 128, &win);
        // (1024 - 256) / 128 + 1 = 7
        assert_eq!(frames.len(), 7);
    }

    #[test]
    fn test_welch_peak_at_signal_frequency() {
        let n = 4096;
        let fs = 1000.0;
        let f0 = 100.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).sin())
            .collect();

        let (freqs, psd) = welch_psd(&signal, 512, 256, fs);

        // Find peak frequency
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - f0).abs() < 5.0,
            "Peak at {} Hz, expected ~{}",
            peak_freq,
            f0
        );
    }
}
