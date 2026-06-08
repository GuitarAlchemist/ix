//! Spectral / time-domain audio features — thin reductions over `ix-signal`'s
//! transforms (no new DSP). These mirror the feature set GA's
//! `compute-spectral-profile.js` computes, so IX can recompute or cross-check the
//! same descriptor vector. Pure `f64`, offline.

use ix_signal::correlation;
use ix_signal::fft;

/// One-sided magnitude spectrum (bins `0..=n/2`) of a power-of-two `frame`.
/// (The underlying `rfft` zero-pads non-power-of-two frames, which shifts the
/// bin→frequency mapping — pass a power-of-two frame for calibrated bins.)
pub fn magnitude_spectrum(frame: &[f64]) -> Vec<f64> {
    let spectrum = fft::rfft(frame);
    let mags = fft::magnitude_spectrum(&spectrum);
    let half = frame.len() / 2 + 1;
    mags.into_iter().take(half).collect()
}

/// Spectral centroid (the "center of mass" of the spectrum), in Hz — a primary
/// brightness descriptor. `mags` are one-sided bin magnitudes; `fft_size` is the
/// frame length the spectrum came from.
pub fn spectral_centroid(mags: &[f64], sample_rate: f64, fft_size: usize) -> f64 {
    let bin_hz = sample_rate / fft_size as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (k, &m) in mags.iter().enumerate() {
        num += (k as f64 * bin_hz) * m;
        den += m;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Spectral rolloff: the frequency below which `fraction` (e.g. 0.85) of the
/// total spectral energy lies, in Hz.
pub fn spectral_rolloff(mags: &[f64], sample_rate: f64, fft_size: usize, fraction: f64) -> f64 {
    let total: f64 = mags.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let bin_hz = sample_rate / fft_size as f64;
    let threshold = fraction.clamp(0.0, 1.0) * total;
    let mut cumulative = 0.0;
    for (k, &m) in mags.iter().enumerate() {
        cumulative += m;
        if cumulative >= threshold {
            return k as f64 * bin_hz;
        }
    }
    (mags.len().saturating_sub(1)) as f64 * bin_hz
}

/// Spectral flux between two consecutive magnitude frames: the L2 norm of the
/// positive (onset) part of the bin-wise difference. Higher = more spectral
/// change / jitter.
pub fn spectral_flux(prev: &[f64], curr: &[f64]) -> f64 {
    let n = prev.len().min(curr.len());
    let mut s = 0.0;
    for i in 0..n {
        let d = curr[i] - prev[i];
        if d > 0.0 {
            s += d * d;
        }
    }
    s.sqrt()
}

/// Root-mean-square level (a loudness/energy proxy).
pub fn rms(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt()
}

/// Estimate the fundamental frequency (Hz) by picking the highest
/// autocorrelation peak whose lag corresponds to a period in `[1/fmax, 1/fmin]`.
/// Returns `None` when no positive in-range peak exists (e.g. noise/silence).
pub fn autocorrelation_f0(signal: &[f64], sample_rate: f64, fmin: f64, fmax: f64) -> Option<f64> {
    if signal.len() < 4 || fmin <= 0.0 || fmax <= fmin || sample_rate <= 0.0 {
        return None;
    }
    // Two-sided normalized ACF, length 2n-1, center (index n-1) is the zero lag.
    let full = correlation::autocorrelation(signal);
    let n = signal.len();
    let acf = &full[n - 1..]; // one-sided: acf[0] = 1.0, acf[lag] for lag = 0..n-1

    let min_lag = (sample_rate / fmax).floor() as usize;
    let max_lag = ((sample_rate / fmin).ceil() as usize).min(acf.len().saturating_sub(1));
    if min_lag < 1 || max_lag <= min_lag {
        return None;
    }
    let mut best_lag = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (offset, &val) in acf[min_lag..=max_lag].iter().enumerate() {
        if val > best {
            best = val;
            best_lag = min_lag + offset;
        }
    }
    if best <= 0.0 || best_lag == 0 {
        return None;
    }
    Some(sample_rate / best_lag as f64)
}

/// Build a mel filterbank: `n_mels` triangular filters equally spaced on the mel
/// scale between `f_min` and `f_max`, as weights over the one-sided FFT bins
/// (`fft_size/2 + 1`). Mel warping matches cochlear resolution (dense low, coarse
/// high), so a mel/MFCC distance weights error the way the ear does.
pub fn mel_filterbank(
    n_mels: usize,
    fft_size: usize,
    sample_rate: f64,
    f_min: f64,
    f_max: f64,
) -> Vec<Vec<f64>> {
    let n_bins = fft_size / 2 + 1;
    if n_mels == 0 || n_bins < 2 || sample_rate <= 0.0 {
        return Vec::new();
    }
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (mel_lo, mel_hi) = (hz_to_mel(f_min.max(0.0)), hz_to_mel(f_max));
    let bin_of = |hz: f64| ((hz * fft_size as f64 / sample_rate).round() as usize).min(n_bins - 1);
    // n_mels + 2 mel-spaced edge points → Hz → nearest FFT bin.
    let edges: Vec<usize> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_lo + (mel_hi - mel_lo) * i as f64 / (n_mels + 1) as f64;
            bin_of(mel_to_hz(mel))
        })
        .collect();

    let mut fb = vec![vec![0.0; n_bins]; n_mels];
    for (m, filt) in fb.iter_mut().enumerate() {
        let (lo, ctr, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        for (k, slot) in filt.iter_mut().enumerate() {
            if k >= lo && k < ctr && ctr > lo {
                *slot = (k - lo) as f64 / (ctr - lo) as f64;
            } else if k >= ctr && k < hi && hi > ctr {
                *slot = (hi - k) as f64 / (hi - ctr) as f64;
            }
        }
    }
    fb
}

/// Apply a mel `filterbank` to a one-sided magnitude spectrum, returning per-band
/// POWER (`Σ weight · magnitude²`). Take `ln` of these for a log-mel spectrum.
pub fn mel_spectrum(mags: &[f64], filterbank: &[Vec<f64>]) -> Vec<f64> {
    filterbank
        .iter()
        .map(|filt| filt.iter().zip(mags).map(|(w, &m)| w * m * m).sum::<f64>())
        .collect()
}

/// MFCCs: DCT-II of the log mel-power spectrum, truncated to `n_coeffs` — a coarse
/// timbral-envelope descriptor, robust to fine pitch jitter.
pub fn mfcc(mel_power: &[f64], n_coeffs: usize) -> Vec<f64> {
    if mel_power.is_empty() {
        return Vec::new();
    }
    let log_mel: Vec<f64> = mel_power.iter().map(|&e| e.max(1e-10).ln()).collect();
    let coeffs = ix_signal::dct::dct2_normalized(&log_mel);
    let k = n_coeffs.min(coeffs.len());
    coeffs.into_iter().take(k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn tone(freq_hz: f64, sample_rate: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|t| (2.0 * PI * freq_hz / sample_rate * t as f64).sin())
            .collect()
    }

    #[test]
    fn centroid_tracks_a_pure_tone() {
        let sr = 1024.0;
        let frame = tone(128.0, sr, 1024); // 128 Hz over a 1024-pt frame
        let mags = magnitude_spectrum(&frame);
        let c = spectral_centroid(&mags, sr, 1024);
        assert!(
            (c - 128.0).abs() < 20.0,
            "centroid {c} should be near 128 Hz"
        );
    }

    #[test]
    fn rolloff_increases_with_higher_tone() {
        let sr = 1024.0;
        let low = magnitude_spectrum(&tone(100.0, sr, 1024));
        let high = magnitude_spectrum(&tone(300.0, sr, 1024));
        let r_low = spectral_rolloff(&low, sr, 1024, 0.85);
        let r_high = spectral_rolloff(&high, sr, 1024, 0.85);
        assert!(
            r_high > r_low,
            "rolloff should rise with tone: {r_low} -> {r_high}"
        );
    }

    #[test]
    fn f0_recovers_a_sine() {
        let sr = 8000.0;
        let sig = tone(220.0, sr, 2048);
        let f0 = autocorrelation_f0(&sig, sr, 80.0, 1000.0).expect("f0 found");
        assert!((f0 - 220.0).abs() < 5.0, "f0 {f0} should be ≈220 Hz");
    }

    #[test]
    fn rms_of_unit_sine_is_about_root_half() {
        let sig = tone(50.0, 1000.0, 1000);
        assert!((rms(&sig) - (0.5f64).sqrt()).abs() < 0.02);
    }

    #[test]
    fn mel_filterbank_has_expected_shape() {
        let fb = mel_filterbank(8, 1024, 8000.0, 50.0, 4000.0);
        assert_eq!(fb.len(), 8);
        assert_eq!(fb[0].len(), 1024 / 2 + 1);
        assert!(
            fb.iter().all(|f| f.iter().any(|&w| w > 0.0)),
            "every mel filter must have positive weight"
        );
    }

    #[test]
    fn mfcc_distinguishes_bright_from_dark() {
        let sr = 8000.0;
        let fb = mel_filterbank(20, 1024, sr, 50.0, 4000.0);
        let dark = mfcc(
            &mel_spectrum(&magnitude_spectrum(&tone(200.0, sr, 1024)), &fb),
            13,
        );
        let bright = mfcc(
            &mel_spectrum(&magnitude_spectrum(&tone(2000.0, sr, 1024)), &fb),
            13,
        );
        let dist: f64 = dark
            .iter()
            .zip(&bright)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            dist > 0.1,
            "a bright vs dark tone must give distinct MFCCs, dist={dist}"
        );
    }
}
