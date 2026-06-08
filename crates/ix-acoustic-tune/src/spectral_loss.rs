//! Spectral distance / loss functions — the objective a sound-matcher minimizes.
//!
//! Built on `ix-signal`'s STFT (`spectrogram`); magnitude-only (phase discarded),
//! which is standard for parameter estimation. The headline is
//! [`multi_resolution_stft_loss`] (the "MSS" objective), the proven default for
//! synth sound-matching. Non-differentiable, pure `f64` — fed to the black-box
//! optimizer. A differentiable variant (via `ix-autograd`'s `fft-autograd`) is a
//! deferred, research-grade follow-up.

use ix_signal::spectral;

/// Single-resolution STFT loss between two signals: mean absolute
/// log-magnitude difference + spectral convergence, over a Hann-windowed STFT.
/// Returns `0.0` for identical inputs. `window_size` should be a power of two.
pub fn stft_loss(a: &[f64], b: &[f64], window_size: usize, hop_size: usize) -> f64 {
    let sa = spectral::spectrogram(a, window_size, hop_size, false);
    let sb = spectral::spectrogram(b, window_size, hop_size, false);
    let frames = sa.len().min(sb.len());
    if frames == 0 {
        return 0.0;
    }

    let mut log_mag_sum = 0.0;
    let mut count = 0usize;
    let mut sc_num = 0.0; // Σ (a-b)²
    let mut sc_den = 0.0; // Σ a²
    const EPS: f64 = 1e-7;

    for t in 0..frames {
        let bins = sa[t].len().min(sb[t].len());
        for k in 0..bins {
            let x = sa[t][k];
            let y = sb[t][k];
            log_mag_sum += ((x + EPS).ln() - (y + EPS).ln()).abs();
            sc_num += (x - y) * (x - y);
            sc_den += x * x;
            count += 1;
        }
    }

    let log_mag = if count > 0 {
        log_mag_sum / count as f64
    } else {
        0.0
    };
    let spectral_convergence = if sc_den > 0.0 {
        sc_num.sqrt() / sc_den.sqrt()
    } else {
        0.0
    };
    log_mag + spectral_convergence
}

/// Multi-resolution STFT loss (MSS): the mean of [`stft_loss`] across several
/// FFT resolutions — the standard objective for synth sound-matching, robust to
/// the choice of any single window size. Only resolutions that fit the signal
/// length are used; falls back to a time-domain RMS distance for short signals.
// @ai:invariant multi_resolution_stft_loss is 0 for identical signals and strictly larger for a more distant one, AND for signals shorter than the smallest STFT window it uses a non-zero time-domain RMS fallback (never a degenerate hann(2)=[0,0] STFT that would falsely score two different short signals as 0) [T:test conf:0.9 src:spectral_loss::tests::short_signals_get_a_meaningful_loss]
pub fn multi_resolution_stft_loss(a: &[f64], b: &[f64]) -> f64 {
    const CONFIGS: [(usize, usize); 3] = [(256, 64), (512, 128), (1024, 256)];
    let min_len = a.len().min(b.len());

    let mut total = 0.0;
    let mut used = 0usize;
    for (w, h) in CONFIGS {
        if min_len >= w {
            total += stft_loss(a, b, w, h);
            used += 1;
        }
    }
    if used > 0 {
        return total / used as f64;
    }
    // Signals too short for any STFT resolution (< 256 samples): fall back to a
    // time-domain RMS distance, which is always meaningful. A degenerate
    // tiny-window STFT would be WRONG here — a Hann window of length 2 is
    // `[0, 0]`, which zeroes every frame, so ANY two short signals would falsely
    // score 0.0 (indistinguishable from identical).
    if min_len == 0 {
        return 0.0;
    }
    let mse: f64 = (0..min_len).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>() / min_len as f64;
    mse.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn tone(freq_hz: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|t| (2.0 * PI * freq_hz * t as f64).sin())
            .collect()
    }

    #[test]
    fn identical_signals_have_near_zero_loss() {
        let a = tone(0.05, 4096);
        assert!(multi_resolution_stft_loss(&a, &a) < 1e-9);
    }

    #[test]
    fn closer_signal_has_lower_loss() {
        let target = tone(0.05, 4096);
        let close = tone(0.051, 4096);
        let far = tone(0.20, 4096);
        let l_close = multi_resolution_stft_loss(&target, &close);
        let l_far = multi_resolution_stft_loss(&target, &far);
        assert!(
            l_close < l_far,
            "a spectrally closer signal must score lower: {l_close} !< {l_far}"
        );
    }

    // Review P1 regression: signals shorter than the smallest STFT window must
    // use the time-domain fallback, NOT a degenerate hann(2)=[0,0] STFT that
    // would score any two short signals as 0 (falsely identical).
    #[test]
    fn short_signals_get_a_meaningful_loss() {
        let a: Vec<f64> = (0..100).map(|t| (t as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = (0..100).map(|t| (t as f64 * 0.3).cos()).collect();
        assert!(
            multi_resolution_stft_loss(&a, &a) < 1e-12,
            "identical short signals → 0"
        );
        assert!(
            multi_resolution_stft_loss(&a, &b) > 0.1,
            "different short signals must not falsely score 0"
        );
    }
}
