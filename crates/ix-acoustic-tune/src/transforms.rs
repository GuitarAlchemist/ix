//! Pitch-aware spectral transforms: the **analytic signal / Hilbert envelope** and the
//! **constant-Q transform (CQT) / chroma** — the log-frequency bridge from a raw waveform
//! to pitch classes.
//!
//! `ix-signal`'s FFT is linear-frequency (constant Hz per bin); music is log-frequency
//! (constant ratio per semitone). These add the two transforms that close that gap:
//!
//! - [`envelope`] / [`hilbert`] / [`analytic_signal`] — instantaneous amplitude & phase
//!   via the FFT-domain Hilbert transform (the standard "double the positive frequencies"
//!   construction). Envelope = `|x + i·H{x}|`; for an AM tone it recovers the modulator.
//! - [`cqt`] — a constant-Q (log-spaced, constant `f/Δf`) magnitude spectrum: one bin per
//!   `1/bins_per_octave` of an octave, so every octave occupies the same number of bins.
//! - [`chroma`] — the 12-element pitch-class profile: CQT folded across octaves so a pitch
//!   and all its octave transpositions land in the same bin (octave-invariant).
//!
//! Pure `f64`, offline; reuses `ix-signal`'s FFT. Experimental-tier, like the rest of the crate.

use std::f64::consts::PI;

use ix_signal::fft::{self, Complex};

/// The discrete-time **analytic signal** `xₐ[n] = x[n] + i·H{x}[n]`, same length as `signal`.
///
/// Built the standard FFT way: take the spectrum, zero the negative frequencies and double
/// the positive ones (keeping DC and Nyquist), inverse-transform. `ix-signal`'s FFT zero-pads
/// to a power of two; the filter is applied in that padded domain and the result truncated
/// back to the input length.
pub fn analytic_signal(signal: &[f64]) -> Vec<Complex> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let input: Vec<Complex> = signal.iter().map(|&x| Complex::from_real(x)).collect();
    let mut spec = fft::fft(&input); // length m = next_power_of_two(n)
    let m = spec.len();
    // h[k]: 1 at DC (and Nyquist when m even), 2 on the positive frequencies, 0 on the negative.
    for (k, c) in spec.iter_mut().enumerate() {
        let h = if k == 0 || (m % 2 == 0 && k == m / 2) {
            1.0
        } else if k < m / 2 {
            2.0
        } else {
            0.0
        };
        c.re *= h;
        c.im *= h;
    }
    let analytic = fft::ifft(&spec); // length m
    analytic.into_iter().take(n).collect()
}

/// The **Hilbert transform** `H{x}` — the imaginary part of the [`analytic_signal`]
/// (a −90° phase shift; `H{cos} ≈ sin`). Same length as `signal`.
pub fn hilbert(signal: &[f64]) -> Vec<f64> {
    analytic_signal(signal).iter().map(|c| c.im).collect()
}

/// The **amplitude envelope** `|xₐ[n]|` — instantaneous amplitude. For a constant-amplitude
/// tone this is ~flat at the amplitude; for an AM signal it traces the modulator.
pub fn envelope(signal: &[f64]) -> Vec<f64> {
    analytic_signal(signal)
        .iter()
        .map(|c| c.magnitude())
        .collect()
}

/// Center frequencies (Hz) of the [`cqt`] bins: `f_k = f_min · 2^(k / bins_per_octave)`.
pub fn cqt_frequencies(f_min: f64, bins_per_octave: usize, n_bins: usize) -> Vec<f64> {
    (0..n_bins)
        .map(|k| f_min * 2f64.powf(k as f64 / bins_per_octave as f64))
        .collect()
}

/// **Constant-Q transform** magnitudes: one value per log-spaced bin from `f_min` upward,
/// `bins_per_octave` bins per octave, `n_bins` total. Each bin is a Hann-windowed DFT at its
/// center frequency with a constant quality factor `Q = 1 / (2^(1/bpo) − 1)`, so the window
/// length `N_k = round(Q · sr / f_k)` shrinks with rising frequency — constant resolution in
/// *cents*, unlike the FFT's constant resolution in Hz.
///
/// A bin whose ideal window exceeds the signal length is computed over the available samples
/// (lower resolution there) rather than dropped. Returns one magnitude per bin.
pub fn cqt(
    signal: &[f64],
    sample_rate: f64,
    f_min: f64,
    bins_per_octave: usize,
    n_bins: usize,
) -> Vec<f64> {
    if signal.is_empty() || sample_rate <= 0.0 || f_min <= 0.0 || bins_per_octave == 0 {
        return vec![0.0; n_bins];
    }
    let q = 1.0 / (2f64.powf(1.0 / bins_per_octave as f64) - 1.0);
    cqt_frequencies(f_min, bins_per_octave, n_bins)
        .into_iter()
        .map(|f_k| {
            // Ideal window length for constant Q, clamped to what we actually have.
            let n_k = ((q * sample_rate / f_k).round() as usize).clamp(1, signal.len());
            let theta = 2.0 * PI * f_k / sample_rate;
            let mut acc = Complex::zero();
            for (j, &x) in signal.iter().take(n_k).enumerate() {
                // Hann window over the per-bin support, then a complex sinusoid at f_k.
                let w = if n_k > 1 {
                    0.5 - 0.5 * (2.0 * PI * j as f64 / (n_k as f64 - 1.0)).cos()
                } else {
                    1.0
                };
                let ph = theta * j as f64;
                acc.re += x * w * ph.cos();
                acc.im -= x * w * ph.sin();
            }
            acc.magnitude() / n_k as f64
        })
        .collect()
}

/// **Chroma** (pitch-class profile): a 12-element vector where index 0 is the pitch class of
/// `f_min`. Computes a `12 · n_octaves`-bin CQT (12 bins/octave) from `f_min` and folds every
/// bin onto its pitch class (`bin k → k mod 12`), so a tone and all its octave transpositions
/// reinforce the same entry. Pass a `f_min` on a pitch class (e.g. ~32.70 Hz = C1) so index 0
/// is that pitch class.
pub fn chroma(signal: &[f64], sample_rate: f64, f_min: f64, n_octaves: usize) -> [f64; 12] {
    let n_bins = 12 * n_octaves;
    let mags = cqt(signal, sample_rate, f_min, 12, n_bins);
    let mut out = [0.0f64; 12];
    for (k, m) in mags.into_iter().enumerate() {
        out[k % 12] += m;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine wave: `amp · sin(2π f t)`, `n` samples at `sample_rate`.
    fn sine(freq: f64, amp: f64, n: usize, sample_rate: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    fn cosine(freq: f64, n: usize, sample_rate: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos())
            .collect()
    }

    #[test]
    fn analytic_signal_preserves_real_part_and_length() {
        let x = sine(50.0, 1.0, 256, 1000.0);
        let a = analytic_signal(&x);
        assert_eq!(a.len(), x.len());
        // Re(analytic) == original signal (the filter only fabricates the imaginary part).
        for (c, &xi) in a.iter().zip(x.iter()) {
            assert!((c.re - xi).abs() < 1e-9, "real part must equal the input");
        }
    }

    #[test]
    fn hilbert_of_cosine_is_sine() {
        // H{cos} = sin. Check the interior (FFT-based Hilbert has edge transients), using an
        // integer number of periods over a power-of-two length so there's no spectral leakage.
        let sr = 256.0;
        let x = cosine(8.0, 256, sr); // 8 full periods in 256 samples
        let h = hilbert(&x);
        let want = sine(8.0, 1.0, 256, sr);
        for i in 32..224 {
            assert!(
                (h[i] - want[i]).abs() < 1e-2,
                "H{{cos}}[{i}] = {} expected ~{}",
                h[i],
                want[i]
            );
        }
    }

    #[test]
    fn envelope_of_constant_tone_is_flat_at_amplitude() {
        let sr = 256.0;
        let x = cosine(8.0, 256, sr); // amplitude 1
        let env = envelope(&x);
        // Interior envelope hugs the amplitude (1.0).
        for &e in &env[32..224] {
            assert!((e - 1.0).abs() < 1e-2, "envelope should be ~1.0, got {e}");
        }
    }

    #[test]
    fn envelope_tracks_am_modulation() {
        // x(t) = (1 + 0.5 sin(2π·2t)) · cos(2π·40t): a slow modulator on a fast carrier.
        let sr = 1024.0;
        let n = 1024;
        let carrier = 40.0;
        let modf = 2.0;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (1.0 + 0.5 * (2.0 * PI * modf * t).sin()) * (2.0 * PI * carrier * t).cos()
            })
            .collect();
        let env = envelope(&x);
        // The envelope should recover the modulator 1 + 0.5 sin(2π·2t) in the interior.
        for (i, &e) in env.iter().enumerate().take(896).skip(128) {
            let t = i as f64 / sr;
            let want = 1.0 + 0.5 * (2.0 * PI * modf * t).sin();
            assert!(
                (e - want).abs() < 0.1,
                "envelope[{i}] = {e} expected ~{want}"
            );
        }
    }

    #[test]
    fn cqt_peaks_at_the_tone_bin() {
        // A tone exactly on bin 12 (one octave above f_min) should peak there.
        let sr = 22050.0;
        let f_min = 110.0; // A2
        let bpo = 12;
        let n_bins = 36; // three octaves
        let target_bin = 12;
        let f_target = f_min * 2f64.powf(target_bin as f64 / bpo as f64); // 220 Hz
        let x = sine(f_target, 1.0, 8192, sr);
        let mags = cqt(&x, sr, f_min, bpo, n_bins);
        let peak = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, target_bin, "CQT should peak at the bin of the tone");
    }

    #[test]
    fn chroma_is_octave_invariant() {
        // f_min = C1 (≈32.70 Hz). A tone at C4 and a tone at C5 (an octave apart) must both
        // peak at chroma bin 0 — the property a single-tone test would not catch.
        let sr = 22050.0;
        let c1 = 32.703_195_66;
        let c4 = c1 * 2f64.powi(3); // 3 octaves up
        let c5 = c1 * 2f64.powi(4);
        for f in [c4, c5] {
            let x = sine(f, 1.0, 8192, sr);
            let ch = chroma(&x, sr, c1, 7);
            let peak = ch
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                peak, 0,
                "a C in any octave must peak at chroma bin 0 (got {peak})"
            );
        }
        // A perfect fifth above C (G, pitch class 7) peaks at chroma bin 7.
        let g4 = c1 * 2f64.powi(3) * 2f64.powf(7.0 / 12.0);
        let x = sine(g4, 1.0, 8192, sr);
        let ch = chroma(&x, sr, c1, 7);
        let peak = ch
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 7, "a G must peak at chroma bin 7 (got {peak})");
    }

    #[test]
    fn empty_and_degenerate_inputs_dont_panic() {
        assert!(analytic_signal(&[]).is_empty());
        assert!(hilbert(&[]).is_empty());
        assert!(envelope(&[]).is_empty());
        assert_eq!(cqt(&[], 44100.0, 55.0, 12, 12), vec![0.0; 12]);
        assert_eq!(chroma(&[], 44100.0, 32.7, 7), [0.0; 12]);
    }
}
