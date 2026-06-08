//! Spectral distance / loss functions — the objective a sound-matcher minimizes.
//!
//! Built on `ix-signal`'s STFT (`spectrogram`); magnitude-only (phase discarded),
//! which is standard for parameter estimation. The headline is
//! [`multi_resolution_stft_loss`] (the "MSS" objective), the proven default for
//! synth sound-matching. Non-differentiable, pure `f64` — fed to the black-box
//! optimizer. A differentiable variant (via `ix-autograd`'s `fft-autograd`) is a
//! deferred, research-grade follow-up.

use ix_signal::spectral;
use serde::{Deserialize, Serialize};

use crate::{features, reference};

/// Single-resolution STFT loss between two signals: mean absolute
/// log-magnitude difference + spectral convergence, over a Hann-windowed STFT.
/// Returns `0.0` for identical inputs. `window_size` should be a power of two.
pub fn stft_loss(a: &[f64], b: &[f64], window_size: usize, hop_size: usize) -> f64 {
    let sa = spectral::spectrogram(a, window_size, hop_size, false);
    let sb = spectral::spectrogram(b, window_size, hop_size, false);
    // Iterate the UNION of frames/bins, treating a missing frame/bin as zero
    // energy — so an unmatched tail (a long ring/noise in the longer signal that
    // the other lacks) is penalized, not silently ignored. Comparing only the
    // common prefix would under-score exactly the duration/decay errors this loss
    // exists to catch.
    let n_frames = sa.len().max(sb.len());
    if n_frames == 0 {
        return 0.0;
    }

    let mut log_mag_sum = 0.0;
    let mut count = 0usize;
    let mut sc_num = 0.0; // Σ (a-b)²
    let mut sc_den = 0.0; // Σ a²
    const EPS: f64 = 1e-7;
    let empty: Vec<f64> = Vec::new();

    for t in 0..n_frames {
        let fa = sa.get(t).unwrap_or(&empty);
        let fb = sb.get(t).unwrap_or(&empty);
        let bins = fa.len().max(fb.len());
        for k in 0..bins {
            let x = fa.get(k).copied().unwrap_or(0.0);
            let y = fb.get(k).copied().unwrap_or(0.0);
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

/// Relative weights for the [`layered_perceptual_loss`] terms.
#[derive(Debug, Clone)]
pub struct LossWeights {
    /// Multi-resolution STFT (spectral-shape backbone).
    pub mss: f64,
    /// Log-mel distance (perceptual frequency weighting).
    pub mel: f64,
    /// Per-band decay-slope distance (the dominant naturalness term).
    pub decay: f64,
    /// Inharmonicity-B distance.
    pub inharmonicity: f64,
    /// Relative f0 distance.
    pub f0: f64,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            mss: 1.0,
            mel: 1.0,
            decay: 1.0,
            inharmonicity: 0.5,
            f0: 0.5,
        }
    }
}

/// Centered power-of-two analysis frame (zero-padded if the signal is short).
fn center_frame(s: &[f64], fft: usize) -> Vec<f64> {
    if s.len() >= fft {
        let start = (s.len() - fft) / 2;
        s[start..start + fft].to_vec()
    } else {
        let mut v = s.to_vec();
        v.resize(fft, 0.0);
        v
    }
}

/// Log-mel L1 distance between two signals (perceptual frequency weighting).
fn mel_distance(a: &[f64], b: &[f64], sample_rate: f64) -> f64 {
    const FFT: usize = 1024;
    let fb = features::mel_filterbank(40, FFT, sample_rate, 50.0, sample_rate / 2.0);
    if fb.is_empty() {
        return 0.0;
    }
    let log_mel = |s: &[f64]| -> Vec<f64> {
        features::mel_spectrum(&features::magnitude_spectrum(&center_frame(s, FFT)), &fb)
            .iter()
            .map(|&e| e.max(1e-10).ln())
            .collect::<Vec<f64>>()
    };
    let (la, lb) = (log_mel(a), log_mel(b));
    let n = la.len().max(1) as f64;
    la.iter().zip(&lb).map(|(x, y)| (x - y).abs()).sum::<f64>() / n
}

/// Mean absolute difference of per-band decay slopes — the dominant naturalness
/// term (a single averaged spectrum is blind to differential decay).
fn decay_distance(a: &[f64], b: &[f64], sample_rate: f64) -> f64 {
    let edges = reference::default_band_edges();
    let sa = reference::per_band_decay_slopes(a, sample_rate, &edges);
    let sb = reference::per_band_decay_slopes(b, sample_rate, &edges);
    let n = sa.len().min(sb.len()).max(1) as f64;
    sa.iter().zip(&sb).map(|(x, y)| (x - y).abs()).sum::<f64>() / n
}

/// Relative-f0 and inharmonicity-B distances between two signals.
fn pitch_distance(a: &[f64], b: &[f64], sample_rate: f64) -> (f64, f64) {
    let fa = features::autocorrelation_f0(a, sample_rate, 50.0, 2000.0);
    let fb = features::autocorrelation_f0(b, sample_rate, 50.0, 2000.0);
    let f0d = match (fa, fb) {
        (Some(x), Some(y)) => (x - y).abs() / x.max(1.0),
        _ => 0.0,
    };
    let b_of = |s: &[f64], f: Option<f64>| {
        f.map(|f| reference::inharmonicity_b(s, sample_rate, f, 8, 1024))
            .unwrap_or(0.0)
    };
    // B is tiny (~1e-3); scale so its term is comparable to the others.
    let inh = (b_of(a, fa) - b_of(b, fb)).abs() * 100.0;
    (f0d, inh)
}

/// The **layered perceptual loss** — the MSS spectral backbone plus the
/// perceptual/temporal terms (mel, per-band decay, f0+inharmonicity) that catch
/// what a single magnitude-spectrum distance is blind to. `target` is the
/// reference; `candidate` is the synth render. Lower = closer / more natural.
// @ai:invariant layered_perceptual_loss is ~0 for identical signals and scores a candidate with the WRONG decay envelope (but a similar average spectrum) strictly worse than one with the right envelope — catching the naturalness gap a bare multi-resolution-STFT loss misses [T:test conf:0.85 src:spectral_loss::tests::layered_loss_catches_decay_that_mss_misses]
pub fn layered_perceptual_loss(
    target: &[f64],
    candidate: &[f64],
    sample_rate: f64,
    w: &LossWeights,
) -> f64 {
    let mss = multi_resolution_stft_loss(target, candidate);
    let mel = mel_distance(target, candidate, sample_rate);
    let decay = decay_distance(target, candidate, sample_rate);
    let (f0d, inh) = pitch_distance(target, candidate, sample_rate);
    w.mss * mss + w.mel * mel + w.decay * decay + w.f0 * f0d + w.inharmonicity * inh
}

/// Per-term breakdown of the [`layered_perceptual_loss`] residual between a target
/// and the best candidate — the diagnostic that decides the next lever:
/// **retune** (cheap) vs **extend the synth kernel** (a reviewed code change).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBreakdown {
    pub mss: f64,
    pub mel: f64,
    pub decay: f64,
    pub f0: f64,
    pub inharmonicity: f64,
    pub total: f64,
    /// The largest weighted term — the bottleneck to address next.
    pub dominant: String,
}

impl ResidualBreakdown {
    /// Maps the dominant residual term to the recommended next move (the plan's
    /// retune-vs-extend-kernel §2). A spread-out small residual ⇒ at model capacity.
    pub fn recommendation(&self) -> &'static str {
        match self.dominant.as_str() {
            "decay" => "extend the GA kernel: a tunable/higher-order loop filter — per-band decay is unreachable by the single fixed one-pole",
            "inharmonicity" => "extend the GA kernel: an all-pass cascade for true inharmonicity — the single dispersion all-pass only smears",
            "f0" => "retune dispersion/tuning, or verify the rendered pitch",
            "mel" | "mss" => "retune brightness/spectral params first; if it persists in the low-freq/body region, replace the synthetic body IR (commuted synthesis)",
            _ => "near model capacity — tuning is the right tool",
        }
    }
}

/// Decompose the layered loss into its weighted per-term contributions, flagging
/// the dominant one. Run on the best candidate after convergence to decide
/// whether to keep tuning or extend the synth model (see
/// [`ResidualBreakdown::recommendation`]).
pub fn decompose_residual(
    target: &[f64],
    candidate: &[f64],
    sample_rate: f64,
    w: &LossWeights,
) -> ResidualBreakdown {
    let mss = w.mss * multi_resolution_stft_loss(target, candidate);
    let mel = w.mel * mel_distance(target, candidate, sample_rate);
    let decay = w.decay * decay_distance(target, candidate, sample_rate);
    let (f0d, inh) = pitch_distance(target, candidate, sample_rate);
    let f0 = w.f0 * f0d;
    let inharmonicity = w.inharmonicity * inh;
    let total = mss + mel + decay + f0 + inharmonicity;

    let terms = [
        ("mss", mss),
        ("mel", mel),
        ("decay", decay),
        ("f0", f0),
        ("inharmonicity", inharmonicity),
    ];
    let dominant = terms
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(n, _)| n.to_string())
        .unwrap_or_default();

    ResidualBreakdown {
        mss,
        mel,
        decay,
        f0,
        inharmonicity,
        total,
        dominant,
    }
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

    #[test]
    fn layered_loss_zero_for_identical() {
        let sr = 8000.0;
        let sig: Vec<f64> = (0..8000)
            .map(|t| {
                let tt = t as f64 / sr;
                (-2.0 * tt).exp() * (2.0 * PI * 220.0 * tt).sin()
            })
            .collect();
        let l = layered_perceptual_loss(&sig, &sig, sr, &LossWeights::default());
        assert!(l < 1e-6, "identical signals → ~0, got {l}");
    }

    // The money test: the layered loss catches a wrong DECAY envelope that a bare
    // averaged-spectrum match misses. A no-decay tone has a similar average
    // spectrum to a decaying one but a very different envelope.
    #[test]
    fn layered_loss_catches_decay_that_mss_misses() {
        let sr = 8000.0;
        let n = 8000;
        let make = |decay: f64| -> Vec<f64> {
            (0..n)
                .map(|t| {
                    let tt = t as f64 / sr;
                    (-decay * tt).exp() * (2.0 * PI * 300.0 * tt).sin()
                })
                .collect()
        };
        let target = make(3.0);
        let close = make(3.2); // similar decay
        let no_decay = make(0.0); // sustained — wrong envelope
        let w = LossWeights::default();
        let l_close = layered_perceptual_loss(&target, &close, sr, &w);
        let l_nodecay = layered_perceptual_loss(&target, &no_decay, sr, &w);
        assert!(
            l_nodecay > l_close,
            "the wrong-envelope candidate must score worse: {l_nodecay} vs {l_close}"
        );
        assert!(
            decay_distance(&target, &no_decay, sr) > decay_distance(&target, &close, sr),
            "the per-band decay term must distinguish the wrong envelope"
        );
    }

    // The residual diagnostic must point at the right lever: a wrong-decay
    // candidate's residual is dominated by the decay term, recommending a kernel
    // loop-filter extension (not more tuning).
    #[test]
    fn residual_dominant_term_points_at_the_wrong_decay() {
        let sr = 8000.0;
        let n = 8000;
        let make = |decay: f64| -> Vec<f64> {
            (0..n)
                .map(|t| {
                    let tt = t as f64 / sr;
                    (-decay * tt).exp() * (2.0 * PI * 300.0 * tt).sin()
                })
                .collect()
        };
        let r = decompose_residual(&make(3.0), &make(0.0), sr, &LossWeights::default());
        assert_eq!(
            r.dominant, "decay",
            "a wrong-decay candidate's residual must be decay-dominated, got {r:?}"
        );
        assert!(
            r.recommendation().contains("loop filter"),
            "recommendation should point at the kernel loop filter, got: {}",
            r.recommendation()
        );
    }

    // Codex P2: a candidate that matches the target's prefix but has a long
    // ringing tail the target lacks must be penalized for that tail — not scored
    // ~0 by comparing only the common prefix.
    #[test]
    fn stft_loss_penalizes_unmatched_tail() {
        let sr = 8000.0;
        let n = 2048;
        // target: a brief tone that goes silent after 50 ms.
        let target: Vec<f64> = (0..n)
            .map(|t| {
                let tt = t as f64 / sr;
                if tt < 0.05 {
                    (2.0 * PI * 300.0 * tt).sin()
                } else {
                    0.0
                }
            })
            .collect();
        // candidate: same, but keeps ringing for another 2048 samples.
        let mut candidate = target.clone();
        for t in n..(n + 2048) {
            candidate.push(0.5 * (2.0 * PI * 300.0 * t as f64 / sr).sin());
        }
        let with_tail = stft_loss(&target, &candidate, 256, 64);
        let identical = stft_loss(&target, &target, 256, 64);
        assert!(identical < 1e-9, "identical → 0");
        assert!(
            with_tail > 0.1,
            "an unmatched ringing tail must add loss, got {with_tail}"
        );
    }
}
