//! Reference analyzer — extracts the perceptual descriptor vector from a target
//! recording (or a single note's samples), to **seed** the search (warm start)
//! and to form the **naturalness** terms of the objective (see
//! `docs/plans/2026-06-07-ix-acoustic-tune.md` §"Naturalness objective").
//!
//! The headline is [`per_band_decay_slopes`] — highs decay faster than lows, and
//! that per-band decay trajectory is the timbral fingerprint a single averaged
//! spectrum throws away. All analysis is pure `f64`, offline, over `ix-signal`'s
//! transforms (no new DSP transforms; just reductions). Operates on samples — a
//! WAV reader is deferred (GA passes samples), per the plan.

use serde::{Deserialize, Serialize};

use crate::features;

/// Default STFT window for the decay/feature analysis (power of two).
pub const DEFAULT_WINDOW: usize = 1024;
/// Default STFT hop.
pub const DEFAULT_HOP: usize = 256;

/// Least-squares slope of `y` vs `x` over `(x, y)` points; `0.0` if degenerate.
fn least_squares_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let sx: f64 = points.iter().map(|(x, _)| x).sum();
    let sy: f64 = points.iter().map(|(_, y)| y).sum();
    let sxx: f64 = points.iter().map(|(x, _)| x * x).sum();
    let sxy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-20 {
        0.0
    } else {
        (n * sxy - sx * sy) / denom
    }
}

/// Parabolic interpolation of a spectral peak at bin `b`, returning the sub-bin
/// offset in `[-0.5, 0.5]` for sharper frequency estimates than the bin grid.
fn parabolic_offset(mags: &[f64], b: usize) -> f64 {
    if b == 0 || b + 1 >= mags.len() {
        return 0.0;
    }
    let (l, c, r) = (mags[b - 1], mags[b], mags[b + 1]);
    let denom = l - 2.0 * c + r;
    if denom.abs() < 1e-20 {
        0.0
    } else {
        (0.5 * (l - r) / denom).clamp(-0.5, 0.5)
    }
}

/// Decay slope of `signal` within `[f_lo, f_hi]`, in **log-energy per second**,
/// via a least-squares fit of log band-energy across the STFT frames. **More
/// negative ⇒ faster decay.** Returns `0.0` if the band has too little energy or
/// too few frames. `window_size` should be a power of two.
// @ai:invariant band_decay_slope returns a more-negative slope for a band that decays faster, so per-band slopes capture the differential decay (highs vs lows) that a single averaged spectrum discards [T:test conf:0.9 src:reference::tests::per_band_decay_distinguishes_fast_from_slow_bands]
pub fn band_decay_slope(
    signal: &[f64],
    sample_rate: f64,
    f_lo: f64,
    f_hi: f64,
    window_size: usize,
    hop_size: usize,
) -> f64 {
    if signal.len() < window_size || sample_rate <= 0.0 {
        return 0.0;
    }
    let spec = ix_signal::spectral::spectrogram(signal, window_size, hop_size, false);
    if spec.len() < 2 {
        return 0.0;
    }
    let bin_hz = sample_rate / window_size as f64;
    let n_bins = spec[0].len();
    let k_lo = ((f_lo / bin_hz).floor() as usize).min(n_bins.saturating_sub(1));
    let k_hi = ((f_hi / bin_hz).ceil() as usize).min(n_bins.saturating_sub(1));
    if k_hi <= k_lo {
        return 0.0;
    }

    let frame_dt = hop_size as f64 / sample_rate;
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(spec.len());
    for (i, frame) in spec.iter().enumerate() {
        let energy: f64 = (k_lo..=k_hi).map(|k| frame[k] * frame[k]).sum();
        if energy > 1e-20 {
            points.push((i as f64 * frame_dt, energy.ln()));
        }
    }
    least_squares_slope(&points)
}

/// Per-band decay slopes for the bands bounded by `band_edges`
/// (`[e0, e1, …, en]` ⇒ `n` bands `[e_i, e_{i+1})`), in log-energy per second.
pub fn per_band_decay_slopes(signal: &[f64], sample_rate: f64, band_edges: &[f64]) -> Vec<f64> {
    band_edges
        .windows(2)
        .map(|w| band_decay_slope(signal, sample_rate, w[0], w[1], DEFAULT_WINDOW, DEFAULT_HOP))
        .collect()
}

/// Estimate the inharmonicity coefficient `B` from partial peaks: a stiff string's
/// partials sit at `f_k = k·f0·√(1 + B·k²)`. Picks the (parabolically refined)
/// magnitude peak near each `k·f0` and fits `B` through the origin from
/// `(f_k/(k·f0))² − 1 = B·k²`. Returns `~0` for a perfectly harmonic signal and a
/// positive value for stretched partials. `window_size` should be a power of two.
// @ai:invariant inharmonicity_b reads ≈0 for a perfectly harmonic signal and a clearly higher positive B for stretched (stiff-string) partials, so it distinguishes a natural inharmonic guitar tone from a synthetic perfectly-harmonic one [T:test conf:0.85 src:reference::tests::inharmonicity_separates_stretched_from_harmonic]
pub fn inharmonicity_b(
    signal: &[f64],
    sample_rate: f64,
    f0: f64,
    n_partials: usize,
    window_size: usize,
) -> f64 {
    if f0 <= 0.0 || n_partials < 2 || sample_rate <= 0.0 || signal.len() < window_size {
        return 0.0;
    }
    // Analyze a window centered in the (sustained) signal for stable partials.
    let start = (signal.len() - window_size) / 2;
    let frame = &signal[start..start + window_size];
    let mags = features::magnitude_spectrum(frame);
    let bin_hz = sample_rate / window_size as f64;
    let search = ((f0 / bin_hz) * 0.4).ceil() as i64; // ±40% of f0 around k·f0

    let mut xs = Vec::new(); // k²
    let mut ys = Vec::new(); // (f_k/(k·f0))² − 1
    for k in 1..=n_partials {
        let center = ((k as f64 * f0) / bin_hz).round() as i64;
        let lo = (center - search).max(1) as usize;
        let hi = ((center + search) as usize).min(mags.len().saturating_sub(2));
        if hi <= lo {
            continue;
        }
        let mut best_bin = lo;
        let mut best_mag = mags[lo];
        for (off, &m) in mags[lo..=hi].iter().enumerate() {
            if m > best_mag {
                best_mag = m;
                best_bin = lo + off;
            }
        }
        if best_mag <= 0.0 {
            continue;
        }
        let f_k = (best_bin as f64 + parabolic_offset(&mags, best_bin)) * bin_hz;
        let ratio = f_k / (k as f64 * f0);
        xs.push((k as f64).powi(2));
        ys.push(ratio * ratio - 1.0);
    }
    if xs.len() < 2 {
        return 0.0;
    }
    // Fit ys = B·xs through the origin: B = Σ(x·y) / Σ(x²). Clamp to physical B≥0.
    let num: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    let den: f64 = xs.iter().map(|x| x * x).sum();
    if den < 1e-20 {
        0.0
    } else {
        (num / den).max(0.0)
    }
}

/// Attack time: seconds from the onset (first short-time-RMS frame above 10 % of
/// the peak) to the RMS-envelope peak — where the pick noise / onset burst lives.
pub fn attack_seconds(signal: &[f64], sample_rate: f64, window: usize) -> f64 {
    if signal.is_empty() || sample_rate <= 0.0 || window == 0 {
        return 0.0;
    }
    let env: Vec<f64> = signal
        .chunks(window)
        .map(|c| (c.iter().map(|x| x * x).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    if env.is_empty() {
        return 0.0;
    }
    let (peak_idx, &peak) = env
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    if peak <= 0.0 {
        return 0.0;
    }
    let threshold = 0.1 * peak;
    let onset_idx = env.iter().position(|&e| e >= threshold).unwrap_or(0);
    peak_idx.saturating_sub(onset_idx) as f64 * window as f64 / sample_rate
}

/// The perceptual descriptor vector extracted from a reference note — both the
/// warm-start seed for the search and the naturalness reference the layered loss
/// scores against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceDescriptor {
    pub f0_hz: Option<f64>,
    pub centroid_hz: f64,
    pub rolloff_hz: f64,
    pub rms: f64,
    pub attack_seconds: f64,
    pub inharmonicity_b: f64,
    /// Decay slope (log-energy/sec, more negative = faster) per band in `band_edges`.
    pub band_decay_slopes: Vec<f64>,
    pub band_edges: Vec<f64>,
}

/// Default analysis bands (Hz edges) spanning fundamental → brilliance, roughly
/// mirroring the GA engine's 5 body resonators (110/240/530/1200/2400 Hz).
pub fn default_band_edges() -> Vec<f64> {
    vec![60.0, 180.0, 360.0, 750.0, 1600.0, 3500.0, 8000.0]
}

/// Analyze a single reference note into its [`ReferenceDescriptor`].
pub fn analyze(signal: &[f64], sample_rate: f64) -> ReferenceDescriptor {
    let band_edges = default_band_edges();
    let f0 = features::autocorrelation_f0(signal, sample_rate, 50.0, 2000.0);

    // Feature frame from the centered sustain (power-of-two for calibrated bins).
    let frame_len = DEFAULT_WINDOW.min(signal.len().max(1).next_power_of_two().min(DEFAULT_WINDOW));
    let frame: Vec<f64> = if signal.len() >= frame_len {
        let start = (signal.len() - frame_len) / 2;
        signal[start..start + frame_len].to_vec()
    } else {
        signal.to_vec()
    };
    let mags = features::magnitude_spectrum(&frame);
    let fft_size = frame.len().max(1);

    ReferenceDescriptor {
        f0_hz: f0,
        centroid_hz: features::spectral_centroid(&mags, sample_rate, fft_size),
        rolloff_hz: features::spectral_rolloff(&mags, sample_rate, fft_size, 0.85),
        rms: features::rms(signal),
        attack_seconds: attack_seconds(signal, sample_rate, DEFAULT_HOP),
        inharmonicity_b: f0
            .map(|f| inharmonicity_b(signal, sample_rate, f, 8, DEFAULT_WINDOW))
            .unwrap_or(0.0),
        band_decay_slopes: per_band_decay_slopes(signal, sample_rate, &band_edges),
        band_edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    // The dominant naturalness term: a band that decays faster must yield a more
    // negative slope. Low band (100 Hz, slow) vs high band (1500 Hz, fast).
    #[test]
    fn per_band_decay_distinguishes_fast_from_slow_bands() {
        let sr = 8000.0;
        let n = 8000;
        let sig: Vec<f64> = (0..n)
            .map(|t| {
                let tt = t as f64 / sr;
                (-2.0 * tt).exp() * (TAU * 100.0 * tt).sin()
                    + (-10.0 * tt).exp() * (TAU * 1500.0 * tt).sin()
            })
            .collect();
        let slopes = per_band_decay_slopes(&sig, sr, &[50.0, 300.0, 3000.0]);
        assert_eq!(slopes.len(), 2);
        assert!(
            slopes[0] < 0.0 && slopes[1] < 0.0,
            "both bands decay: {slopes:?}"
        );
        assert!(
            slopes[0] > slopes[1],
            "the slow low band must decay slower (less-negative slope) than the fast high band: {slopes:?}"
        );
    }

    // Inharmonicity: a stretched-partial signal must score B clearly above a
    // perfectly harmonic one (which must be ≈0).
    #[test]
    fn inharmonicity_separates_stretched_from_harmonic() {
        let sr = 8000.0;
        let n = 4096;
        let f0 = 200.0;
        let mk = |b: f64| -> Vec<f64> {
            (0..n)
                .map(|t| {
                    let tt = t as f64 / sr;
                    (1..=6)
                        .map(|k| {
                            let fk = k as f64 * f0 * (1.0 + b * (k as f64).powi(2)).sqrt();
                            (TAU * fk * tt).sin()
                        })
                        .sum::<f64>()
                })
                .collect()
        };
        let harmonic = inharmonicity_b(&mk(0.0), sr, f0, 6, 4096);
        let stretched = inharmonicity_b(&mk(0.004), sr, f0, 6, 4096);
        assert!(
            harmonic < 0.001,
            "harmonic signal should read B≈0, got {harmonic}"
        );
        assert!(
            stretched > 0.001 && stretched > harmonic,
            "stretched partials should read a clearly higher B: {stretched} vs {harmonic}"
        );
    }

    #[test]
    fn attack_time_tracks_a_ramp() {
        // A realistic pluck: 0.05 s linear attack ramp, then exponential decay
        // (a real note decays after the attack — not a constant plateau).
        let sr = 8000.0;
        let attack_n = (0.05 * sr) as usize;
        let n = 8000;
        let sig: Vec<f64> = (0..n)
            .map(|t| {
                let tt = t as f64 / sr;
                let env = if t < attack_n {
                    t as f64 / attack_n as f64
                } else {
                    (-3.0 * (tt - 0.05)).exp()
                };
                env * (TAU * 200.0 * tt).sin()
            })
            .collect();
        let a = attack_seconds(&sig, sr, 64);
        assert!((a - 0.05).abs() < 0.02, "attack {a} should be ≈0.05 s");
    }

    #[test]
    fn analyze_bundles_a_descriptor() {
        let sr = 8000.0;
        let sig: Vec<f64> = (0..8000)
            .map(|t| {
                let tt = t as f64 / sr;
                (-3.0 * tt).exp() * (TAU * 220.0 * tt).sin()
            })
            .collect();
        let d = analyze(&sig, sr);
        assert_eq!(d.band_decay_slopes.len(), d.band_edges.len() - 1);
        assert!(
            d.f0_hz.map(|f| (f - 220.0).abs() < 10.0).unwrap_or(false),
            "f0 ≈ 220"
        );
        assert!(d.rms > 0.0 && d.centroid_hz > 0.0);
    }
}
