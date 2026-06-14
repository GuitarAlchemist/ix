//! Offline reference-recording analyzer (dev harness, not part of the lib API).
//!
//! Turns a real guitar recording into the target [`ReferenceDescriptor`] that the
//! naturalness oracle scores the GA synth against. A 16-bit PCM WAV is decoded to
//! mono `f64`, then either:
//!   * pass 1 (no `--start`): print global stats + the cleanest single-note
//!     windows (onset followed by a confident, stable f0), so you can pick one;
//!   * pass 2 (`--start S --dur D`): analyze exactly that window into a descriptor.
//!
//! Usage:
//!   cargo run -p ix-acoustic-tune --example analyze_reference -- <path.wav>
//!   cargo run -p ix-acoustic-tune --example analyze_reference -- <path.wav> --start 12.5 --dur 1.5
//!
//! No new deps: the WAV reader is a minimal RIFF/PCM16 parser local to this
//! harness (the crate proper still takes samples, per the plan).

use ix_acoustic_tune::features;
use ix_acoustic_tune::reference;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: analyze_reference <path.wav> [--start SEC --dur SEC] [--f0 HZ]");
        std::process::exit(2);
    }
    let path = &args[1];
    let mut start: Option<f64> = None;
    let mut dur: f64 = 1.5;
    let mut force_f0: Option<f64> = None;
    let mut partials = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--partials" => {
                partials = true;
                i += 1;
            }
            "--start" => {
                start = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--dur" => {
                dur = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1.5);
                i += 2;
            }
            "--f0" => {
                force_f0 = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                i += 1;
            }
        }
    }

    let (samples, sr) = match load_wav_mono(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };
    let n = samples.len();
    let dur_total = n as f64 / sr;
    println!("== {path}");
    println!("   samples={n}  sample_rate={sr}  duration={dur_total:.3}s");

    // --- Global stats (also: is this a real recording or an engine render?) ---
    let peak = samples.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let rms_all = features::rms(&samples);
    let clip = samples.iter().filter(|&&x| x.abs() >= 0.999).count();
    // Frame RMS envelope for noise-floor + onset work.
    let win = 2048usize;
    let hop = 512usize;
    let env: Vec<f64> = (0..n.saturating_sub(win))
        .step_by(hop)
        .map(|s| features::rms(&samples[s..s + win]))
        .collect();
    let mut sorted = env.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| {
        if sorted.is_empty() {
            0.0
        } else {
            sorted[((p * (sorted.len() - 1) as f64) as usize).min(sorted.len() - 1)]
        }
    };
    let floor = pct(0.05);
    let median = pct(0.50);
    println!(
        "   peak={peak:.4}  rms={rms_all:.4}  clip_samples={clip}  noise_floor(p05)={floor:.5}  median_env={median:.4}  crest={:.1}dB",
        20.0 * (peak / rms_all.max(1e-9)).log10()
    );
    println!(
        "   realness hint: noise_floor {} 0 (real recordings have a non-zero floor; a synth render is near-silent between notes)",
        if floor > 1e-4 { ">" } else { "~=" }
    );

    // --- Whole-file long-term average spectrum → aggregate brightness ---
    // (robust to polyphony; characterizes the guitar's spectral signature)
    {
        let fsize = 4096usize;
        let fhop = 2048usize;
        let mut acc = vec![0.0f64; fsize / 2 + 1];
        let mut frames = 0usize;
        let mut s = 0usize;
        while s + fsize <= n {
            let mags = features::magnitude_spectrum(&samples[s..s + fsize]);
            for (a, m) in acc.iter_mut().zip(&mags) {
                *a += *m;
            }
            frames += 1;
            s += fhop;
        }
        if frames > 0 {
            for a in &mut acc {
                *a /= frames as f64;
            }
            let centroid = features::spectral_centroid(&acc, sr, fsize);
            let rolloff = features::spectral_rolloff(&acc, sr, fsize, 0.85);
            println!(
                "   LTAS over {frames} frames: centroid={centroid:.0}Hz  rolloff85={rolloff:.0}Hz"
            );
        }
    }

    let _ = (floor, median);
    if let Some(st) = start {
        // Pass 2: analyze the requested window precisely.
        analyze_window(&samples, sr, st, dur, force_f0);
        if partials {
            if let Some(f0) = force_f0 {
                dump_partials(&samples, sr, st, dur, f0);
            } else {
                eprintln!("--partials needs --f0 <hz>");
            }
        }
        return;
    }

    // Pass 1: dense short-window scan → maximal runs of stable, monophonic pitch
    // (a clean single note ringing out before the next pluck overlaps it).
    let wlen = (0.12 * sr) as usize; // 120 ms
    let whop = (0.025 * sr) as usize; // 25 ms
    struct Fr {
        t: f64,
        f0: f64,
        strength: f64,
        rms: f64,
    }
    let mut frs: Vec<Fr> = Vec::new();
    let mut s = 0usize;
    while s + wlen <= n {
        let (f0, strength) = periodicity(&samples[s..s + wlen], sr, 70.0, 1000.0);
        frs.push(Fr {
            t: s as f64 / sr,
            f0: f0.unwrap_or(0.0),
            strength,
            rms: features::rms(&samples[s..s + wlen]),
        });
        s += whop;
    }
    // Group consecutive frames whose pitch is stable (≤4% drift) and clearly
    // periodic (strength ≥ 0.80); each run is one ringing note.
    #[derive(Clone)]
    struct Cand {
        t: f64,
        ring: f64,
        f0: f64,
        strength: f64,
        rms: f64,
    }
    let mut cands: Vec<Cand> = Vec::new();
    let mut i0 = 0usize;
    while i0 < frs.len() {
        if frs[i0].strength < 0.80 || frs[i0].f0 <= 0.0 {
            i0 += 1;
            continue;
        }
        let f_ref = frs[i0].f0;
        let mut j = i0 + 1;
        while j < frs.len()
            && frs[j].strength >= 0.70
            && frs[j].f0 > 0.0
            && (frs[j].f0 - f_ref).abs() / f_ref < 0.04
        {
            j += 1;
        }
        let ring = frs[j.min(frs.len() - 1)].t - frs[i0].t;
        if ring >= 0.35 {
            let med_f0 = {
                let mut v: Vec<f64> = frs[i0..j].iter().map(|f| f.f0).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            let avg_str = frs[i0..j].iter().map(|f| f.strength).sum::<f64>() / (j - i0) as f64;
            let avg_rms = frs[i0..j].iter().map(|f| f.rms).sum::<f64>() / (j - i0) as f64;
            cands.push(Cand {
                t: frs[i0].t,
                ring,
                f0: med_f0,
                strength: avg_str,
                rms: avg_rms,
            });
        }
        i0 = j.max(i0 + 1);
    }
    cands.sort_by(|a, b| {
        let sa = a.strength * a.ring.min(1.0) * a.rms;
        let sb = b.strength * b.ring.min(1.0) * b.rms;
        sb.partial_cmp(&sa).unwrap()
    });
    println!("\n  cleanest single-note runs (stable f0 × periodicity × rms):");
    println!(
        "   {:>7}  {:>6}  {:>7}  {:>8}  {:>6}",
        "t(s)", "ring", "f0(Hz)", "periodic", "rms"
    );
    for c in cands.iter().take(15) {
        println!(
            "   {:>7.3}  {:>6.2}  {:>7.1}  {:>8.3}  {:>6.4}",
            c.t, c.ring, c.f0, c.strength, c.rms
        );
    }

    // Auto-analyze the top few clean notes (forcing the run's own f0, which the
    // generic detector rails on for short windows).
    for c in cands.iter().take(3) {
        println!(
            "\n  [auto] note at t={:.3}s (f0≈{:.1}Hz, ring {:.2}s):",
            c.t, c.f0, c.ring
        );
        analyze_window(&samples, sr, c.t, c.ring.min(0.8), Some(c.f0));
    }
}

fn analyze_window(samples: &[f64], sr: f64, start_s: f64, dur_s: f64, force_f0: Option<f64>) {
    let n = samples.len();
    let s0 = ((start_s * sr) as usize).min(n.saturating_sub(1));
    let s1 = (((start_s + dur_s) * sr) as usize).min(n);
    if s1 <= s0 + 1024 {
        eprintln!("window too short");
        return;
    }
    let seg = &samples[s0..s1];
    let d = reference::analyze(seg, sr);
    let f0 = force_f0.or(d.f0_hz);
    println!(
        "   window [{start_s:.3}..{:.3}]s  ({} samples)",
        start_s + dur_s,
        seg.len()
    );
    // detected_f0 is the autocorrelation pitch (pitch-preservation guardrail);
    // cents vs the forced/known f0 quantifies any dispersion-induced flattening.
    let cents = match (d.f0_hz, force_f0) {
        (Some(det), Some(known)) if known > 0.0 => 1200.0 * (det / known).log2(),
        _ => 0.0,
    };
    println!(
        "   detected_f0={:?}Hz  ({:+.1} cents vs forced)  centroid={:.0}Hz  rolloff85={:.0}Hz  rms={:.4}",
        d.f0_hz, cents, d.centroid_hz, d.rolloff_hz, d.rms
    );
    // Inharmonicity wants frequency resolution: a 1024-pt window can't separate a
    // 110 Hz note's partials (bin≈47 Hz). Use the largest power-of-two ≤ seg/2.
    if let Some(f) = f0 {
        let mut wsize = 1024usize;
        while wsize * 2 <= seg.len() && wsize < 16384 {
            wsize *= 2;
        }
        let b = reference::inharmonicity_b(seg, sr, f, 10, wsize);
        println!(
            "   attack={:.4}s  inharmonicity_B@{:.1}Hz={:.3e}  (FFT window {})",
            d.attack_seconds, f, b, wsize
        );
    } else {
        println!(
            "   attack={:.4}s  inharmonicity_B(no f0)={:.3e}",
            d.attack_seconds, d.inharmonicity_b
        );
    }
    println!("   per-band decay slopes (log-energy/s, more negative = faster):");
    for (w, slope) in d.band_edges.windows(2).zip(&d.band_decay_slopes) {
        println!("      {:>5.0}-{:<5.0}Hz : {:>9.3}", w[0], w[1], slope);
    }
}

/// Print the measured partial frequencies, their stretch ratio fₖ/(k·f0), and the
/// implied per-partial Bₖ = ((fₖ/(k·f0))²−1)/k². A stiff string gives a flat Bₖ
/// (the k² law); the k=1 row is the precise fundamental → exact pitch readout.
fn dump_partials(samples: &[f64], sr: f64, start_s: f64, dur_s: f64, f0: f64) {
    let n = samples.len();
    let s0 = ((start_s * sr) as usize).min(n.saturating_sub(1));
    let s1 = (((start_s + dur_s) * sr) as usize).min(n);
    if s1 <= s0 + 2048 {
        return;
    }
    let seg = &samples[s0..s1];
    let mut wsize = 1024usize;
    while wsize * 2 <= seg.len() && wsize < 16384 {
        wsize *= 2;
    }
    let start = (seg.len() - wsize) / 2;
    let mags = features::magnitude_spectrum(&seg[start..start + wsize]);
    let bin_hz = sr / wsize as f64;
    let search = ((f0 / bin_hz) * 0.4).ceil() as i64;
    let parab = |b: usize| -> f64 {
        if b == 0 || b + 1 >= mags.len() {
            return 0.0;
        }
        let (l, c, r) = (mags[b - 1], mags[b], mags[b + 1]);
        let d = l - 2.0 * c + r;
        if d.abs() < 1e-20 {
            0.0
        } else {
            (0.5 * (l - r) / d).clamp(-0.5, 0.5)
        }
    };
    println!("   partials (FFT window {wsize}, bin={bin_hz:.2}Hz):");
    println!(
        "   {:>2}  {:>9}  {:>8}  {:>10}",
        "k", "f_k(Hz)", "mag", "B_k(naive)"
    );
    // Collect (k, f_k, mag) for strong, in-range peaks.
    let mut pts: Vec<(f64, f64, f64)> = Vec::new(); // (k, f_k, mag)
    let mut max_mag = 0.0f64;
    for k in 1..=12i64 {
        let center = ((k as f64 * f0) / bin_hz).round() as i64;
        let lo = (center - search).max(1) as usize;
        let hi = ((center + search) as usize).min(mags.len().saturating_sub(2));
        if hi <= lo {
            continue;
        }
        let (mut best_bin, mut best) = (lo, mags[lo]);
        for (off, &m) in mags[lo..=hi].iter().enumerate() {
            if m > best {
                best = m;
                best_bin = lo + off;
            }
        }
        let f_k = (best_bin as f64 + parab(best_bin)) * bin_hz;
        max_mag = max_mag.max(best);
        pts.push((k as f64, f_k, best));
        let ratio = f_k / (k as f64 * f0);
        let b_k = (ratio * ratio - 1.0) / (k as f64).powi(2);
        println!("   {k:>2}  {f_k:>9.2}  {best:>8.3}  {b_k:>10.2e}");
    }

    // Robust joint (f0, B) fit. Linearize f_k = k·f0·√(1+B·k²):
    //   (f_k/k)² = f0² + (f0²·B)·k²   ⇒  Y = c0 + c1·X,  X=k², Y=(f_k/k)².
    //   f0 = √c0,  B = c1/c0.  Magnitude-weighted, with iterative outlier rejection
    //   (a spurious peak — body mode, polyphonic bleed — must not bend the fit).
    let mut keep: Vec<usize> = (0..pts.len())
        .filter(|&i| pts[i].2 >= 0.02 * max_mag) // drop only very weak partials
        .collect();
    let mut f0_fit = f0;
    let mut b_fit = 0.0;
    for _round in 0..4 {
        if keep.len() < 3 {
            break;
        }
        // weighted linear regression Y = c0 + c1 X
        let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &i in &keep {
            let (k, f_k, mag) = pts[i];
            let x = k * k;
            let y = (f_k / k).powi(2);
            let w = mag;
            sw += w;
            sx += w * x;
            sy += w * y;
            sxx += w * x * x;
            sxy += w * x * y;
        }
        let denom = sw * sxx - sx * sx;
        if denom.abs() < 1e-9 {
            break;
        }
        let c1 = (sw * sxy - sx * sy) / denom;
        let c0 = (sy - c1 * sx) / sw;
        if c0 <= 0.0 {
            break;
        }
        f0_fit = c0.sqrt();
        b_fit = c1 / c0;
        // residuals in Hz; drop the worst if it's a clear outlier (> 3× median).
        let mut resid: Vec<(usize, f64)> = keep
            .iter()
            .map(|&i| {
                let (k, f_k, _) = pts[i];
                let pred = k * f0_fit * (1.0 + b_fit * k * k).max(0.0).sqrt();
                (i, (f_k - pred).abs())
            })
            .collect();
        let mut rs: Vec<f64> = resid.iter().map(|&(_, r)| r).collect();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = rs[rs.len() / 2].max(1e-6);
        resid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if resid[0].1 > 5.0 * median && keep.len() > 4 {
            let drop = resid[0].0;
            keep.retain(|&i| i != drop);
        } else {
            break;
        }
    }
    let cents = 1200.0 * (f0_fit / f0).log2();
    println!(
        "   robust joint fit: f0={f0_fit:.2}Hz ({cents:+.1} cents vs forced) B={b_fit:.3e}  [{} partials kept]",
        keep.len()
    );
}

/// Normalized-autocorrelation periodicity: returns (f0, peak strength in [0,1]).
fn periodicity(signal: &[f64], sr: f64, fmin: f64, fmax: f64) -> (Option<f64>, f64) {
    if signal.len() < 64 {
        return (None, 0.0);
    }
    // zero-mean
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let x: Vec<f64> = signal.iter().map(|s| s - mean).collect();
    let energy: f64 = x.iter().map(|v| v * v).sum();
    if energy < 1e-12 {
        return (None, 0.0);
    }
    let min_lag = (sr / fmax).floor() as usize;
    let max_lag = ((sr / fmin).ceil() as usize).min(x.len() - 1);
    if min_lag < 1 || max_lag <= min_lag {
        return (None, 0.0);
    }
    let mut best = 0.0f64;
    let mut best_lag = 0usize;
    for lag in min_lag..=max_lag {
        let mut acc = 0.0;
        for j in 0..x.len() - lag {
            acc += x[j] * x[j + lag];
        }
        let norm = acc / energy;
        if norm > best {
            best = norm;
            best_lag = lag;
        }
    }
    if best_lag == 0 {
        (None, 0.0)
    } else {
        (Some(sr / best_lag as f64), best.clamp(0.0, 1.0))
    }
}

/// Minimal RIFF/WAVE reader: 16-bit PCM, mono or stereo (averaged to mono).
/// Returns (mono samples in [-1,1], sample_rate).
fn load_wav_mono(path: &str) -> Result<(Vec<f64>, f64), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let u16le = |b: &[u8], o: usize| (b[o] as u16) | ((b[o + 1] as u16) << 8);
    let u32le = |b: &[u8], o: usize| {
        (b[o] as u32)
            | ((b[o + 1] as u32) << 8)
            | ((b[o + 2] as u32) << 16)
            | ((b[o + 3] as u32) << 24)
    };
    let mut i = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (audio_fmt, channels, sr, bits)
    let mut data: Option<(usize, usize)> = None; // (offset, len)
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let clen = u32le(&bytes, i + 4) as usize;
        let body = i + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            fmt = Some((
                u16le(&bytes, body),
                u16le(&bytes, body + 2),
                u32le(&bytes, body + 4),
                u16le(&bytes, body + 14),
            ));
        } else if id == b"data" {
            let len = clen.min(bytes.len().saturating_sub(body));
            data = Some((body, len));
        }
        i = body + clen + (clen & 1);
    }
    let (afmt, ch, sr, bits) = fmt.ok_or("no fmt chunk")?;
    let (off, len) = data.ok_or("no data chunk")?;
    if afmt != 1 || bits != 16 {
        return Err(format!(
            "only 16-bit PCM supported (got fmt={afmt} bits={bits})"
        ));
    }
    let ch = ch.max(1) as usize;
    let n_frames = len / (2 * ch);
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let base = off + f * 2 * ch;
        let mut acc = 0.0f64;
        for c in 0..ch {
            let o = base + c * 2;
            let s = (u16le(&bytes, o) as i16) as f64 / 32768.0;
            acc += s;
        }
        out.push(acc / ch as f64);
    }
    Ok((out, sr as f64))
}
