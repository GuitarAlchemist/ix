---
title: Measuring guitar inharmonicity — fixed-f0 fit is corrupted by pitch error; the engine is not "too harmonic"
date: 2026-06-08
category: math-correctness
tags: [acoustic-tune, inharmonicity, dsp, oracle, guitar-engine, negative-result, estimator]
components: [ix-acoustic-tune, ga/guitar-web-wasm-demo]
status: resolved
---

# Inharmonicity measurement: the pitch-error confound, and an overturned premise

## Context
Goal: make the GA Karplus–Strong guitar engine "as natural as a recorded sample"
(`ga/Rust/guitar-web-wasm-demo/reference/by-the-lake.wav`, a real 114 s acoustic
recording). A multi-agent workflow + adversarial review concluded the #1
naturalness gap was **inharmonicity** — the engine read `B ≈ 0` (perfectly
harmonic = the "synthetic/organ" tell), the real guitar should be `B ≈ 1e-4`.
We built the full render→measure loop and went to add a dispersion allpass.

## The surprise (oracle overturned the premise)
With a **rigorous** estimator the engine is **not** too harmonic. Measured on the
dry string, early window, robust fit:

| | Engine baseline | Real guitar |
|---|---|---|
| Bass B (E2/G2) | **7–9 × 10⁻⁵** | ~5.5 × 10⁻⁵ |

The engine's loop filters (one-pole damping + the 2-point averaging filter)
already produce guitar-scale inharmonicity. Adding a dispersion allpass (which
works — it drives B from 7e-5 → 2.56e-4 as the coefficient goes 0 → −0.9) moves
the engine **away** from the recording. **The dispersion change was not shipped.**

## Root cause of the original "B ≈ 0" — three estimator bugs
1. **Fixed-f0 fit folds pitch error into a spurious B.** `inharmonicity_b` holds
   `f0` fixed; the engine is ~10 cents flat, so the `k=1` partial contributes
   `B₁ = (ratio²−1) ≈ −1.2e-2`, dragging the least-squares fit. A few cents of
   f0 error fabricates `B ≈ 1e-2`. The "B≈0" was symmetric scatter averaging out.
2. **Window too coarse for bass.** A 1024-pt FFT (bin ≈ 47 Hz) cannot separate a
   98 Hz note's partials. Needs ≥ ~8·sr/f0 (16384 pts for a low note).
3. **Wrong analysis window.** The engine's highs decay fast; a mid-note window
   leaves only 2–3 partials. Use the **early** window (first ~300 ms) where high
   partials still exist.

## The fix — `reference::fit_inharmonicity` (joint f0+B, robust)
Linearize `f_k = k·f0·√(1+B·k²)` ⇒ `(f_k/k)² = f0² + (f0²·B)·k²`, regress
`Y=(f_k/k)²` vs `X=k²` (magnitude-weighted), iteratively dropping the single
worst-residual partial (> 5× median) to reject spurious peaks (body modes,
polyphonic bleed). Intercept → f0², slope/intercept → B. **Pitch error cancels.**
Test `joint_fit_is_robust_to_a_wrong_f0_guess` proves it recovers f0+B with a
26-cent-wrong guess where the fixed-f0 fit inflates B > 5×.

## Hard limit worth remembering
At the real-guitar level (B ~ 5e-5) inharmonicity is **near the measurement noise
floor** on both the synth and the recording (a real performance's partials
scatter from vibrato/body/polyphony). Precise B-matching is **below the noise** —
don't tune to it. Large B (≥ 2e-4, piano-scale) is cleanly measurable; small
guitar B is not.

## What the oracle said the real gaps ARE
- **Tuning: ±11 cents** across strings (flat low / sharp high). Cause: linear
  fractional-delay interpolation (HF) + uncompensated loop-filter phase (LF).
  Fix shipped to the worktree: **Thiran first-order allpass FD** (flat group
  delay, no HF loss) + per-note loop-filter phase compensation
  (`0.5·(1−brightness)·lp_group_delay(f0)`). Result: **±11 → ±6 cents**, most
  strings ±3. Full ±1c needs a 2nd-order Thiran + per-string calibration.
- **HF sustain: engine highs decay ~2× too fast** (1.6–8 kHz: engine −8/−9 vs
  recording −3.4/−4.8 log-energy/s, clean single-note 0.5 s windows). The loop
  lowpass is too aggressive at HF. Fix = relax loop HF damping; best done as a
  CMA-ES match of the engine's (decay, brightness, damping) to the reference
  per-band decay profile (what `ix-acoustic-tune` is for). Not yet done.

## Method that worked (reusable)
Native render harness (`guitar_engine` as `rlib` + `examples/ix_render.rs`,
`engine_set_dry_out` to bypass body/reverb/clip) → IX `analyze_reference`
example (`fit_inharmonicity` + per-band decay + LTAS centroid). Every claim is a
number from a render, adversarially cross-checked against the recording — not an
opinion. The opinion (workflow + initial framing) was wrong; the oracle was right.
