# IX ↔ GA acoustic auto-tuning (`ix-acoustic-tune`)

**Date:** 2026-06-07
**Status:** Phase 1 — contract **signed off 2026-06-07**; optimizer core (CMA-ES, ask/tell, features, spectral loss) shipped in `ix-acoustic-tune`; orchestrator + MCP skills in progress.

**Sign-off (2026-06-07) — the one-way doors, decided:**
- **Transport:** JSON-on-disk sidecar (`tune-request.json` / `tune-result.json`), the established GA↔IX pattern.
- **Result payload:** scalar `spectral_score` (Phase A) **+ optional `features`** block (forward-compatible to Phase B with no schema break).
- **Objective:** **A-then-B** — validate against GA's as-is `SPECTRAL_SCORE` first, then a corrected mel-MSS+centroid loss as a human-merged upgrade.
- **`guitar_type`:** outer enumeration (tune the 5 continuous params per type; CMA-ES stays purely continuous).
- **Param-vector order:** `[decay, brightness, dispersion, attack_decay, reverb_mix]` with the GA clamp ranges (table below).
**Repos:** ix (optimizer + offline DSP analysis), ga (`Rust/guitar-web-wasm-demo`, the realtime synth + renderer)
**Enables:** turn GA's guitar-synth "iterate by ear / textual suggestions" loop into a
governed **propose → executable-oracle → human-merge** parameter search: IX proposes
synth params, GA renders + scores the audio against a reference recording, IX consumes
the score and proposes the next vector — until the spectral distance to the reference is
minimized.

## Problem / who is in pain

GA's `guitar-web-wasm-demo` is a pure-Rust **Karplus–Strong** acoustic-guitar synth
compiled to WASM (`rust-engine/src/lib.rs`), with an auto-iteration critic
(`full-auto.ps1` → `scripts/run-spectral-critic.js`) that renders the synth, computes a
spectral profile vs a reference (`reference/by-the-lake.wav`), and emits a scalar
`SPECTRAL_SCORE` + **textual** suggestions ("reduce brightness", "lower flux"). Two
weaknesses: (1) the critic **hand-rolls** an FFT + Hann-window + spectral
centroid/rolloff/flux in JS (`scripts/compute-spectral-profile.js`), and (2) the loop
does **not** close — a human reads the hints and re-tweaks. The synth exposes a tiny
continuous param space; matching it to a target is a textbook black-box optimization
that nobody is actually running.

## Key finding (corrects an earlier mischaracterization)

IX's `optimize` **skill** only exposes named benchmark functions
(`handlers.rs:172` hardcodes `sphere`/`rosenbrock`/`rastrigin`) — but the **crate**
underneath already optimizes arbitrary objectives:
`ClosureObjective<F: Fn(&Array1<f64>) -> f64>` (`ix-optimize/src/traits.rs:38`) wraps any
Rust closure into PSO / Simulated Annealing as a true black-box objective. So IX needs
**less** extension than first assumed: the optimizer engine and every raw DSP transform
(`ix-signal`: fft/stft/spectrogram/welch_psd/dct/autocorrelation/window/resampling)
already exist and are tested. The genuine, source-verified gaps:

| Gap | Lives in | Note |
|---|---|---|
| No **ask/tell** driver | `ix-optimize` | the closed `minimize()` owns the eval loop in-process; the render here is **external** |
| No high-level **audio features** | `ix-signal` | only raw transforms; centroid/rolloff/flux/MFCC/f0/decay absent |
| No **CMA-ES / Nelder-Mead** | `ix-optimize` | only PSO + SA today |
| No packaged **spectral loss** | `ix-signal` | multi-resolution STFT loss must be assembled |
| No **WAV I/O** | `ix-io` (csv/json/tcp/http/ws only) | **deferred** — keep WAV decode in GA |

## Decision — the ask/tell external-objective architecture

The crux of "optimize an external, render-required objective" is to **hoist `evaluate()`
out of IX's loop**:

```
loop:
  params  = optimizer.ask()          # IX proposes (one vector, or a CMA-ES generation)
  # ---- GA side: render WASM synth(params) -> audio -> spectral score vs reference ----
  optimizer.tell(params, score)      # IX updates its state from the EXTERNAL score
recommend() -> best params + history
```

No change to `ObjectiveFunction`/`ClosureObjective` — the driver wraps the existing
PSO/SA (and new CMA-ES) *step* engines so the external renderer, not IX, supplies each
score. This is IX's signature **propose → executable-oracle → govern** loop, and it
**generalizes** beyond guitar to any sibling-repo parameter tuning (router thresholds,
chatbot, …).

## Extension inventory (ranked; reuse-heavy)

1. **`ask_tell` driver** — `ix-optimize/src/ask_tell.rs` (M, high). `trait AskTell { fn ask(&mut self) -> Vec<Array1<f64>>; fn tell(&mut self, &[(Array1<f64>, f64)]); fn recommend(&self) -> OptimizeResult; }`. Refactor PSO/SA to step from externally-supplied scores. *Reuses pso.rs/annealing.rs state + `ClosureObjective` unchanged.*
2. **`features.rs`** — `ix-signal/src/features.rs` (M, high). Centroid, rolloff(85%), flux, bandwidth, flatness, band-energy ratios, RMS/decay envelope, autocorrelation-f0 (peak-pick). Each a few-line reduction over `magnitude_spectrum`/`welch_psd`/`spectrogram`. Field-named to mirror GA's `compute-spectral-profile.js`.
3. **`spectral_loss.rs`** — `ix-signal/src/spectral_loss.rs` (S, high). Multi-resolution STFT loss (log-mag L1 + spectral-convergence over 2–3 FFT sizes) + low-weight centroid term. Magnitude-only (phase discarded) — standard for param estimation.
4. **CMA-ES** — `ix-optimize/src/cmaes.rs` (M, high). The de-facto standard for ~5-param non-convex synth matching; a step engine behind `AskTell`, box-constrained to the GA clamps. PSO/SA stay as fallbacks. *Reuses the `ObjectiveFunction` trait + ix-math eigendecomposition for the covariance update.*
5. **MCP `#[ix_skill]` surface** — `ix-agent/src/skills/acoustic_tune.rs` (M, med). `acoustic_tune_{init,ask,tell,recommend}` + stateless `spectral_features`/`spectral_distance`. Template: `skills/assumption_graph.rs`.
6. **`ix-acoustic-tune`** crate (lib + bin) — orchestrates ask → [host renders] → tell over the contract; thin.
7. *(DEFERRED, demand-gated)* `ix-io/src/wav.rs` (only if IX must own extraction) and a DDSP differentiable spectral loss via `ix-autograd` `fft-autograd` (research-grade; not needed at 5 params).

## Build order

1. **Lock the GA↔IX contract** (one-way door — below). Both repos build against it.
2. `ix-signal::features` (+ unit tests vs known signals — reuse the "welch peak at f0" test shape).
3. `ix-signal::spectral_loss` (start with plain L1-on-STFT — near-best + simplest for plucked-string; escalate to mel-MSS+centroid only if timbrally off).
4. `ix-optimize::ask_tell` + `ix-acoustic-tune` (closes the loop end-to-end with PSO/SA IX already has).
5. `ix-optimize::cmaes` (becomes the default; heuristic init from target analysis: rolloff→brightness, f0→delay, RMS-decay→damping).
6. `ix-agent` skill surface + instrumentation (baseline `SPECTRAL_SCORE`, clip/silence guardrail).
7. *(deferred)* WAV I/O, DDSP loss.

## Contract DRAFT — `urn:ix:acoustic-tune:v0.1-draft`

JSON-on-disk handoff (the canonical GA↔IX pattern, transported via `ix-io::json_io`).
v0.1.x is a **draft** — freeze only at the named Phase-4 milestone; use
`links.supersedes` for any baseline shift (mirror `optick-sae-artifact` discipline).

**`tune-request.json`** (IX → GA, "render these"):
```json
{
  "schema": "urn:ix:acoustic-tune:v0.1-draft/request",
  "session_id": "by-the-lake-2026-06-07",
  "iteration": 7,
  "optimizer": "cmaes",
  "param_names": ["decay", "brightness", "dispersion", "attack_decay", "reverb_mix"],
  "candidates": [
    { "id": "c0", "params": [0.997, 0.42, 0.18, 0.985, 0.25] }
  ],
  "guitar_type": 0,
  "render": { "note_hz": 110.0, "velocity": 0.9, "seconds": 4.0, "sample_rate": 48000 }
}
```

**`tune-result.json`** (GA → IX, "here is what they sounded like" — the metric+guardrail pair):
```json
{
  "schema": "urn:ix:acoustic-tune:v0.1-draft/result",
  "session_id": "by-the-lake-2026-06-07",
  "iteration": 7,
  "reference": "reference/by-the-lake.wav",
  "scores": [
    {
      "id": "c0",
      "params": [0.997, 0.42, 0.18, 0.985, 0.25],
      "spectral_score": 0.8123,            // metric (higher = closer)
      "loss": 0.1877,                       // = 1 - spectral_score (what IX minimizes)
      "feature_deltas": {                   // guardrails / diagnostics
        "centroid_hz": -42.0, "rolloff_hz": 110.0, "flux": 0.01,
        "band_low": 0.0, "band_mid": 0.0, "band_high": -0.03
      },
      "guardrail": { "clipped": false, "silent": false, "rms": 0.21 }
    }
  ]
}
```

### Param-vector convention (the actuator interface — locked once)

Index order + units must match the GA FFI clamps (`rust-engine/src/lib.rs:521–566`,
verified):

| idx | name | range (GA clamp) | setter |
|---|---|---|---|
| 0 | `decay` | [0.95, 0.9999] | `engine_set_decay` |
| 1 | `brightness` | [0.0, 1.0] | `engine_set_brightness` |
| 2 | `dispersion` | [0.0, 0.5] | `engine_set_dispersion` |
| 3 | `attack_decay` | [0.95, 0.999] | `engine_set_attack_decay` |
| 4 | `reverb_mix` | [0.0, 0.9] | `engine_set_reverb_mix` |
| — | `guitar_type` | categorical {0..3} | `engine_set_guitar_type` |

**`guitar_type` is NOT a CMA-ES continuous dimension** — handle as an outer enumeration
(tune the 5 continuous params per type) or a rounded/penalized integer. Box-constrain
CMA-ES to the ranges above.

## Open decision (needs your call) — objective calibration

GA's current `SPECTRAL_SCORE` (`run-spectral-critic.js`) **zero-weights** `bandLow`/`bandMid`
and blends a self-declared **placeholder `wav2vec2`** term. This is a governed,
Goodhart-sensitive, one-way calibration choice:

- **Option A — optimize GA's as-is objective.** Pro: validates the loop end-to-end against
  the existing score (parity); fastest to first result. Con: optimizes a known-imperfect
  metric (the zero-weighted bands + placeholder model) — risk of "winning" a flawed score.
- **Option B — optimize a corrected mel-MSS + centroid loss** (the best-practice recipe).
  Pro: principled, perceptually grounded. Con: a new objective definition that must be
  validated; bigger one-way door.

**Recommendation: A then B.** Phase A proves the harness against the as-is score
(cheap, reversible parity check); Phase B swaps in the corrected loss as a *governed*
upgrade with human-merge, surfacing GA's band-weighting + placeholder as `@ai:assumption [U]`
rather than silently reweighting (safe-RSI: the oracle defines the surface but is still
Goodhart-vulnerable — the human gate is non-optional).

## Naturalness objective (Phase B+) — the real engine

Minimizing a single averaged magnitude-spectrum distance will **not** sound
natural: magnitude losses are phase-blind by design and an averaged spectrum
discards the cues that define a guitar — attack transient, inharmonicity,
frequency-dependent decay, body resonance (Goodhart, documented for audio:
*The PESQetarian*, arXiv:2406.03460). Naturalness needs two things:

**(1) A layered, perceptual, temporal objective** (replaces the single scalar;
MSS backbone per DDSP/auraloss, then weighted terms):
- (a) multi-frame MSS over the decay — keep per-frame structure, don't average;
- (b) attack-window weighting (first ~30–50 ms — pick noise + onset);
- (c) mel/MFCC perceptual weighting (cochlear frequency warping; IX has DCT,
  needs a mel filterbank);
- (d) **per-band decay-slope matching** — highs decay faster than lows; *the*
  timbral fingerprint the scalar throws away (the single highest-leverage new term);
- (e) f0 + inharmonicity (stiff-string partial stretching; IX has f0, needs a
  B-estimator).

**(2) Respect the model-capacity ceiling.** Tuning the 5 params reaches
brightness/sustain/attack-balance/reverb. It **cannot** reach true inharmonicity,
frequency-dependent per-band decay, or a real body — those need GA **kernel
extensions** (all-pass cascade; tunable loop filter; commuted-synthesis body IR;
richer excitation). IX's **residual** (per-term loss of reference − best)
diagnoses *which*: residual on decay-slope → extend the loop filter (stop
tuning); on inharmonicity → extend the all-pass; on body → swap the IR; broadly
small → at capacity, tuning is right. This is the safe-RSI oracle
([[reference_safe_rsi_loop_validated]]) telling you retune (cheap) vs extend-kernel
(reviewed change).

**The loop:** analyze reference → seed params from descriptors (warm start) →
CMA-ES on the layered loss (GA returns the `features` blob; IX owns the
objective) → residual-diagnose → extend/retune. **Match multiple notes** (not
one) to avoid single-pitch overfit.

**Validation:** every term is a proxy CMA-ES will exploit → a **human ABX test
is the merge gate** (metric proposes, human disposes), paired with the
`clipped`/`silent` guardrails — the non-optional human-merge of safe-RSI.

**IX builds:** reference-analyzer (`reference.rs` + `bin/analyze_reference`),
`layered_perceptual_loss()`, mel-filterbank + per-band-decay-slope +
inharmonicity-B estimators, a frame-weighted STFT variant, `decompose_residual()`,
warm-start wiring, a backward-compatible multi-note contract field.
**GA extends (only after the residual proves it):** tunable loop filter →
all-pass cascade → measured body IR → richer excitation; expose `engine_set_*` +
add to `param_names` for each; ship each only through the ABX gate.

## Reversibility & one-way doors

- **Two-way:** `AskTell` trait, CMA-ES, `features.rs`, `spectral_loss.rs`,
  `ix-acoustic-tune` internals — all additive; existing PSO/SA/`ObjectiveFunction` untouched.
- **One-way (sign-off required):**
  1. The `tune-request`/`tune-result` **contract schema** (cross-repo, GA-consumed).
  2. The **param-vector ordering + units** (actuator interface).
  3. The **MCP tool schemas** for `acoustic_tune_*` (stable-surface once agents bind;
     additions = minor, removals = major-bump).
  4. The **objective definition** (Option A vs B above) — governed, human-merge.
- **Revisit trigger:** freeze the contract at Phase 4 (`urn:` → public `$id`); re-open if
  GA adds synth params (extends the vector) or changes the score weighting.

## Do NOT build (reuse)

FFT/rfft/magnitude_spectrum (`ix-signal/fft.rs`), STFT/spectrogram/welch_psd
(`spectral.rs`), DCT-for-MFCC (`dct.rs`), windows (`window.rs`), autocorrelation
(`correlation.rs`), resampling (`sampling.rs`), `ObjectiveFunction`/`ClosureObjective`
(`traits.rs` — **unchanged**), PSO/SA step engines, `#[ix_skill]` macro + `ix-registry`,
`ix-io::json_io` for the handoff, and `ix-autograd` ops (only if the deferred DDSP path is
ever wanted). **The realtime f32 Karplus–Strong kernel + WAV decode stay in GA** — pass
feature vectors / magnitude frames as JSON; IX renders nothing and needs no WAV reader on
the critical path.

## Honest boundary

IX stays **offline f64 analysis + the tuning brain**. It never enters the audio thread.
The realtime/offline split (sentrux = realtime, ix-signal/ix-optimize = offline) holds.
IX *can* design/validate filter coefficients offline (`fir_filter`/`svd`/spectral tools)
that then get hand-coded into GA's f32 kernel — design in IX, run in the kernel.
