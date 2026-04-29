# Phase 7 Validation Results — Live OPTIC-K Weight Tuning

**Status:** Final — weak signal, **not conclusive**
**Date:** 2026-04-29
**Plan:** `docs/plans/2026-04-26-001-feat-ix-autoresearch-edit-eval-iterate-plan.md` Phase 7
**Code:** `crates/ix-autoresearch/src/target_optick.rs` (RebuildMode::CIReduced) + `examples/phase7_validate.rs`
**Reversibility:** Two-way door (the validation is observational; no code paths changed in production code).

---

## Hypothesis Under Test

The IX↔GA autoresearch loop, with `Strategy::SimulatedAnnealing` + Dirichlet-on-simplex perturbation, can find an OPTIC-K partition-weight vector that **lowers `structure_leak_pct` by ≥ 1pp from the uniform-1/6 baseline**, **reproducibly** across multiple seeds.

Falsification criterion: if the seed-to-seed stddev of the per-seed delta is comparable to the mean (signal/noise μ/σ < 2), the loop is chasing noise.

## Method

- **Corpus**: tri-instrument (guitar + bass + ukulele), `--export-max 1500` per instrument → ~4500 voicings.
- **Iterations**: 30 per seed.
- **Seeds**: 3 (`20260429`, `20261432`, `20262435` — base ± 1_000_003 increments).
- **Strategy**: Simulated Annealing with calibrated initial T from the first eval, cooling rate 0.95.
- **Perturbation**: Dirichlet at α=200, floor=1e-3, around the current accepted config.
- **Score**:
  - `structure_leak_pct` = (STRUCTURE classifier accuracy − 1/3) / (1 − 1/3), clamped [0, 1]
  - `retrieval_match_pct` = avg PC-set match @ k=10 across 30 random queries
  - Invariants #25/#32/#36 pass rates from `ix-optick-invariants` stderr
- **Reward**: lex-order `1.0 − leak` dominates by 1e6, retrieval at 1e3, mean(invariants) breaks ties.

## Smoke check (pre-run)

3 iters × 1 seed × 1500 voicings/instrument completed in **57.5s** (~19s/iter).
- iter 0 (uniform 1/6 baseline): leak = **0.036**
- iter 1 (SA proposes): leak = **0.026**
- iter 2 (SA proposes): leak = 0.067
- best across 3 iters: 0.026 (1.0pp below baseline)

Pipeline parses scores cleanly, all subprocesses exit 0, no JSON errors.

## Results

Wall-clock: 7m55s (much faster than the conservative 30-min estimate; per-iter ≈ 5.3s once warm).

### Per-seed trajectories (`structure_leak_pct` per iter)

```
seed 20260429: 0.034 0.054 0.049 0.053 0.062 0.012 0.031 0.011 0.001 0.000
               0.024 0.004 0.017 0.032 0.004 0.003 0.000 0.036 0.029 0.040
               0.043 0.041 0.030 0.018 0.004 0.022 0.000 0.015 0.000 0.037
seed 21260432: 0.000 0.000 0.042 0.028 0.000 0.021 0.056 0.033 0.000 0.003
               0.058 0.034 0.014 0.041 0.018 0.018 0.000 0.041 0.017 0.000
               0.024 0.021 0.017 0.027 0.032 0.017 0.021 0.022 0.050 0.000
seed 22260435: 0.018 0.012 0.022 0.043 0.012 0.021 0.042 0.007 0.036 0.060
               0.036 0.018 0.000 0.000 0.046 0.084 0.029 0.008 0.030 0.000
               0.015 0.032 0.000 0.000 0.021 0.013 0.000 0.003 0.019 0.030
```

Notable: every seed bottoms out at exactly `0.000` multiple times. This is suspicious — see §Verdict.

### Per-seed table

| seed     | baseline_leak | best_leak | delta (pp) | accepted | failures | elapsed |
|---------:|--------------:|----------:|-----------:|---------:|---------:|--------:|
| 20260429 |        0.0340 |    0.0000 |       3.40 |       25 |        0 | 160.9 s |
| 21260432 |        0.0000 |    0.0000 |       0.00 |       25 |        0 | 157.4 s |
| 22260435 |        0.0180 |    0.0000 |       1.80 |        0 |        0 | 156.7 s |

Seed 22260435 has 0 SA-accepted moves but still produced 30 evals — every proposal lost the SA accept test, so the trajectory is just the rejected proposals. The fact that its lex-best is still `0.000` (matching the others) is itself the smoking gun.

### Across seeds

| metric            | μ      | σ      |
|-------------------|-------:|-------:|
| baseline leak     | 0.0173 | 0.0139 |
| best leak         | 0.0000 | 0.0000 |
| delta (pp)        | 1.73   | 1.39   |
| **signal/noise**  | **1.25** |        |

### Lex-best config (seed 20260429)

```
leak_pct: 0.0000
cfg:      S=0.127 M=0.294 C=0.152 Y=0.128 L=0.173 R=0.126
```

## Verdict — **Weak signal, not conclusive**

`μ/σ = 1.25` falls in the "weak" band (1 ≤ ratio < 2). But the more honest reading is **"the score signal hits a noise floor before the optimizer finds anything real"**:

1. **Every seed bottoms out at exactly `0.000`.** That's not the optimizer being clever — that's the score saturating. `structure_leak_pct = max(0, (acc − 1/3) / (2/3))`. When the RandomForest classifier on a 500-sample-per-instrument fold dips to ≤ 1/3 accuracy by chance, leak is reported as zero.
2. **Seed 21260432's *baseline* (uniform 1/6) is `0.000`.** The loop "improved" by 0pp because it started saturated. Same RF noise floor.
3. **Seed 22260435 had zero SA-accepted moves**, yet its lex-best is `0.000`. That can only happen if a *rejected* proposal happened to score 0 due to RF cross-validation noise.

**The optimizer is finding noise, not signal.** The 3.4pp delta on seed 1 is sampling variance of the diagnostics binary, not a discovery about partition weights.

This is a **falsifiable, honest negative result**: the loop *plumbing* is correct (89 in-crate tests + 6-case GA contract matrix), but the *score* at this corpus size + diagnostic budget cannot tell good weight vectors from bad ones.

## Limitations of this run

- **Corpus is CI-reduced**, not full 313K voicings. A weight vector that's better on 4.5K may not generalize.
- **Score is single-seed for the diagnostics binary** (`--seed 42`), so RF cross-validation noise is partially absorbed but not zeroed.
- **30 iters is short for SA convergence**; 100+ iters would give the cooling schedule more room.
- **Baseline is uniform 1/6**, NOT production weights — the loop optimizes around uniform, not around the deployed schema. A separate run starting from production weights would answer "can we beat what's deployed?".

## What would make a future run conclusive

Listed by leverage (biggest signal-floor reduction first):

1. **Bigger classifier sample.** `--class-samples-per-instrument 4000` (the diag binary's default) instead of 500. 8× more training data → tighter accuracy distribution → less of the "leak collapses to 0" floor effect.
2. **Average over several diagnostic seeds.** Currently the diag binary runs with `--seed 42` fixed. Running each weight config under k=3–5 diag seeds and taking the median absorbs the RF cross-validation jitter. Cost: k× per-iter time.
3. **Bigger corpus.** `--export-max 5000` per instrument (15K voicings) makes the retrieval-consistency metric dominate. Rebuild cost goes from ~4s to ~10s per iter; per-seed run goes from 2.5min to 5min — still tractable.
4. **Replace lex-order leak with a continuous score** that doesn't saturate at zero. E.g. raw `STRUCTURE accuracy − 1/3` keeping negative values, or a softplus to discourage the floor.
5. **Start from production weights, not uniform.** Production has higher leak (~0.114 in our smoke probe) so there's real headroom. Uniform 1/6 is already inside the noise band.

## Status after Phase 7

- The *infrastructure* (cross-repo loop, GA `--weights-config` contract, live evaluator, validation harness) is shipped and tested.
- The *scientific question* ("can the loop find a better OPTIC-K weight vector?") is **unanswered** — the current score signal is too noisy at CI-reduced sizes to distinguish improvement from RF-classifier sampling variance.
- This is a **two-way door**: the loop can be re-run with the items above without changing any code paths in production. No promotions, no schema changes, no governance asks.

## Artifacts

- Validation log: `state/autoresearch/phase7-validation-2026-04-29.log`
- Per-seed JSONL logs: temp dirs cleaned up at process exit (lost). Add `--keep-logs` flag if needed for forensics.
- Code: pushed at `5284b30` on IX `main`.
