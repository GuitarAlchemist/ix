# Jarvis J4 tracer bullet — GBDT vs persistence on a ga `state/quality/` series

- **Issue:** [GuitarAlchemist/ix#221](https://github.com/GuitarAlchemist/ix/issues/221)
- **Date:** 2026-07-06
- **Verdict:** **PAUSED — insufficient data (honest-pause rule, Karpathy R4).** No ga
  `state/quality/` series has enough real transitions to train and evaluate a
  learned transition model against persistence. Beating persistence was never
  assumed; the epic pauses with the exact counts below as committed evidence.

## What was built

The smallest end-to-end slice of the J4 "learned predictive world model" epic —
data ingest → hand-crafted features → GBDT → persistence baseline → held-out
eval → committed report — as one thin vertical, reusing existing ix machinery:

- **Harness:** `crates/ix-quality-trend/src/forecast.rs` (module `forecast`).
  Loads one ga `state/quality/<category>/*.json` series, builds six recency
  features, frames the target as the **direction of the next step**
  (`Down`/`Flat`/`Up`), trains a GBDT, and compares it against persistence on a
  chronological held-out tail. Honest-pause guardrail returns
  `Verdict::PausedInsufficientData` with exact counts when a series is too thin.
- **Model:** `ix_ensemble::gradient_boosting::GradientBoostedClassifier`
  (workspace-internal — **no new external dependency**; the fit is an exhaustive
  threshold search, so results are reproducible with no RNG).
- **Baseline:** persistence = "last value carries forward". In the
  direction-classification decision space that is exactly "always predict
  `Flat`", so the two are directly comparable.
- **Eval bin:** `crates/ix-quality-trend/examples/gbdt_vs_persistence.rs`.

### Framing note (why classification, not regression)

The issue's reference technique is Facebook predictive test selection
(ICSE-SEIP 2019) — a *classifier* over history features ("will this regress?"),
matching the CI ~99:1 class-imbalance note. `ix-ensemble`'s GBDT is likewise
classifier-only (softmax), so the target is the next-step **direction**, and
persistence maps to the constant `Flat` prediction.

### The six features (from what the snapshots actually contain)

The named "change class / files touched / author count" features **do not
exist** in the ga quality snapshots — those are daily/periodic metric dumps with
no per-commit metadata. The features are therefore recency-style, derived from
the series itself:

| # | feature | meaning |
|---|---|---|
| 1 | `last_value` | level at t-1 |
| 2 | `last_delta` | momentum: v[t-1] − v[t-2] |
| 3 | `flat_run_len` | consecutive flat steps ending at t-1 |
| 4 | `steps_since_change` | steps since the last non-flat step (recency) |
| 5 | `last_direction_sign` | sign of `last_delta` ∈ {−1, 0, 1} |
| 6 | `last_was_real` | 1.0 fresh measurement, 0.0 carried/degraded |

## Data reality (the reason for the pause)

Read path: `../ga/state/quality/` (sibling clone). `real` = fresh measurement;
`carried/degraded` = the producer literally copied the previous value.

| series | metric | snapshots w/ value | real | carried/degraded | distinct values | transitions | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| chatbot-qa | `pass_pct` | 25 | **2** | 23 | 3 | 3 | PAUSED (n_real=2 < 8) |
| embeddings | `leak_detection_full_classifier_accuracy` | 35 | **3** | 32 | 4 | 2 | PAUSED (n_real=3 < 8; transitions=2 < 3) |
| voicing-analysis | `voicing_analysis_avg_pass_rate` | 38 | 38 | 0 | **1** | 0 | PAUSED (transitions=0 < 3) |

- **chatbot-qa** — the issue's headline series: of 33 dated files, only **two**
  carry a non-null `pass_pct` (75.0 on 2026-06-15, 7.69 on 2026-06-19). The rest
  are backend-degraded nulls; 23 carry a last-known-good forward. Two fresh
  points cannot support a train/test split.
- **embeddings** — 35 points but piecewise-constant: ~4 distinct values dominated
  by long carried-forward / degraded runs, only **2** real transitions
  (0.747→0.752 in April, 0.752→0.831 on 2026-06-25). The producer *already
  implements persistence* via `carried_forward: true`. The transition points are
  exogenous (OPTIC-K rebuilds), unpredictable from series-history features.
- **voicing-analysis** — 38 points, all `metric_value = 1.0`. Zero variance:
  persistence is trivially perfect and there is nothing to learn.

Across every candidate series, none provides both usable target variance **and**
the per-change features the technique needs. That is the honest pause.

## Reproduce

```bash
cd ix
# Eval against the sibling ga clone (adjust path as needed):
cargo run -p ix-quality-trend --example gbdt_vs_persistence -- \
    --snapshots-dir ../ga/state/quality \
    --out state/jarvis/2026-07-06-j4-gbdt-vs-persistence.eval.json

# Harness unit tests (no ga checkout required — uses in-crate fixtures):
cargo test -p ix-quality-trend --lib forecast
```

Machine-readable evidence: `state/jarvis/2026-07-06-j4-gbdt-vs-persistence.eval.json`.

## Revisit trigger (two-way door)

Unpark J4 when **either**: (a) a ga `state/quality/` series accumulates ≥ 8 fresh
measurements with ≥ 3 real transitions, **or** (b) a series is enriched with the
per-change commit features (change class, files touched, author count) the
Facebook technique actually relies on — at which point re-run the harness. Any
growth into a real ML surface requires the Demerzel tribunal per the issue.
