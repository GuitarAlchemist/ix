---
title: ix-math::inference — the statistical-inference primitive (two-sample tests, divergences, moments)
type: feat
status: draft
date: 2026-06-21
origin: investigation "are we missing any ML/math tool in DuckDB/IX?" (2026-06-21)
reversibility: one-way (new public ix-math surface + new ix-duck UDFs) — see "One-way door" below
revisit-trigger: first telemetry regression-gate that needs distribution-shift detection, or any consumer importing ix_math::inference
---

# ix-math::inference — the missing statistical layer

> One-page design produced from the 2026-06-21 gap analysis of the DuckDB/IX surface.
> Of six gaps found, five are *exposure* gaps (wire an existing IX algorithm through a
> UDF). **This is the only *algorithm* gap** — the primitive is absent from `ix-math`
> *and* from DuckDB built-ins. It is therefore the highest-leverage piece and the one
> that needs design before code.

## Problem / why

`ix-duck` is an analyst bench over **JSONL time-series telemetry** (routing, OOD, loop,
chatbot, quality lenses). The single most-used analyst question over such data is
*"did this metric's distribution shift versus baseline?"* — the core of any RSI /
regression gate. Today neither side can answer it:

- **`ix-math::stats`** has only `mean / variance / std_dev / median / covariance_matrix /
  correlation_matrix / min_max`. No higher moments, no quantiles, no divergences, no tests.
- **DuckDB built-ins** give `quantile`, `stddev`, `corr`, `regr_*` — but **no** two-sample
  test (KS, Mann–Whitney, t-test), **no** divergence (KL, JS), **no** skew/kurtosis/MAD,
  **no** entropy / mutual-information.

So the `maintain` gate currently compares distributions with hand-rolled thresholds on
means. That is exactly the Goodhart-prone shortcut the RSI work warned against: a mean can
hold steady while the *distribution* degrades (variance blow-up, bimodality, tail shift).

Who is in pain: the maintain-gate / RSI loop (and any future telemetry gate) — it cannot
make a principled "distributions differ" call. What changes: a fail-closed gate can assert
"today's `coverage_max` distribution differs from the 30-day baseline at p<0.01 (KS)" as a
**bound, testable** verdict instead of a tuned threshold.

## What we're building

A new pure-Rust module `ix-math::inference` (no new deps — built on `ndarray` + the
existing `stats` primitives), then a thin UDF layer in `ix-duck`. Two tiers:

### Tier 1 — descriptive (distribution shape over one column)
| fn | signature | notes |
|----|-----------|-------|
| `quantile` | `(x, q) -> f64` | linear interpolation; `q∈[0,1]` |
| `iqr`, `mad` | `(x) -> f64` | robust spread (MAD = median abs deviation) |
| `skewness`, `kurtosis` | `(x) -> f64` | 3rd/4th standardized moments (excess kurtosis) |
| `zscore` | `(x) -> Array1` | `(xᵢ − mean)/std`; the normalize the bench keeps re-deriving |
| `shannon_entropy` | `(p) -> f64` | over a normalized histogram / probability vector |

### Tier 2 — inferential (two-sample: today vs baseline)
| fn | signature | returns | notes |
|----|-----------|---------|-------|
| `ks_two_sample` | `(a, b) -> TestResult` | `{statistic, p_value}` | **the primary gate primitive** — distribution-free, no normality assumption |
| `mann_whitney_u` | `(a, b) -> TestResult` | `{u, p_value}` | rank-based location shift |
| `welch_t_test` | `(a, b) -> TestResult` | `{t, df, p_value}` | unequal-variance means |
| `kl_divergence` | `(p, q) -> f64` | nats | over aligned histograms; asymmetric |
| `js_divergence` | `(p, q) -> f64` | nats | symmetric, bounded — safer default than KL |

`TestResult { statistic: f64, p_value: f64 }` — one small public struct.

### UDF layer (`ix-duck`, after the module lands)
- Scalars over `DOUBLE[]`: `ix_quantile(x, q)`, `ix_skewness(x)`, `ix_kurtosis(x)`,
  `ix_mad(x)`, `ix_entropy(p)`, `ix_kl(p,q)`, `ix_js(p,q)`.
- Table fn `ix_two_sample(a DOUBLE[], b DOUBLE[], test VARCHAR) -> (statistic, p_value)`
  so SQL can pick KS / Mann–Whitney / Welch by name in one call.

## Tracer-bullet slice (Karpathy r2 / aihero)

Smallest end-to-end cut, **all layers**: `ks_two_sample` only →
`ix_two_sample(..., 'ks')` UDF → one `maintain`-gate query that compares today's
`coverage_max` vs the baseline window → assert a known-shifted fixture trips p<0.01 and a
same-distribution fixture does not. Ship that, get feedback, *then* fan out the rest of the
table. Do **not** build all 10 functions before the first SQL call works.

## Instrumentation (CLAUDE.md "instrument before you ship")

- **Baseline**: the current maintain-gate verdict series in
  `state/thinking-machine/maintain-gate.jsonl`.
- **Expected direction**: false-flip rate of the gate should *drop* (mean-threshold
  flapping replaced by a distribution test); detection of a genuine seeded regression
  should *fire*.
- **Guardrail**: KS must not flag two samples drawn from the same fixture (TNR check) —
  the asymmetric fail-closed discipline from `feedback_llm_judge_panel_failclosed`. A test
  primitive that cries wolf is worse than none.

## One-way door — sign-off required

New **public `ix-math` API** (a stable-tier crate) + new public `TestResult` type. Per
CLAUDE.md the stable-surface gate (`pub `-line hash) will trip on the added `pub fn`s →
this needs the version-bump / beta-demote dance and operator sign-off before merge. The
**p-value implementations** are the risk: KS/Mann–Whitney asymptotic tails and the
t-distribution CDF must be validated against a reference (SciPy values pinned in tests),
or the gate is "green but wrong." No LLM may stand in as the correctness oracle here —
pin numeric fixtures.

Two-way (cheap to revert): the `ix-duck` UDF names — internal-tier, no downstream contract.

## Out of scope

- Multiple-comparison correction, bootstrap CIs, Bayesian tests — add only if a consumer asks.
- Replacing DuckDB's native `quantile`/`stddev`/`corr` — we *complement*, never shadow them.

## Links

- Gap analysis: this investigation (2026-06-21).
- RSI guardrail discipline: `docs/solutions/**` + the maintain-gate ledger.
- Sibling exposure-gaps (separate issues): ix-signal time-series UDFs, t-SNE/MDS projection,
  supervised fit/predict, graph centrality.
