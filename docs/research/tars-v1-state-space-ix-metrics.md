# State-space and control metrics for agent loops (TARS V1 → IX)

**Issue:** [#193](https://github.com/GuitarAlchemist/ix/issues/193) · Parent epic #189 · Depends on #190
**Companion, not duplicate:** [`tars-v1-advanced-math-ix-gap-matrix.md`](tars-v1-advanced-math-ix-gap-matrix.md)

**Status:** design note with executable fixtures. Four metrics proposed; all four are
implemented as test-local compositions and validated by
[`crates/ix-chaos/tests/loop_metrics.rs`](../../crates/ix-chaos/tests/loop_metrics.rs)
(10 tests, all passing). Numbers quoted below are measured, not illustrative.

## What this document is, and what it is not

The gap matrix's [section A](tars-v1-advanced-math-ix-gap-matrix.md#a-state-space-and-control-theory)
covers state-space and control **algorithms** — the linear model `x_{k+1}=Ax_k+Bu_k+w_k` (A1),
the non-linear case (A2), Kalman (A3), MPC (A4), observability/controllability tests (A5),
Lyapunov stability certificates (A6). That layer is about *machinery*.

This note is the layer above it: **metrics you can compute over episodes and traces** to say
whether an agent loop is converging, oscillating, stalling or diverging. It deliberately does
not restate the matrix, and it does not require A1 to exist. Every metric proposed here runs
on a sequence of per-step observables, using primitives IX has already shipped.

The distinction matters practically: A1 and A5 are marked `missing_algorithm` in the matrix,
so a design that depends on them cannot be built today. The four metrics below can be built
now, which is why they are the deliverable.

## Source notes

- **Gap matrix rows reused**: A3 (Kalman, `implemented_needs_exposure`), A6 (Lyapunov —
  *exponent* shipped, *function* missing). Rows A1/A2/A4/A5/A7 are **not** dependencies of
  anything below.
- **A correction to the brief.** `crates/ix-graph/src/state_space.rs` was suggested as a
  candidate to reuse. It is not usable here, and the matrix's A1 row already flags why: the
  file defines `trait State`, `SearchResult<S>`, `astar()` and `beam_search()` — a *discrete
  search space*, not a linear time-invariant system. Verified directly in this worktree.
  A `state_space` grep is a false positive for this work.
- **A primitive the brief did not mention, and the best find in this survey.**
  `crates/ix-signal/src/timeseries.rs` already ships two concept-drift detectors —
  `ddm_detect` (Drift Detection Method) and `page_hinkley_detect` (Page–Hinkley) — plus
  `rolling_std`, `rolling_mean`, `ewma`, `difference` and `pct_change`. The issue asks for
  "drift metrics"; IX has had drift detection all along. M1 is a direct reuse with **no new
  mathematics**.
- **Naming discipline inherited from A6.** IX ships the Lyapunov *exponent*
  (`crates/ix-chaos/src/lyapunov.rs`), not a Lyapunov *function*. M3 below is careful not to
  claim otherwise.

## The observable: what a loop metric actually consumes

Every metric here reads a per-step sequence. The common model is deliberately thin — one
scalar and one boolean per step is enough for three of the four metrics:

```text
episode := [ step_0, step_1, ..., step_n ]
step    := { residual: f64, step_failed: bool, action_signature: f64 }
```

`residual` is whatever the loop is trying to drive to zero: failing-test count, unresolved
diagnostics, diff distance to target, review comments outstanding. The metrics do not care
which, only that it is monotone-meaningful and comparable across steps of the same episode.

## Candidate metrics

### M1 — Progress drift · reuses `ix_signal::timeseries::ddm_detect`

**Question:** is the loop getting *worse* than it was at its best?

DDM tracks a binary error rate and flags when it degrades significantly from the running
minimum. An agent loop's `step_failed` flag has exactly that shape, so this is a direct reuse.

```rust
fn first_drift_step(step_failed: &[bool], config: DdmConfig) -> Option<usize> {
    ddm_detect(step_failed, config)
        .iter()
        .position(|s| s.state == DriftState::Drift)
}
```

**Measured** (150 steps at 10% failure, then 150 at 70%, `DdmConfig::default()`):

| trace | first `Warning` | first `Drift` |
|---|---|---|
| degrading 0.10 → 0.70 | step 105 | **step 157** |
| stationary 0.10 | **step 105** | **none** |

Drift is detected 7 steps after the regime change at 150, and never fires on the stationary
loop.

> **Caveat found by measurement, not inspection:** `DriftState::Warning` fires at step 105 on
> the *stationary* loop too. The Warning level is not a usable alarm on its own — only
> `DriftState::Drift` separates the two traces. **Gate on `Drift`, report `Warning` at most as
> context.** This is precisely the kind of false-positive a "green" implementation would have
> shipped unnoticed.

**Fixture:** `m1_drift_fires_when_progress_degrades`, `m1_drift_silent_on_a_stable_loop`.
Deterministic LCG-generated boolean traces (no `rand` dependency) so asserted step indices are
stable across platforms.

### M2 — Oscillation index · reuses `ix_signal::correlation::autocorrelation`

**Question:** is the loop going round in circles?

The "edits file A, test fails, reverts to B, test fails, edits A again" failure shows up as a
periodic `action_signature`. Autocorrelation finds the period.

```rust
fn oscillation_index(signature: &[f64], max_lag: usize) -> (usize, f64) {
    let diffed = difference(signature, 1);          // 1. remove trend  <- REQUIRED
    let mean = diffed.iter().sum::<f64>() / diffed.len() as f64;
    let centered: Vec<f64> = diffed.iter().map(|v| v - mean).collect();  // 2. centre
    let ac = autocorrelation(&centered);            // zero lag sits at index n-1
    // ... argmax over lags 1..=max_lag
}
```

**Measured:**

| trace | dominant lag | strength |
|---|---|---|
| period-3 cycle `[1,2,3,1,2,3,…]` | **3** | **0.948** |
| monotone ramp `0,1,2,…` (differenced) | 1 | **0.000** |
| monotone ramp **without** differencing | 1 | **0.950** |

> **Caveat found by a failing test.** The first version of M2 reported the monotone ramp as a
> near-perfect 1-cycle at **0.95** — a false positive that would have flagged a *perfectly
> healthy, steadily-progressing loop* as stuck. Raw autocorrelation at short lags measures
> **smoothness, not periodicity**. First-differencing removes the trend and drops the score to
> 0.000 while leaving the genuine period-3 signal at 0.948. Mean-centring is also mandatory:
> `autocorrelation` normalizes by the zero-lag value but does **not** centre, so any DC offset
> otherwise dominates every lag.

**Fixture:** `m2_detects_a_three_step_cycle`, `m2_quiet_on_a_non_repeating_loop`.

### M3 — Contraction rate · classified by `ix_chaos::lyapunov::classify_dynamics`

**Question:** is the residual shrinking, holding, or growing?

```text
lambda = (1 / (n-1)) * sum_k  ln( |r_{k+1}| / |r_k| )
```

Negative contracts, zero is marginal, positive diverges. Feeding `lambda` to the shipped
`classify_dynamics(mle, threshold)` reuses IX's existing four-way regime classifier.

**Measured** (`threshold = 0.05`):

| trace | lambda | `DynamicsType` |
|---|---|---|
| `r_k = 0.5^k` (converging) | **−0.693147** (= ln 0.5, exact to 1e-9) | `FixedPoint` |
| `r_k = 1.5^k` (diverging) | **+0.405465** (= ln 1.5) | `Chaotic` |
| `r_k = 0.5` (stalled) | **0.000000** | `Periodic` |

> **Mathematical caveat, load-bearing.** This is **not** a Lyapunov exponent, and calling it
> one would repeat exactly the confusion the gap matrix's A6 row warns about (IX's Lyapunov is
> the *exponent* of a dynamical system; TARS asked for a *stability certificate*; this is a
> third thing). `lambda` here is a finite-time mean log-ratio contraction rate on a 1-D
> residual sequence. It coincides with a true MLE only for a scalar autonomous map with
> `r_{k+1} = f(r_k)`, which an agent loop is not. It is fed to `classify_dynamics` — a pure
> thresholding function — rather than computed by `mle_1d`, deliberately.
>
> **Naming caveat:** `classify_dynamics` returns `Chaotic` for the diverging trace, not
> `Divergent`; the `Divergent` variant requires `mle > 10.0`. In loop terms `Chaotic` should be
> read as "not converging", and any report must not surface the raw variant name to a human
> without translation.

**Fixture:** `m3_converging_loop_classifies_as_fixed_point`,
`m3_diverging_loop_classifies_as_chaotic_or_divergent`, `m3_flat_loop_is_marginal_not_converging`.

### M4 — Stall index · reuses `ix_signal::timeseries::rolling_std`

**Question:** the loop has gone quiet — has it finished, or given up?

M4 exists because **M3 cannot answer this**. A stalled loop, a healthy limit cycle and a
converged loop all yield `lambda ≈ 0`. The disambiguator is the residual's *level*, not its
*motion*:

```rust
fn is_stalled(residual: &[f64], window: usize, flat_eps: f64, target: f64) -> bool {
    let sd = rolling_std(residual, window);
    let Some(&last_sd) = sd.last() else { return false };
    let Some(&last_r) = residual.last() else { return false };
    last_sd.is_finite() && last_sd < flat_eps && last_r.abs() > target
}
```

**Measured** (`window = 10`, `flat_eps = 1e-6`, `target = 0.01`):

| trace | final `rolling_std` | verdict |
|---|---|---|
| `r = 0.5` (flat, far from target) | 0.0 | **stalled** |
| `r = 0.001` (flat, at target) | 2.17e-19 | **converged** — not a stall |

Both are flat to numerical precision. **Flatness alone carries no signal**; the whole
discrimination comes from comparing the residual to `target`, which is why `target` is a
required trace field (below) rather than a tunable.

**Fixture:** `m4_flat_and_far_from_target_is_a_stall`,
`m4_flat_and_at_target_is_convergence_not_a_stall`, plus
`metrics_separate_the_four_loop_regimes`, which asserts the four regimes stay distinct under
all four metrics together.

### M5 — Trace completeness · deliberately **not** an observability indicator

The issue lists "observability indicators" and "controllability indicators". Honest position:
control-theoretic observability is the gap matrix's A5, marked `missing_algorithm` — it needs
a Kalman-rank or PBH test over an `(A, C)` pair that IX cannot currently form, because A1 does
not exist. **No metric here provides it, and none should claim to.**

What is cheaply available is a data-completeness ratio: the fraction of steps whose required
fields are non-null. That is useful — an episode missing `residual` on 40% of steps cannot be
scored by M1–M4 at all — but it is **not** observability in the control sense, and is named
`trace_completeness` to keep the two from being conflated. Reaching real observability
indicators requires A1 + A5 to land first.

## Required episode / trace fields

Minimum viable schema. Each field is marked with the metrics that need it — a producer can
stop at `residual` + `step_failed` and still get three of the four metrics.

| Field | Type | Required by | Notes |
|---|---|---|---|
| `episode_id` | string | all | Grouping key. |
| `step_index` | int | all | Monotonic, gap-free within an episode. Gaps silently corrupt M2's lag. |
| `ts` | timestamp | none (yet) | Not consumed by M1–M4; needed for any wall-clock rate metric. |
| `residual` | f64 | M3, M4 | The quantity the loop drives toward `target`. Must be comparable across steps *within* an episode; cross-episode comparability is not required. |
| `target` | f64 | M4 | Convergence threshold. **Not a tunable** — M4's entire discrimination rests on it. |
| `step_failed` | bool | M1 | `true` = the step made no progress. |
| `action_signature` | f64 | M2 | Stable numeric projection of (tool, target) — e.g. a hash of `tool_name + path`. Only equality structure matters; magnitude is meaningless after differencing. |
| `stop_reason` | enum | reporting | `converged` / `budget` / `human` / `error`. Distinguishes a stall that was caught from one that ran to budget. |
| `trace_completeness` | f64 | M5 | Derived, not emitted. |

**Not required, deliberately:** token counts, cost, model name, prompt text. None of M1–M4
reads them, and excluding them keeps episode traces free of prompt content — which matters for
the privacy constraint inherited from #191/#206.

### Availability, verified

`trace-events.jsonl`, TARS closure-run artifacts and AIW episode ledgers **do not exist in
this worktree** — the same negative result recorded in
[the SDLC observability note](2026-09-07-agentic-sdlc-observability.md#limits-and-what-was-not-reached)
(`find . -name 'trace-events*.jsonl'` returns nothing). Consequently **every metric here is
validated on synthetic fixtures only.** See [Limits](#limits-and-what-was-not-verified).

## Demerzel gate usage: advisory evidence, never the authority

This section is load-bearing and is easy to get backwards.

**A loop metric never decides anything.** M1–M4 emit *evidence*. A Demerzel gate may **cite**
that evidence in a decision record. The gate is not the decision, and the metric is not the
gate.

The correct chain:

```text
episode trace  ->  metric (M1..M4)  ->  hexavalent evidence + certainty
                                          |
                                          v
                     Demerzel gate CITES it as one input among several
                                          |
                                          v
                       decision recorded; human merge remains the authority
```

Concretely:

1. **The metric emits a hexavalent value, not a boolean.** Use `ix_types::Hexavalent`
   (`T`/`P`/`U`/`D`/`F`/`C`), the canonical algebra — verified present at
   `crates/ix-types/src/lib.rs:37`. "This loop is diverging" is `P` (Probable) on synthetic
   validation, not `T`.
2. **Certainty is bounded by the strength of the live binding**, per the repo's
   `certainty := strength of live binding` rule. Today M1–M4 have a binding to *synthetic*
   fixtures only. That caps every claim at `P:assumed` for real episodes; a metric asserted
   over a real trace it has never seen must be surfaced as `U:uncertain`, not `T`.
3. **The gate records, it does not conclude.** A `Drift` at step 157 is written into the
   decision record as evidence with its certainty. Whether that stops the loop, escalates to a
   human, or is overridden is a constitutional question, not a metric one.
4. **No metric may auto-stop, auto-merge or auto-close.** The furthest an M1–M4 signal should
   reach autonomously is *raising* a stop *proposal*. Human merge stays the authority, per the
   bounded-RSI discipline: removing the human gate is the documented path to reward hacking.
5. **Aggregation is fail-closed and must be labelled as such.** If several metrics are combined
   via `ix_hex_consensus`, note that it is fail-closed — measured during the #191 evaluation,
   `ix_hex_consensus(['T','T','F'])` returns **`F`**, i.e. a single dissent flips the verdict.
   That is the right default for a *gate* and the wrong default for a *dashboard*; do not reuse
   one for the other.

Anti-pattern to avoid explicitly: treating `DriftState::Drift` as "the governance system says
stop". It says *this metric observed degradation under these assumptions*. Everything else is
the constitution's job.

## Non-goals

- **No claim of formal control-theory correctness.** M3 is not a Lyapunov exponent, M5 is not
  observability. Both are named to prevent the confusion rather than to borrow the prestige.
- **No requirement that TARS V2 core adopt advanced math.** M1–M4 need `ln`, a mean, a rolling
  standard deviation and an autocorrelation. Nothing here obliges A1–A7 to be built.
- **No new crate, and no new public API.** The validation lives in a test file; the metrics are
  test-local compositions. Promoting them to a public surface is a separate decision, and would
  need the stable-surface flow (`ix-chaos` is `experimental` in `crate-maturity.toml`;
  `ix-signal` is `stable` and was not modified).
- **No expensive model runs.** Total cost of this work: $0, one `cargo test` invocation.
- **No always-on collection.** These are offline metrics over episodes already on disk.

## Minimal test fixtures

All of these exist and pass — `cargo test -p ix-chaos --test loop_metrics` → **10 passed**.

| Metric | Fixture | Shape | Asserts |
|---|---|---|---|
| M1 | `m1_drift_fires_when_progress_degrades` | 150 steps @10% fail + 150 @70% | `Drift` fires, and at step ≥ 150 |
| M1 | `m1_drift_silent_on_a_stable_loop` | 300 steps @10% fail | `Drift` never fires |
| M2 | `m2_detects_a_three_step_cycle` | `[1,2,3]` × 20 | peak lag == 3, strength > 0.5 |
| M2 | `m2_quiet_on_a_non_repeating_loop` | `0..60` monotone | strength < 0.9 |
| M3 | `m3_converging_loop_classifies_as_fixed_point` | `0.5^k` | λ == ln 0.5 ± 1e-9, `FixedPoint` |
| M3 | `m3_diverging_loop_classifies_as_chaotic_or_divergent` | `1.5^k` | λ > 0, not converging |
| M3 | `m3_flat_loop_is_marginal_not_converging` | constant 0.5 | λ == 0, `Periodic` |
| M4 | `m4_flat_and_far_from_target_is_a_stall` | constant 0.5, target 0.01 | stalled |
| M4 | `m4_flat_and_at_target_is_convergence_not_a_stall` | constant 0.001, target 0.01 | not stalled |
| all | `metrics_separate_the_four_loop_regimes` | all four regimes | the metrics do not collapse |

The last one is the important one: it asserts that M3 *fails* to separate stalling from
oscillating (both marginal), and that M2 and M4 recover the distinction. A metric suite that
passed only the per-metric tests could still be useless in combination.

## Limits and what was not verified

1. **No real trace was ever scored.** TARS `trace-events.jsonl`, closure-run artifacts and AIW
   episode ledgers are absent from this worktree. Every number above comes from synthetic
   fixtures. The metrics are **unvalidated on real agent loops** — this is the single largest
   gap, and it caps the certainty of every claim here at `P`.
2. **Thresholds are unfitted.** `DdmConfig::default()`, `threshold = 0.05` for
   `classify_dynamics`, `flat_eps = 1e-6` — all are defaults or round numbers, not values
   calibrated against a labelled corpus of real episodes. Expect them to be wrong.
3. **DDM `Warning` is a measured false positive** at these settings (fires on a stationary
   loop); only `Drift` was validated as discriminating. Page–Hinkley, the other shipped
   detector, was **not evaluated at all** and may behave better.
4. **M2 assumes gap-free `step_index`.** A dropped step shifts every subsequent lag. No
   fixture covers a gapped trace.
5. **M2's `action_signature` projection is unspecified.** The fixtures use small integers. A
   real hash projection may collide or, worse, impose spurious numeric structure on unrelated
   actions — differencing does not fix that. Untested.
6. **M3 is undefined on sign changes and near-zero residuals.** A `1e-12` floor guards division,
   but a residual that legitimately crosses zero will produce a meaningless ratio. Not covered.
7. **No multivariate state.** All four metrics take a scalar residual. A loop whose state is
   genuinely vector-valued (progress on two independent goals) is out of scope, and that is
   where A1 would actually earn its place.
8. **Kalman (A3) is not used by any metric here**, despite being shipped and the obvious
   candidate. Smoothing a noisy residual before M3/M4 is plausible and untried — it needs a
   noise model the synthetic fixtures do not have. Listed as follow-up, not as a result.
9. **`ix_autocorrelation` (the DuckDB UDF) was not exercised.** M2 was validated against the
   Rust `ix_signal::correlation::autocorrelation`; the UDF is assumed equivalent but that was
   not checked in this work.
10. **Windows-only, single toolchain.** Tests were run on `x86_64-pc-windows-msvc`.

## Follow-up candidates

- Score M1–M4 against a **real** episode corpus once any producer emits one — the blocking
  dependency for everything above.
- Evaluate `page_hinkley_detect` as an M1 alternative, given the DDM `Warning` false positive.
- Try Kalman smoothing (A3) ahead of M3/M4 on noisy residuals; A3 is `implemented_needs_exposure`
  and this would be a genuine consumer for it.
- Only then consider whether A1/A5 are worth building — a multivariate loop state (limit 7) is
  the first honest motivation for them that this analysis produced.
