# Fractal mutation and crossover operators vs Gaussian — benchmark (ix#204)

**Question (#204):** do fractal operators — Takagi-noise mutation, a Takagi
amplitude schedule, de Rham recursive crossover — beat the Gaussian baseline in
`ix-evolution`, and if so, is the gain *fractal* or something cheaper?

**Answer: no fractal benefit survives the controls.** The one arm that wins
broadly, `takagi_schedule`, is matched by plain linear annealing, and its only
edge over linear annealing appears on objectives whose optimum sits at the
centre of the search box — an edge that reverses on the shifted control.

## Setup

```text
cargo run --release -p ix-evolution --example fractal_operator_bench -- --csv out.csv
```

- Commit `19c5f40` (branch `feat/ix-204-fractal-operators`), run 2026-09-13.
- dim 10, population 60, 300 generations, mutation rate 0.2, seeds 0..29.
- Every arm uses the same seeds, bounds and per-gene step scale; Takagi noise
  is standardised to mean 0 / sd 1, so differences are noise *structure*, not
  step length.
- Comparisons are **paired within a seed** (sign test on 30 pairs). Seed-to-seed
  variance exceeds the effects measured, so unpaired medians would mostly
  measure the seeds.
- The run is deterministic: re-running the command reproduces the table.

Objectives form a 2×2 plus a control:

|               | separable   | coupled      |
|---------------|-------------|--------------|
| **unimodal**  | sphere      | rosenbrock   |
| **multimodal**| rastrigin   | ackley       |

`shifted_rastrigin` moves the optimum off-centre, so an operator that merely
drifts toward the centre of the box cannot win there.

## Results — each arm vs Gaussian

Wins/losses are per seed (lower best fitness wins); verdict at p < 0.05.

| objective | arm | W/L | median delta | p | verdict |
|---|---|---|---|---|---|
| sphere | takagi_iid | 9/21 | +0.0015 | 0.043 | worse |
| sphere | takagi_correlated | 4/26 | +0.0023 | 0.0001 | worse |
| sphere | takagi_schedule | 30/0 | −0.0029 | <0.0001 | better |
| sphere | sched_ctrl_constant | 7/23 | +0.0008 | 0.005 | worse |
| sphere | sched_ctrl_linear | 30/0 | −0.0029 | <0.0001 | better |
| sphere | de_rham_crossover | 9/21 | +0.0010 | 0.043 | worse |
| rosenbrock | takagi_iid | 19/11 | −0.66 | 0.20 | — |
| rosenbrock | takagi_correlated | 22/8 | −1.14 | 0.016 | better |
| rosenbrock | takagi_schedule | 25/5 | −0.74 | 0.0003 | better |
| rosenbrock | sched_ctrl_constant | 19/11 | −0.17 | 0.20 | — |
| rosenbrock | sched_ctrl_linear | 23/7 | −1.08 | 0.005 | better |
| rosenbrock | de_rham_crossover | 12/18 | +0.57 | 0.36 | — |
| rastrigin | takagi_iid | 8/22 | +2.50 | 0.016 | worse |
| rastrigin | takagi_correlated | 15/15 | +0.02 | 1.0 | — |
| rastrigin | takagi_schedule | 18/12 | −0.60 | 0.36 | — |
| rastrigin | sched_ctrl_constant | 15/15 | +0.31 | 1.0 | — |
| rastrigin | sched_ctrl_linear | 21/9 | −2.76 | 0.043 | better |
| rastrigin | de_rham_crossover | 2/28 | +15.49 | <0.0001 | worse |
| ackley | takagi_iid | 8/22 | +0.044 | 0.016 | worse |
| ackley | takagi_correlated | 8/22 | +0.019 | 0.016 | worse |
| ackley | takagi_schedule | 30/0 | −0.093 | <0.0001 | better |
| ackley | sched_ctrl_constant | 13/17 | +0.006 | 0.58 | — |
| ackley | sched_ctrl_linear | 30/0 | −0.084 | <0.0001 | better |
| ackley | de_rham_crossover | 16/14 | −0.002 | 0.86 | — |
| shifted_rastrigin | takagi_iid | 12/18 | +1.48 | 0.36 | — |
| shifted_rastrigin | takagi_correlated | 16/14 | −1.27 | 0.86 | — |
| shifted_rastrigin | takagi_schedule | 16/14 | −0.52 | 0.86 | — |
| shifted_rastrigin | sched_ctrl_constant | 16/14 | −0.41 | 0.86 | — |
| shifted_rastrigin | sched_ctrl_linear | 22/8 | −4.29 | 0.016 | better |
| shifted_rastrigin | de_rham_crossover | 7/23 | +9.69 | 0.005 | worse |

## Head-to-head: is the Takagi schedule fractal, or just annealing?

`T(0) = T(1) = 0`, so a Takagi-modulated amplitude shrinks toward the end of a
run — that is annealing. Both controls are matched to the schedule's *measured*
mean multiplier.

| objective | takagi_schedule vs | W/L | median delta | p | verdict |
|---|---|---|---|---|---|
| sphere | constant | 30/0 | −0.0035 | <0.0001 | better |
| sphere | linear | 28/2 | −0.0001 | <0.0001 | better |
| rosenbrock | constant | 24/6 | −0.68 | 0.001 | better |
| rosenbrock | linear | 14/16 | +0.12 | 0.86 | — |
| rastrigin | constant | 19/11 | −2.40 | 0.20 | — |
| rastrigin | linear | 11/19 | +1.48 | 0.20 | — |
| ackley | constant | 30/0 | −0.100 | <0.0001 | better |
| ackley | linear | 28/2 | −0.007 | <0.0001 | better |
| shifted_rastrigin | constant | 13/17 | +1.83 | 0.58 | — |
| shifted_rastrigin | linear | 9/21 | +4.96 | 0.043 | **worse** |

## Reading

Each arm isolates one property, so each result answers one question:

1. **Distribution shape (`takagi_iid`): hurts.** Same step scale, independent
   genes, Takagi-shaped marginal — worse than Gaussian on sphere, rastrigin and
   ackley, never better.
2. **Cross-gene correlation (`takagi_correlated`): no reliable benefit.** It
   wins on rosenbrock (22/8, p = 0.016), the coupled objective where the
   hypothesis predicts it should — but loses on ackley, the other coupled
   objective, and on sphere. One win at p = 0.016 among 40 comparisons is
   within what chance produces (~2 false positives expected at α = 0.05 with no
   correction), so it is not evidence on its own.
3. **Amplitude schedule (`takagi_schedule`): annealing, not fractal.** It beats
   Gaussian on sphere, rosenbrock and ackley, but `sched_ctrl_linear` — a plain
   linear decay matched to the same mean amplitude — beats Gaussian on those
   *and* on both rastrigins. Head-to-head, the Takagi schedule's remaining edge
   over linear is confined to sphere and ackley, both centred at the origin;
   on rastrigin it ties, and on the off-centre control it **loses** (9/21).
   That pattern points to a centre-seeking side effect, not a multi-scale
   benefit.
4. **Recursive crossover (`de_rham_crossover`): harmful on multimodal
   landscapes** (2/28 and 7/23 on the rastrigins), neutral elsewhere.

**Recommendation.** Do not make any fractal operator a default. If the
operators stay in the crate, document them as evaluated and not beneficial,
and point users who want the schedule's gain at linear annealing, which gets
all of it and more. The benchmark and its controls are the reusable part: it is
the harness that caught the annealing confound, and it is how any future
operator claim in `ix-evolution` should be tested.

**Limits.** One dimension (10), one budget (300 generations), one population
(60), no correction for multiple comparisons. A different budget could move
the annealing comparison; the shape and crossover results are large enough that
it is unlikely to flip them.
