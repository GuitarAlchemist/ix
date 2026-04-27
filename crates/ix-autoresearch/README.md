# ix-autoresearch

Karpathy-style edit-eval-iterate kernel for IX subsystems.

## What it does

Runs an autonomous experiment loop:

1. **Perturb** a configuration (target-defined; e.g. partition weights, grammar weights).
2. **Evaluate** it (with a soft deadline; optional hard timeout for shell-out targets).
3. **Decide** accept / reject via a strategy (`Greedy`, `SimulatedAnnealing`, `RandomSearch`).
4. **Append** an `Iteration` event to a JSONL log.
5. **Update** best-so-far on `(reward: f64, iteration: usize)`.
6. **Repeat.**

Three target adapters ship in priority order: grammar (smoke test, sub-second), chatbot QA threshold tuning (~30 s/iter), OPTIC-K self-tuning (~140 s/iter, cross-language). Phase 1 ships the kernel only; targets land in Phases 2-7 of the plan.

## Quickstart

```rust
use ix_autoresearch::{run_experiment, Strategy, TimeBudget, Experiment, AutoresearchError};
use std::time::Duration;

struct MyTarget;
impl Experiment for MyTarget {
    type Config = f64;
    type Score = f64;
    fn baseline(&self) -> f64 { 1.0 }
    fn perturb(&mut self, c: &f64, _: &mut rand_chacha::ChaCha8Rng) -> f64 { c + 0.1 }
    fn evaluate(&mut self, c: &f64, _: std::time::Instant) -> Result<f64, AutoresearchError> {
        Ok(-c * c) // optimum at c = 0
    }
    fn score_to_reward(&self, s: &f64) -> f64 { *s }
}

let mut t = MyTarget;
let outcome = run_experiment(
    &mut t,
    Strategy::Greedy,
    100,
    TimeBudget::soft(Duration::from_secs(5)),
    std::path::Path::new("state/autoresearch/runs"),
    42,
).unwrap();

println!("best config: {:?}, reward: {:?}", outcome.best_config, outcome.best_reward);
```

## Resume

Interrupted runs continue from the JSONL log:

```rust
ix_autoresearch::resume_experiment(
    &mut target,
    &outcome.log_path,
    50, // additional iterations
    TimeBudget::soft(Duration::from_secs(5)),
    seed,
)?;
```

For SA strategies the resume reads the last logged temperature and continues cooling from there.

## Milestone promotion

A successful run can be promoted into a tracked milestones tree with sanitization (slug regex, secret-pattern scan, atomic rename + `.complete` sentinel):

```rust
ix_autoresearch::promote_run(
    std::path::Path::new("state/autoresearch/runs"),
    std::path::Path::new("state/autoresearch/milestones"),
    &outcome.run_id.as_string(),
    "first-overnight-tune",
    /* force = */ false,
)?;
```

## Locked decisions

See `docs/plans/2026-04-26-001-feat-ix-autoresearch-edit-eval-iterate-plan.md` for the full design. Highlights:

- Numeric kernel only in v1; LLM-perturb is a v2 plug-in via the reserved `Strategy::Custom` variant.
- Sequential `for` loop; `Strategy::Parallel(n)` reserved for v2.
- JSONL log uses tagged `LogEvent` enum (`RunStart` / `Iteration` / `RunComplete`) with `schema_version: u32` everywhere.
- Best-so-far tracked on `(reward: f64, iteration)`, not `Score: PartialOrd` (PartialOrd is partial).
- Hard-kill cascade abort threshold = 3 consecutive `HardKilled` errors.
- Cost ledger on `RunComplete`: `total_elapsed_ms`, `cache_hit_count`, `eval_failure_count`, `rejected_count`.
- Promote sanitization scrubs `sk-ant-`, `Bearer `, AWS / GitHub / Google API keys, absolute Windows + Unix paths.

## Governance

The `autoresearch-driver` persona at `governance/demerzel/personas/autoresearch-driver.persona.yaml` ratifies the kernel's affordances, constraints, and behavioral-test contract per Demerzel invariant #34.

## License

MIT, per workspace.
