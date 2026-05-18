# ix-optimize Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `ObjectiveFunction::gradient` default implementation calls `ix_math::calculus::numerical_gradient` with `eps = 1e-7`. Overriding with an analytic gradient must match within numerical tolerance or downstream `SGD` / `Adam` convergence claims silently break.
- `ParticleSwarm::run` minimizes the objective. To maximize, negate the objective at the caller; do not assume the optimizer auto-detects direction.
- `SimulatedAnnealing` accept probability is `exp(-ΔE / T)` for uphill moves; `ΔE <= 0` always accepts. `CoolingSchedule::Logarithmic` ignores the `alpha` field (uses `T0 / ln(1 + k)`).
- `Optimizer::step` returns NEW params; it does not mutate the input slice. `Momentum`/`Adam` track internal velocity state across calls — instances are stateful and not interchangeable across optimization runs.
- `OptimizeResult.converged = true` means convergence criterion was met BEFORE `max_iterations`; `false` means the loop hit `max_iterations`. It is NOT "best > some threshold".
- `ClosureObjective::dim()` returns the caller-supplied `dimensions` field verbatim; the closure is never inspected.

## Concurrency contracts
- Optimizers (`SGD`, `Momentum`, `Adam`) are `!Sync` due to interior mutable velocity state; clone per thread.
- `ParticleSwarm::run` is single-threaded by design (seeded RNG sequentially advanced). Parallelizing requires per-particle RNG split and is out of scope for the v1 stable surface.

## Failure contracts
- No optimizer returns `Result` — all infallible by signature. Pathological objectives returning NaN propagate NaN into `OptimizeResult.best_value`. Callers MUST check `.is_finite()`.
- `ParticleSwarm` clamps positions to `bounds` on every step; particles outside bounds at init are silently clamped, not rejected.

## Determinism contracts
- `ParticleSwarm` is fully deterministic given `seed`; identical seeds + bounds + dims produce identical trajectories.
- `SimulatedAnnealing` is deterministic given `seed`. Switching `CoolingSchedule` variants changes the random-walk acceptance pattern even with the same seed.
- Gradient optimizers (`SGD`, `Momentum`, `Adam`) are deterministic given identical initial params + gradients.
- Default seed for `ParticleSwarm` / `SimulatedAnnealing` is `42`; calling `.with_seed(42)` is a no-op but explicit.

## Memory contracts
- `Optimizer::step` allocates one new `Array1<f64>` per call (size = param dim). No buffer reuse.
- `ParticleSwarm::run` allocates `num_particles` particles each holding two `Array1<f64>` (position + velocity); peak memory is `O(num_particles * dim)`.
- `Momentum` and `Adam` lazily allocate velocity on first `step` call (not at construction).
