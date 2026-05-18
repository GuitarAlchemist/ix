# ix-rl Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `EpsilonGreedy::select_arm` returns a UNIFORM random arm with probability `epsilon`, else argmax of `q_values`. Ties broken by FIRST occurrence (lowest index). `epsilon` is NOT clamped to `[0, 1]` — values outside silently bias selection.
- `EpsilonGreedy::update` uses incremental mean: `q_values[arm] += (reward - q_values[arm]) / n`. Equivalent to running sample mean — no learning-rate parameter. Calling `update` BEFORE first `select_arm` is allowed and bumps the count.
- `UCB1::select_arm` plays each arm AT LEAST ONCE first (sequential warm-up). After warm-up, picks `argmax(q + sqrt(2 * ln(total) / count))`. The exploration term diverges if any arm has `count == 0` after warm-up — but warm-up guarantees this can't happen.
- `UCB1::select_arm` is `&self` (NOT `&mut self`) — does not advance internal state. Caller must call `update` to advance counts. EpsilonGreedy and Thompson `select_arm` are `&mut self` (consume RNG).
- `ThompsonSampling::select_arm` PANICS via `Normal::new(...).unwrap()` if `variance` produces a non-positive stddev — guarded by `.sqrt().max(0.01)` (minimum stddev floor of 0.01). Do NOT set `variances[i]` to NaN.
- `ThompsonSampling::update` resets `variances[arm] = 1.0 / n` for `n > 1` — this is a simplified update, NOT the conjugate Normal-Inverse-Gamma posterior. Documented as "Simplified variance update" in source; downstream consumers needing proper Bayesian updates must wrap externally.
- `QLearning::select_action_index` argmax tie-break = lowest action index (`max_by` with `partial_cmp` is stable in Rust 1.62+).
- `QLearning::update` is off-policy: bootstraps from `max(Q[next_state, :])`, NOT from `Q[next_state, next_action]`. SARSA (on-policy) is a separate function — do not confuse.
- `bandit::*::update(arm, reward)` PANICS on out-of-range `arm` index (Vec indexing). No validation.

## Concurrency contracts
- All bandit types contain `StdRng` and are `!Sync` — `select_arm` requires `&mut self`. Clone per thread.
- `UCB1` could be `Sync` (no RNG) but is conservatively `!Sync` for trait consistency. `select_arm` is `&self`, so `Arc<UCB1>` reads work; `update` still needs `&mut`.
- `QLearning::q_table` is a public `Array2<f64>` — direct mutation possible but bypasses learning-rate logic.

## Failure contracts
- No `Result` returns. Out-of-range arm indices panic. NaN rewards corrupt `q_values` silently (no NaN guard).
- `Normal::new` unwrap inside `ThompsonSampling::select_arm` is the only documented panic site; protected by the 0.01 stddev floor.

## Determinism contracts
- All bandits take an explicit `seed: u64` at construction. Identical seed + identical (arm, reward) sequence = identical decisions.
- `UCB1` has NO seed — it's deterministic given its q/count state (no RNG). Two `UCB1` instances with identical histories choose identically.
- `QLearning::select_action_index` is deterministic given its `q_table` for the greedy path; epsilon-greedy exploration uses the seeded RNG.
- `ThompsonSampling::select_arm` samples from `Normal(mean, sqrt(var).max(0.01))` — deterministic given seed and history.

## Memory contracts
- All bandit types allocate three `Vec`s of length `n_arms` at construction. No further allocation after that point.
- `QLearning::q_table` is a single `(num_states, num_actions)` `Array2<f64>` — memory is fixed at construction. No dynamic state-space expansion.
- `select_arm` on Thompson allocates a fresh `Vec<f64>` of samples per call — for hot loops on large arm counts, consider externally-pooled buffers.
- `traits::Environment` (in `env.rs`) is trait-only — does not impose memory layout on implementations.
