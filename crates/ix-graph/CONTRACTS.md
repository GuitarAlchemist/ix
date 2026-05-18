# ix-graph Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `MarkovChain::new(transition)` returns `Err(String)` (not a typed error) on: non-square matrix, any row not summing to 1.0 within `1e-6`. Negative entries are NOT rejected by `MarkovChain::new` but are by `HiddenMarkovModel::new`.
- `HiddenMarkovModel::new` validates: square transition matching N, emission N rows, initial sums to 1.0 within `1e-6`, all probabilities non-negative, transition + emission rows sum to 1.0. ALL validation failures return `Err(String)` — error MESSAGES are not stable across versions, but their `is_err()` status is.
- `MarkovChain::state_distribution(initial, steps)` allocates `Array1::clone(initial)` and right-multiplies `steps` times. `steps = 0` returns initial unchanged (no copy elision but mathematically identity).
- `MarkovChain::stationary_distribution` uses power iteration; terminates EARLY when L1 delta < `tol`. May return a non-stationary distribution if `max_iter` is hit first — caller cannot distinguish convergence vs cutoff from the return value alone.
- `MarkovChain::simulate` advances RNG `steps` times; identical seed + identical chain + identical start_state = identical trajectory.
- `hmm::viterbi` returns the MOST-LIKELY hidden state sequence under the model. Tie-breaks (equal-probability paths) follow argmax-first-wins, which depends on state ordering.
- `hmm::baum_welch` (forward-backward EM) MUTATES the HMM in place and returns the final log-likelihood. It does NOT guarantee monotonic improvement across iterations near convergence due to numerical underflow.
- `SkillRouter::add_skill` does NOT validate that `skill.dependencies` reference registered skill IDs. Forward references must be resolved before calling `build_graph`, else edges silently drop.
- `routing::SkillRouter::route` returns shortest dependency-respecting plan by total `token_cost`. Cycles produce undefined behavior (no cycle detection at route time).

## Concurrency contracts
- `MarkovChain` and `HiddenMarkovModel` are `Send + Sync` — pure data structures with `&self` methods only. Cloning is cheap (small matrices).
- `SkillRouter` holds `HashMap`s and is `!Sync` for mutation; safe to share `&SkillRouter` across threads for read-only routing.

## Failure contracts
- `MarkovChain::new` and `HiddenMarkovModel::new` return `Result<Self, String>` — stringly-typed errors. This is the v1 surface; do not rely on parsing the error string.
- Internal numerical methods (`viterbi`, `baum_welch`) do not return Result — they degrade gracefully (underflow → log-space) and may return finite-but-wrong values rather than panic.
- `MarkovChain::simulate` with `steps = 0` returns `vec![start_state]` (length 1).

## Determinism contracts
- Every randomized method takes an explicit `seed: u64`. No `thread_rng()` use on public surface.
- `viterbi` ties broken by lowest state index (stable across runs).
- `SkillRouter::route` iterates skills in `HashMap` order INTERNALLY but exposes results sorted by topological order — output order is stable; intermediate computation is not.
- `Markov::stationary_distribution` starts from uniform `1/n` distribution (not random) — fully deterministic for a given chain.

## Memory contracts
- HMM forward/backward allocate two `(T, N)` matrices where T = observation length, N = state count. Memory = O(T*N). Not streaming; full sequence must be in memory.
- `MarkovChain::simulate` returns `Vec<usize>` of length `steps + 1` (start state included).
- `SkillRouter` stores full skill objects (not references) in `HashMap<String, SkillNode>` and dual `id_to_index`/`index_to_id` maps — memory = 3x skill count.
