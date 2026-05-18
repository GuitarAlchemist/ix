# ix-game Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `BimatrixGame::new(a, b)` PANICS via `assert_eq!` on shape mismatch — it does NOT return Result. Callers must pre-check `payoff_a.shape() == payoff_b.shape()`.
- `BimatrixGame::zero_sum(a)` constructs `payoff_b = -a` — the negation, not the transpose. For symmetric zero-sum, callers must construct B explicitly.
- `best_response_a` / `best_response_b` use `1e-10` absolute tolerance for tie detection. Ties are split UNIFORMLY across all near-optimal strategies (mixed best response).
- `is_nash_equilibrium(profile, tolerance)` checks whether NEITHER player can improve by deviation by more than `tolerance` — tolerance is in payoff units, not probability units.
- `first_price_auction` ties broken by FIRST bidder in `bids` slice (lowest input index wins). Same for second_price.
- `second_price_auction` with `bids.len() < 2` falls back to first-price (no separate error path). Empty bids returns `None`.
- `second_price_auction` winner pays the second-HIGHEST bid (sorted descending, index 1). Vickrey-truthful only if all bidders bid their valuation.
- `cooperative::shapley_value` requires the coalition value function to be MONOTONIC for the result to be meaningful, but the API does not validate this.
- `evolutionary::replicator_dynamics` is discrete-time per-step update; small step sizes approximate continuous dynamics. Step size > 1.0 is mathematically defined but can produce non-stochastic strategy vectors (negative probabilities).
- `mechanism::vcg_payment` returns each bidder's externality cost. Sum across bidders MAY exceed total welfare — that's the VCG revenue, not a bug.

## Concurrency contracts
- All game-theory types (`BimatrixGame`, `StrategyProfile`) are `Send + Sync` value types with `&self` query methods.
- `evolutionary` simulators mutate `&mut self`; serialize across threads.
- No internal RNG on the public Nash/auction surface — fully thread-safe for read.

## Failure contracts
- `BimatrixGame::new` panics on shape mismatch. No Result variant.
- `first_price_auction` / `second_price_auction` return `Option<AuctionResult>` — `None` iff `bids.is_empty()`.
- Internal `partial_cmp` on bid amounts uses `unwrap` — NaN bids panic. Caller MUST filter NaN.
- `cooperative` algorithms over `n` players iterate `2^n` coalitions — silently exponential. No guard.

## Determinism contracts
- All Nash / auction / cooperative / mechanism functions are deterministic functions of input — no internal RNG.
- `best_response_*` returns a mixed strategy that is a function of the f64 payoff matrix only — bit-identical across runs.
- `evolutionary::replicator_dynamics` is deterministic given initial state + payoff matrix + step count.
- Tie-breaks are stable: lowest-index wins for auctions; uniform-split for best response.

## Memory contracts
- `BimatrixGame` stores two owned `Array2<f64>` matrices — no shared references.
- `best_response_*` allocates one `Vec<f64>` (size = num_strategies) and one `Array1<f64>` output per call.
- `cooperative` Shapley-value computation allocates a coalition-value cache of size `2^n` — DO NOT call for n > ~20.
- Auction `all_bids` field clones the input bid vector into the result (`bids.to_vec()`) — O(n) extra memory per auction call.
