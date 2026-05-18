# ix-search Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `astar` returns `None` iff no path exists (frontier exhausted). It does NOT return `None` on heuristic mistakes; an inadmissible heuristic produces a SUBOPTIMAL but extant path silently.
- `astar` requires an ADMISSIBLE heuristic for optimality. Inadmissibility is the caller's responsibility — the API cannot detect it.
- `weighted_astar(start, h, 1.0)` is exactly `astar`. `weight > 1.0` returns a path whose cost is at most `weight * optimal_cost`.
- `astar` / `weighted_astar` path: `path[0]` = start, `path[last]` = goal. `actions[i]` is the action taken from `path[i]` to `path[i+1]`. `actions.len() == path.len() - 1`.
- `SearchState::successors` MUST return `(action, successor, step_cost)` with `step_cost >= 0`. Negative costs break the priority-queue invariant and produce undefined results (no panic, just wrong answers).
- `bfs` returns the shortest path BY EDGE COUNT, not by cost. Use `astar` with `h = 0` for shortest path by cost.
- `mcts_search` returns the action with the HIGHEST visit count at the root (not highest mean reward). Ties broken by first-encountered.
- `MctsState::reward()` MUST be in `[0.0, 1.0]` (0=loss, 0.5=draw, 1=win). Out-of-range values bias UCB1 selection.
- `local::hill_climb` returns the BEST seen state, not the last state — local maxima are honored.

## Concurrency contracts
- All search algorithms are single-threaded by design. No internal rayon/threading.
- `SearchState: Clone + Eq + Hash` — these traits' impls MUST be consistent across threads if the caller intends to parallelize over different start states. Hash/Eq inconsistency corrupts the closed set silently.

## Failure contracts
- No `Result` returns. Failure to find a path = `None`. Pathological inputs (e.g. `successors()` panicking, NaN heuristic) propagate the panic.
- A heuristic returning NaN breaks `partial_cmp` in `SearchNode::Ord` — falls back to `Ordering::Equal`, causing non-deterministic node ordering. Callers MUST ensure heuristic is finite.

## Determinism contracts
- `astar`, `bfs`, `dfs` are deterministic IF `SearchState::successors()` returns successors in deterministic order AND `Hash` is consistent (`HashMap` iteration is NOT used for selection — only `BinaryHeap` priority, which is total on f_cost).
- `mcts_search` requires `seed: u64` and is deterministic given seed + identical `MctsState` implementation.
- F-cost ties in A*'s heap are broken by `BinaryHeap`'s internal order — NOT a stable tie-break across crate versions. Callers needing tie-break stability must embed a tiebreaker in `f_cost`.
- `weighted_astar(.., 1.0)` is byte-identical to `astar` (literally delegates).

## Memory contracts
- A* / BFS allocate `g_scores` and `came_from` HashMaps that grow with the explored frontier. No upper bound — pathological graphs OOM.
- `SearchResult.path` and `.actions` are owned `Vec`s (cloned from the state during reconstruction). Large states = large clones.
- `mcts_search` stores nodes in a flat `Vec<MctsNode>`; children stored as indices (no `Rc`/`Box` per node). Memory = O(iterations) in nodes generated.
