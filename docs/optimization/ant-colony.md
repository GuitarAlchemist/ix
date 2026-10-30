# Ant Colony Optimization

> Ants find the shortest path to food without a map. Each ant lays pheromone on the way it walked. Short paths are walked more often, so they collect more pheromone, and more ants follow them. The trail on long detours evaporates.

**Prerequisites:** [Probability & Statistics](../foundations/probability-and-statistics.md), [Particle Swarm](particle-swarm.md)

---

## The Problem

A delivery van leaves the depot, visits every shop once, and comes back. In what order should it visit them so the loop is as short as possible? This is the **travelling salesman problem** (TSP). With 20 shops there are more than 10^16 possible routes, so trying them all is out of the question. Greedy "go to the nearest shop next" is fast, but on the two TSPLIB instances measured below it is 24 to 27% longer than the best route on average (over every starting city).

Ant Colony Optimization (ACO) builds many routes in parallel. It learns from the good ones which roads belong in a short loop, and it concentrates the next routes on those roads.

---

## The Intuition

Picture a colony of ants, each building a complete delivery route:

1. **Each ant walks a full loop.** At each shop it picks the next one at random, but not uniformly. Near shops are favoured (the *heuristic*), and so are roads with a strong pheromone trail (the colony's *memory*).
2. **Pheromone evaporates.** After every round, all trails weaken a little. A road nobody uses fades away.
3. **Good loops leave stronger trails.** An ant deposits pheromone on every road of its loop, and the amount is inversely proportional to the loop's length. Short loops mark their roads more strongly.

Round after round, the roads of short loops accumulate pheromone, and the ants concentrate their search around them.

---

## How It Works

### Choosing the Next City

From city `i`, an ant moves to an unvisited city `j` with probability

```
p(i -> j) = tau[i][j]^alpha * eta[i][j]^beta  /  sum over unvisited k of the same
```

**In plain English:** `tau` is the pheromone on the road `i-j`, and `eta = 1 / distance` rewards near cities. `alpha` weighs the colony's memory and `beta` weighs the map. With `alpha = 0` the ants ignore each other and act like randomized greedy searchers.

### Evaporation and Deposit

```
tau <- (1 - rho) * tau              (every road, every round)
tau[a][b] += 1 / L                  (every road a-b of a loop of length L)
```

**In plain English:** Evaporation (`rho`) makes the colony forget old decisions. Deposits reward short loops. The two variants differ in **who** deposits:

- **Ant System** (`AntColony::new()`): every ant deposits. It is simple, but on larger instances the colony keeps rewarding mediocre loops and stalls.
- **MAX-MIN Ant System** (`AntColony::max_min()`): only the best ant of the round deposits. Every pheromone value is then clamped to `[tau_min, tau_max]`. The clamp keeps every road possible (no trail drops to zero) and stops one route from taking over (no trail grows without limit).

### 2-opt Local Search

A loop that crosses itself is never optimal: uncrossing the two crossing roads always shortens it. **2-opt** tries every pair of roads, swaps the pair whenever that shortens the loop, and repeats until no swap helps. With `with_local_search(true)`, every ant's loop is cleaned up this way before it is scored. The ants then explore good regions and 2-opt polishes each result, which is much faster than letting pheromone do all the work.

---

## In Rust

### Planning a Delivery Route

```rust
use ix_optimize::aco::AntColony;
use ndarray::Array2;

let shops: [(f64, f64); 6] = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (0.0, 1.0)];
let km = Array2::from_shape_fn((shops.len(), shops.len()), |(i, j)| {
    let (dx, dy) = (shops[i].0 - shops[j].0, shops[i].1 - shops[j].1);
    (dx * dx + dy * dy).sqrt()
});

let route = AntColony::max_min_2opt().with_seed(42).solve_tsp(&km);

assert_eq!(route.tour[0], 0); // tours start at the depot
assert_eq!(route.length, 6.0); // around the grid's edge: the shortest loop
```

This example is the doctest of `crates/ix-optimize/src/aco.rs`, so `cargo test -p ix-optimize --doc` runs it.

The distance matrix must be square, symmetric, finite and non-negative; `solve_tsp` panics otherwise. Any distance works: kilometres, minutes or euros.

### Presets

| Preset | Variant | Local search | `rho` | Ants | Iterations |
|--------|---------|--------------|-------|------|------------|
| `AntColony::new()` | Ant System | no | 0.5 | one per city | 200 |
| `AntColony::max_min()` | MAX-MIN | no | 0.02 | one per city | 1000 |
| `AntColony::max_min_2opt()` | MAX-MIN | 2-opt | 0.2 | 25 | 200 |

**Start with `max_min_2opt()`.** It is the best of the three in every measurement below. The settings are those Dorigo & Stützle (2004) recommend for each case.

### Understanding the Return Value

`solve_tsp` returns a `TourResult`:

| Field     | Type         | Meaning                                                        |
|-----------|--------------|----------------------------------------------------------------|
| `tour`    | `Vec<usize>` | Every city exactly once, starting at city 0                    |
| `length`  | `f64`        | The closed length, including the road back to city 0           |
| `history` | `Vec<f64>`   | The best length so far after each iteration (never increases) |

`ix_optimize::aco::tour_length(&distances, &tour)` measures any tour the same way.

---

## Measured on TSPLIB

`crates/ix-optimize/examples/aco_tsplib.rs` runs each configuration with seeds 1 to 10 on two TSPLIB instances whose optimum is proven. The gap is `length / optimum - 1`. Times are for a release build on one desktop machine and only indicate orders of magnitude.

```text
cargo run -p ix-optimize --release --example aco_tsplib
```

**kroA100** (100 cities, optimum 21282):

| Configuration | Iterations | Mean gap | Optimum reached | Time per run |
|---------------|-----------:|---------:|----------------:|-------------:|
| Ant System | 1000 | 7.01% | 0/10 | ~1.9 s |
| MAX-MIN | 1000 | 0.43% | 1/10 | ~1.8 s |
| Ant System + 2-opt | 50 | 0.07% | 3/10 | ~0.4 s |
| `max_min_2opt()` | 25 | 0.33% | 3/10 | ~0.08 s |
| `max_min_2opt()` | 50 | 0.00% | 10/10 | ~0.1 s |

**berlin52** (52 cities, optimum 7542):

| Configuration | Iterations | Mean gap | Optimum reached |
|---------------|-----------:|---------:|----------------:|
| Ant System | 200 | 1.68% | 0/10 |
| MAX-MIN | 200 | 9.62% | 0/10 |
| MAX-MIN | 1000 | 0.00% | 10/10 |
| `max_min_2opt()` | 25 | 0.00% | 10/10 |

Two lessons from these runs:
- **MAX-MIN needs a large budget without local search.** At 200 iterations it is worse than Ant System, because with `rho = 0.02` it has not converged yet. That is why `max_min()` defaults to 1000 iterations.
- **With local search, MAX-MIN needs different settings.** With the no-local-search `rho = 0.02` it lags behind Ant System + 2-opt. With `rho = 0.2` and 25 ants it reaches the optimum on every seed.

---

## When To Use This

| Situation | Use ACO? |
|-----------|----------|
| Shortest loop through a set of places (routing, drilling, picking) | Yes -- this is what `solve_tsp` does |
| Up to a few hundred cities | Yes -- `max_min_2opt()` is fast and near-optimal there |
| You need a *proven* optimum | No -- ACO gives no certificate; use an exact solver (e.g. Concorde) |
| Thousands of cities | Cautious -- each iteration costs O(ants * n^2); expect seconds to minutes |
| Asymmetric distances (one-way streets) | No -- `solve_tsp` requires a symmetric matrix |
| Continuous parameters (hyperparameter tuning) | No -- use [Particle Swarm](particle-swarm.md) |

---

## Key Parameters

### Variant and Local Search

- Use `max_min_2opt()` unless you are studying the algorithm itself.
- `with_local_search(true)` also works with `AntColony::new()`. It is the single biggest improvement for Ant System, too.

### Iterations (`with_max_iterations`)

- Each iteration builds one tour per ant. Its cost is O(ants * n^2), plus the cost of 2-opt when local search is on.
- `history` shows when the search stopped improving. If its last entries are all equal, fewer iterations would have done.

### Ants (`with_ants`)

- The default is one ant per city. With local search, 25 ants are enough, because 2-opt does the fine-tuning.

### `alpha`, `beta` (`with_alpha`, `with_beta`, defaults 1 and 2)

- A higher `beta` trusts the map more: the search is greedier and converges faster, but may converge on the wrong loop.
- A higher `alpha` trusts the colony more: the search converges faster, with a higher risk of stagnation.

### Evaporation (`with_evaporation`, `rho`)

- A high `rho` means a short memory: the colony forgets fast and follows the latest good loops.
- A low `rho` means a long memory: the search is broad but slow. MAX-MIN without local search needs a low `rho` *and* many iterations.

### Seed (`with_seed`)

- The same seed with the same matrix gives the same tour. For important routes, run 5 to 10 seeds and keep the shortest.

---

## Pitfalls

**Judging MAX-MIN at a small budget.** With `rho = 0.02`, 200 iterations are not enough: the colony is still exploring. Compare variants at the budget they were designed for, or use local search.

**Keeping no-local-search settings after turning local search on.** The best `rho` changes when 2-opt is added (0.02 becomes 0.2 for MAX-MIN). The `max_min_2opt()` preset carries the right values.

**Asymmetric or invalid matrices.** `solve_tsp` checks the matrix and panics on a non-square matrix, an asymmetric one, or a negative, NaN or infinite entry. Clean the data first.

**Distances rounded differently from a reference.** TSPLIB rounds each Euclidean distance to the nearest integer. Compare your lengths with published optima only if your matrix uses the same rule.

---

## Going Further

- **See it measured:** [`crates/ix-optimize/examples/aco_tsplib.rs`](../../crates/ix-optimize/examples/aco_tsplib.rs) prints the tables above.
- **Swarm on continuous spaces:** [Particle Swarm](particle-swarm.md) uses a similar "shared memory" idea for real-valued parameters.
- **Single-agent alternative:** [Simulated Annealing](simulated-annealing.md) can also search tours, with one agent and a cooling temperature.
- **Population alternative:** [Genetic Algorithms](../evolutionary/genetic-algorithms.md) evolve solutions by crossover and mutation.
- **GPU:** no GPU version exists yet. [ix#362](https://github.com/GuitarAlchemist/ix/issues/362) records why this CPU version comes first: any `wgpu` kernel in `ix-gpu` would be checked against it.
- **References:** M. Dorigo & T. Stützle, *Ant Colony Optimization*, MIT Press, 2004. T. Stützle & H. H. Hoos, "MAX-MIN Ant System", *Future Generation Computer Systems* 16(8), 2000.
