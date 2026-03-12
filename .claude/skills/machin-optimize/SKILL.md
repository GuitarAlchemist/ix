---
name: machin-optimize
description: Optimize a function using SGD, Adam, PSO, or simulated annealing
---

# Optimize

Run optimization using the machin workspace algorithms.

## When to Use
When the user asks to optimize, minimize, maximize, find best parameters, or tune hyperparameters.

## Method Selection
- **`sgd`** or **`adam`** — Smooth differentiable functions, gradient-based
- **`pso`** (Particle Swarm) — Multi-modal or noisy landscapes, no gradient needed
- **`annealing`** (Simulated Annealing) — Discrete or combinatorial problems, escapes local minima

## Execution
```bash
cargo run -p machin-skill -- optimize --algo <method> --function <fn> --dim <n> --max-iter <iter>
```

Built-in test functions: `sphere`, `rosenbrock`, `rastrigin`

## Programmatic Usage
```rust
use machin_optimize::gradient::{sgd, adam};
use machin_optimize::pso::pso_minimize;
use machin_optimize::annealing::simulated_annealing;
use machin_optimize::traits::ClosureObjective;
```

## Interpretation
- Report best solution found, final objective value, and iterations to convergence
- For PSO/annealing, note that results are stochastic — suggest running multiple seeds
- Compare methods when user is unsure which to use
