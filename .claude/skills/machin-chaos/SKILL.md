---
name: machin-chaos
description: Chaos theory analysis — Lyapunov exponents, bifurcation, attractors, fractals
---

# Chaos Analysis

Analyze dynamical systems and time series for chaotic behavior.

## When to Use
When the user has time series data that might be chaotic, wants to study dynamical systems, or needs fractal dimension estimates.

## Capabilities
- **Lyapunov exponents** — Positive = chaos, zero = periodic, negative = stable
- **Bifurcation diagrams** — How system behavior changes with a parameter
- **Strange attractors** — Lorenz, Rössler, Chen system integration
- **Fractal dimensions** — Box-counting, correlation dimension, Hurst exponent
- **Delay embedding** — Reconstruct attractor from scalar time series (Takens' theorem)
- **Poincaré sections** — Reduce continuous dynamics to discrete maps
- **Chaos control** — OGY method, Pyragas time-delay feedback

## Programmatic Usage
```rust
use machin_chaos::lyapunov::{mle_1d, classify_dynamics};
use machin_chaos::bifurcation::bifurcation_diagram;
use machin_chaos::attractors::{lorenz, integrate};
use machin_chaos::fractal::box_counting_dimension_2d;
use machin_chaos::embedding::{delay_embed, optimal_delay};
```

## Quick Check
For the logistic map `x_{n+1} = r * x_n * (1 - x_n)`:
- r < 3.0: stable fixed point
- r ≈ 3.57: onset of chaos
- r = 4.0: fully chaotic
