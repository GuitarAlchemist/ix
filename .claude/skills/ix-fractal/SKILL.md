---
name: ix-fractal
description: Fractal generation — Takagi curves, space-filling curves, Morton codes
disable-model-invocation: true
---

# Fractals

Generate fractal curves and space-filling curve data.

## When to Use
When the user needs fractal curve data, space-filling curve coordinates, or Morton Z-order encoding/decoding.

## Capabilities
- **Takagi (Blancmange) curve** — Nowhere-differentiable continuous fractal curve
- **Hilbert curve** — Space-filling curve mapping 1D → 2D, preserves locality
- **Peano curve** — Original space-filling curve with ternary structure
- **Morton encoding** — Z-order curve encode/decode for spatial hashing
- **IFS chaos game** — Sierpinski, Barnsley fern, Koch snowflake
- **L-systems** — Dragon, Sierpinski arrowhead, Koch curve

## Programmatic Usage
```rust
use ix_fractal::takagi::takagi_series;
use ix_fractal::space_filling::{hilbert_curve, peano_curve, morton_encode, morton_decode};
use ix_fractal::ifs::{ifs_iterate, sierpinski_maps};
use ix_fractal::lsystem::{dragon_curve, interpret};
```

## MCP Tool
Tool name: `ix_fractal`
Operations: `takagi`, `hilbert`, `peano`, `morton_encode`, `morton_decode`
