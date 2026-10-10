---
name: ix-topo
description: Topological data analysis — persistent homology, Betti numbers, point cloud topology
disable-model-invocation: true
---

# Topological Data Analysis

Compute topological features of point cloud data using persistent homology.

## When to Use
When the user has point cloud data and wants to detect topological features (connected components, loops, voids), compute Betti numbers, or generate persistence diagrams.

## Capabilities
- **Persistent homology** — Birth-death pairs across filtration radii
- **Betti numbers at radius** — Count of H_0 (components), H_1 (loops), H_2 (voids) at a given scale
- **Betti curve** — How Betti numbers evolve as radius increases
- **Vietoris-Rips complex** — Build simplicial complex from distance threshold

## Key Concepts
- β₀ = connected components, β₁ = loops/tunnels, β₂ = voids
- Long-lived features (large death-birth) are topologically significant
- Short-lived features are likely noise

## Programmatic Usage
```rust
use ix_topo::pointcloud::{persistence_from_points, betti_at_radius, betti_curve};
use ix_topo::simplicial::{rips_complex, SimplexStream};
```

## MCP Tool
Tool name: `ix_topo`
Operations: `persistence`, `betti_at_radius`, `betti_curve`
