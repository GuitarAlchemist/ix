---
name: ix-ktheory
description: Algebraic K-theory for graphs — Grothendieck K0/K1, Mayer-Vietoris sequences
disable-model-invocation: true
---

# K-Theory

Algebraic K-theory applied to graph structures.

## When to Use
When the user asks about K-theory computations on graphs, Grothendieck groups, K0/K1 invariants, or Mayer-Vietoris exact sequences for graph decompositions.

## Capabilities
- **Graph K0** — Grothendieck group K0 from adjacency matrix, generator/relation computation
- **Graph K1** — K1 computation via determinant maps on graph automorphisms
- **Mayer-Vietoris** — Exact sequence for graph decompositions, connecting homomorphisms
- **Spectral sequences** — Basic spectral sequence page computation

## Programmatic Usage
```rust
use ix_ktheory::graph_k::{graph_k0, graph_k1, GraphKTheory};
use ix_ktheory::mayer_vietoris::{MayerVietoris, ExactSequence};
```

## MCP Tool Reference
Not yet available as a dedicated MCP tool. Use the Rust API directly.
