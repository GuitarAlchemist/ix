---
name: ix-sedenion
description: Hypercomplex algebra — sedenion/octonion multiplication, Cayley-Dickson construction
disable-model-invocation: true
---

# Hypercomplex Algebra

Operations on sedenions (16D), octonions (8D), and arbitrary Cayley-Dickson algebras.

## When to Use
When the user needs hypercomplex number operations, wants to explore non-associative algebras, or needs Cayley-Dickson multiplication tables.

## Capabilities
- **Sedenion multiplication** — 16-dimensional Cayley-Dickson product
- **Octonion multiplication** — 8-dimensional non-associative product
- **Cayley-Dickson construction** — Generic 2^n-dimensional algebra multiplication
- **Conjugate and norm** — For any Cayley-Dickson algebra element
- **BSP tree** — Binary space partition for nearest-neighbor queries

## Key Concepts
- Complex → Quaternion → Octonion → Sedenion (each doubles dimension)
- Octonions lose associativity; sedenions also lose alternativity
- Zero divisors appear at dimension 16+

## Programmatic Usage
```rust
use ix_sedenion::sedenion::Sedenion;
use ix_sedenion::octonion::Octonion;
use ix_sedenion::cayley_dickson::{double_multiply, double_conjugate, double_norm};
```

## MCP Tool
Tool name: `ix_sedenion`
Operations: `multiply`, `conjugate`, `norm`, `cayley_dickson_multiply`
