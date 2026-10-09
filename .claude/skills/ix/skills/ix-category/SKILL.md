---
name: ix-category
description: Category theory primitives — monad laws verification, free-forgetful adjunction
disable-model-invocation: true
---

# Category Theory

Verify monad laws and explore adjunctions with concrete implementations.

## When to Use
When the user wants to verify monad laws, explore the Free ⊣ Forgetful adjunction, or understand category theory through executable examples.

## Capabilities
- **Monad laws verification** — Left unit, right unit, associativity for Option/Result
- **Free-Forgetful adjunction** — Free functor wraps elements, Forgetful flattens back
- **Round-trip verification** — Forget(Free(S)) == S

## Key Concepts
- **Left unit**: bind(unit(a), f) == f(a)
- **Right unit**: bind(m, unit) == m
- **Associativity**: bind(bind(m, f), g) == bind(m, |x| bind(f(x), g))
- Free functor: Set → Mon (wraps each element in singleton list)
- Forgetful functor: Mon → Set (flattens/concatenates)

## Programmatic Usage
```rust
use ix_category::monad::{Monad, OptionMonad, ResultMonad, FreeForgetfulAdj};
```

## MCP Tool
Tool name: `ix_category`
Operations: `monad_laws`, `free_forgetful`
