---
name: ix-grammar
description: Formal grammars — weighted CFGs, grammar evolution, constrained generation
disable-model-invocation: true
---

# Grammar

Formal grammar systems for structured text generation and analysis.

## When to Use
When the user asks about context-free grammars, weighted production rules, grammar-guided generation, or evolving grammars with replicator dynamics.

## Capabilities
- **Weighted CFGs** — Context-free grammars with weighted production rules, sampling
- **Grammar Evolution** — Replicator dynamics to evolve grammar rule weights over fitness landscape
- **Constrained Generation** — Generate strings satisfying grammar constraints
- **Grammar Search** — Search for optimal grammars matching target distributions

## Programmatic Usage
```rust
use ix_grammar::weighted::{WeightedGrammar, Production};
use ix_grammar::replicator::GrammarReplicator;
use ix_grammar::constrained::ConstrainedGenerator;
```

## MCP Tool Reference
Tools: `ix_grammar_weights`, `ix_grammar_evolve`, `ix_grammar_search`
