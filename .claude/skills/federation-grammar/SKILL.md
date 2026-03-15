---
name: federation-grammar
description: Delegate grammar weighting and evolution tasks across ix (local Rust) and TARS (F# engine)
---

# Federation Grammar

Combine ix-grammar (fast Rust) with TARS grammar engine (advanced pattern promotion) for grammar tasks.

## When to Use
When working with weighted grammars, replicator dynamics, grammar-guided search, or pattern promotion.

## Routing Logic
- **Simple weighting/evolution**: Use ix locally (`ix_grammar_weights`, `ix_grammar_evolve`, `ix_grammar_search`)
- **Pattern promotion**: Delegate to TARS (`tars_pattern_promote`)
- **Trace-based learning**: Export traces to TARS (`tars_trace_ingest`)

## Example Workflow
1. Run grammar evolution locally with `ix_grammar_evolve`
2. Export successful patterns as traces
3. Feed traces to TARS for promotion analysis
4. Import promoted patterns back into ix grammar weights
