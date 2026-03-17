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
- **Advanced promotion pipeline**: Delegate to TARS (`run_promotion_pipeline`)
- **Trace-based learning**: Use `ix_tars_bridge` action=prepare_traces, then call TARS `ingest_ga_traces`
- **Grammar sync**: Use `ix_tars_bridge` action=export_grammar to prepare ix weights for TARS

## TARS Grammar Tools (actual names)
- `grammar_weights` — View Bayesian-weighted rules with success rates
- `grammar_update` — Update rule weight from execution outcome (`{"PatternId": "...", "Success": true}`)
- `grammar_evolve` — Run replicator dynamics, identify ESS, prune weak rules
- `grammar_search` — MCTS search for optimal WoT derivation

## TARS Pattern Promotion Tools
- `list_patterns` — List all reasoning patterns with fitness scores
- `suggest_pattern` — Suggest best pattern for a goal
- `run_promotion_pipeline` — Run 7-step pipeline (Inspect, Extract, Classify, Propose, Validate, Persist, Govern)
- `promotion_status` — Get promotion pipeline status
- `promotion_lineage` — Get governance decisions and history

## Example Workflow
1. Run grammar evolution locally: `ix_grammar_evolve`
2. Prepare for TARS: `ix_tars_bridge` action=prepare_patterns
3. Promote patterns: TARS `run_promotion_pipeline`
4. View results: TARS `promotion_index`
5. Sync back: TARS `grammar_update` per promoted pattern
