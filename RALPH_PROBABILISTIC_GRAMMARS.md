# Ralph Prompt: Probabilistic Grammars for ix

## Goal

Add probabilistic grammar infrastructure to ix so TARS can delegate
grammar-weighted MCTS search, replicator dynamics simulation, and Bayesian
weight updates to fast Rust implementations via the `ix-skill` CLI and
`ix-agent` MCP server.

## Context

TARS (F# repo) has implemented a probabilistic grammar layer:
- **WeightedGrammar**: Beta-Binomial Bayesian weight updates, softmax selection
- **ReplicatorDynamics**: Evolutionary game theory for grammar rule competition
- **MCTS for WoT derivations**: UCB1 + random rollout tree search over workflow graphs
- **Constrained decoding**: EBNF grammar -> vLLM guided_decoding via xgrammar/outlines

ix already has: `ix-search` (MCTS with UCB1), `ix-game`
(Nash equilibria, evolutionary dynamics), `ix-rl` (bandits with Thompson
sampling). The goal is to wire these into a grammar-aware layer.

## Architecture

### Crate: `ix-grammar` (IMPLEMENTED)

Located at `crates/ix-grammar/` with:

```
src/
  lib.rs           -- pub mod weighted, replicator, constrained;
  weighted.rs      -- WeightedRule, bayesian_update, softmax, select_weighted
  replicator.rs    -- GrammarSpecies, replicator_step, detect_ess, simulate
  constrained.rs   -- EBNF loading, grammar-guided MCTS state adapter
```

**weighted.rs** — Port from TARS `WeightedGrammar.fs`:
- `WeightedRule { id, alpha, beta, weight, level, source }`
- `bayesian_update(rule, success: bool) -> WeightedRule` (Beta-Binomial)
- `softmax(rules, temperature) -> Vec<(RuleId, f64)>`
- `select_weighted(rules, rng) -> RuleId`

**replicator.rs** — Port from TARS `ReplicatorDynamics.fs`:
- `GrammarSpecies { id, proportion, fitness, is_stable }`
- `replicator_step(species, dt) -> Vec<GrammarSpecies>` (dx_i/dt = x_i * (f_i - f_avg))
- `detect_ess(species, threshold) -> Vec<GrammarSpecies>`
- `simulate(species, steps, dt, prune_threshold) -> SimulationResult`

**constrained.rs** — Grammar-guided MCTS adapter:
- `EbnfGrammar` struct (load from file, parse productions)
- `GrammarMctsState` implementing `ix-search::MctsState` trait
- Actions = grammar production choices, reward = structural validity + weight
- Bridge function: `search_derivation(grammar, config) -> MctsResult`

### MCP Tools (ix-agent)

3 tools registered in `crates/ix-agent/src/tools.rs`:

1. **`ix_grammar_weights`** — Bayesian update + softmax query
   - Input: `{ rules: [...], observation: { rule_id, success }, temperature }`
   - Output: `{ updated_rules: [...], probabilities: [...] }`

2. **`ix_grammar_evolve`** — Replicator dynamics simulation
   - Input: `{ species: [...], steps, dt, prune_threshold }`
   - Output: `{ final_species: [...], trajectory: [...], ess: [...] }`

3. **`ix_grammar_search`** — Grammar-guided MCTS
   - Input: `{ grammar_ebnf: "...", max_iterations, exploration, max_depth }`
   - Output: `{ best_derivation: [...], reward, iterations }`

### CLI (ix-skill)

```
ix grammar weights --rules rules.json --observe rule1:success --temperature 1.0
ix grammar evolve --species species.json --steps 100 --dt 0.1
ix grammar search --grammar grammar.ebnf --iterations 1000
```

## Cross-Repo Integration

TARS `MctsBridge.fs` can call:
```
ix grammar search --grammar cortex.ebnf --iterations 5000
```
instead of the F# fallback, getting 10-100x speedup from Rust.

TARS `MachinBridge.fs` can call:
```
ix grammar evolve --species species.json --steps 1000
```
for fast replicator dynamics simulation.

## Status

IMPLEMENTED — ix-grammar crate builds, tests pass, MCP tools registered.
