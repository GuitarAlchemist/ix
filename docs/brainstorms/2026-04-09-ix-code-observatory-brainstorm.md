---
date: 2026-04-09
topic: ix-code-observatory
mode: multi-ai-team
providers: [codex-gpt5.4, gemini-2.5, claude-opus-4.6]
---

# ix-code: From Lightweight Analyzer to Code Observatory

## What We're Building

Evolve ix-code from a 20-feature keyword-based code analyzer into a 5-layer **Code Observatory** — ML-powered, temporally aware, topologically informed, with governance-grade uncertainty quantification. Ship all 5 layers together.

## Multi-Perspective Analysis

### Provider Contributions
| Provider | Key Contribution | Unique Insight |
|----------|-----------------|----------------|
| Codex (GPT-5.4) | Two-lane architecture, scorecard indirection, 3 feature families, narrow MCP tools | Provenance on every MCP output (version feature schema + model + analyzer). Review Routing Oracle. |
| Gemini 2.5 | Adaptive Resolution / Code-LOD, constraint engine for LLMs, fractal metrics | Analyze YAML/Protobuf/JSON — config complexity is underserved. Holographic heatmaps for VLM. Token entropy as zero-shot bug detection. |
| Claude Opus 4.6 | "Lightweight is the moat", metric metabolism, topological code structure | Call-graph persistent homology via ix-topo — Betti numbers capture structural complexity nobody else measures. Hexavalent governance verdicts. |

### Cross-Provider Convergence
- Don't replace the lightweight engine — keep it as universal fast layer
- tree-sitter > rust-code-analysis if adding AST
- Function-level is the right primary unit
- Join code metrics with git/change data for ML
- Gate on risk deltas, not absolutes
- Governance needs indirection (Metric -> Signal -> Scorecard -> Policy -> Decision)

## Architecture: 5 Layers

```
Layer 1: ix-code (existing)     -> fast 20-feature vectors, 11 languages
Layer 2: ix-code-semantic (new) -> tree-sitter for Rust/Python/TS call-graph + AST features
Layer 3: ix-code-trajectory     -> git-history x metrics -> EWMA, velocity, acceleration
Layer 4: ix-code-topology       -> call-graph -> ix-topo persistent homology -> Betti numbers
Layer 5: Governance gates       -> Risk delta scorecards -> hexavalent verdicts (T/P/U/D/F/C)
```

## Key Decisions

- **11 languages supported**: Python, JS, TS, Java, C/C++, C#, Go, Rust, PHP, Ruby, F#
- **tree-sitter for selective depth**: Rust, Python, TypeScript first (call-graph extraction)
- **Metric Metabolism**: temporal derivatives of code metrics over git history via ix-signal
- **Topological Code**: call-graph -> persistent homology -> Betti numbers -> Prime Radiant 3D viz
- **Risk Delta Gates**: gate PRs on complexity increase with hexavalent governance verdicts
- **Function-level canonical unit**: roll up to file/module with distribution-preserving aggregation (p90, max, Gini coefficient — not just means)
- **Narrow MCP tools**: code_metrics_file, code_metrics_symbol, code_risk_score, code_hotspots, code_quality_gate, code_trajectory, code_topology
- **Every scoring tool returns**: score + confidence + top contributing features + provenance

## MCP Tools (Target)

| Tool | Layer | Input | Output |
|------|-------|-------|--------|
| `ix_code_analyze` | 1 | source + language | 20-feature vector, per-function metrics |
| `ix_code_semantic` | 2 | file path | AST-derived features, call-graph edges |
| `ix_code_trajectory` | 3 | file path + git range | metric velocity, acceleration, EWMA |
| `ix_code_topology` | 4 | directory/module | Betti numbers, persistent homology |
| `ix_code_risk_score` | 5 | diff/PR/file | risk delta, hexavalent verdict, confidence |
| `ix_code_hotspots` | 3+4 | repo path glob | ranked risky areas by trajectory + topology |
| `ix_code_quality_gate` | 5 | diff + policy ID | pass/warn/fail/escalate + violated rules |

## Ecosystem Integration

- **Octopus Level 2**: `/octo:review` -> `ix_code_analyze` + `ix_code_risk_score`
- **Prime Radiant**: `ix_code_topology` -> 3D visualization (ga#47)
- **Dogfooding**: analyze ix/tars/ga repos themselves
- **Federation**: cross-repo tool calls via MCP protocol

## Open Questions
- tree-sitter grammar quality for F# — may need custom queries
- Label acquisition strategy for ML: git-blame correlation vs manual annotation
- Parquet/Arrow for feature store or keep it JSON-based?
- How to handle incremental analysis (only changed files) in CI?

## Next Steps
-> `/ce:plan` for implementation details across all 5 layers
