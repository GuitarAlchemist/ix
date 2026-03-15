---
name: federation-discover
description: Find capabilities across the GuitarAlchemist ecosystem (ix, tars, ga) by domain or keyword
---

# Federation Discovery

Query the capability registry to find tools across all repos in the ecosystem.

## When to Use
When you need a capability that might exist in another repo, or when the user asks "can we do X?" and you're unsure which repo provides it.

## Available Domains
- **ix**: math, ml, optimization, search, game-theory, signal, chaos, topology, governance
- **tars**: grammar, pattern-promotion, neuro-symbolic, metacognition, workflow
- **ga**: music-theory, chord-analysis, fretboard, spectral, trace-export

## MCP Tool
Tool name: `ix_federation_discover`
Input: `{ "domain": "music-theory" }` or `{ "query": "optimize" }`
Output: Matching tools with their server, description, and invocation pattern

## Cross-Repo Invocation
Once a tool is discovered, invoke it directly — all servers are registered in `.mcp.json`:
- ix tools: `ix_*` prefix
- tars tools: `tars_*` prefix
- ga tools: `ga_*` prefix

## Roadblock Resolution
When stuck on "unknown unknowns", use the exploratory-fan-out strategy:
1. Query all repos simultaneously for relevant capabilities
2. Mark uncertain claims as Unknown (tetravalent logic)
3. Synthesize evidence from all sources
