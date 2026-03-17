---
name: federation-discover
description: Find capabilities across the GuitarAlchemist ecosystem (ix, tars, ga) by domain or keyword
---

# Federation Discovery

Query the capability registry to find tools across all repos in the ecosystem.

## When to Use
When you need a capability that might exist in another repo, or when the user asks "can we do X?" and you're unsure which repo provides it.

## Available Domains
- **ix** (33 crates, 42+ tools): math, ml, optimization, search, game-theory, signal, chaos, topology, governance, code-analysis
- **tars** (16 tools): grammar, pattern-promotion, neuro-symbolic, metacognition, workflow, trace-analysis
- **ga** (50+ tools): music-theory, chord-analysis, atonal, fretboard, dsl, trace-export, chat

## MCP Tools
- `ix_federation_discover` — Query registry by domain or keyword
- `ix_tars_bridge` — Prepare data for TARS (traces, patterns, grammar export)
- `ix_ga_bridge` — Convert GA music data to ML features (chords, progressions, scales)

## Cross-Repo Tool Names

### TARS (actual MCP tool names)
- Grammar: `grammar_weights`, `grammar_update`, `grammar_evolve`, `grammar_search`
- Traces: `ingest_ga_traces`, `ga_trace_stats`, `promotion_index`, `export_insights`
- Patterns: `list_patterns`, `get_pattern`, `suggest_pattern`, `run_promotion_pipeline`
- Workflow: `tars_compile_plan`, `tars_execute_step`, `tars_validate_step`, `tars_complete_plan`
- Memory: `tars_memory_op`

### GA (actual MCP tool names)
- Chords: `GaParseChord`, `GaChordIntervals`, `GaTransposeChord`, `GaDiatonicChords`, `GaCommonTones`, `GaChordSubstitutions`
- Atonal: `GaChordToSet`, `GaSetClassSubs`, `GaPolychord`, `GaIcvNeighbors`
- Analysis: `GaAnalyzeProgression`, `GaKeyFromProgression`, `GaProgressionCompletion`
- Keys: `GetAllKeys`, `GetKeySignatureInfo`, `GetKeyNotes`, `CompareKeys`
- Scales: `GetAvailableScales`, `GaScaleById`, `GaScaleByName`
- Guitar: `GaArpeggioSuggestions`, `GaEasierVoicings`
- DSL: `EvalGaScript`, `TranspileGaScript`, `ListGaClosures`
- Chat: `AskChatbot`

## Roadblock Resolution
When stuck on "unknown unknowns", use the exploratory-fan-out strategy:
1. Query all repos simultaneously for relevant capabilities
2. Mark uncertain claims as Unknown (tetravalent logic)
3. Synthesize evidence from all sources
