# ga-chatbot Agent Spec

**Status:** Draft v2 (rewrite — v1 was castle-on-sand)
**Date:** 2026-04-16
**Owner:** spareilleux

## What went wrong with v1

V1 designed an elaborate QA pipeline for a chatbot that couldn't answer basic questions. The chatbot searched static JSON by string matching. That's fundamentally broken. GA already has a 216-dimension OPTIC-K embedding schema and real music theory computation in C#. The chatbot should use those, not reinvent chord theory in Rust.

## Architecture

```
User: "Drop-2 voicings for Cmaj7 on guitar"
    │
    ▼
LLM (GPT-4o / Claude) with tool use
    │ calls GA tools as needed:
    │
    ├── GaParseChord("Cmaj7")
    │   → {root: C, quality: maj7, intervals: [0,4,7,11]}
    │
    ├── GA OPTIC-K embedding search
    │   query: {quality: maj7, symbolic: drop-2, instrument: guitar}
    │   → 216-dim vector → cosine similarity against voicing index
    │   → top-k real voicings with fret diagrams + scores
    │
    ├── GaChordIntervals("Cmaj7")
    │   → interval analysis, voice leading options
    │
    ├── verify_voicing(frets, instrument)
    │   → physical playability check
    │
    └── GaDiatonicChords(key, mode)
        → for progression context (ii-V-I etc.)
    │
    ▼
Grounded answer with REAL voicings from GA's engine
```

Three layers, clear responsibilities:

1. **LLM** — conversational layer. Translates natural language to tool calls. Formats results. NEVER invents voicings.
2. **GA MCP server** — music theory layer. Parses chords, computes intervals, generates OPTIC-K embeddings, searches voicing index. The source of truth.
3. **ix** — validation layer. Adversarial QA verifies GA's answers against the corpus. Governance gates the pipeline. NOT the answer engine.

## OPTIC-K Integration

GA's existing embedding schema (v1.6, 216 dimensions):

| Partition | Dims | Weight | What it captures |
|-----------|------|--------|-----------------|
| IDENTITY | 0-5 | filter | Object type (voicing/scale/etc) |
| STRUCTURE | 6-29 | 0.45 | Pitch-class set, ICV, consonance |
| MORPHOLOGY | 30-53 | 0.25 | Fretboard geometry, fingering, span |
| CONTEXT | 54-65 | 0.20 | Harmonic function, voice leading tendency |
| SYMBOLIC | 66-77 | 0.10 | Tags: Drop-2, shell, Hendrix, etc. |
| EXTENSIONS | 78-95 | info | Register, spread, density |
| SPECTRAL | 96-108 | info | DFT phase geometry |
| MODAL | 109-148 | 0.10 | Modal flavors |

Reference: `ga/Common/GA.Business.ML/Embeddings/EmbeddingSchema.cs`

When user asks "Drop-2 Cmaj7 on guitar":
1. GA builds a 216-dim query vector: STRUCTURE from Cmaj7 pitch classes, SYMBOLIC with drop-2 flag, MORPHOLOGY filtered to guitar
2. Cosine similarity against pre-indexed voicing embeddings
3. Returns top-k matches with real fret positions, NOT hallucinated ones

## GA MCP Tools (already exist)

From `ga/GaMcpServer/Tools/`:

| Tool file | What it does |
|-----------|-------------|
| `GuitaristProblemTools.cs` | Core chord/voicing queries |
| `InstrumentTool.cs` | Instrument specs (strings, tuning, range) |
| `KeyTools.cs` | Key signatures, diatonic functions |
| `ScaleTool.cs` | Scale degrees, modes |
| `ModeTool.cs` | Modal interchange |
| `ChordAtonalTool.cs` | Set class analysis |
| `AtonalTool.cs` | Pitch class operations |
| `ContextualChordsTool.cs` | Context-aware chord suggestions |
| `ChatTool.cs` | Existing chat wrapper |
| `SceneControlTool.cs` | Prime Radiant scene control |
| `GaDslTool.cs` | GA domain-specific language |
| `GaScriptTool.cs` | Script evaluation |

The chatbot calls these via MCP (stdio JSON-RPC), same as Claude Code does today.

## Implementation

### Phase 1: Wire GA MCP to the chatbot (2 days)

The ga-chatbot Rust HTTP server spawns the GA MCP server as a child process and communicates via stdio JSON-RPC. When the LLM requests a tool call, `execute_tool` sends a JSON-RPC request to the GA child process and returns the result.

```
ga-chatbot HTTP server (Rust, port 7184)
    │
    ├── OpenAI/Claude API (tool use loop)
    │
    └── GA MCP child process (C#, stdio)
        ├── GaParseChord
        ├── GaDiatonicChords
        ├── GaChordIntervals
        ├── GaEasierVoicings
        ├── GetAvailableInstruments
        ├── GetTuning
        └── ... (all 40+ GA tools)
```

What to build:
1. `ga-chatbot serve --http 7184` spawns `dotnet run --project GaMcpServer.csproj` as a child
2. On startup, sends `tools/list` to discover available tools
3. Converts GA tool schemas to OpenAI function-calling format
4. `execute_tool` sends `tools/call` JSON-RPC to the child, returns result
5. LLM tool-use loop runs up to 5 rounds

### Phase 2: OPTIC-K voicing search (3 days)

Add an OPTIC-K search tool to the GA MCP server:
1. Pre-compute 216-dim embeddings for the voicing corpus (guitar/bass/ukulele)
2. Store in Qdrant or MongoDB (GA already uses both)
3. New GA MCP tool: `SearchVoicingsByEmbedding(chord_query, instrument, top_k)`
4. The chatbot calls this for any voicing lookup

This replaces the broken `search_voicings` Rust function with real embedding-based retrieval.

### Phase 3: Adversarial QA with Octopus (2 days)

NOW the QA pipeline makes sense — it validates a chatbot that actually works:
1. Send graduated prompts to the chatbot
2. Chatbot calls GA tools, returns grounded answers
3. Octopus dispatches answers to 3 judge personas
4. Judges verify: did the LLM correctly translate the GA tool results? Did it hallucinate anything beyond what GA returned?
5. Hexavalent aggregation, Shapley attribution

The QA tests the LLM's translation accuracy, not its music theory knowledge (GA handles that).

## What ix provides (validation, not answers)

- `ix-sanitize` — input sanitization before LLM
- `ix-governance` — hexavalent verdict aggregation for QA
- `ix-game::shapley` — prompt attribution for QA corpus pruning
- `ix-voicings` corpus — ground truth for cross-checking GA's output
- `ix-topo` — topology drift detection on voicing relationships
- Adversarial QA pipeline — CI gate on PRs

## What GA provides (the actual music theory)

- Chord parsing, interval computation, voice leading
- OPTIC-K embeddings for semantic voicing search
- Instrument specifications (string count, tuning, range)
- Diatonic analysis, modal interchange
- Fretboard visualization data

## What the LLM provides (conversation, not computation)

- Natural language understanding
- Tool call orchestration
- Result formatting and explanation
- Multi-turn context

## MVP scope

**Phase 1 only.** Wire GA MCP to the chatbot. No OPTIC-K yet (Phase 2), no QA pipeline yet (Phase 3). Just: user asks → LLM calls GA tools → real answer.

Success criteria: "Drop-2 voicings for Cmaj7 on guitar" returns voicings computed by GA's C# engine, not hallucinated by the LLM.

## Kill criteria

- If GA MCP server startup takes >10s, the child-process approach is too slow. Fall back to pre-built GA executable.
- If GA doesn't expose a voicing-search tool, add one to GaMcpServer before proceeding.
- If LLM ignores tool results and hallucinates anyway, add a post-processing step that strips any voicing not returned by a tool call.

## Effort

- Phase 1: 2 days (Rust HTTP ↔ C# MCP plumbing)
- Phase 2: 3 days (OPTIC-K embedding pipeline + Qdrant index)
- Phase 3: 2 days (Octopus QA wiring, already partially built)
- Total: 7 days

## What we already built (reusable)

- `crates/ga-chatbot` — HTTP server, CLI, QA harness, aggregation module (keep all of this)
- `crates/ix-sanitize` — input sanitization (keep)
- `crates/ix-governance` hexavalent extension (keep)
- Adversarial corpus — 77 prompts across 8 categories (keep, expand)
- `.claude/skills/adversarial-qa/SKILL.md` — Octopus skill (keep)
- `.github/workflows/adversarial-qa.yml` — CI workflow (keep)
- React frontend at `/chatbot` (keep, just fix the model chip)

## What we throw away

- `search_voicings` Rust function (static JSON grep — fundamentally wrong)
- `parse_chord_pitch_classes` Rust function (reinventing what GA already does)
- The idea that ix provides answers (it provides validation)
