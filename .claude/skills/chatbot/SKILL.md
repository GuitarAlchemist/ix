---
name: chatbot
description: Ask the GA Chatbot music theory questions — grounded voicing answers via real GA+IX MCP tools, no hallucination
---

# GA Chatbot Skill

Ask music theory questions and get answers grounded in real computation — GA computes chord theory and OPTIC-K embeddings, IX provides structural analysis, the LLM orchestrates.

## When to Use

- Ask about chord voicings, voice leading, or instrument-specific fingerings
- Search for voicings by similarity (OPTIC-K embedding search)
- Analyze voicing relationships (clustering, topology, transitions)
- Validate chatbot answers against real data

## Invocation

```
/chatbot "Drop-2 voicings for Cmaj7 on guitar"
/chatbot "smoothest transition from Dm7 to G7" --instrument guitar
/chatbot "compare Am7 voicings guitar vs ukulele"
```

## Instructions for Claude

When the user invokes `/chatbot`, follow these steps:

### Step 1: Parse the question

Extract the music theory question from the arguments. If an `--instrument` flag is provided, use it; otherwise default to guitar.

### Step 2: Check if ga-chatbot is running

```bash
curl -sf http://localhost:7184/api/chatbot/status 2>/dev/null
```

If not running, start it:

```bash
# Stub mode (fast, no MCP servers needed):
cd /c/Users/spare/source/repos/ix && cargo run -p ga-chatbot -- serve --http 7184 --stub &

# Live mode (real GA+IX MCP tools — requires GA and IX MCP servers):
cd /c/Users/spare/source/repos/ix && cargo run -p ga-chatbot -- serve-live \
  --port 7184 \
  --ga-command dotnet --ga-args run --ga-args --project --ga-args /c/Users/spare/source/repos/ga/GaMcpServer \
  --ix-command cargo --ix-args run --ix-args -p --ix-args ix-agent &
```

Wait a few seconds for startup, then verify with the status endpoint.

### Step 3: Send the question

```bash
curl -s -X POST http://localhost:7184/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"THE_QUESTION"}]}'
```

### Step 4: Present the response

Parse the JSON response. Display:
- The answer text (formatted with markdown)
- Any voicing diagrams in a table
- Which tools were called (if visible in the response)

### Step 5: Offer follow-ups

Suggest related questions the user might ask, like:
- "Want to see the voice leading between those voicings?"
- "Should I cluster these voicings to find families?"
- "Want the same voicings on a different instrument?"

## Available Tools (via MCP bridge)

### GA tools (ga__ prefix)
| Tool | What it does |
|------|-------------|
| `ga__GaGenerateVoicingEmbedding` | Full (uncompacted) OPTIC-K embedding for a voicing diagram; its `dimension` field is the schema total, not the index's compact dimension, so it is not directly a valid `ix_optick_search` query |
| `ga__GaGetEmbeddingSchema` | Schema info: partitions, dimensions, weights |
| `ga__GaParseChord` | Parse chord name → intervals, pitch classes |
| `ga__GaChordIntervals` | Interval analysis for a chord |
| `ga__GaEasierVoicings` | Simpler voicing alternatives |
| `ga__GaSearchTabs` | Search tab/voicing database |
| `ga__GetAvailableInstruments` | List instruments + tunings |

### IX tools (ix__ prefix)
| Tool | What it does |
|------|-------------|
| `ix__ix_optick_search` | OPTIC-K cosine similarity search over voicing index; query length must equal the index header dimension (returned as `index_dimension`, stated in the mismatch error) |
| `ix__ix_kmeans` | Cluster voicings into families |
| `ix__ix_topo` | Persistent homology on voicing point clouds |
| `ix__ix_search` | A* voice leading (minimal finger movement) |
| `ix__ix_graph` | Transition cost graphs between voicings |
| `ix__ix_grammar_search` | Parse chord progressions against grammar |
| `ix__ix_stats` | Statistical profiling of voicing corpus |

## Example Session

```
User: /chatbot "Drop-2 voicings for Cmaj7 on guitar"

→ Chatbot calls ga__GaParseChord("Cmaj7") → {root: C, quality: maj7, intervals: [0,4,7,11]}
→ Chatbot calls ga__GaEasierVoicings("Cmaj7", instrument="guitar") → real voicings
→ Response: "Here are the Drop-2 Cmaj7 voicings on guitar:
   x-3-2-0-0-0 (root position, open)
   x-3-5-4-5-3 (barré, 3rd fret)
   8-x-9-9-8-x (8th position)
   All computed by GA's engine — not hallucinated."
```

## Three-Brain Architecture

1. **GA** (C# MCP) — music theory computation: parses chords, computes OPTIC-K embeddings, generates voicings
2. **IX** (Rust MCP) — structural analysis: clusters, topology, voice leading, grammar
3. **LLM** — conversation: translates natural language to tool calls, formats results, NEVER invents voicings
