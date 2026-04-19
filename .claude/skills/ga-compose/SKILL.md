---
name: ga-compose
description: Compose a chord progression with smooth voice leading — chains ga_generate_progression → ga_voice_leading_pair → ga_search_voicings. Use when the user asks for a progression in a key / style with playable voicings (e.g. "compose ii-V-I in Bb with smooth voicings").
---

# GA Compose Skill

End-to-end composition pipeline: template → chord symbols → smooth-voiced arrangement. Wraps three MCP tools into one workflow so the user gets a playable progression in one turn instead of managing the chain manually.

## When to use

Trigger this skill when the user asks for a composed progression with voicings — phrases like:

- "compose a ii-V-I in Bb"
- "give me 12-bar blues in E with smooth voice leading"
- "rhythm changes in F, playable on guitar"
- "andalusian in A minor"
- "canon progression in D"

Do **not** invoke for:
- Single-chord voicing lookups → use `mcp__ga__ga_search_voicings` directly.
- Reharmonization of an existing progression → that's a separate workflow (`/ga-reharm` if built later).
- Fuzzy mood/style queries with no structural template → use `mcp__ga__ga_search_voicings` with `allowSampling: true`.

## Arguments

Parse the user's request into:

| Argument | Required | Example | Notes |
|---|---|---|---|
| `root` | yes | `C`, `Bb`, `F#` | Canonical roots per `mcp__ga__ga_voicing_vocabulary` |
| `template` | yes | `ii-V-I`, `12-bar-blues` | See template list below |
| `length` | no | `8` | Override natural template length (truncates or loops) |
| `instrument` | no | `guitar`, `bass`, `ukulele` | Default: guitar |

**Templates** (from `ga_generate_progression`):

| Template | Domain | Length |
|---|---|---|
| `ii-V-I` | jazz | 3 |
| `circle-of-fifths` | jazz | 4 |
| `rhythm-changes-a` | jazz | 8 |
| `I-V-vi-IV` | pop axis | 4 |
| `I-vi-IV-V` | doo-wop | 4 |
| `canon` | Pachelbel | 8 |
| `12-bar-blues` | blues | 12 |
| `minor-vamp` | minor | 3 |
| `andalusian` | flamenco | 4 |

If the user's phrase maps to a style not listed (e.g. "bossa nova"), pick the closest template (`ii-V-I` for most jazz styles, `canon` for folk, `12-bar-blues` for blues) and mention the substitution in your reply.

## Workflow

### Step 1 — Generate the progression

```
mcp__ga__ga_generate_progression(root, template, length?)
```

Returns an array of `{roman, symbol, degree, quality, pitchClasses}`. If the template name is unknown, the tool returns an error with `availableTemplates` — show those to the user and ask them to pick.

### Step 2 — Find smooth-voiced transitions

For each consecutive chord pair `(chords[i], chords[i+1])`, call:

```
mcp__ga__ga_voice_leading_pair(
  fromChord: chords[i].symbol,
  toChord: chords[i+1].symbol,
  limit: 3,
  instrument: <user-specified or "guitar">,
  candidatesPerChord: 15
)
```

Fire these pair calls **in parallel** — they're independent. For an N-chord progression you'll have N-1 pair calls.

### Step 3 — Stitch the voicing chain

Build a single voicing sequence:

1. Start with the `from` voicing of pair[0] (best candidate by voice-leading distance).
2. For each subsequent pair i, the `to` voicing of pair[i] becomes the `from` of the next chord. But pair[i+1].from is an independent retrieval, not constrained to match pair[i].to.

This means the naive chain — `[pair[0].from, pair[0].to, pair[1].to, pair[2].to, ...]` — has a gap: pair[0].to and pair[1].from are not guaranteed to be the same voicing. In practice they often are (same chord → same top candidates) but not always.

**Resolution**: for a strictly continuous chain, walk the pair results and pick the best pair[i].to that equals (same MIDI notes as) pair[i+1].from. If no match exists, accept the gap and flag it in the output.

For most use cases (pedagogical / quick composition), the simpler "concatenate top pairs" produces a readable arrangement even with small gaps. Start with that and only escalate to strict-chain matching if the user asks for it.

### Step 4 — Format the reply

Present the composed arrangement as:

```markdown
**<template> in <root>** — <length> chords on <instrument>

| # | Roman | Chord | Diagram | MIDI notes | Voice-leading → next |
|---|---|---|---|---|---|
| 1 | ii7 | Dm7 | x-5-3-5-3-x | [62, 57, 53, 45] | 7 semitones |
| 2 | V7 | G7 | x-x-x-5-8-7 | [55, 53, 47] | 5 semitones |
| 3 | Imaj7 | Cmaj7 | 8-8-9-9-x-8 | [72, 67, 64, 59, 48] | — (final) |

Total voice motion: 12 semitones across 2 transitions.
```

Include:
- The template + key + length in a header
- A table with chord, diagram, MIDI notes, transition cost
- Total voice motion as a summary metric

### Step 5 — Offer follow-ups

Short menu the user can redirect to:
- "Want me to try a different template?"
- "Reharmonize with tritone substitutions?" (requires `ga_chord_substitutions`)
- "Realize on bass/ukulele instead?"
- "Generate an alternate voicing-leading path through the same chords?"

## Examples

### Example 1 — jazz ii-V-I

```
User: /ga-compose ii-V-I in Bb
```

- Call `ga_generate_progression(root: "Bb", template: "ii-V-I")` → Cm7 / F7 / Bbmaj7
- Fire `ga_voice_leading_pair(Cm7, F7)` + `ga_voice_leading_pair(F7, Bbmaj7)` in parallel
- Format as a 3-chord arrangement with transition costs

### Example 2 — rhythm changes

```
User: /ga-compose rhythm changes in F
```

- Template: `rhythm-changes-a` (first 8 bars of rhythm-changes)
- 8 chords, 7 parallel pair calls
- Format as 8-row table

### Example 3 — mismatch fallback

```
User: /ga-compose bossa nova in D
```

- `bossa nova` not a template. Pick `ii-V-I` and mention: "Using `ii-V-I` — bossa-nova-specific templates aren't in the library yet. Flag this if you want a bossa template added."

## Failure modes

**Tool returns `"error": "unknown template"`** — show the `availableTemplates` list and ask the user to pick.

**Tool returns `"error": "unknown root"`** — likely a typo. Suggest the closest canonical root (e.g. "Did you mean `Bb`?").

**`ga_voice_leading_pair` returns fewer pairs than expected** — candidate retrieval was thin. Re-run with `candidatesPerChord: 30` for that single transition. Flag which transition needed the boost in your reply.

**The strategy / MCP server is cold** — first call can take ~500 ms. Subsequent calls are ~10 ms. Don't retry on first-call latency.

## Why this exists

We shipped three primitives — `ga_generate_progression`, `ga_voice_leading_pair`, `ga_search_voicings` — but composing them by hand is tedious: 1 generate call + N-1 voice-leading calls + stitching + formatting = 10+ tool calls for a simple ii-V-I. This skill reduces it to a single invocation. It's the difference between "bricks" and "a house."
