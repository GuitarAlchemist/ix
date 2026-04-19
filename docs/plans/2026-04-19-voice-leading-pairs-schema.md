---
date: 2026-04-19
reversibility: one-way-door
revisit-trigger: voice-leading queries reach >5% of chatbot volume, or corpus storage approaches 1.5M pairs ceiling
status: design — NOT YET IMPLEMENTED
---

# Voice-leading pairs — schema design for OPTK v5

## Problem

The OPTIC-K corpus answers "what is Cmaj7?" but cannot answer "what chord moves
smoothly to Cmaj7?" — the single-voicing retrieval surface is blind to
sequential relationships. The music/product octopus panel on 2026-04-18 rated
voice-leading as the #1 missing feature for working guitarists:

> A jazz comper types "ii-V-I in Bb with smooth top voice" or "what moves well
> from Dm9 to G13b9"; a singer-songwriter types "chord that bridges C to Am
> without the obvious Em". The current corpus answers "what is Dm9" — it
> cannot answer "what comes next". Every other gap (alternate tunings,
> extended jazz) is a vocabulary request; voice-leading is the actual verb
> guitarists use when composing or reharmonizing. Missing this makes the
> chatbot a dictionary, not a collaborator.

## Proposed artifact: OPTK **v5** pair index

A separate mmap file at `state/voicings/optick-pairs.index` containing
voicing→voicing ordered pairs with per-pair geometric features. The single-
voicing `optick.index` (v4-pp) remains as the chord-retrieval surface;
the pair index is a second artifact co-located under the same DI-bound
"voicing search" umbrella.

### Pair identity

A pair is `(voicing_a_id, voicing_b_id)` ordered — voice-leading is directional.
Total enumeration space: N² ≈ 313k² ≈ 98 billion. Unusable at full density.

Pair selection rules (reduce to ~1M):

- **Common-tone threshold:** both voicings share ≥1 pitch class (otherwise
  voice-leading is strained). Drops ~60% of combinations.
- **Max voice displacement:** sum of absolute-semitone motion across voices
  ≤ 8 semitones. Drops "jumpy" pairs that aren't musically smooth.
- **Same-instrument:** pairs are per-instrument (guitar↔guitar only). A
  guitarist transitioning Cmaj7→Am7 doesn't care what bass voicings exist.
- **Structural-equivalence bucket:** pairs group by (chord_A_quality,
  chord_B_quality) so retrieval can find "any good Dm9→G13b9 transition" cheaply.

Estimate: 313k voicings × ~3 plausible neighbors each ≈ 1M pairs. At ~200
bytes per pair record (see schema below), ~200 MB mmap. Within the 1.5 GB
ceiling the architect panel set.

## v5 schema (compact)

Extends OPTK v4-pp layout with a new partition family for pair features.
**New schema hash, new file format** — this is the one-way door.

### Per-pair record (proposed 192 bytes)

```
struct PairRecord {
    u32  voicing_a_idx;          // offset into optick.index
    u32  voicing_b_idx;
    u32  instrument : 2;         // guitar|bass|ukulele
    u32  _pad : 30;

    // Geometric features (64 dims × f32 = 256 bytes — too much).
    // Compress to 32 dims × f32 = 128 bytes via partition-weighted projection
    // of the underlying voicing-pair difference.
    f32  features[32];

    // Per-pair metrics (16 bytes)
    f32  total_displacement;     // Σ|semitone motion|
    f32  max_voice_motion;       // largest single voice move
    f32  contrary_motion_ratio;  // fraction of voices moving against melody
    f32  common_tone_count;      // # of pitch classes held across both
}  // 192 bytes exactly
```

### Pair features (32 dims) — partition breakdown

| Partition | Dims | What it encodes |
|---|---|---|
| HARMONIC_MOTION | 8 | ICV(A) − ICV(B); chroma-diff vector; root-motion bucket |
| VOICE_LEADING_SMOOTHNESS | 8 | voice-pair motion histogram; parallel/contrary ratios |
| TONAL_FUNCTION | 6 | A.HarmonicFunction × B.HarmonicFunction transition class |
| DYNAMIC | 6 | tension A→B, resolution A→B, register shift |
| SYMBOLIC_PAIR | 4 | common tags (both jazz? both drop-2? shell→drop-2?) |

Partition weights (TBD, tune empirically):
- HARMONIC_MOTION: 0.35 — "what harmonic move is this"
- VOICE_LEADING_SMOOTHNESS: 0.35 — "is this smooth"
- TONAL_FUNCTION: 0.15
- DYNAMIC: 0.10
- SYMBOLIC_PAIR: 0.05

Per-partition normalization (same v4-pp discipline).

## New MCP surface

Three tools, all in `GaMcpServer.Tools.VoicingPairSearchTool`:

```
ga_search_voicing_pairs
  query: { fromChord, toChord, style?, maxDisplacement? }
  → top-K ordered pairs with diagrams, transition scores, voice-leading metrics

ga_find_bridge_chord
  query: { fromChord, toChord, excludeObvious? }
  → top-K candidate intermediate chords whose (from→bridge) and
     (bridge→to) both score well on voice-leading smoothness

ga_reharmonize
  query: { progression: [chord], style? }
  → re-rank each chord's voicings to minimize total-progression voice-leading
```

Resources:

- `ga://voicings/pairs-vocabulary` — harmonic-motion category names
  (circle-of-fifths, chromatic-mediant, tritone-sub, etc.).

## Generator changes

- New file: `ga/Demos/Music Theory/FretboardVoicingsCLI/OptickPairIndexWriter.cs`
  mirrors `OptickIndexWriter` shape (same mmap discipline, new schema seed
  `optk-v5-pairs:...`).
- New CLI flag: `--export-pairs` runs the pair enumeration + feature
  extraction + writes the .pairs.index file.
- Generator orchestration: typically called AFTER `--export-embeddings` so
  the voicing index is fresh; pairs index references voicing indices.

## Encoder changes

- New `MusicalPairQueryEncoder` composes a 32-dim pair query from
  `StructuredPairQuery { fromChord, toChord, style?, directionality? }`.
- Same deterministic-first, LLM-fallback structure as the single-voicing
  extractor.

## Reader changes (ix-optick or new ix-optick-pairs crate)

- If staying in `ix-optick`: bump crate major version, new PairIndex type.
- Cleaner: new crate `ix-optick-pairs` with its own mmap reader + search
  binary. Shares the OPTK header conventions but different schema hash.

## Size / cost estimates

- 1M pairs × 192 bytes ≈ 192 MB mmap. Reasonable.
- Generation: ~3× the voicing-generation cost (130s × 3 ≈ 7 min).
- Query latency: 1M × 32-dim dot product ≈ 10 ms on SIMD, sub-50ms with
  partition filtering pre-scan.
- Storage cost vs. value: one "what comes next" query type is unanswerable
  without this; cost is acceptable if voice-leading UX materializes.

## Rollout phases

**Phase 1 (foundation — 1 session):**
- Decide pair-selection rules (common-tone threshold, max displacement).
- Prototype feature extraction on 1k voicings × 10 neighbors = 10k pairs.
- Hand-tune partition weights against a curated set of "good" voice-leading
  examples from jazz literature.

**Phase 2 (generator + reader — 2 sessions):**
- Build OptickPairIndexWriter.
- Build ix-optick-pairs reader + search binary.
- Generate 1M-pair index, verify retrieval quality.

**Phase 3 (MCP surface + agent — 1 session):**
- Three MCP tools above.
- Extend `VoicingAgent` or introduce `VoiceLeadingAgent` as a peer specialist.
- SKILL.md for pair-search UX.

**Phase 4 (polish — opportunistic):**
- `ga_reharmonize` progression optimizer (needs multi-chord coordination;
  could leverage `ix-optimize` or `ix-search` MCTS).
- Tension / release modeling via `ix-chaos` Lyapunov if overkill becomes
  appealing.

## Risks / one-way doors

1. **Schema hash (v5 pairs):** separate hash from v4-pp. Bumping partition
   layout on pairs is a second one-way door. Design the pair partitions
   carefully before first write.
2. **Pair-selection rules:** common-tone + max-displacement are proxies.
   If they drop musically valid pairs, those are permanently missing until
   regeneration with new rules.
3. **Directionality:** (A, B) ≠ (B, A). Storage doubles if we precompute
   both directions. Decision: store ONE direction; flip on query time.

## When to revisit

- Chatbot telemetry shows >5% of queries are transition-shaped.
- Storage budget ceiling reached (≥1 GB total mmap).
- Better pair-selection heuristic proposed (e.g., learned pair-relevance
  classifier via ix-ml or ix-supervised).

## Not in scope for this doc

- Cross-instrument pairs (guitar voicing → bass voicing).
- N-voicing sequences (just pairs, not triplets+).
- Learned pair features (hand-crafted first; ML later).

## Related

- `ix/docs/plans/2026-04-18-optic-k-v4-pp-per-partition-norm.md` — the
  predecessor normalization-fix plan.
- `ga/.claude/skills/optic-k-rebuild/SKILL.md` — runbook pattern this
  pair-index rebuild would follow.
