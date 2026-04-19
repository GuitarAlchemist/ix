---
date: 2026-04-19
reversibility: two-way-door
revisit-trigger: new invariants added to catalog, or baseline-diagnostics leak-test numbers regress without explanation
status: design — NOT YET IMPLEMENTED
---

# OPTK invariant checker — reads the real corpus, not synthetic exemplars

## Gap

`ix-invariant-produce` runs the 10 Phase-1 checkers against a synthetic
pitch-class-set exemplar corpus hard-coded in
`crates/ix-invariant-coverage/src/producer.rs`. Post-2026-04-18 v4-pp
rebuild, the leak-test baseline showed CONTEXT leak eliminated and
STRUCTURE leak substantially reduced — but the invariant-coverage verdict
didn't change because the produce step doesn't load the actual
`optick.index`.

That gap means we have INDIRECT evidence (leak accuracy dropping) for
invariants #25 (cross-instrument STRUCTURE equality) and #32 (cross-octave
STRUCTURE cosine = 1.0) but no direct verification. A dedicated checker
that reads the real mmap is the right shape.

## Proposed artifact

New crate or binary: **`ix-optick-invariants`** at
`ix/crates/ix-optick-invariants/`. Reuses `ix-optick` reader. Emits
`firings.json` in the same shape `ix-invariant-produce` does, so
`ix-invariant-coverage` consumes it identically.

## What it checks (Phase 1 — 4 data-driven invariants)

### #25 — Cross-instrument STRUCTURE equality

For each distinct pitch-class-set that appears in ≥2 instruments (common:
major triads, m7 chords), compare the STRUCTURE slice (compact dims 0-23)
across instruments. Under v4-pp they should be bit-identical.

```
Pseudocode:
  index_by_pcs = group voicings by (pitch_class_set)
  for pcs, voicings in index_by_pcs:
      if len({v.instrument for v in voicings}) < 2: continue
      first_struct = voicings[0].structure_slice
      for v in voicings[1:]:
          if v.structure_slice != first_struct (within 1e-6):
              fire invariant_25 violation (pcs, instruments)
  report: #violations / #cross-instrument-PCs-tested
```

Success condition: 0 violations. v4-pp per-partition norm guarantees this
at the math level; checker proves it at the data level.

### #26 — MIDI-octave invariance

For each voicing V, compare V.structure_slice to the structure of its
octave-transpose V+12 (if present in corpus). Should be bit-identical.

Requires enumerating "same PC-set, ICV, rootPC, but shifted MIDI by 12"
pairs. Implementation: index by (PC-set + rootPC); for each group, find
pairs whose min-MIDI differs by exactly 12. Test should succeed on v4-pp.

### #27 — Per-partition norm bounds

Under v4-pp, each partition slice has L2 norm = sqrt(weight_p) after
sqrt-weight scaling. Expect:
- STRUCTURE slice norm ≈ sqrt(0.45) ≈ 0.6708
- MORPHOLOGY slice norm ≈ sqrt(0.25) = 0.5
- CONTEXT slice norm ≈ sqrt(0.20) ≈ 0.4472
- SYMBOLIC slice norm ≈ sqrt(0.10) ≈ 0.3162
- MODAL slice norm ≈ sqrt(0.10) ≈ 0.3162

Within 1e-4. Failures indicate writer drift or partition corruption.

### #32 — Cross-octave STRUCTURE cosine == 1.0

For each same-PC-set-across-octaves pair: compute cosine(A.structure,
B.structure). Expect 1.0 (identical structure) within 1e-6.

## CLI interface (mirrors `ix-invariant-produce`)

```
ix-optick-invariants --index path/to/optick.index [--out firings.json]
                     [--sample-voicings 50000]  // default: all
                     [--pretty]
```

Output: same JSON schema `ix-invariant-coverage` already ingests.

## Why a separate binary, not in ix-invariant-produce?

`ix-invariant-produce` is Phase-1 (synthetic, no external deps). Mixing
corpus-reading into it couples two different input surfaces. Separate
binary keeps each focused:
- `ix-invariant-produce` — algebraic invariants over pitch-class sets.
- `ix-optick-invariants` — data-level invariants over the real corpus.

`ix-invariant-coverage` merges both firings files seamlessly by invariant
number.

## Expected runtime

- 313k voicings × group by PC-set: <1 s.
- For each multi-instrument group (thousands), compare 24-dim STRUCTURE
  slices: dominated by mmap paging. Should finish under 30 s for #25.
- Cross-octave pairs (#26, #32): fewer — maybe 10k pairs — so <10 s.
- Total budget: 1 min.

## Rollout

**Phase 1 (1 session):**
- Scaffold crate.
- Implement invariant #25 first (most important, biggest signal for the
  v4-pp fix).
- Verify it reports 0 violations on the current v4-pp corpus.

**Phase 2 (follow-up):**
- Add #26, #27, #32.
- Wire into `ga/.claude/skills/optic-k-rebuild/SKILL.md` as part of the
  verification sweep after every rebuild.

**Phase 3 (opportunistic):**
- Chord-name consistency checker (#33) — requires loading metadata msgpack
  and comparing `chordName` strings across same-PC-set voicings.

## Known edge cases

- Some PC-sets only appear on one instrument (e.g., 6-note chords missing
  from ukulele). Those skip invariant #25 correctly (no cross-instrument
  pair to compare).
- Same-PC-set voicings might differ in `rootPitchClass` (Cmaj7 vs C7/E
  reinterpretation). STRUCTURE SHOULD differ if the root bit is set
  differently — invariant needs clarification of whether root-invariant
  or root-sensitive comparison is intended.

## Related

- `ga/docs/methodology/invariants-catalog.md` — the invariant catalog
  this binary feeds firings for.
- `ix/crates/ix-invariant-coverage/src/producer.rs` — companion synthetic
  producer.
- `ix/docs/plans/2026-04-18-optic-k-v4-pp-per-partition-norm.md` — the
  normalization-fix plan whose corpus-level invariants this validates.
