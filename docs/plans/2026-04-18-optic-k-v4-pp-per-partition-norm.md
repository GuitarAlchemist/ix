---
date: 2026-04-18
reversibility: one-way-door
revisit-trigger: leak-test shows STRUCTURE/CONTEXT/SYMBOLIC/MODAL accuracy still > 1/3 + 3σ after rebuild, or invariant #25/#28/#32 still fail
---

# OPTIC-K v4 → v4-pp: per-partition normalization

## Decision

Change the OPTK on-disk vector semantics from **global L2 normalization** (v4) to
**per-partition L2 normalization followed by sqrt-weight scaling** (v4-pp).

Schema layout unchanged (still 112 compact dims, same partition boundaries,
same weights). Only the normalization step differs. Schema-hash bumped via
layout-string prefix `optk-v4-pp:` so old v4 indexes fail validation and force
rebuild.

## Problem being solved

The 2026-04-18 baseline-diagnostics run against `state/voicings/optick.index`
showed four invariants failing:

| # | Invariant | Accuracy / status |
|---|---|---|
| 25 | Cross-instrument STRUCTURE equality (same PC-set → same STRUCTURE vector) | FAIL |
| 28 | 3-class partition leak test (no partition accuracy > 1/3 + 3σ) | FAIL across CONTEXT/SYMBOLIC/MODAL |
| 32 | Same PC-set across octaves → cosine(STRUCTURE) = 1.0 | FAIL |
| 33 | ChordName consistency across instruments | FAIL |

Leak test showed STRUCTURE classified instrument at **59.6%** (vs random 33%),
MODAL at 49.8%, SYMBOLIC at 61.4%, CONTEXT at 52.2%.

### Root cause

`OptickIndexWriter.ExtractAndNormalize` (v4) applies sqrt(partition-weight)
scaling to each dim, then divides every compact dim by the **global** L2 norm.

For two voicings A, B with identical PC-set:
- STRUCTURE raw slices identical (TheoryVectorService is deterministic).
- MORPHOLOGY slices differ per instrument (guitar 6 strings, ukulele 4).
- `global_norm_A = sqrt(w_S·|S|² + w_M·|M_A|² + ...)`
- `global_norm_B = sqrt(w_S·|S|² + w_M·|M_B|² + ...)`
- **After normalization**, `compact_STRUCTURE_A = S·√w_S / norm_A ≠ compact_STRUCTURE_B = S·√w_S / norm_B`.

MORPHOLOGY variation leaks into the STRUCTURE slice. Random forest learns
`|compact_structure|` as instrument signal. Same mechanism breaks cross-octave
invariance because octave transposition shifts MORPHOLOGY (fretboard geometry)
while leaving STRUCTURE identical.

### Fix

Per-partition L2 normalize **before** sqrt-weight scaling. Each partition slice
has unit length × sqrt(weight_p) after the pipeline. Total vector norm² =
Σ_p weight_p (not 1), but dot product of two such vectors = weighted partition
cosine directly:

```
dot(A, B) = Σ_p Σ_dim A_p[d] · B_p[d]
         = Σ_p (√w_p)² · dot(unit_A_p, unit_B_p)
         = Σ_p w_p · cos(A_p, B_p)
```

No global renormalization needed. Cross-partition variations cannot leak.

## Coordinated changes

| File | Change |
|---|---|
| `ga/Common/GA.Business.ML/Embeddings/EmbeddingSchema.cs` | Layout string `optk-v4:` → `optk-v4-pp:`. CRC32 changes, schema hash bumps. |
| `ga/Demos/Music Theory/FretboardVoicingsCLI/OptickIndexWriter.cs` | `ExtractAndNormalize` implements per-partition norm. |
| `ga/Common/GA.Business.ML/Search/MusicalQueryEncoder.cs` | `ExtractCompactAndNormalize` mirrors writer semantics. |
| `ix/crates/ix-optick/src/lib.rs` | `SCHEMA_SEED` updated to `optk-v4-pp:…` to match. |
| Tests | `Reader_Vector_IsUnitLength` replaced with `Reader_Vector_NormMatchesPerPartitionSchema`; encoder norm expectation updated. |

## Also bundled in this rebuild window

1. **SYMBOLIC tag enrichment** (`VoicingTagEnricher`) — derives mood/register/
   style/technique tags from metrics the analyzer already computes. Fixes the
   observed `Cmaj7 jazz` < `Cmaj7` regression (corpus had thin SYMBOLIC bits).
2. **Extended chord qualities** — added to `ChordPitchClasses`:
   `mmaj7`, `minmaj7`, `maj13`, `m11`, `m13`, `9#11`, `maj9#11`, `13#11`,
   `7b9`, `7#9`, `7b5`, `7#5`, `aug7`, `+7`, `13b9`, `7alt`, `69`, `6/9`,
   `sus4add9`, `sus2add11`. Parser slash-handling fixed so `6/9` doesn't
   collapse to `6` with a `9` bass.

## Deployment path

The on-disk `state/voicings/optick.index` is now incompatible with the new
reader. `OptickIndexReader` throws `InvalidDataException` with clear schema-
hash mismatch message. 10 retrieval tests skip cleanly with rebuild
instructions until the rebuild runs.

Rebuild:

```powershell
cd C:\Users\spare\source\repos\ga\Demos\Music Theory\FretboardVoicingsCLI
dotnet run -- --export-embeddings --out state/voicings/optick.index
```

Expected post-rebuild verifications:

1. `baseline-diagnostics.exe --index state/voicings/optick.index` — STRUCTURE
   leak accuracy should drop from 0.596 toward 0.33 (random-chance); MODAL/
   CONTEXT/SYMBOLIC should also approach random under cross-instrument test.
2. `ix-invariant-produce.exe` + `ix-invariant-coverage.exe` — #25/#28/#32
   should flip from FAIL to PASS.
3. `Cmaj7 jazz` vs `Cmaj7` — the tagged query should now rank higher
   (SYMBOLIC partition has content after enrichment, not just after write).
4. Live MCP `ga_search_voicings` through Claude — top-5 for any chord
   should be near-identical whether the user phrases it `"Cmaj7"` or
   `"C maj 7"` or other canonical form.

## Revisit trigger

Re-evaluate the normalization approach if:
- Leak test still shows > 40% accuracy on any non-MORPHOLOGY partition
  after rebuild. Likely means the partition services themselves encode
  instrument signal (not a normalization issue).
- Retrieval quality (PC-set match) drops below 92% (current baseline on v4).
- Someone proposes a different cosine weighting (per-partition Mahalanobis,
  learned partition weights, etc.) — then revisit both normalization and
  weighting together.

## Notes

- CLAUDE.md flags OPTIC-K partition layout as a one-way door. Partition
  *layout* hasn't changed (same dim ranges, same weights). Only the
  *normalization semantics* shift. Schema hash change forces rebuild —
  that's the enforcement mechanism.
- ix-optick reader doesn't need logic changes for correctness: dot product
  over on-disk vectors still yields the correct similarity metric (weighted
  partition cosine). Only `SCHEMA_SEED` needed updating so its hash matches.
