# `ix_voicing_similarity` — harmonic nearest-neighbours over the voicing corpus

A runnable demo of **similarity retrieval** on the DuckDB analyst bench, with a built-in
**correctness oracle**. Companion to [`voicing-mesh.md`](voicing-mesh.md) (which does
*correlation/clustering*); this one does *nearest-neighbour retrieval + outlier scoring*.

> _Version française : [`docs/fr/pas-a-pas/similarite-voicings.md`](../fr/pas-a-pas/similarite-voicings.md)._

```bash
cargo run -p ix-duck --example ix_voicing_similarity --features duck
```

## The question

> **Given a chord's set-class, what are its harmonically-nearest neighbours by interval
> content — and which set-classes in the real repertoire are harmonically isolated (no
> close neighbour)?**

Each Forte set-class present in the corpus is one point. The distance is **`ix_icv_l1(a, b)`**
— the L1 distance between the two PC-sets' **interval-class vectors** (the Grothendieck
harmonic cost). A single self-join computes the full N×N harmonic-distance table on the
bench; `ix_forte_number` annotates and `any_value(midiNotes)` picks a representative
(the ICV is a set-class invariant, so any voicing of a set-class gives the same vector).

## Nearest neighbours

```text
 3-11 → 2-3 (d=2), 2-4 (d=2), 2-5 (d=2), 3-3 (d=2), 3-4 (d=2)
 4-27 → 4-12 (d=2), 4-13 (d=2), 4-18 (d=2), 4-26 (d=2), 4-Z15 (d=2)
4-Z15 → 4-Z29 (d=0), 4-11 (d=2), 4-12 (d=2), 4-13 (d=2), 4-14 (d=2)
```

The highlight is `4-Z15`: its **nearest neighbour is `4-Z29` at distance 0**. These are the
two **all-interval tetrachords** — distinct chord shapes with *identical* interval content.
The retrieval surfaces that automatically, which is exactly what the validation formalises.

## Validation — the Z-relation oracle

This is the analog of the mesh's null model, but matched to a *retrieval* demo: instead of
a significance test, a **ground-truth correctness oracle**.

Two *distinct* set-classes have an identical ICV **iff** they are **Z-related** (same
cardinality + same interval vector, different prime form). So **every pair at harmonic
distance 0 must be `ix_z_related`** — if any weren't, `ix_icv_l1` would be unfaithful.
The demo checks this against the corpus:

```text
validation — 19 distinct-set-class pairs at harmonic distance 0:
   4-Z15 ≡ 4-Z29,  5-Z12 ≡ 5-Z36,  6-Z29 ≡ 6-Z50,  … (all hexachordal Z-pairs)
   ✅ ORACLE PASS: every distance-0 pair is ix_z_related
```

The count is itself a **textbook match**: 12-TET has exactly **19 Z-related pairs** (1
tetrachord + 3 pentachord + 15 hexachord). The demo recovers the *complete* set from the
guitar corpus — so every Z-related set-class actually occurs in real fingerings — and
confirms `ix_icv_l1` and `ix_z_related` agree on all of them. That is both an external
ground-truth check (the 19, and the famous `4-Z15/4-Z29`) and a cross-UDF consistency check.

## Harmonic isolation

Among set-classes common enough to be "really used" (≥ 500 voicings), the most
harmonically **isolated** — largest nearest-neighbour distance — are:

```text
 5-7: nn = 4  (10 546 voicings)      5-33: nn = 4  (4 241 voicings)
5-31: nn = 4  ( 7 994 voicings)      4-28: nn = 3  (1 749 voicings)
5-15: nn = 4  ( 4 952 voicings)
```

The pentachords `5-7`, `5-31`, `5-15`, `5-33` sit farthest from everything else by
interval content while still appearing thousands of times — harmonically distinctive
chords guitarists nonetheless reach for.

## Scope and caveats

- **Advisory only** (like all `ix_duck` lenses; see ADR-0002 / ADR-0004) — not a gate.
- The ICV space is **dense and quantised**: distances are small integers (nn ∈ {2, 3, 4}
  here), so "isolation" differentiates only weakly. The oracle (distance 0) is exact; the
  isolation ranking is a soft signal, not a sharp partition.
- The metric is harmonic (interval content) only — it ignores voicing/register/playability.
  Pair it with the position/stretch axes of `ix_voicing_mesh` for the fretboard view.
- Full corpus is gitignored (110 MB); on a fresh checkout the demo falls back to the
  tracked 500-voicing sample (fewer set-classes, so fewer Z-pairs — the oracle still holds).
