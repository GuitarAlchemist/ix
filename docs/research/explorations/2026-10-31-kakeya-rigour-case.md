# Kakeya in ℝ³ as a teaching case and a test of agent rigour

Exploration note, low priority (behind the WMUX hookup). It starts no engine work and announces no
audio or RTX gain. French: [`docs/fr/research/explorations/2026-10-31-kakeya-rigour-case.md`](../../fr/research/explorations/2026-10-31-kakeya-rigour-case.md).

Status words: **source-read** (checked against the cited page here), **recalled** (classical
background, not rechecked in this note), **open**.

## Why IX cares

The result is a clean test of three distinctions an agent blurs easily: dimension vs measure,
simulation vs proof, established result vs extrapolation. The proposed exercise: one agent explains,
an adversary hunts the errors, then every claim is checked against the sources below.

## Questions, answers, traps

**Q1. What exactly was proved?** Every Kakeya set in ℝ³ has Hausdorff and Minkowski dimension 3.
*Source-read:* H. Wang, J. Zahl, "Volume estimates for unions of convex sets, and the Kakeya set
conjecture in three dimensions", [arXiv:2502.17655](https://arxiv.org/abs/2502.17655), v1, 24 Feb
2025, abstract. *Trap:* the arXiv page lists one version and no journal reference: say "preprint",
not "peer-reviewed", unless a later source shows otherwise.

**Q2. Does dimension 3 mean the set has positive volume?** No. *Recalled:* Besicovitch constructed
Kakeya sets of Lebesgue measure zero, in the plane and in every ℝⁿ, n ≥ 2. Such a set in ℝ³ has
volume zero and, by Q1, dimension 3. *Trap:* "full dimension" read as "fills space".

**Q3. What does the δ-thickened picture show?** Thicken each unit segment into a tube of radius δ.
For a measure-zero Kakeya set E, the volume of the δ-neighbourhood tends to 0 as δ → 0; dimension 3
says it tends to 0 more slowly than any power δ^ε. *Trap:* an animation runs over a finite range of
δ, where δ^0.1 and a constant are hard to tell apart: it cannot separate dimension 3 from 2.9, so it
illustrates and never demonstrates.

**Q4. What about the plane and higher dimensions?** *Recalled:* in ℝ² every Kakeya set has dimension
2 (Davies, 1971). For n ≥ 4 the conjecture is **open** as far as this note's sources go. *Trap:*
extrapolating "3 in ℝ³" to "n in ℝⁿ".

**Q5. What is the abstract's own route?** *Source-read:* a volume estimate for unions of δ-tubes in
ℝ³ when not too many tubes fit inside a common convex set; the dimension result is stated as a
consequence. *Trap:* summarising the proof beyond the abstract without reading the 127 pages.

## The adversary's checklist

- Every number or year carries a source or the word *recalled*.
- "Dimension" always names Hausdorff or Minkowski; "measure" or "volume" never stands in for it.
- A picture or simulation is called an illustration.
- "Proved" applies to ℝ² (recalled) and ℝ³ (preprint); nothing is said proved for n ≥ 4.

## Decision this note prepares (not taken)

Whether a Streeling visualisation (thickened segments, rescaling) is worth building. Revisit only if
the Q&A above survives an explainer/adversary round with no unsourced claim left.
