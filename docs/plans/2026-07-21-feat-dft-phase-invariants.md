---
title: DFT-phase invariants — separate the Z-related pairs ICV cannot
type: feat
status: draft
date: 2026-07-21
issue: W2 of docs/research/2026-07-21-ktheory-duckdb-supercharge.md
reversibility: one-way (3 new public DuckDB UDF names become locked API) — sign-off required before merge
revisit-trigger: F3 Z-pair separation count returns < 23/23 → do NOT ship, the phase invariant earns nothing
---

# DFT-phase invariants — separate the Z-related pairs ICV cannot

- **Issue:** W2 from `docs/research/2026-07-21-ktheory-duckdb-supercharge.md`.
- **Date:** 2026-07-21
- **Status:** proposal (tracer-bullet)
- **Reversibility:** **one-way door** — introduces 3 new public DuckDB UDF names (`ix_dft_mag`, `ix_dft_phase`, `ix_phase_aligned_sim`). New UDF surface is a locked API. Requires sign-off before merge. Revisit trigger: if the Z-pair separation count (falsifier F3) comes back 0, do NOT ship — the invariant earns nothing.

## Who is in pain

Anyone using the ICV bridge (`ix_grothendieck_delta`, `ix_icv_l1`, GA's `ga_chord_to_set` → ICV) to judge chord similarity or identity. ICV is **provably blind** to homometric (Z-related) pairs: `ix-bracelet/src/grothendieck.rs:192` enumerates **23 unordered Z-pairs** that share an interval-class vector and therefore collapse to identical ICV / Grothendieck-delta output despite being genuinely different set-classes. Every ICV-keyed lookup silently conflates each such pair.

## The math (why this is the ONE honest finer invariant)

For a pitch-class set S ⊆ ℤ₁₂, its discrete Fourier coefficients are

  F_k(S) = Σ_{p∈S} e^{−2πi k p / 12},  k = 0..6  (k>6 are conjugates).

**Amiot's theorem:** the interval-class vector determines exactly the *magnitudes* |F_k(S)|² — and nothing more. Two sets are homometric (Z-related) **iff** they share all magnitudes but differ in **phase** arg(F_k). So the phase vector is precisely the invariant that separates the 23 pairs ICV cannot. This is Fourier analysis on ℤ₁₂ (Quinn / Amiot / Lewin), not K-theory — but it is the real "supercharge" the K-theory question was reaching for, and it is the only finer-invariant win the audit found.

## Tracer-bullet slice (end-to-end, thinnest, every layer)

One thin vertical slice, math → crate → UDF → falsifier, before any expansion:

1. **`crates/ix-bracelet/src/fourier.rs`** — a small module on the existing `PcSet(u16)` bitset:
   - `dft(set: PcSet) -> [Complex<f64>; 7]` — F_0..F_6.
   - `dft_magnitudes(set) -> [f64; 7]` — |F_k| (the ICV-equivalent half; used for parity check).
   - `dft_phases(set) -> [f64; 7]` — arg(F_k) in radians (the *new* information).
   - `phase_aligned_similarity(a, b) -> f64` — magnitude-weighted phase agreement, transposition-invariant (align on F_1 phase first, since a T_n transposition rotates arg(F_k) by 2πkn/12).
2. **`crates/ix-duck/src/bracelet.rs`** — three scalar UDFs alongside the existing `ix_icv` / `ix_prime_form` (same `register_scalar_function` pattern, `register()` at line 151):
   - `ix_dft_mag(pcs)` → list of 7 doubles.
   - `ix_dft_phase(pcs)` → list of 7 doubles.
   - `ix_phase_aligned_sim(pcs_a, pcs_b)` → double in [0,1].
3. Wire `register()` to add the three; bump the ix-duck UDF count oracle if one exists.

No SAE, no GA changes, no schema. Pure additive slice.

## Falsifiers (goal-driven success criteria)

1. **F1 — Amiot magnitude identity.** For all 224 set-classes, `dft_magnitudes(S)²` reconstructs the ICV via the known linear map (ICV_k = Σ contributions of |F|²). If magnitudes disagree with `ix_bracelet::icv`, the DFT is wrong. Unit test over all classes.
2. **F2 — GA parity on magnitudes.** `ix_dft_mag` magnitudes must be transposition-invariant and agree with GA's ICV-derived quantities for the shared corpus (per the ICV bridge contract; **never** bridge on Forte number — `grothendieck.rs:28-31`).
3. **F3 — Z-pair separation count (the ship/no-ship gate).** Feed all 23 Z-pairs from `ix-bracelet/src/grothendieck.rs:192`. `ix_phase_aligned_sim` must return **< 1.0** for every pair (they are different) while `ix_icv_l1` returns 0 for all of them. **Report the exact count separated.** If it is not 23/23, the phase invariant does not do what the theory says and the feature is rejected — no partial ship.

F3 is the whole justification. A green build that separates 0 pairs is green-but-dead (`feedback_green_but_dead`); the test must assert the count, not merely run.

## One-way-door log

- New locked UDF names: `ix_dft_mag`, `ix_dft_phase`, `ix_phase_aligned_sim`. Once published, downstream (GA, notebooks) may bind them; renaming becomes a breaking change.
- Sign-off condition: F3 returns 23/23 **and** an owner accepts the 3-UDF surface expansion.

## Out of scope

- A DuckDB UDAF that aggregates phase over a group (blocked anyway by the duckdb-rs no-UDAF constraint — see `project_epistemic_sql_ixduck`).
- Extending phase similarity into OPTIC-K voicing search ranking (a separate, larger slice — only after F3 proves the invariant earns its keep).
