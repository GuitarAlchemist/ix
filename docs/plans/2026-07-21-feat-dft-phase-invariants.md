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

---

## Research Insights (deepened 2026-07-21)

_Parallel research + numerical/perf review (Fable-5 agents) + institutional-learnings scan. The core theory holds; three findings below are **load-bearing corrections**, not enrichment._

### Enhancement summary
- **Theory confirmed** (independent 4096-subset sweep, 0 mismatches) — but the homometry "iff" (line 28) needs a T/I-exclusion qualifier, and the exact ICV↔magnitude formula should be F1's oracle verbatim.
- **CORRECTION 1 — "align on arg(F₁)" is broken** and must be replaced with a discrete group-max in complex arithmetic (no `atan2`).
- **CORRECTION 2 — the similarity range is [−1, 1], not [0, 1]** (§Tracer-bullet item 2 is wrong as written).
- **CORRECTION 3 — the metric has a closed form** (= normalized max common-tone overlap, Lewin), giving an independent integer cross-check and turning F3 into a *theorem*.
- Numerical, DuckDB-API, dependency, and falsifier hardening below.

### Theory — confirmed, with two qualifiers (agent: DFT-phase theory)
- The plan's "Amiot's theorem" is the pc-set instance of Wiener–Khinchin: `iv(d)` is the autocorrelation of `1_S`, so its DFT is exactly `|F_k|²`. **Exact oracle for F1** (verified over all 4096 subsets): `|F_k(S)|² = n + 2·Σ_{d=1..5} icv_d·cos(2πkd/12) + 2·icv_6·cos(πk)`, `n = |S|`. Note the factor-2 on `icv_6` (a tritone contributes both directions). State this formula normatively in §The math and use it verbatim in F1 (do **not** derive the map from the DFT code — that's circular).
- **Qualifier (line 28):** "homometric iff equal magnitudes, differ in phase" is imprecise — T/I-related sets *also* share magnitudes. Correct: **equal magnitudes ⟺ homometric; Z-related = homometric AND not T/I-equivalent.** Cite Amiot 2016 ch. 3 + Rosenblatt & Seymour, *SIAM J. Alg. Disc. Meth.* 3(3), 1982.
- Consequences to record in module docs: `F_0 = |S|` (no new info, exclude from sim); k=7..11 are conjugates of 5..1 (so 0..6 is complete); **`F_6` is real** ⇒ `arg(F_6) ∈ {0, π}`.

### CORRECTION 1 — replace continuous phase alignment (agents: DFT-theory + numerical)
"Align on arg(F₁) first" is **undefined precisely on inputs the feature exists for**: `|F₁| = 0` for `{0,6}`, diminished sevenths, whole-tone sets, and any T₆-symmetric set. (Empirically, 22 of the 46 Z-pair member sets have some `|F_k| = 0` at k∈{3,4,6}; F₁ happens to be nonzero on the F3 corpus but not on general UDF inputs.) The transposition group is **discrete** — use an exact max over group elements in pure complex arithmetic, never a fitted continuous rotation and never a phase extraction:

```
sim(A,B) = max_{n∈0..11 [, ±conj]}  Re Σ_{k=1..6} w_k · F_k(A) · conj(e^{−2πikn/12} F_k(B))
                                     / ( ‖F_A‖_w · ‖F_B‖_w )
```

This never divides by `|F₁|`, never calls `atan2` (no mod-2π wraparound bugs), is exact, and makes near-zero coefficients contribute ~0 weight *by construction* (the magnitude-weighting the plan wanted). **Decide the group explicitly:** ICV is D₁₂-invariant (T *and* I), so to be the honest refinement of `ix_icv_l1`, maximise over all 24 D₁₂ elements (12 rotations × optional conjugation). Both T-only and T/I variants separate all 23 pairs with identical margins.

### CORRECTION 2 + 3 — range and closed form (agent: DFT-phase theory)
- With Plancherel weights `w = (0,2,2,2,2,2,1)` for k=0..6, the metric **collapses via Parseval to normalized maximal common-tone overlap** (Lewin 1959): `sim(A,B) = (12·maxCT − |A||B|) / sqrt((12|A|−|A|²)(12|B|−|B|²))`, where `maxCT` = max common tones over the 24 alignments. This lives in **[−1, 1]** (a genuine cosine; goes negative for disjoint-ish pairs) — the plan's "double in [0,1]" (§Tracer-bullet item 2) is a bug; either document [−1,1] or map `(1+s)/2`. It also gives a **brute-force integer cross-check** (must agree to 1e-12) that doesn't share the DFT code path.
- **F3 becomes a theorem:** with strictly positive weights on k=1..6, `sim = 1` forces Cauchy–Schwarz equality ⇒ `A = g·B` for `g ∈ D₁₂`; Z-pairs are by definition not D₁₂-equivalent, so `sim < 1` is guaranteed for *any correct* implementation. Empirically over the 23 pairs, `sim ∈ {1/3, 2/3}` exactly — **max = 0.667**, an enormous margin (worst pair `[0,1,2,4,5,7]`/`[0,1,2,3,5,8]`).

### Numerical & performance (agent: DFT numerical/perf review)
- **Precompute the 12 roots of unity as a `const` table** indexed by `(k·p) mod 12` (coords in `{0, ±½, ±1, ±√3/2}`) — not per-element `cos/sin`. Naive trig turns true-zero coefficients into ~6e-17 noise with random-looking phase. There is a provable gap: nonzero `|F_k| ≥ 1/12³ ≈ 5.8e-4`, const-table noise ≤ ~3e-15, so a **zero-threshold τ = 1e-6** sits inside an 11-order gap and can never misclassify. Name τ as a constant with this justification.
- **Degeneracy is the common case, not an edge case.** For `ix_dft_phase`, return a **NULL list element** where `|F_k| < τ` (0.0 is a lie — a valid phase; NaN poisons aggregates). For sim, define `0/0` (∅ or full aggregate) as **SQL NULL**, not 0.0/1.0 — a leaked NaN reads as `< 1.0` = false and would make F3 pass wrongly.
- Per-row cost is trivial (≤84 table lookups); **do not** build a 4096-entry LUT (speculative, Karpathy r2). `LIST<DOUBLE>` return is the right shape but **untrodden in ix-duck** (existing VScalars return DOUBLE/VARCHAR; list outputs only in table fns) — start the tracer-bullet with a minimal list-returning-VScalar spike and name the VARCHAR fallback (à la `ix_icv`) before the one-way door closes. **The return types are locked API too** — add them to §One-way-door log, not just the names.

### DuckDB API — confirmed against the pinned crate (agent: DFT-phase theory)
- Verified against `duckdb = "=1.10503.1"` source: a list-returning **scalar** UDF is fully supported — `WritableVector::list_vector()` (`arrow.rs:583`), `ListVector::{child, set_entry, set_len, set_null}` (`vector.rs:228-325`). **Copy the existing write pattern `emit_coord_rows` at `crates/ix-duck/src/tablefn.rs:241`** (LIST<DOUBLE>, fixed length per row) nearly verbatim; the only delta is `output.list_vector()` (no column index).
- Signature: `exact(vec![list_bigint()], LogicalTypeHandle::list(&Double))`, mirroring `bracelet.rs:28`.
- **NULL gap:** the existing `read_pcsets` (`bracelet.rs:35`) never checks input validity — decide NULL-in → NULL-out explicitly via `FlatVector::row_is_null` (`vector.rs:62`); don't copy the gap silently.

### Falsifier hardening (agents: theory + numerical) — each property is a theorem, so exact-tolerance
- **F1:** run the exact §1 formula over **all 4096 raw subsets** (instant), not just 224 classes — catches bitset-decode bugs on non-prime-form inputs the UDF will see. Assert the loop saw exactly 224 classes / `z_related_pairs().len() == 23` before iterating (guards silent shrinkage of `grothendieck.rs:192`).
- **F3:** assert `sim ≤ 0.67` (not `< 1.0` — a bare `<1.0` gate passes a regression that degrades separation to 0.999; `feedback_green_but_dead` applies to margins) and **report min/max margin**. Pair with a **positive control** `sim(S, g·S) > 1 − 1e-9` for all 224 × 24 — without it F3 is passed by a broken sim returning 0.42 for everything.
- **New property tests** P1 `F_0 = |S|`; P2 T-covariance (compare as complex, ≤1e-12, avoids mod-2π); P3 inversion conjugates; P4 complement negates `F_k` (⇒ hexachord theorem, explains why 15/23 pairs are complement hexachords); P5 `F_6` real; P6 sim is a proper similarity over all 24 `g`; P7 closed-form integer cross-check; P8 degenerate ∅/aggregate → NULL (fixture `[0,1,2,5,6,7]`, `F_4=F_6=0`); P9 converse — `sim(S,g·S) ≥ 1−1e-12`, separation only from genuine inequivalence.
- **F2 stays magnitude-only** — "phases agree with GA" is ill-defined (phases are representative-dependent). Restate as a pure-ix numeric test `|F_k(T_n S)| = |F_k(S)| = |F_k(I S)|` (≤1e-12 over 224×12×{id,I}), which *is* the mathematical content of GA parity; demote live-GA comparison to a manual step. (Learnings: **never bridge on Forte number** — use explicit PC-set pairs.)

### Dependency & institutional notes
- **`num-complex` is nowhere in the workspace**; `ix-bracelet` is a leaf crate (thiserror + ix-search only). Return `[(f64, f64); 7]` or a 3-line local complex struct — do **not** add a dep to a leaf crate. (Tier `internal`, so no gate trips, but state the choice in the plan.)
- Keep ix-duck **terse — do not run `cargo fmt`** (dense SQL-adjacent style is intentional; `reference_ix_transforms_and_ixql`).
- If ix-duck grows a UDF-count oracle, bump it in the same PR; check `docs/guides/` for a UDF-listing doc (`reference_ix_agent_parity_cascade`).
- Annotate the standing claim honestly until F3 passes: `// @ai:invariant phase distinguishes Z-pairs [T:test conf:unknown src:F3]` (`certainty := strength of live binding`).

### References
Amiot, *Music Through Fourier Space* (Springer, 2016, ch. 1–3); Lewin, "Intervallic Relations Between Two Collections of Notes," *JMT* 3(2), 1959; Quinn, "General Equal-Tempered Harmony," *PNM* 44–45, 2006–07; Yust, "Schubert's Harmonic Language and Fourier Phase Space," *JMT* 59(1), 2015; Rosenblatt & Seymour, *SIAM J. Alg. Disc. Meth.* 3(3), 1982; Ballinger et al., "The Continuous Hexachordal Theorem," MCM 2009.
