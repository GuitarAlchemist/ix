---
title: IX↔GA phase-similarity reconciliation — one name, two semantics
type: arch
status: draft
date: 2026-07-31
issue: follow-on to docs/plans/2026-07-21-feat-dft-phase-invariants.md (ix #277 merged, ga #579 open)
reversibility: one-way (fixes the IX↔GA bridge invariant and the canonical weighting/invariance group) — sign-off required
revisit-trigger: the agreement test (F5) reports any pair disagreeing by > 1e-9 → the two implementations have diverged, stop and reconcile before either is used as an oracle
---

# IX↔GA phase-similarity reconciliation — one name, two semantics

- **Date:** 2026-07-31
- **Status:** proposal (decision required; no code in this slice)
- **Parent:** [`2026-07-21-feat-dft-phase-invariants.md`](2026-07-21-feat-dft-phase-invariants.md) — ix-side tracer-bullet. Its step 1 (`crates/ix-bracelet/src/fourier.rs`) **shipped** in ix #277 (merge `8b95889`). Its steps 2–3 (the three DuckDB UDFs) remain **unshipped and still a one-way door**.
- **Reversibility:** **one-way door.** Naming and the canonical (invariance group × weighting) pair become load-bearing across two repos. Once GA or a notebook binds either, changing the semantics is a silent numerical break, not a compile error.

## Who is in pain

Anyone comparing pitch-class sets across the IX↔GA boundary. Two implementations of "phase-aligned similarity" now exist, **with the same name and different answers**, and nothing checks that they agree.

| | ix #277 (merged) | ga #579 (open since 2026-07-22) |
|---|---|---|
| Location | `crates/ix-bracelet/src/fourier.rs` (Rust, +370) | `Theory/Atonal/SpectralPhaseAlignment.cs` (C#, +183) |
| Invariance group | **D₁₂ only** (T + I), single function | **Two modes**: `Similarity` = T-only; Theorem 4 = TnI |
| Z-pairs (homometry) | separated | separated |
| Chirality (major vs minor) | **conflated → scores exactly 1.0** | **separated** (the stated purpose) |
| Weights `w_k` | hardcoded Plancherel `[0,2,2,2,2,2,1]` | **optional; uniform when `null`** |
| Zero tolerance | `ZERO_TOL = 1e-6` (documented ~11-order gap) | `Epsilon = 1e-9` |
| Return | `Option<f64>` | record: similarity + aligning transpositions + inverted flag |

### The chirality gap is real, and ix's own test asserts it

`fourier.rs` test **P6** asserts similarity `= 1.0` across the whole D₁₂ orbit *including inversions*:

```rust
for g in [transpose(s, n), invert(transpose(s, n))] {
    assert!((sim - 1.0).abs() < 1e-9, "sim(S, g·S)={sim}");
}
```

A major triad and a minor triad are inversionally related, so `phase_aligned_similarity` returns exactly `1.0` for them. **IX therefore fixes one of the ICV's two blind spots (homometry), not both.** GA #579 fixes both. This is not a defect in ix — D₁₂-invariance is a deliberate, documented choice — but it is not what the shared name implies.

### Two concrete divergences, neither of which errors

1. **Wrong-pair matching.** ix's function corresponds to ga's **Theorem 4**, *not* to ga's default `Similarity`. A consumer matching by name gets chirality-blind where it expected chirality-preserving.
2. **Different weights.** Even correctly matched to Theorem 4, ix hardcodes Plancherel weights while ga defaults to uniform. Same inputs → different numbers, silently. ix's Plancherel choice is load-bearing: it is what makes the parent plan's CORRECTION 3 closed form (normalised max common-tone overlap, Lewin 1959) a valid independent oracle. Uniform weights do **not** have that closed form.

## The contract question (the actual one-way door)

`compound-engineering.local.md:19` states: *"IX is a pure ML layer; musical/domain semantics live in GA. The IX↔GA bridge is on the interval-class vector (ICV) — never bridge on Forte number."* (Note this rule lives in the review-context file, **not** in `CLAUDE.md` — worth knowing, since a cross-repo contract this load-bearing is currently recorded only in reviewer guidance.)

Both #277 and ga #579 exist **because the ICV is provably insufficient** — being the autocorrelation of the chroma, it fixes exactly the DFT magnitudes and nothing more (Wiener–Khinchin; parent plan §Theory). GA now carries the strictly richer invariant. So:

> **Decision required:** does the bridge stay ICV-on-the-wire with phase as a GA-local refinement, or does it widen to (magnitudes + phases)?

Options, with the cost of each:

- **(A) Bridge stays ICV.** Cheapest, no contract change. Cost: the bridge remains blind to 23 Z-pairs and to chirality, and every cross-repo similarity claim inherits that. The two phase implementations stay repo-local and *must not* be compared or substituted.
- **(B) Bridge widens to magnitudes + phases.** Honest to the math. Cost: a real contract change with a schema/wire impact, and it forces the canonical (group × weights) decision below.
- **(C) Bridge stays ICV for identity, adds phase as an explicit, separately-named refinement channel.** Keeps existing consumers working; the refinement is opt-in and unambiguous.

**Recommendation: (C).** It is the only option that neither pretends ICV is sufficient nor breaks existing ICV consumers, and it makes the naming collision impossible by construction.

## Canonical semantics (falls out of the decision above)

Whichever option is chosen, these must be named, not defaulted:

1. **Invariance group is part of the function name, never a default.** Two distinct names — a set-class (TnI) form and a chirality-preserving (T-only) form. IX currently has **only** the former; the latter is a genuine gap on the ix side if any consumer needs major ≠ minor.
2. **Weights are explicit at every call site.** Plancherel is canonical *because* it admits the Lewin closed-form oracle; uniform is a legitimate alternative (Quinn quality semantics) but has no oracle and must never be a silent default.

## Falsifier (goal-driven success criterion)

Extends the parent plan's **F2**, which checks GA parity on *magnitudes only* and predates ga #579.

**F5 — cross-implementation agreement.** Over a shared fixture set (at minimum: all 23 Z-pairs from `ix-bracelet/src/grothendieck.rs:192`, plus the major/minor triad pair, plus the T₆-symmetric degenerates — `{0,6}`, diminished sevenths, whole-tone), assert:

- ix `phase_aligned_similarity(a, b)` **==** ga Theorem 4 with Plancherel weights **passed explicitly**, to within `1e-9`;
- ix's function returns `1.0` for the major/minor pair and ga's T-only `Similarity` returns `< 1.0` — i.e. the chirality difference is **asserted as intended behaviour**, not discovered later as a bug.

Report the exact count agreeing. A test that runs without asserting the count is green-but-dead (`feedback_green_but_dead`). Until F5 exists there are two oracles and no referee.

## Out of scope

- Shipping the three DuckDB UDFs (`ix_dft_mag`, `ix_dft_phase`, `ix_phase_aligned_sim`) — still governed by the parent plan's one-way-door log and its F3 23/23 gate.
- Adding a chirality-preserving mode to ix. Warranted only if a consumer needs it; noted here so the gap is recorded rather than assumed absent.
- Merging or closing ga #579. That PR is independently valid; this plan only says the two must not be treated as interchangeable.

## One-way-door log

- **Locked by this decision:** the bridge invariant (ICV vs magnitudes+phases vs ICV+refinement channel), and the canonical (invariance group × weighting) pair for anything named "phase-aligned similarity" in either repo.
- **Sign-off condition:** an owner picks A/B/C **and** F5 is implemented and passing before either implementation is cited as an oracle for the other.
- **Revisit trigger:** any F5 pair disagreeing by more than `1e-9`, or a third implementation appearing in any repo.
