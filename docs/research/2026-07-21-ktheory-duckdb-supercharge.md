# Where K-theory could supercharge IX + DuckDB — a research verdict

- **Issue:** Deep research (Fable-subagent fan-out) — "where could algebraic / topological / Grothendieck K-theory *genuinely* improve the IX Rust workspace and its DuckDB integration?"
- **Date:** 2026-07-21
- **Verdict:** **NO** for K-theory-as-a-new-feature. The K-theoretic constructs IX already ships (`ix-ktheory`, `ix-duck::grothendieck`) collapse, under IX's actual data shapes, to primitives DuckDB already has (SUM / COUNT / set cardinality) or to invariants IX already computes by other means (Betti numbers, ICV L1). The research did surface **three genuine wins** — but each is a *reframe or a finer invariant*, not "add K-theory." Two verified bugs fell out of the audit (filed as #247, #248).

---

## Q — the question, sharpened

The seductive pitch is: "IX manipulates monoids (pitch-class multisets, resource vectors, DAGs); K-theory is the universal machine for turning monoids into groups and extracting invariants; therefore K-theory should unlock new capability in IX and in the DuckDB warehouse the ecosystem queries."

Sharpened into three falsifiable sub-questions, one per research angle:

1. **Algebraic K₀/K₁ (graph & resource):** does `K₀ = coker(I−Aᵀ)`, `K₁ = ker(I−Aᵀ)` give IX a structural invariant that its existing graph modules (`ix-graph`, `ix-topo`, `ix-search`) don't already give — and that DuckDB can't get from a GROUP BY?
2. **Grothendieck completion (music / ICV):** does ℤ⁶ group-completion of the ℕ⁶ interval-class-vector monoid buy anything over the L1 metric IX already ships, and does topological K-theory add anything over Betti numbers?
3. **Additivity / Mayer–Vietoris (warehouse):** is there a K-theoretic invariant that is *distributive over DuckDB shards* and therefore lets the warehouse compute something globally-correct from local pieces that a naive aggregate would get wrong?

## H — hypotheses going in

- H1 (optimistic): K₁ rank is a cheap cycle/feedback detector that beats DFS.
- H2 (optimistic): Grothendieck ℤ⁶ diffs separate homometric (Z-related) chord pairs that ICV alone cannot.
- H3 (optimistic): a Mayer–Vietoris additivity law lets the warehouse reconcile sharded SAE / voicing aggregates.

## M — method

Three adversarial Fable-5 subagents, one per angle, each instructed to try to *kill* its own hypothesis with a concrete counterexample or a collapse-to-primitive proof. Every load-bearing claim was then re-verified by me directly against source (`ix-ktheory/src/*.rs`, `ix-duck/src/grothendieck.rs`, `ix-bracelet/src/grothendieck.rs`) and against the eigenvalue math. No claim in this doc rests on an unverified agent assertion.

## E — evidence & findings

### The collapse theorem (why K-theory-as-feature is redundant here)

> **Claim.** Every K-theoretic invariant IX currently computes is, on IX's actual data shapes, either (a) a valuation — a signed weighted count — hence collapses to DuckDB `SUM`/`COUNT`; or (b) already computed by IX via a cheaper classical route (Betti numbers, ICV L1).

Three supporting legs, each independently checked:

**Leg 1 — Additivity ⇒ valuations ⇒ SUM.** The additivity theorem says a sharding-distributive invariant on an abelian category is a valuation: `v(B) = v(A) + v(B/A)`. Over IX's finite, free, torsion-free data (counts of voicings, activations, rows), K₀ *is* rank *is* a count. `ix-ktheory/src/mayer_vietoris.rs:29` already documents this — its `consistency_check` is literally `|A| + |B| − |A∩B| = |A∪B|`, i.e. inclusion–exclusion on cardinalities. That is a DuckDB `COUNT(DISTINCT …)` with a GROUP BY, not a UDF. **Do not ship it as a UDF.**

**Leg 2 — Chern character kills topological K-theory.** `ch: K⁰(X) ⊗ ℚ ≅ H^even(X; ℚ)` is an isomorphism. IX already computes Betti numbers (`ix-topo`). Rational topological K-theory therefore carries *strictly no information* IX doesn't already have from homology — it is the even Betti numbers repackaged. Torsion (where K-theory could differ) does not arise in IX's simplicial/graph complexes at the scale it operates.

**Leg 3 — Grothendieck ℤ⁶ = ICV L1 in disguise.** `ix-duck::grothendieck`'s `ix_grothendieck_delta` group-completes ℕ⁶ → ℤ⁶ and reports signed interval-class diffs; `ix_icv_l1` is their L1 norm. Group completion of a *cancellative* commutative monoid (which ℕ⁶ is) is just the ambient ℤ⁶ lattice — the "completion" adds nothing the lattice embedding didn't already give. The signed diff is a coordinate subtraction. Useful as ergonomics; not new math.

**Verdict on the theorem:** CONFIRMED for IX's data regime. K-theory would only start to pay if IX grew *torsion* (non-free modules — genuine quotient relations with finite order) or *non-cancellative* monoids. It has neither today.

### The three genuine wins (what the audit *did* find)

| # | Win | Kind | Cost | Ship vehicle |
|---|-----|------|------|-------------|
| W1 | **Additivity reconciliation gate** | reframe | ~30 lines SQL, **zero Rust** | `docs/plans/2026-07-21-arch-additivity-reconciliation-gate.md` |
| W2 | **DFT-phase Fourier invariants** | finer invariant | small `ix-bracelet` module + 3 UDFs | `docs/plans/2026-07-21-feat-dft-phase-invariants.md` |
| W3 | **Two hygiene bugs** | correctness | filed | #247, #248 |

**W1 — Additivity as a *reconciliation gate*, not a feature.** The collapse theorem is not purely negative: the fact that valid warehouse invariants *must* be additive over shards is itself a governance oracle. If a sharded aggregate (SAE activations split train/val, voicings split by partition) fails `v(A) + v(B/A) = v(whole)`, the split is lossy or mis-joined. This is exactly the failure mode behind bug #248 (train-split-only join key silently drops ~5% of the corpus). A DuckDB reconciliation query that asserts additivity would have *caught #248 automatically*. This is the single highest-value finding: K-theory's additivity axiom, used as a fail-closed data-integrity check rather than a computed feature.

**W2 — DFT phase is the one place a finer invariant genuinely exists.** Amiot's theorem: the interval-class vector (ICV) determines the *magnitudes* `|F_k|²` of the Fourier coefficients of a pitch-class set, but **not** their *phases*. Homometric (Z-related) pairs share an ICV precisely because they share all magnitudes — and are separated by phase. `ix-bracelet/src/grothendieck.rs:192` enumerates 23 unordered Z-pairs that ICV (hence `ix_grothendieck_delta`, hence GA's ICV bridge) provably *cannot* distinguish. The DFT phase vector separates them. This is a real capability gap in the current ICV-only bridge — and it is Fourier analysis, not K-theory, but it is the honest "supercharge" the question was fishing for. Falsifiable via the Amiot magnitude-identity, GA-parity on shared magnitudes, and a Z-pair-separation count.

**W3 — Two bugs the audit surfaced (both filed):**
- **#247** — `ix_k1`'s `@ai:invariant` at `ix-duck/src/grothendieck.rs:445` claims "rank>0 iff feedback cycles exist (a DAG → 0)". False. K₁ = ker(I−Aᵀ) detects *eigenvalue-1 circulation*, not cycles. Counterexample: the bidirected path P3 (a–b–c with both directions) is not a DAG yet its I−Aᵀ has eigenvalues {√2, 0, −√2}, so K₁ = 0 despite abundant cycles; conversely a pure directed cycle gives K₁ rank 1. The invariant is confidently wrong with a live `[T]` binding — the worst kind.
- **#248** — the 2026-07-20 optick-sae quality snapshot is contract-incomplete (missing `feature_manifest.jsonl`, `optick-sae-artifact.json`, `coverage` vs the complete 2026-06-14 set), and `train.py:557` keys `feature_activations.parquet` on `train_idx` only (297,395 rows into a 313,047-row corpus) — a silent ~5% join gap that W1's reconciliation gate is designed to catch.

## V — verdict & next steps

**K-theory as a headline feature for IX/DuckDB: rejected**, with a theorem-shaped reason (collapse to valuations + Chern isomorphism + cancellative-monoid triviality). Revisit trigger: **only** if IX acquires torsion modules or non-cancellative monoids — log this as the one-way-door condition, not a date.

Shipped from this research:
1. This doc (Q→H→M→E→V).
2. #247, #248 filed.
3. W1 spec — `docs/plans/2026-07-21-arch-additivity-reconciliation-gate.md` (the highest-value item; zero Rust).
4. W2 spec — `docs/plans/2026-07-21-feat-dft-phase-invariants.md` (the one honest finer-invariant win).

**Recommended order:** W1 first (catches #248-class bugs, cheapest, pure SQL), then W2 as a tracer-bullet slice through `ix-bracelet` → `ix-duck` if the Z-pair separation proves it earns its three new UDF names (a one-way door — new public UDF surface).
