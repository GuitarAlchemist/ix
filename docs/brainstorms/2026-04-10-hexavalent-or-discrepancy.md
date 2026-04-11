# Hexavalent OR discrepancy — spec tables vs. De Morgan derivation

**Status:** decision pending
**Discovered:** 2026-04-10 while starting Item A (hexavalent migration) in `docs/brainstorms/2026-04-10-ix-harness-primitives.md`
**Blocker for:** hexavalent migration, `ix-approval`, any new hexavalent consumer
**Scope:** pre-existing bug in `ix_types::Hexavalent::or` vs. inconsistency in the canonical spec

---

## The finding

`crates/ix-types/src/lib.rs:105-107` implements hexavalent OR via De Morgan:

```rust
pub const fn or(self, other: Self) -> Self {
    self.not().and(other.not()).not()
}
```

This is mathematically clean: for any boolean algebra / De Morgan lattice, `a OR b ≡ !(!a AND !b)` holds, and deriving OR from AND keeps the two operators automatically consistent.

**But `governance/demerzel/logic/hexavalent-logic.md` specifies OR with an explicit truth table that is NOT the De Morgan dual of its AND table.**

### The specific cells that disagree

| `a` | `b` | Spec OR table | Current code (De Morgan) |
|---|---|---|---|
| U | P | **P** | U |
| U | D | **D** | U |
| P | U | **P** | U |
| D | U | **D** | U |
| P | D | **P** | *(need to verify)* |
| D | P | **P** | *(need to verify)* |

At least four cells definitively disagree. The U/P/D subspace is where the spec's OR tables intentionally treat directional evidence as dominant over Unknown, which the De Morgan derivation from the (correct) AND tables cannot reproduce.

### Proof the current code disagrees with the spec on `U OR D`

Spec `hexavalent-logic.md` OR table:

```
| OR  | T | P | U | D | F | C |
|-----|---|---|---|---|---|---|
| T   | T | T | T | T | T | T |
| P   | T | P | P | P | P | C |
| U   | T | P | U | D | U | C |
| D   | T | P | D | D | D | C |
| F   | T | P | U | D | F | C |
| C   | T | C | C | C | C | C |
```

`U OR D` = row U, column D = **D**.

Current code trace:
- `U.not()` = `U` (per NOT table: T↔F, P↔D, U→U, C→C)
- `D.not()` = `P`
- `U.and(P)` = `U` (per spec AND table row U col P, and per current `and` impl which matches the spec)
- `.not()` = `U`
- **Result: `U`**. Spec wants `D`.

### The spec is self-inconsistent

The spec has *three* internally conflicting statements about OR:

1. **The explicit truth tables** (row U col D = D, etc.)
2. **The prose claim** *"Design: T is absorbing (any OR true = true). Symmetric to AND via De Morgan."* — this claims De Morgan symmetry, which is the opposite of what the tables actually encode.
3. **The lattice diagram**:
   ```
           T
          / \
         P   |
         |   |
         U   C
         |   |
         D   |
          \ /
           F
   ```
   Under the truth ordering `F < D < U < P < T`, OR should be the **lattice join** (least upper bound). But `LUB(U, D) = U` (since D < U in this ordering), while the spec's OR table says `U OR D = D`. **The spec's OR table is not a lattice join operation.**

So within the spec itself:
- Tables: not De Morgan dual, not a lattice join
- Prose: claims De Morgan symmetry
- Lattice diagram: implies a different structure still

One of these must give. All three cannot be simultaneously correct.

### How both interpretations fail

| Property | De Morgan interpretation (current code) | Spec-table interpretation |
|---|---|---|
| De Morgan laws hold | ✅ by construction | ❌ breaks for U/P/D |
| Lattice join (monotone, LUB) | ✅ | ❌ breaks (result smaller than inputs) |
| `!(a OR b) == (!a AND !b)` | ✅ | ❌ |
| Associative `(a OR b) OR c == a OR (b OR c)` | ✅ | ❓ needs verification |
| Commutative `a OR b == b OR a` | ✅ | ✅ (table is symmetric) |
| `a OR F == a` (F is identity) | ✅ | ✅ (spec row U col F = U, row D col F = D) |
| `a OR T == T` (T absorbs) | ✅ | ✅ (spec row T all = T) |
| Matches "weak evidence beats unknown" epistemology | ❌ | ✅ |
| Matches Kleene K6 standard references | ❓ need citation check | ❓ need citation check |

Neither is cleanly right. The De Morgan version sacrifices epistemic intuition; the spec-table version sacrifices algebraic structure.

---

## Where this came from

The `ix_types::Hexavalent` type was introduced for new code (per the `feedback_hexavalent_logic.md` memory). The `and` function was implemented carefully against the spec's AND table. The `or` function was implemented as a one-liner via De Morgan, presumably on the assumption that the spec's OR table IS the De Morgan dual. That assumption was never verified against the actual tables.

This is the kind of bug that compiles, tests pass (because the 2 smoke tests only check absorption properties that both interpretations satisfy), and ships to production quietly. It only surfaces when someone is doing a careful migration and reads both the code and the spec side by side.

---

## Three options

### Option A — Trust the spec tables (code is wrong, fix it)

Rewrite `Hexavalent::or` with an explicit 36-cell match expression matching the spec's truth table exactly. Accept non-De-Morgan, non-lattice semantics.

**Pros:**
- Matches the spec "copy tables directly" instruction
- Encodes the epistemic intuition (directional evidence beats unknown)
- The spec is treated as canonical; code catches up to the spec

**Cons:**
- Breaks `!(a OR b) == (!a AND !b)` law — any code relying on this is now wrong
- Not a lattice join; some formal-methods tooling will reject it
- Must update the spec's prose ("Symmetric to AND via De Morgan") which is now misleading
- The spec's lattice diagram is also misleading under this interpretation; either remove or add a note

**Effort:** ~1 hour. Rewrite `or`, add 6×6 table test, fix spec prose.

### Option B — Trust the De Morgan derivation (spec is wrong, fix it)

Keep the current code. Update the spec's OR table to be the De-Morgan-dual of the AND table. This is lattice-consistent and algebraically clean.

**Pros:**
- Code unchanged; no downstream correctness concerns
- De Morgan, associativity, lattice join all hold
- Matches the spec's own prose claim
- Matches the spec's own lattice diagram
- Easier to reason about formally

**Cons:**
- Sacrifices the epistemic intuition ("U OR weak-evidence" stays U)
- The spec's prose author presumably had a reason for the table they wrote; overriding that reason without understanding it is presumptuous
- Requires modifying `governance/demerzel/logic/hexavalent-logic.md` which is the Demerzel-canonical source (cross-repo impact)

**Effort:** ~0.5 hour. Spec edit only, no code change. But the spec change needs Demerzel buy-in since it's the canonical source for the whole GuitarAlchemist ecosystem.

### Option C — Both code and spec are wrong, design a new OR

The real insight: the spec wants a *non-monotone* OR operator where directional evidence beats unknown, but the rest of the spec (lattice diagram, De Morgan claim) assumes a lattice-monotone OR. These are mutually exclusive. Neither the spec's table nor the De Morgan derivation is "right" — they're each internally inconsistent with other parts of the framework.

A third option: define two OR operators and pick the right one per use case:

- `Hexavalent::or` — lattice-monotone, De Morgan dual of AND (for formal reasoning, governance policy composition)
- `Hexavalent::or_evidential` — spec-table, non-monotone (for belief revision, where directional evidence should prevail)

Downstream consumers explicitly choose which semantics they want.

**Pros:**
- Acknowledges the genuine tradeoff rather than papering over it
- Both use cases are supported
- Name carries the semantics, reducing surprise

**Cons:**
- Doubles the API surface for ORs (and presumably ANDs too?)
- Consumers may pick the wrong one silently
- The spec needs significant rework to explain both

**Effort:** ~1 day. Full rewrite of hexavalent operators, spec expansion, migration guidance.

---

## Recommendation

**Option B (trust De Morgan, fix the spec)** is my recommendation, for three reasons:

1. **Code churn is zero.** No downstream consumers need to be touched. The ix-context and Context DAG work currently in progress uses Hexavalent labels but not directly the OR operator; all consumers get correct behavior without changes.
2. **The spec's prose already claims De Morgan symmetry.** Fixing the tables to match the prose is lower-friction than rewriting the prose, lattice diagram, and the implicit formal properties to match the tables.
3. **Lattice-consistent OR is compositional.** The harness primitives roadmap has `ix-middleware` composing governance verdicts, `ix-approval` classifying actions, and `ix-session` tracking belief transitions. All of these benefit from an OR that's a well-behaved lattice join. A non-monotone OR breaks the composition.

But this is a cross-repo decision — the spec lives in `governance/demerzel/`, which is a submodule with its own governance. Any change to `hexavalent-logic.md` is a Demerzel change and should be reviewed accordingly.

**Alternative if cross-repo coordination is expensive:** Option A (rewrite the code to match the spec). It's a one-file edit in `ix-types`, the migration pain is localized, and it defers the cross-repo conversation. But it bakes non-lattice semantics into the ecosystem's foundational truth-value type, which makes primitives #2, #3, and #7 of the harness primitives meta-doc harder to reason about.

---

## What blocks on this decision

The hexavalent migration (Item A in the harness primitives meta-doc) cannot proceed cleanly until this is resolved, because:

1. **The migration would encode the bug into more places.** Currently only ix_types has the suspect OR. After migration, `ix-governance` modules that use `or` chains for governance reasoning would inherit the bug. Shipping the migration without a decision makes the eventual fix harder.

2. **`ix-approval` (primitive #3) depends on composing hexavalent verdicts.** The prompt's blast-radius classifier logic is `Tier 2 if and(edit_scope, trajectory_stability, context_confidence) >= Probable`. Which OR/AND semantics applies there is a load-bearing question.

3. **Tests cannot assert correctness** without knowing which interpretation is canonical. Any 6×6 table test has to match *something*, and right now there are two "somethings" that disagree.

---

## Open questions for the decision

1. Which interpretation does Demerzel's governance team actually want? (Cross-repo question.)
2. Are there any OTHER hexavalent consumers in TARS or GA (the F# and C# repos in the ecosystem) that have already made an assumption one way?
3. Are there any OR cells beyond the U/P/D subspace that also disagree? (I verified four cells; a full 6×6 comparison is needed.)
4. What does the Kleene K6 literature (the spec cites it) actually specify for OR? Is one interpretation closer to the canonical academic source?
5. Should the fix (whichever way) ship as a breaking change in `ix-types`, or with a deprecated `or` alongside a new `or_v2`?

---

## Decision needed from

Whoever owns the Demerzel governance framework's hexavalent-logic.md spec (ultimately `governance/demerzel/` submodule maintainer). Within ix, the `ix-types` crate owner.

Next session should either:
- Resolve the decision in 30 minutes and proceed with the hexavalent migration per whichever option was chosen
- Escalate to Demerzel via Galactic Protocol directive if cross-repo buy-in is needed
- Defer the migration indefinitely and document the OR inconsistency as known

---

## References

- `crates/ix-types/src/lib.rs:105-107` — current De-Morgan-derived `or`
- `crates/ix-types/src/lib.rs:78-101` — current `and` (spec-faithful)
- `crates/ix-types/src/traits.rs:316-330` — current tests (only smoke tests)
- `governance/demerzel/logic/hexavalent-logic.md` — canonical spec (self-inconsistent)
- `docs/brainstorms/2026-04-10-ix-harness-primitives.md` Item A — the migration plan this blocks
- `feedback_hexavalent_logic.md` memory — ecosystem-wide guidance
