# Defer a `Lens` trait in `ix-duck` — keep heterogeneous lenses concrete

Status: accepted (2026-06-20)

## Context

`ix-duck` exposes five analyst **lenses** over GA telemetry — `chatbot`, `routing`,
`loops`, `ood`, `maintain`. An architecture review (`/improve-codebase-architecture`,
2026-06-20) noted that `maintain::evaluate` orchestrates four of them by calling each
one's bespoke API individually (`chatbot::check_regressions` → `GateReport`,
`loops::oscillating_loops` → tuples, `ood::flag_ood` → tuples), and that each lens
returns a different result shape (struct / tuple / type-alias) with no common envelope.
A `trait Lens { fn ingest(&Connection) -> Result<usize>; fn signal(&Connection) ->
Signal; }` was raised as a possible deepening: maintain could then iterate lenses
uniformly and a new signal would be "add a `Lens`," not "edit the orchestrator."

## Decision

**Do not introduce a `Lens` trait now.** Keep the lenses concrete and let `maintain`
call each one's API directly. Revisit only when a *sixth* maintain signal actually
needs to be fused.

The same review's **ingestion** finding *was* acted on — the duplicated
graceful-degrade scanner, `sql_list`, and `fnv1a64` were pulled into a private
`telemetry` module (this is the real seam, with four/two existing adapters). The
`Lens` trait is the part deliberately deferred.

## Why (the trade-off)

- **The lenses are genuinely heterogeneous, not parallel.** `chatbot`'s guardrail is a
  *hard* gate (a regression fails the verdict); `ood` is *advisory* (drift only flags);
  `loops` is *iteration-scoped* (correlated to a specific commit via `IterationScope`).
  A single `signal()` shape would either lose this structure or grow options until it
  re-encodes each lens's specialness — an abstraction that earns nothing.
- **It would make shallow lenses *look* deep without adding leverage.** A trait over
  four call-sites that one function makes is a hypothetical seam, not a real one
  (one adapter ≠ a seam). Per the deletion test, deleting the trait today would just
  move the four calls back inline — complexity moves, it doesn't concentrate.
- **`maintain`'s per-lens error framing is load-bearing.** `MaintainError` distinguishes
  `Chatbot` ("guardrail lens error") / `Loops` ("convergence lens error") / `Ood`
  ("drift lens error") via distinct `From` impls — which is *why* the ingestion refactor
  kept the per-lens `*Error` types instead of collapsing them. A uniform `Lens` trait
  pushes against that grain.
- **CLAUDE discipline.** Karpathy r2 (simplicity, no speculative future-proofing) and
  YAGNI both say: don't abstract four heterogeneous call-sites on spec.

## Consequences

- Adding a *fifth* maintain signal today means editing `maintain::evaluate` directly.
  That is the accepted cost — it is a few lines and keeps each lens's contract explicit.
- The ingestion seam (`telemetry::{collect_dir, sql_list, fnv1a64, LensError}`) stands
  on its own and does not depend on this decision.

## Revisit trigger

When a **sixth** lens/signal needs fusing into the maintain verdict — at that point the
orchestrator's per-lens glue is paying real recurring cost, the trait would *concentrate*
complexity (deletion test flips to "yes"), and the heterogeneity will have shown its
actual axes (hard/advisory/scoped), so the trait can be shaped to them rather than guessed.
