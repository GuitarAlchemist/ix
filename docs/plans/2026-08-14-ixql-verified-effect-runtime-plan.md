---
title: IXQL verified effect runtime tracer bullet
type: arch
status: active
date: 2026-08-14
reversibility: additive public seam; adoption by production callers or authoritative adapters requires a separate decision
revisit-trigger: a strict-path negative control passes unexpectedly, a receipt cannot be replayed byte-for-byte, or a production caller needs a capability this seam rejects
---

# IXQL Verified Effect Runtime Plan

**Status:** tracer-bullet implementation complete; independent review and production adapters remain.

## Implementation evidence

- `ix-ixql` now exposes the phased `compile -> verify -> plan -> commit` seam.
- Strict verification covers capability allowlists, statement/effect budgets, freshness
  evidence, and complete mutation metadata.
- The in-memory effect adapter commits an entire batch atomically, detects idempotency
  conflicts, checks expected state, and emits deterministic receipts.
- `ix-pipeline` now offers an opt-in strict path with content-addressed cache identities and
  cooperative cancellation; no production caller has adopted it yet. Real same-level
  parallelism also fixes the shared legacy path.
- Focused compiler/runtime tests, strict clippy, the full workspace test suite, Streeling,
  and the supervised-loop preflight pass. On Windows, the full workspace suite requires
  `CARGO_BUILD_JOBS=1` to avoid linker memory exhaustion.

This evidence does **not** establish production readiness. `run_source` remains a legacy
compatibility path, cancellation cannot preempt an already-running opaque closure, freshness
attestations are supplied by the caller, and only the in-memory effect adapter exists. No
Postgres adapter, DuckDB authority, Gaia bus integration, installation, or deployment was
added.

## One-way-door authorization and compatibility

The operator explicitly accepted the seven-part architecture in this plan on 2026-08-14,
including the new public compiler, verifier, runtime-control, effect-plan, adapter, and receipt
types. That authorizes their additive tracer-bullet introduction, subject to independent
Standards and Spec approval before commit.

The existing `execute()` cache format remains byte-for-byte compatible with its former
`pipeline:{node}:{fnv64}` identity. Only the new opt-in `execute_with_options()` path emits the
versioned `pipeline:sha256:{digest}` identity, so this change does not migrate or invalidate an
existing cache namespace. Deprecating `run_source`, adopting the strict runtime in a production
caller, or changing either cache namespace remains a separate one-way-door decision.

Strict planning currently rejects model invocation and compound operations. Neither has a
receiptable effect adapter, so evaluating either while producing an `EffectPlan` would create an
untracked effect or silently discard one. Supporting them requires a separately reviewed adapter,
idempotency contract, authority check, and receipt shape.

Schema values are necessarily checked during evaluation because they do not exist during pure
verification. The verifier's responsibility is to bind a verified program to the expected
schema-gate identity; the evaluator must prove that exact gate is installed before planning.

## Problem

IXQL can already parse and execute the governance binding/record dialect, but its current
`Executor::run_source` interface combines parsing, checking, evaluation, and persistence.
That makes capabilities implicit, permits a later failure after an earlier write, and leaves
cache identity and cancellation to a separate runtime with incomplete contracts.

Gaia needs a small, inspectable seam that keeps language intent separate from runtime effects
and never mistakes DuckDB analytics for distributed authority.

## Constraints

- Compilation and verification are pure and deterministic.
- No effect reaches an adapter before the whole program passes verification and evaluation.
- Every mutation declares authority, idempotency, expected state, compensation, and receipt
  identity before it can run under strict policy.
- `ix-pipeline` owns DAG execution, parallelism, cache identity, and cooperative cancellation.
- DuckDB/IX remains an advisory analyst's bench.
- CAS, leases, fencing, and multi-process authority belong to an authoritative-store adapter.
- Gaia keeps exactly `register`, `send`, `inbox`, `ack`, `heartbeat`, and `handoff`.

## Design It Twice

### A. Minimal compiler interface

```rust
compile(source) -> Result<TypedProgram, Diagnostics>
verify(program, policy) -> Result<VerifiedProgram, Diagnostics>
evaluate(program, inputs) -> Result<EffectPlan, Diagnostics>
commit(plan, adapter) -> Result<ExecutionReceipt, CommitError>
```

This has the deepest caller-facing interface and makes phase ordering explicit. Its limitation
is that extension data must live in the program, policy, plan, or receipt types rather than new
entry points.

### B. Unified engine interface

```rust
Engine::run(source, policy, adapter) -> Result<ExecutionReceipt, RunError>
```

The common case is trivial, but the interface hides phase boundaries that Gaia must inspect,
cache, review, and replay independently. Rejected as the primary seam; it may remain a safe
convenience wrapper after the phased interface exists.

### C. Extensible plugin interface

```rust
Compiler<Grammar, TypeSystem, CapabilityResolver>
Verifier<PolicySet>
Runtime<Scheduler, Cache, EffectAdapter, ReceiptStore>
```

This maximizes variation but exposes implementation choices and produces a shallow public
surface before two real adapters exist. Rejected for the tracer bullet.

## Decision

Use **A** as the external seam, with internal ports/adapters from **C** only where two concrete
implementations exist. Keep the existing `run_source` as a compatibility wrapper during the
migration, but route strict Gaia execution through the phased interface.

## Vertical slices

1. Pure compiler emits typed program metadata, capability requirements, deterministic
   diagnostics, and a source digest.
2. Verifier checks capability allowlists, statement/effect budgets, forbidden effects, and
   declared freshness evidence.
3. Evaluation stages effects; a commit adapter applies one effect batch and emits a receipt.
4. `ix-pipeline` cache keys include logic identity; cancellation is cooperative and observable.
5. DuckDB, authoritative-store, and Gaia-bus adapters implement the contracts without changing
   the language grammar.

Each slice starts with a public-seam failure test and is independently reviewable.

## Adapter contracts

These are behavioral contracts, not permission to install or connect infrastructure.

| Adapter | Minimal interface | Required guarantees | Explicit non-guarantees |
| --- | --- | --- | --- |
| Analyst bench | `query(verified_query, immutable_inputs) -> AdvisoryArtifact` | read-only inputs, units/nulls/provenance, deterministic replay where declared | authority, locks, workflow scheduling |
| Effect adapter | `commit(effect_plan) -> ExecutionReceipt` | atomic batch or fail before mutation, idempotency conflict detection, expected-state check, authority and receipt preservation | consensus unless the concrete adapter proves it |
| Authoritative store | effect adapter plus CAS/lease/fence primitives | durable monotone fencing, bounded leases, atomic CAS, defined crash/partition behavior | analytics or agent reasoning |
| Gaia bus | the existing six verbs only | `register`, `send`, `inbox`, `ack`, `heartbeat`, `handoff`; provenance and zero implicit authority transfer | new verbs, automatic routing/spend/approval |

DuckDB/IX supplies the analyst-bench adapter. The first effect adapter is the atomic in-memory
implementation used for tests. A Postgres adapter is justified only when a live multi-process
writer requires CAS/leases/fencing; until then it remains a named port, not a dependency. The
Gaia adapter is deferred to M4 and must consume a verified advisory artifact without widening
the bus.

## Reversibility and revisit triggers

The phased interfaces are additive until `run_source` deprecation, so the first slices are
two-way doors. A grammar change, cross-repo schema, cache-format change, or authoritative-store
choice is a one-way door and requires separate explicit review.

Revisit the seam if two real effect adapters cannot implement the same batch/receipt semantics,
or if corpus coverage shows that static capability extraction cannot describe existing IXQL
without grammar widening.
