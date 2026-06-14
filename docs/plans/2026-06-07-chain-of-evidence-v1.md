# Chain of Evidence v1 — minimal provenance for IX ML pipelines

**Date:** 2026-06-07 · **Status:** approved, pre-implementation · **Scope:** ix-pipeline

## Problem

Whoever later asks *"did this pipeline output actually come from those inputs, run
by that skill, under that constitution?"* — an operator debugging a run, an
auditor verifying a governance claim, a future agent deciding whether to trust a
derived artifact — cannot answer it today. PR #84 made producer→consumer **bindings**
provably valid at *compile* time (output schemas), but at *runtime* nothing proves
the binding was honored. Three distinct gaps (not conflated):

- **(a)** no materialized per-artifact provenance record — `PipelineResult`/`NodeResult`
  (`executor.rs:31-50`) is the only lineage object and is dropped after the run;
  `ix.lock` hashes the *static template* args (`lock.rs:52`), so two runs with
  different upstream-fed values produce identical lock entries.
- **(b)** no content-hashing of intermediate data — outputs are `Value`, echoed to
  stdout, never hashed.
- **(c)** no certainty propagation through data-flow.

## Decision

**Build (a)+(b) now; defer (c).** (a)+(b) are one cheap, high-value unit, almost
entirely reuse. (c) has no reusable substrate (see "Deferred" below) and would ship
uncalibrated magic numbers — green-but-dead.

## Design (v1)

**Integration point — reuse, don't add an executor hook.** The single attach point
is the stage compute closure `lower.rs:102-130` — the only place holding `stage_id`,
`skill`, the **resolved** args (post `resolve_from_refs`), the upstream `inputs` map
(lineage), and the **output**. The governance gate already fires here, proving the
seam carries side-effects. Inject `Arc<dyn ProvenanceSink>` per node exactly as
`SharedGate` is injected (`gate.rs:41-47`), wired in `run()` (`pipeline.rs:480/496`)
alongside `ConstitutionGate`. Do **not** add a per-stage callback to `execute()`.

**Record — extend `LockedStage` → `ix-lock/v2` (additive):**

```
LockFile (v2)        { schema: "ix-lock/v2", generated, run_id, spec_hash, stages: [LockedStage] }
LockedStage (v2)     { skill, args_hash, deps, duration_ms, cache_hit,   // v1, unchanged
                       resolved_args_hash, output_hash, inputs: [InputBinding] }   // NEW
InputBinding         { name, from_stage, from_key, upstream_output_hash }
```

- `resolved_args_hash` = `fnv1a64:<hex>` over canonicalized **resolved** args (reuse
  `lock.rs:canonicalize`; only the input differs vs v1's template hash).
- `output_hash` = `sha256:<hex>` over canonical-JSON of the stage output (one-way door §below).
- `inputs[]` are free from `gather_inputs` + `input_map` `(sourceNode, outputKey)`;
  `upstream_output_hash` links each consumed edge to the producer's `output_hash` —
  the verifiable "this output came from these inputs" chain.

**Storage.** Keep writing `ix.lock` (latest run) where it's written today; **also**
append the v2 record as one line to `state/thinking-machine/provenance.jsonl`
(append-only, torn-line-tolerant, best-effort-never-block — same convention as
`hits.jsonl`). `run_id` keys the durable trail.

**Certainty field.** Reserve an **optional** `certainty` slot, left unpopulated in v1.

## Reuse map

- `lower.rs` compute closure → the sink attaches here (full tuple).
- `SharedGate` injection pattern → copy for `ProvenanceSink`.
- `LockFile`/`LockedStage` + `canonicalize`/`hash_json` → extend in place.
- `gather_inputs`/`input_map` → lineage edges for free.
- `hits.jsonl` JSONL conventions → `provenance.jsonl`.
- Constitution sha256-pin pattern → template for pinning the v2 schema (CI gate).
- Hexavalent T/P/U/D/F/C → the scale the deferred certainty field will use.

## One-way doors (signed off 2026-06-07)

| Door | Decision | Reversibility | Revisit trigger |
|---|---|---|---|
| Output/integrity hash algo | **sha256** (`sha256:<hex>`) — aligns with constitution-pin + `ix-harness-signing`, NOT the FNV cache key | One-way for persisted history; two-way forward only via a versioned `algo:` prefix | A cryptographic weakness in sha256, or a signing requirement forcing a different primitive |
| Canonical-JSON for hashing | Reuse `lock.rs:canonicalize` **exactly** (key sort, number/whitespace normalization) — frozen | One-way once hashes persist | Never, except a v3 break |
| `ix-lock/v2` schema | Additive bump; new fields optional | Two-way while additive; one-way for renames/removals/type changes (→ v3) | A field rename/removal need |
| `run_id` format (ulid vs uuid) | pick one (low stakes) | Two-way-ish (mixed logs tolerable) | — |
| `provenance.jsonl` location | `state/thinking-machine/` | Two-way (path move + migration cheap) | — |

A `docs/contracts/2026-06-07-provenance-record.contract.md` holds the v2 schema
(draft v0.1; freeze only at its named Phase 4).

## Verifiable success criteria

1. **Cardinality** — a multi-stage run emits a v2 file whose `stages.len()` equals the
   executed-stage count, each with non-empty `output_hash`/`resolved_args_hash`.
2. **Determinism** — re-running a deterministic spec yields identical `output_hash`es.
3. **Input-change detection** — mutating one seed input dirties *exactly* its
   downstream cone's hashes; unaffected stages stay identical.
4. **Edge integrity** — `verify_chain(&LockFile)` confirms every `InputBinding.upstream_output_hash`
   equals the `output_hash` recorded for `from_stage`; `Ok` on a clean run.
5. **Append-only history** — N runs → ≥N parseable lines, distinct `run_id`s, torn-line tolerant.
6. **Back-compat** — v1-field consumers still find `skill/args_hash/deps/duration_ms/cache_hit`.

## Deferred / out of scope for v1

- **Certainty propagation (c).** IX has an `Opinion{b,d,u}` carrier and *same-claim*
  fusion (`ix-fuzzy merge_all`) but **no `discount`/derivation operator** to propagate
  certainty *through* a computation, and the `from_truth` 0.6 / `projected` 0.5 are
  unvalidated magic numbers. Building it now = inventing uncalibrated semantics.
  **Revisit when** there is (1) a certainty-gated decision demanding it and (2) an
  `empirical` vs `deterministic` tag to discount against. Then: deterministic skill →
  pass parent opinion through; empirical/llm → SL-discount toward `u`; multi-parent →
  fuse via existing `merge_all`.
- **Signing / Merkle-chaining the trail** — `ix-harness-signing` exists but unwired;
  v1 records are hashed, not signed. Designed so signing is a later pure-add.
- **Gating execution on provenance** — v1 only *records*; `verify_chain` is provided
  but not wired to fail a run.
- **System-A (`ix_pipeline_run {steps}`, ungated) ↔ System-B unification** — v1
  instruments System B (the real thinking machine) only.
- **Data-quality evidence**, **large/binary output streaming**, and fixing the
  `PipelineResult::final_outputs()` dead stub — unrelated, surgical-scope.
