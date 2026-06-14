# Contract: pipeline provenance record (`ix-lock/v2`)

**Version:** v0.1.0 (DRAFT — Phase 0; freeze only at the named Phase 4 milestone)
**Date:** 2026-06-07 · **Owner:** ix (`ix-pipeline`) · **Plan:** `docs/plans/2026-06-07-chain-of-evidence-v1.md`

Defines the on-disk provenance record IX emits per pipeline run, so a consumer can
answer *"did this output come from these inputs, under this skill?"*. Additive
extension of `ix-lock/v1` (`crates/ix-pipeline/src/lock.rs`).

## Locations

- `ix.lock` (workspace-relative, latest run; overwritten each run — back-compat).
- `state/thinking-machine/provenance.jsonl` (append-only durable trail; one JSON
  object per run, keyed by `run_id`; torn-line-tolerant per the `hits.jsonl` convention).

## Schema

```jsonc
// LockFile (ix-lock/v2)
{
  "schema": "ix-lock/v2",
  "generated": "<ISO-8601 UTC>",
  "run_id": "<ulid|uuid>",          // NEW — unique per run
  "spec_hash": "sha256:<hex>",      // NEW — sha256 of the canonicalized PipelineSpec
  "stages": [ /* LockedStage */ ]
}

// LockedStage
{
  // v1 fields — UNCHANGED (locked field set; renames are breaking → v3):
  "skill": "<dotted skill name>",
  "args_hash": "fnv1a64:<hex>",       // over canonicalized TEMPLATE args (spec)
  "deps": ["<stage_id>", ...],
  "duration_ms": 0,
  "cache_hit": false,
  // v2 additions (optional for forward-compat readers):
  "resolved_args_hash": "fnv1a64:<hex>", // canonicalized RESOLVED args (post from-ref)
  "output_hash": "sha256:<hex>",         // canonical-JSON of the stage output Value
  "inputs": [ /* InputBinding */ ],
  "certainty": null                       // RESERVED — unpopulated in v1 (see plan §Deferred)
}

// InputBinding — one per consumed cross-stage edge
{
  "name": "<consumer input arg name>",
  "from_stage": "<sourceNodeId | __input__:KEY>",
  "from_key": "<output key within source output ('*' = whole output)>",
  "upstream_output_hash": "sha256:<hex>"  // == output_hash of from_stage in the same run
}
```

## Locked fields & hashing rules (one-way doors — signed off 2026-06-07)

1. **`output_hash` / `spec_hash` / `upstream_output_hash` use sha256**, prefixed
   `sha256:`. (Distinct from the FNV-1a64 cache/template hashes, which stay FNV for
   back-compat.) Algorithm change for *new* records only via a versioned prefix.
2. **Canonical-JSON for all hashes = `lock.rs::canonicalize` exactly** (key sort +
   number/whitespace normalization). Frozen; any change orphans persisted hashes.
3. **`ix-lock/v2` is additive.** New optional fields are non-breaking; renames /
   removals / type changes require a v3 bump and cross-consumer coordination.

## Invariants (the verifiable chain)

- **Edge integrity:** for every `InputBinding`, `upstream_output_hash` equals the
  `output_hash` recorded for `from_stage` in the same run. A `verify_chain()` over a
  `LockFile` returns `Ok` iff this holds for all edges.
- **Determinism:** a deterministic spec + identical seed inputs ⇒ identical
  `output_hash` for every stage across runs.
- **Localized change:** mutating one seed input changes the `resolved_args_hash` /
  `output_hash` of fed stages and the `upstream_output_hash` of their consumers, and
  nothing else.

## Out of scope for v0.1

Signing the trail (`ix-harness-signing` — designed as a later pure-add), certainty
propagation (`certainty` reserved but null), provenance-gated execution
(`verify_chain` records, does not block), and System-A lineage unification.
