# ix-autoresearch JSONL Event Schema (Hari boundary contract)

**Status:** Phase 1 — IX-side contract pinned. Phase 2 (Hari replay) tracked in `agent-blackbox/docs/ix-real-problems-plan.md` Workflow 3.

This document pins the JSONL event shape that downstream consumers — chiefly Hari's `hari-extractor` `hari-from-ix-autoresearch` adapter — read without custom hand-shaping. It is a **publishable contract**, not just an internal record format. Changes to the **raw** layer must remain additive on minor versions; breaking changes bump `schema_version`.

## Two layers

There are two contract layers, both stable:

1. **Raw kernel JSONL** (this crate emits) — append-only run log, one event per line, three event kinds (`run_start`, `iteration`, `run_complete`). This is what gets written to `<state_dir>/<run-id>/log.jsonl`.
2. **Derived semantic event view** — what Hari and other epistemic consumers see *per iteration line* after projecting. Required fields are listed below in §"Derived event view". This shape exists so consumers can validate without depending on `ix-autoresearch` Rust types.

The JSON schema file (`jsonl-event.schema.json`) covers layer 1 in full. Layer 2 is documented here for downstream implementers.

## Layer 1 — Raw kernel JSONL

Every line is exactly one JSON object. Lines are separated by `\n` (single newline; no `\r\n` even on Windows — see `log.rs`). The discriminator is the `event` field with values `"run_start"`, `"iteration"`, or `"run_complete"`. Every event carries `schema_version: 1`.

### `run_start`

First line of every run. Exactly one per log.

```json
{
  "event": "run_start",
  "schema_version": 1,
  "run_id": "01958d6a-...",        // UUIDv7 (lexicographically time-ordered)
  "timestamp": "2026-05-17T10:00:00Z",
  "target": "ix_autoresearch::target_grammar::GrammarTarget",
  "strategy": { ... },              // serde-tagged Strategy enum
  "seed": 42,
  "git_sha": "<40 hex>" | null,
  "git_sha_reason": null | "not a git checkout" | "git not on PATH" | "malformed sha",
  "baseline_config": { ... },       // target-specific Config payload
  "eval_inputs_hash": null | "<hex>"
}
```

### `iteration`

One per evaluated candidate. This is the line Hari converts to a `ResearchEvent`.

```json
{
  "event": "iteration",
  "schema_version": 1,
  "iteration": 0,                   // usize, monotone within a log
  "timestamp": "2026-05-17T10:00:01Z",
  "config": { ... },                // target-specific Config payload (the candidate)
  "config_hash": "autoresearch:<64 hex>" | "<64 hex>",
  "score": { ... } | null,          // target-specific Score payload; null on eval error
  "reward": <f64> | null,           // scalar projection of score; null on eval error
  "accepted": true | false,
  "previous_hash": "<hash>" | null,
  "error": null | "hard-killed: ..." | "timed out after ..." | "eval failed: ...",
  "elapsed_ms": 12,
  "strategy_state": { ... } | null,  // e.g. { "temperature": 1.23 } for SA
  "cache_hit": false
}
```

### `run_complete`

Last line on graceful exit. Absence means the run was interrupted (replay-tolerant).

```json
{
  "event": "run_complete",
  "schema_version": 1,
  "timestamp": "2026-05-17T10:00:10Z",
  "iterations": 20,
  "accepted": 7,
  "best_iteration": 12 | null,
  "best_reward": 0.85 | null,
  "consecutive_kills_at_abort": null | <usize>,
  "cost": {
    "total_elapsed_ms": 250,
    "cache_hit_count": 1,
    "eval_failure_count": 0,
    "rejected_count": 13
  } | null
}
```

### Acceptance criteria (from `agent-blackbox/docs/ix-real-problems-plan.md` Workflow 3)

- **Append-only**: writers MUST open `O_APPEND`; readers MUST process events in file order.
- **Deterministic replay**: running the same log through a deterministic consumer (e.g. `hari-from-ix-autoresearch` then `hari-core replay`) MUST produce identical output for identical input. Tested in `tests/jsonl_contract.rs`.
- **Contradictory findings preserved**: when two iteration events on the *same* `config_hash` carry different `accepted` values, the consumer (Hari) preserves the contradiction as `HexValue::Contradictory` rather than averaging. Verified by `hari-core`'s combined-evidence semantics (`crates/hari-core/src/lib.rs` §`process_research_trace`).
- **Crash tolerance**: trailing parse failure is silently discarded as crash-truncation; mid-stream parse failure is a hard error.

## Layer 2 — Derived semantic event view

This is the projection Hari (and any other epistemic consumer) sees per `iteration` line. It is **derived from layer 1 fields**, NOT a separate emitted format. Documented here so consumers in other languages can implement the same projection without re-reading the IX Rust source.

For each `iteration` line, the derived view is:

| Derived field    | Type                                     | Source from layer 1                                                                              |
| ---------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `event_id`       | string (monotone-ordered within a log)   | `format!("{run_id}#{iteration}")` — `run_id` from `run_start`, `iteration` from this line        |
| `timestamp`      | RFC3339                                  | `iteration.timestamp`                                                                            |
| `target`         | string                                   | `run_start.target` (propagated to every derived event in the run)                                |
| `claim`          | string                                   | `format!("{target}/config-{config_hash_short}-is-an-improvement")` — see `hari_from_ix_autoresearch.rs` |
| `evidence`       | array of `{kind, value}` objects         | `[{kind: "reward", value: <reward>}, {kind: "elapsed_ms", value: <elapsed_ms>}, {kind: "config_hash", value: <full hash>}, ...]` |
| `confidence`     | float in [0.0, 1.0]                      | `if accepted { 0.66 } else if error.is_some() { 0.10 } else { 0.33 }` — pegged to HexValue rank  |
| `contradicted_by`| array of `event_id` references           | The set of *prior* `event_id`s in the same log whose `claim` matches this line's `claim` AND whose `accepted` differs. Empty for the first occurrence. Computed by the consumer. |
| `disposition`    | enum `pending` \| `confirmed` \| `refuted` \| `contradictory` | `if contradicted_by.is_empty() && !accepted { "refuted" } else if contradicted_by.is_empty() && accepted { "confirmed" } else if !contradicted_by.is_empty() { "contradictory" } else { "pending" }` |

### Why the projection lives in the consumer

The raw IX log is the canonical wire format. The derived view is a *reading discipline*, not a separate emission, because:

- `contradicted_by` requires looking across multiple lines (consumer scope).
- `confidence` is a downstream interpretation, not a measurement (each consumer may pick its own mapping; ours is documented above).
- Keeping IX free of HexValue / belief vocabulary preserves the layer boundary — IX is the experiment runner; Hari is the epistemic state layer (per Hari docs).

### Example projection

Given these two iteration lines (same `config_hash`, different `accepted`):

```jsonl
{"event":"iteration","schema_version":1,"iteration":5,"timestamp":"2026-05-17T10:00:05Z","config":{"...":"..."},"config_hash":"autoresearch:abc123def456...","score":{"...":"..."},"reward":0.42,"accepted":true,...}
{"event":"iteration","schema_version":1,"iteration":11,"timestamp":"2026-05-17T10:00:11Z","config":{"...":"..."},"config_hash":"autoresearch:abc123def456...","score":{"...":"..."},"reward":0.40,"accepted":false,...}
```

The derived view is:

```json
[
  {
    "event_id": "01958d6a-.../iteration-5",
    "timestamp": "2026-05-17T10:00:05Z",
    "target": "target_grammar",
    "claim": "target_grammar/config-abc123def456-is-an-improvement",
    "evidence": [{"kind":"reward","value":0.42},{"kind":"elapsed_ms","value":12},{"kind":"config_hash","value":"autoresearch:abc123def456..."}],
    "confidence": 0.66,
    "contradicted_by": [],
    "disposition": "confirmed"
  },
  {
    "event_id": "01958d6a-.../iteration-11",
    "timestamp": "2026-05-17T10:00:11Z",
    "target": "target_grammar",
    "claim": "target_grammar/config-abc123def456-is-an-improvement",
    "evidence": [{"kind":"reward","value":0.40},{"kind":"elapsed_ms","value":14},{"kind":"config_hash","value":"autoresearch:abc123def456..."}],
    "confidence": 0.33,
    "contradicted_by": ["01958d6a-.../iteration-5"],
    "disposition": "contradictory"
  }
]
```

Hari's BeliefNetwork then consolidates these as `HexValue::Contradictory` for the canonical proposition `target_grammar/config-abc123def456-is-an-improvement`.

## Versioning

- `schema_version: 1` — current. Bumped only on non-additive changes.
- New optional fields on layer 1 events are NOT a version bump (annotated with `#[serde(default)]`).
- The derived view does not carry a version; it is a function of the layer 1 contract.

## Consumers

- **Hari** (`crates/hari-extractor` `hari-from-ix-autoresearch` bin) reads layer 1 directly. The mapping it uses is documented in that file's module-level comment.
- **agent-blackbox** consumes the resulting `ResearchReplayReport` JSON (the "belief diff") as evidence — see Workflow 3 in `agent-blackbox/docs/ix-real-problems-plan.md`.

## Validation

- JSON schema: `crates/ix-autoresearch/jsonl-event.schema.json`.
- Round-trip integration test: `crates/ix-autoresearch/tests/jsonl_contract.rs`.
- Example producer: `crates/ix-autoresearch/examples/grammar_pinned_contract.rs`.
