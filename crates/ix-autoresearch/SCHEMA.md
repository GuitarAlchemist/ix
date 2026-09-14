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
- **Deterministic replay**: running the same log through a deterministic consumer (e.g. `hari-from-ix-autoresearch` then `hari-core replay`) MUST produce identical output for identical input. IX side: same seed ⇒ same `config_hash` sequence, tested in `tests/jsonl_contract.rs`. Hari side: committed run reports regenerate byte-for-byte (after CRLF normalisation) from committed logs, tested by `committed_run_reports_regenerate_byte_for_byte_from_committed_logs` in hari's `crates/hari-extractor/tests/ix_autoresearch_replay.rs`.
- **Contradictory findings preserved**: when two iteration events carry the same derived `claim` — the same `config_hash` judged against the same incumbent (see the `claim` row below) — and different `accepted` values, the consumer (Hari) should preserve the contradiction as `HexValue::Contradictory` rather than averaging. **Status: untested against, and unobserved on, real IX data.**
  - **What can and cannot produce one.**
    - The incumbent is part of the claim because a repeat of the same `config_hash` alone is *not* a contradiction. Under Greedy (`candidate_reward > prev_reward`) the incumbent's reward never decreases. A repeat of an already-accepted config is therefore correctly rejected once the incumbent has moved to it or past it. Keyed on `config_hash` alone, every such repeat would read as a spurious contradiction.
    - With the incumbent in the claim, a deterministic target under Greedy cannot produce one at all, since the same config against the same incumbent gets the same reward and the same decision. A genuine conflict needs a nondeterministic evaluator.
    - Under SA or random search, `accepted` is not an improvement test, so a conflict there reflects the accept rule rather than the evidence.
  - **Measured on the grammar target only.** It perturbs by continuous Gaussian noise, so no `config_hash` repeats within a run: 500 of 500 claims are distinct under Greedy and under SA. That is pinned by `a_seeded_grammar_run_never_repeats_a_claim_so_nothing_is_contradictory` in `tests/jsonl_contract.rs`. Recorded grammar runs replayed through Hari end with zero `Contradictory` beliefs (GuitarAlchemist/hari#37).
  - **Other targets were not measured.** `target_chatbot` clamps its perturbation to bounds, and `target_optick` falls back to the renormalised weights on a degenerate Dirichlet, so either can repeat a config. Both declare deterministic evaluation, though, so by the reasoning above a repeat there should not conflict either.
  - **The only exercises are synthetic.** In hari (`crates/hari-extractor/tests/ix_autoresearch_replay.rs`), one test has a genuine conflict (same config, same incumbent, rewards straddling the incumbent's) and one pins that a Greedy re-evaluation of the incumbent is not a conflict. Here, the two-line log in `contradictory_findings_preserved_in_derived_view`. None of these is evidence about IX runs.
- **Crash tolerance**: trailing parse failure is silently discarded as crash-truncation; mid-stream parse failure is a hard error.

## Layer 2 — Derived semantic event view

This is the projection Hari (and any other epistemic consumer) sees per `iteration` line. It is **derived from layer 1 fields**, NOT a separate emitted format. Documented here so consumers in other languages can implement the same projection without re-reading the IX Rust source.

For each `iteration` line, the derived view is:

| Derived field    | Type                                     | Source from layer 1                                                                              |
| ---------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `event_id`       | string (monotone-ordered within a log)   | `format!("{run_id}#{iteration}")` — `run_id` from `run_start`, `iteration` from this line        |
| `timestamp`      | RFC3339                                  | `iteration.timestamp`                                                                            |
| `target`         | string                                   | `run_start.target` (propagated to every derived event in the run)                                |
| `claim`          | string                                   | `format!("{target}/config-{config_hash_short}-is-an-improvement-over-{incumbent_short}")`. `target` is `run_start.target` verbatim. `config_hash_short` is the first 12 hex characters after stripping the `autoresearch:` prefix. `incumbent_short` is the same shortening of the *previous* iteration line's `previous_hash` (the config this candidate was judged against), or `baseline` for a log's first iteration. See hari `crates/hari-extractor/src/ix_autoresearch.rs`. |
| `evidence`       | array of `{kind, value}` objects         | `[{kind: "reward", value: <reward>}, {kind: "elapsed_ms", value: <elapsed_ms>}, {kind: "config_hash", value: <full hash>}, ...]` |
| `confidence`     | float in [0.0, 1.0]                      | `if accepted { 0.66 } else if error.is_some() { 0.10 } else { 0.33 }` — pegged to HexValue rank  |
| `contradicted_by`| array of `event_id` references           | The set of *prior* `event_id`s in the same log whose `claim` (config *and* incumbent) matches this line's `claim` AND whose `accepted` differs. Empty for the first occurrence. Computed by the consumer. |
| `disposition`    | enum `pending` \| `confirmed` \| `refuted` \| `contradictory` | `if contradicted_by.is_empty() && !accepted { "refuted" } else if contradicted_by.is_empty() && accepted { "confirmed" } else if !contradicted_by.is_empty() { "contradictory" } else { "pending" }` |

### Why the projection lives in the consumer

The raw IX log is the canonical wire format. The derived view is a *reading discipline*, not a separate emission, because:

- `contradicted_by` requires looking across multiple lines (consumer scope).
- `confidence` is a downstream interpretation, not a measurement (each consumer may pick its own mapping; ours is documented above).
- Keeping IX free of HexValue / belief vocabulary preserves the layer boundary — IX is the experiment runner; Hari is the epistemic state layer (per Hari docs).

### Example projection

This example is **synthetic**. It needs a nondeterministic evaluator: the same config is judged twice against the same incumbent (`autoresearch:9f8e7d6c5b4a...`, reward 0.41) and gets rewards on opposite sides of it. A deterministic target cannot produce this (see the acceptance criterion above). Under Greedy the first evaluation (0.40) is rejected and the incumbent stays. The second (0.42) is accepted.

```jsonl
{"event":"iteration","schema_version":1,"iteration":5,"timestamp":"2026-05-17T10:00:05Z","config":{"...":"..."},"config_hash":"autoresearch:abc123def456...","score":{"...":"..."},"reward":0.40,"accepted":false,"previous_hash":"autoresearch:9f8e7d6c5b4a...",...}
{"event":"iteration","schema_version":1,"iteration":11,"timestamp":"2026-05-17T10:00:11Z","config":{"...":"..."},"config_hash":"autoresearch:abc123def456...","score":{"...":"..."},"reward":0.42,"accepted":true,"previous_hash":"autoresearch:abc123def456...",...}
```

Both lines' preceding iteration lines (4 and 10) carry `previous_hash: "autoresearch:9f8e7d6c5b4a..."`, so both candidates were judged against the same incumbent. The derived view is:

```json
[
  {
    "event_id": "01958d6a-.../iteration-5",
    "timestamp": "2026-05-17T10:00:05Z",
    "target": "ix_autoresearch::target_chatbot::ChatbotTarget",
    "claim": "ix_autoresearch::target_chatbot::ChatbotTarget/config-abc123def456-is-an-improvement-over-9f8e7d6c5b4a",
    "evidence": [{"kind":"reward","value":0.40},{"kind":"elapsed_ms","value":12},{"kind":"config_hash","value":"autoresearch:abc123def456..."}],
    "confidence": 0.33,
    "contradicted_by": [],
    "disposition": "refuted"
  },
  {
    "event_id": "01958d6a-.../iteration-11",
    "timestamp": "2026-05-17T10:00:11Z",
    "target": "ix_autoresearch::target_chatbot::ChatbotTarget",
    "claim": "ix_autoresearch::target_chatbot::ChatbotTarget/config-abc123def456-is-an-improvement-over-9f8e7d6c5b4a",
    "evidence": [{"kind":"reward","value":0.42},{"kind":"elapsed_ms","value":14},{"kind":"config_hash","value":"autoresearch:abc123def456..."}],
    "confidence": 0.66,
    "contradicted_by": ["01958d6a-.../iteration-5"],
    "disposition": "contradictory"
  }
]
```

Hari's belief network then consolidates both as `HexValue::Contradictory` for that one proposition.

## Versioning

- `schema_version: 1` — current. Bumped only on non-additive changes.
- New optional fields on layer 1 events are NOT a version bump (annotated with `#[serde(default)]`).
- The derived view does not carry a version; it is a function of the layer 1 contract.

## Consumers

- **Hari** (`crates/hari-extractor` `hari-from-ix-autoresearch` bin; projection in `src/ix_autoresearch.rs`) reads layer 1 directly. The mapping it uses is documented in that file's module-level comment. It departs from the confidence column above in one place: an errored line becomes `Unknown` (no evidence either way), not a low-confidence refutation. `--report` replays the stream under Hari's arms beside IX's own `accepted` flag. Sample reports live in hari's `fixtures/ix-real-or-synthetic/`.
- **agent-blackbox** consumes the resulting `ResearchReplayReport` JSON (the "belief diff") as evidence — see Workflow 3 in `agent-blackbox/docs/ix-real-problems-plan.md`.

## Validation

- JSON schema: `crates/ix-autoresearch/jsonl-event.schema.json`.
- Round-trip integration test: `crates/ix-autoresearch/tests/jsonl_contract.rs`.
- Example producer: `crates/ix-autoresearch/examples/grammar_pinned_contract.rs`.
