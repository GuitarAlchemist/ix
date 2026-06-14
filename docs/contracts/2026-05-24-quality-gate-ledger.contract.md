# Quality Gate Ledger v1 — Cross-Repo Contract

**Status:** Draft (Phase 0)
**Date:** 2026-05-24
**Schema id:** `quality-gate-ledger-v1`
**Owners:** ix (Rust producer/consumer) + sentrux (Rust producer) + ga (PowerShell producer + dashboard consumer)
**Reversibility:** Two-way door (we can rev `schema_version` without freezing the file). One-way once we publish a `v1` aggregator that downstream dashboards bind to.

## Why

Today three systems write their own gate-pass/fail history with different schemas:

| System | Path (per-repo) | Shape |
|---|---|---|
| ix-quality-trend | `state/quality/gate-ledger.jsonl` | chatbot-PR-shaped row (`pr`, `branch`, `gates.{tests,agentToolReview,octoReview,tribunal}`, `decision`) — see `ga/docs/schemas/gate-ledger.schema.json` |
| sentrux regression gate | (none — emits stdout + exit code) | n/a |
| GA dashboard quality runs | `state/quality/{chatbot-qa,council,e2e,test-plans}/<date>.json` | per-domain, schema-tolerant; aggregated by `ix-quality-trend` |

Aggregators have to special-case each source. New sources mean code changes in every consumer. We want **one append-only JSONL substrate per repo**, with a unified shape, that any producer can append to and any consumer can fold over.

## Schema (one entry per line in `state/quality/gate-ledger.jsonl`)

```json
{
  "schema_version": 1,
  "schema": "quality-gate-ledger-v1",
  "id": "01HXYZ...",
  "run_at": "2026-05-24T18:00:00Z",
  "source": "ix-quality-trend",
  "domain": "structural",
  "decision": "pass",
  "metric": {
    "name": "quality_signal",
    "value": 3015.0,
    "threshold": 2500.0,
    "trend": "improving"
  },
  "evidence": {
    "kind": "file",
    "ref": "state/quality/embeddings/2026-05-24.json"
  },
  "supersedes": [],
  "operator_ack": null,
  "extra": {}
}
```

### Field reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | integer | yes | Set to `1`. Absence = legacy v0 (PR-shaped chatbot row). |
| `schema` | string | yes | Always `"quality-gate-ledger-v1"`. Lets one file mix versions during the transition. |
| `id` | string | yes | UUID v7 preferred (sortable). UUID v4 acceptable. |
| `run_at` | string (RFC3339 UTC) | yes | When the gate was evaluated, not when the row was appended. |
| `source` | string | yes | Producer id. Enum (extensible): `ix-quality-trend`, `sentrux`, `chatbot-qa`, `council`, `e2e`, `test-plans`, `tribunal`. Add new values by PR. |
| `domain` | string | yes | What was measured. Open enum, current values: `structural`, `tests`, `invariants`, `coverage`, `chatbot`, `routing`, `voicings`, `harness`, `governance`. |
| `decision` | string | yes | `pass`, `fail`, `warn`, `skip`. `skip` covers degraded environments (e.g., backend unavailable). |
| `metric.name` | string | yes | Producer-defined metric label (e.g., `quality_signal`, `cycles`, `coverage_pct`, `pass_pct`, `findings_count`). |
| `metric.value` | number | yes | Observation. `f64` on the wire. |
| `metric.threshold` | number | no | The threshold the gate was checking against. Omit when the decision is qualitative. |
| `metric.trend` | string | no | `improving`, `stable`, `degrading`, `unknown`. Producer fills if it has history; otherwise omit. |
| `evidence.kind` | string | no | `url`, `file`, `sha`, `run-id`, `pr`. |
| `evidence.ref` | string | no | The actual reference (URL, repo-relative path, commit SHA, GH Actions run id, PR number-as-string). |
| `supersedes` | array of strings | no | List of prior `id`s this entry replaces (e.g., rebaseline events). Empty array OK. |
| `operator_ack` | object \| null | no | When a human ack'd a `fail`/`warn`. Shape: `{ "by": "spareilleux", "at": "2026-05-24T18:30:00Z", "note": "..." }`. |
| `extra` | object | no | Producer-specific extension blob. Consumers MUST ignore unknown keys. Use this for source-shaped detail (e.g., the full chatbot PR-row goes here when `source=chatbot-qa-merge`). |

### Backward compatibility

Old chatbot-PR-shaped rows in `state/quality/gate-ledger.jsonl` (no `schema_version` field) are treated as **legacy v0**. Both formats coexist on the same file. New aggregators MUST:

1. Read line-by-line.
2. If `schema_version == 1`, parse as v1.
3. Else, parse as legacy v0 (chatbot PR row).
4. Project both into a common in-memory view if needed.

The existing `Scripts/gate-ledger-write.ps1` (GA) keeps writing v0 for one release cycle. A new sibling script `Scripts/gate-ledger-write-v1.ps1` writes v1, and the dashboard middleware projects both.

## Producers (Phase 0 wiring)

| Producer | Trigger | Source value | Domain |
|---|---|---|---|
| `ix-quality-trend` | After daily snapshot ingest | `ix-quality-trend` | `structural` / `coverage` (one row per category) |
| `ix-sentrux-gate-writer` (new) | Wraps `sentrux gate` | `sentrux` | `structural` |
| `Scripts/chatbot-qa-snapshot.ps1` (or equivalent) | After QA run | `chatbot-qa` | `chatbot` |
| `Scripts/council-snapshot.ps1` (TBD) | After council run | `council` | `chatbot` |
| `Scripts/e2e-snapshot.ps1` (TBD) | After Playwright e2e | `e2e` | `tests` |

## Consumers (Phase 0)

| Consumer | Read path | Purpose |
|---|---|---|
| `ix_quality_gate_history` MCP tool | `state/quality/gate-ledger.jsonl` (any repo) | Filtered tail queries for agents (`source=...&domain=...&since=...&limit=...`) |
| `/dev-data/quality-gates` Vite middleware (ga) | `state/quality/gate-ledger.jsonl` | Dashboard tile aggregation |
| `ix-quality-trend` report | `state/quality/gate-ledger.jsonl` | Roll into the existing markdown trend report (future) |

## File location convention

One ledger per repo at `state/quality/gate-ledger.jsonl`. Cross-repo aggregation is the consumer's job (read both ix and ga ledgers, merge by `(source, domain, run_at)`).

## Open questions (resolve before freezing as v1.0)

1. Should `id` be content-addressed (sha256 of canonical-JSON minus `id`) so re-emissions dedup naturally? **Tentative: no — producers may legitimately re-evaluate the same metric.**
2. Should `extra` have a sub-schema per `(source, domain)` pair? **Tentative: no — keep it opaque; if a consumer needs structure, it should bind to top-level fields only.**
3. Cross-repo merge: do we ship a `state/quality/gate-ledger-merged.jsonl` aggregate, or always merge in-memory in consumers? **Tentative: in-memory only for now.**

## Revisit triggers

- Any new producer that needs a field outside `extra` → bump to v2.
- Cross-repo aggregator goes from "in-memory" to "on-disk" → freeze v1.0.
- Operator workflow needs richer `operator_ack` than `{by, at, note}` → bump.
