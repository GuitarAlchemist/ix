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
| ix-quality-trend | `state/quality/gate-ledger.jsonl` | chatbot-PR-shaped row (`pr`, `branch`, `gates.{tests,agentToolReview,octoReview,tribunal}`, `decision`) — see `ga/docs/schemas/gate-ledger.schema.json`. Written in **ga** only; in ix this path had no producer at all until `ix doctor` (2026-09-17). |
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
| `source` | string | yes | Producer id. Enum (extensible): `ix-doctor`, `ix-quality-trend`, `sentrux`, `chatbot-qa`, `council`, `e2e`, `test-plans`, `tribunal`. Add new values by PR. |
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

| Producer | Trigger | Source value | Domain | Wired? |
|---|---|---|---|---|
| `ix doctor` | Every run of the pre-PR gate (`crates/ix-skill/src/doctor/mod.rs`) | `ix-doctor` | `harness` | **yes** |
| `ix-quality-trend` | After daily snapshot ingest | `ix-quality-trend` | `structural` / `coverage` (one row per category) | no |
| `ix-sentrux-gate-writer` | Wraps `sentrux gate` | `sentrux` | `structural` | binary exists, no caller — needs a `sentrux` on PATH |
| `Scripts/chatbot-qa-snapshot.ps1` (or equivalent) | After QA run | `chatbot-qa` | `chatbot` |
| `Scripts/council-snapshot.ps1` (TBD) | After council run | `council` | `chatbot` |
| `Scripts/e2e-snapshot.ps1` (TBD) | After Playwright e2e | `e2e` | `tests` |

## Consumers (Phase 0)

| Consumer | Read path | Purpose |
|---|---|---|
| `ix_quality_gate_history` MCP tool | `state/quality/gate-ledger.jsonl` (any repo) | Filtered tail queries for agents (`source=...&domain=...&since=...&limit=...`) |
| `/dev-data/quality-gates` Vite middleware (ga) | `state/quality/gate-ledger.jsonl` | Dashboard tile aggregation |
| `ix-quality-trend` report | `state/quality/gate-ledger.jsonl` | Roll into the existing markdown trend report (future) |

## Consumer obligation: an absent ledger is not a pass

A consumer that reports only a row count cannot be read correctly. `count: 0`
means *either* "no producer has ever run here" *or* "producers ran and nothing
matched your filter", and those are opposite answers — the first is the absence
of evidence, the second is evidence. Read as a pass, the first one turns a
never-wired gate into a reassuring green.

Every consumer MUST therefore report presence separately from count.
`ix_quality_gate_history` returns:

| `ledger_status` | Meaning |
|---|---|
| `absent` | No file. No gate has recorded a run. **Not** evidence that gates passed. |
| `empty` | File exists, no rows. Same epistemic weight as `absent`. |
| `present` | At least one row (v1 or legacy v0) exists. A `count: 0` here means the filter excluded everything. |

`absent` and `empty` also carry a `note` naming the producer that fixes it.

## Append durability and bounded growth

- **One write syscall per row.** `append_entry` concatenates the JSON and its
  newline and issues a single `write_all` on an `O_APPEND` handle, so
  concurrent producers on one host cannot interleave half-lines. Splitting the
  payload from its `\n` (what `writeln!` does) is what makes a torn line
  possible.
- **Readers tolerate damage.** Blank lines are skipped; a partially-shaped v1
  line degrades to `LedgerLine::LegacyV0` rather than failing the read. Worst
  case is a dropped row, never an unreadable file.
- **Rotation, not unbounded growth.** At `LEDGER_MAX_BYTES` (4 MiB, ~10k rows)
  the live file is renamed to `gate-ledger.1.jsonl` and a fresh one started.
  One generation is kept; consumers read the live file only. Rotation is a
  floor on disk use, not an archive — anything that must outlive it belongs in
  a dated snapshot under `state/quality/`.

## File location convention

One ledger per repo at `state/quality/gate-ledger.jsonl`. Cross-repo aggregation is the consumer's job (read both ix and ga ledgers, merge by `(source, domain, run_at)`).

`ix_quality_gate_history` resolves the default path against the **ix workspace
root**, not the process working directory — the MCP server is launched from
wherever the client happens to sit. Pass `ledger_path` explicitly to read a
sibling repo's ledger.

**Tracked in ga, ignored in ix.** ga's ledger is committed because a CI script
writes it. ix's is gitignored (`.gitignore`): `ix doctor` runs on every
contributor's machine, so tracking it would put an append-only JSONL on the
merge path of every PR. The consequence is deliberate and worth stating — ix's
ledger is per-checkout history, and a fresh clone starts `absent` until the
gate runs once. Publishing ix gate history beyond a checkout needs a CI job
that uploads or commits the file, which this contract does not yet specify.

## Open questions (resolve before freezing as v1.0)

1. Should `id` be content-addressed (sha256 of canonical-JSON minus `id`) so re-emissions dedup naturally? **Tentative: no — producers may legitimately re-evaluate the same metric.**
2. Should `extra` have a sub-schema per `(source, domain)` pair? **Tentative: no — keep it opaque; if a consumer needs structure, it should bind to top-level fields only.**
3. Cross-repo merge: do we ship a `state/quality/gate-ledger-merged.jsonl` aggregate, or always merge in-memory in consumers? **Tentative: in-memory only for now.**

## Revisit triggers

- Any new producer that needs a field outside `extra` → bump to v2.
- Cross-repo aggregator goes from "in-memory" to "on-disk" → freeze v1.0.
- Operator workflow needs richer `operator_ack` than `{by, at, note}` → bump.
