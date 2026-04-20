---
date: 2026-04-19
reversibility: two-way-door (format migratable via version tag); one-way-door on "in-process file persistence vs delegate to DB" architectural choice
revisit-trigger: graph size exceeds 100k nodes, hydration-on-boot exceeds 2s, or cross-process concurrent writers appear
status: runtime MVP shipped 2026-04-19 (GraphPersistence.fs + GraphTools.fs wiring in tars). Smoke tests pass: add-restart-query + clear-restart-stats. Unit tests pending. Open questions (1) + (2) accepted with proposed defaults.
affects: tars/v2/src/Tars.Tools/GraphTools.fs (public contract), state/graph/ (new storage location)
---

# TARS Knowledge-Graph persistence

## Problem

`Tars.Tools.Graph.GraphTools` maintains the knowledge graph in two `ConcurrentDictionary` instances (`nodes`, `edges`) created at module-load time. Every TARS MCP server restart wipes the graph. Today's session confirmed this concretely: the 17:40 rebuild of `Tars.Interface.Cli` cleared any graph state accumulated during the previous session.

That in-memory design is fine for demos and unit tests. It blocks any use-case where claims must persist across sessions — which includes every cross-session feature the ecosystem has been designing toward: chatbot-claim contradiction detection, pattern promotion over a real trace window, belief-state continuity, Demerzel governance audit trails.

## Status quo — what exists today

```fsharp
// src/Tars.Tools/GraphTools.fs:10-14
let private nodes = ConcurrentDictionary<string, JsonElement>()
let private edges = ConcurrentDictionary<string, JsonElement>()
```

Eight MCP tools operate on these dictionaries: `graph_add_node`, `graph_add_edge`, `graph_get_neighborhood`, `graph_query`, `graph_stats`, `graph_export_json`, `graph_clear`, `graph_find_contradictions`. None persist.

The surrounding ecosystem does have persistent artifacts (`state/knowledge/*.json`, `~/.ga/traces/*.json`, Demerzel beliefs under `governance/demerzel/state/beliefs/*.belief.json`), but none of them feed back into this graph. The graph is an island.

## Proposed design

### Storage

Two append-only JSONL files plus a small snapshot file, all under `state/graph/` in the TARS working directory:

| File | Purpose | Format |
|------|---------|--------|
| `state/graph/nodes.jsonl` | Append-only node log | one JSON object per line: the exact payload passed to `graph_add_node` plus an internal `_op`/`_ts` envelope |
| `state/graph/edges.jsonl` | Append-only edge log | same shape |
| `state/graph/snapshot.json` | Optional fast-boot cache | `{schema_version, nodes: [...], edges: [...]}` written every N appends or on graceful shutdown |

The append-only logs are the authority. The snapshot is a cache — delete it and the graph still rebuilds correctly from the logs. This is the same crash-safety shape Kafka, PostgreSQL WAL, and SQLite all use.

Envelope shape:

```json
{"_op":"add_node","_ts":"2026-04-19T21:15:02.334Z","id":"b:101","type":"Belief","label":"X causes Y","confidence":0.85}
{"_op":"update_node","_ts":"2026-04-19T21:16:10.118Z","id":"b:101","confidence":0.91}
{"_op":"remove_node","_ts":"2026-04-19T22:04:45.002Z","id":"b:101"}
```

`_op` lets us encode updates and removals without rewriting the file — important because `graph_clear` should stay O(1) from the user's perspective (write a `clear_all` marker; hydration handles the rest).

### Hydration on boot

On `McpServer` startup, before any tool call is serviced:

1. If `state/graph/snapshot.json` exists and its `schema_version` matches: load it into the dictionaries.
2. Replay `nodes.jsonl` and `edges.jsonl` entries with `_ts` newer than the snapshot's `_ts`.
3. If snapshot is missing or schema-mismatched: replay from the start of both logs.

For a realistic chatbot-claims volume (estimate: ~500 tool-call claims per active day → ~15k/month), the full replay is milliseconds. Above 100k nodes, switch to a real embedded DB (revisit trigger).

### Write semantics

- `graph_add_node` / `graph_add_edge`: append to the log **before** updating the in-memory dictionary. The log is the source of truth; the dictionary is a cache.
- Use `FileStream` with `FileShare.Read` + manual `Flush(true)` — not async — so an exception between log-write and dictionary-update leaves the log consistent. Cost: ~0.1ms per append on NVMe, negligible for the expected write rate.
- Do **not** add batched writes in v1. Single-writer, single-flush. Batching is a premature optimization; if it ever matters we'll see it in telemetry.
- Snapshots are written on `graph_clear`, on graceful shutdown, and every 1000 appends. The 1000-append threshold is arbitrary — revisit after the first month of real data.

### Concurrency

`GraphTools` already uses `ConcurrentDictionary` for reader parallelism. For the log, a single `lock (logLock)` around the append path is simpler than lock-free approaches and good enough for expected write volume (tens of ops/second at peak). If multiple TARS processes ever need to share the graph, this design breaks and we move to an embedded DB.

### New/changed MCP surface

Public MCP tool signatures do not change. One internal change: `graph_clear` now writes a `clear_all` log entry and truncates the dictionary, rather than just clearing the dictionary.

One new internal function (not MCP-exposed): `Graph.ensureHydrated()` called from `McpServer.start`. This is additive.

## Alternatives considered

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **JSONL append-only** (proposed) | No deps, crash-safe, trivially diffable, aligns with `state/telemetry/*.jsonl` and `state/knowledge/*.json` conventions already present | Slower for large graphs (>100k nodes) | ✅ Chosen |
| **SQLite (Microsoft.Data.Sqlite)** | ACID, fast for large graphs, good tooling | Adds a native dep and a migration story; overkill for <100k nodes | Deferred — revisit at scale trigger |
| **LiteDB** (pure .NET embedded doc DB) | Single-DLL embed, LINQ-friendly | Less common in F# codebases; another dep; file format not human-inspectable | Rejected — no decisive advantage over JSONL |
| **Neo4j embedded / external** | Real graph semantics, Cypher queries | Heavy dep, JVM, out of proportion | Rejected |
| **EventStoreDB / custom CQRS** | Event-sourced, principled | Massive over-engineering | Rejected |

The JSONL choice is deliberately conservative: the graph's access patterns today (add_node, add_edge, BFS neighborhood, filter-scan query) do not exercise anything a DB would give us. The file format matches existing ecosystem conventions, which makes debugging and manual inspection easy.

## Scope & non-goals

**In scope:**
- Persistent node/edge storage across TARS process lifetimes.
- Crash safety (append-before-update).
- Boot-time hydration with snapshot acceleration.
- Schema version tag for forward migration.

**Out of scope (explicitly):**
- Multi-process / multi-writer graphs. Single-writer TARS process only.
- Distributed graph federation across ix/tars/ga. That's a downstream design.
- Query-planner / indexing improvements. Current `graph_query` scans all nodes; that's fine until it isn't.
- Migration from tetravalent to hexavalent semantic on existing nodes. Explicitly out of scope per the paused `ix_governance::TruthValue` → `ix_types::Hexavalent` migration (blocked on the `Hexavalent::or` spec discrepancy, `docs/brainstorms/2026-04-10-hexavalent-or-discrepancy.md`). Chatbot claim tracer on the GA side emits `T/F/U/C`, matching what `BeliefStateService` and IxQL already speak.
- Any change to node/edge schemas themselves — this plan is only about persistence, not about what chatbot claims look like.

## Sizing

| Task | Effort |
|------|--------|
| `GraphPersistence.fs` (log writer, snapshot, hydration) | 0.5 day |
| Plumb into `McpServer.start` + thread safety review | 0.25 day |
| Unit tests (append/replay/snapshot/corruption/clear) | 0.5 day |
| Manual smoke: restart TARS mid-session, confirm graph survives | 0.25 day |
| Documentation update (README + this plan flipped to `status: shipped`) | 0.1 day |
| **Total** | **~1.6 days** |

## Risks & mitigations

1. **Silent data loss if append fails but dictionary update succeeds.** Mitigation: append-before-update ordering; wrap in try/finally; surface append failures as tool-call errors.
2. **Snapshot corruption.** Mitigation: snapshot is non-authoritative; delete-and-rebuild from logs.
3. **Schema drift between snapshot and current code.** Mitigation: `schema_version` integer field; mismatch → ignore snapshot, replay logs.
4. **Log file unbounded growth.** Out of scope for v1. Revisit at 100MB.

## Open questions — require sign-off

1. **Location**: `state/graph/` (TARS working dir, per-checkout) vs. `~/.tars/graph/` (user home, per-user). I propose **`state/graph/`** to match the existing convention (`state/telemetry/`, `state/knowledge/`). The `~/.ga/traces/` convention used by `GaTraceBridge.fs` is an exception I'd argue is a mistake, not a template — it makes debugging harder and conflates per-checkout and per-user state.
2. **Should `graph_clear` preserve history?** Today it wipes. I propose it writes a `clear_all` marker and keeps the log (for audit), but truncates the in-memory dictionary. An explicit `graph_vacuum` tool would actually delete logs — not in v1.
3. **JSON Lines vs JSON Array?** JSONL is append-friendly; arrays require rewrite. Proposal: JSONL.
4. **One-way-door framing check.** Is "file-based in-process persistence" really a one-way-door? Argument for: once chatbots and adversarial-qa depend on it, migrating to a DB requires coordinated downtime and a dump/load tool. Argument against: the migration path (read JSONL → write to DB) is mechanical and one-shot. I'm calling it **two-way-door on format** (versioned migration), **one-way-door on "in-process"** (moving to a network DB would require every TARS caller to become async-aware — expensive).

Seeking sign-off on (1) and (2) specifically. (3) and (4) are lower-stakes.

## Downstream unblocks

With this shipped:
- `2026-04-19-chatbot-kg-integration` (to be written) — the main payload: `ClaimTracer` in GaMcpServer + `ChatbotClaimsBridge.fs` in TARS.
- Persistent `temporal_detect_contradictions` results across sessions.
- Audit-trail continuity for Demerzel governance actions.
- Real `graph_stats` numbers for the Prime Radiant 3D viz integration (project memory: `project_prime_radiant_integration.md`).
