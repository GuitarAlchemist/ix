# #485 inventory: the context seams IX already has

Research note for [#485](https://github.com/GuitarAlchemist/ix/issues/485) (map
[#483](https://github.com/GuitarAlchemist/ix/issues/483)). Local, not published. Source-read only,
on origin/main 99f2f869: nothing below was run unless it says **tested**. It records what exists;
the operator decides #485.

## The seams, against #485's questions

| Seam | What it is | Scope / durability | Version pin | Concurrent update | Missing input |
|---|---|---|---|---|---|
| `ix-agent-core` `ReadContext` (`context.rs:51`) | immutable projection of a session event log at an `ordinal`; writes go through `WriteContext` as new events | private to one session; the crate ships `VecEventSink` (memory, tests only); durability is `ix-session` | `(session_id, ordinal)`, plus a `replay_seed` | none: append-only events, no shared document | n/a |
| `ix-session` `SessionLog` (`log.rs`) | JSONL event sink | local disk; `flush` of a `BufWriter` per emit, no `sync_all` found | log position | not read | not read |
| `ix-context` `ContextBundle` + `ContextCache` (`cache.rs`) | deterministic code-context walk with a replayable `walk_trace` | in-process `Mutex<HashMap>` only | key `(git HEAD, file hash, manifest hash, strategy, root)`; SipHash, **not** cryptographic | manual `invalidate_path`, no watcher | n/a |
| `ix-recover` `Store` (`lib.rs`) | publish-once result per caller-chosen run id: intent before effect, attempt files + manifest, commit by hard link | local disk; Unix fsyncs directories, Windows makes **no** power-loss claim | `sha256:` per file and of the canonical bindings | single writer: owner lock; different bindings for one run id → `Error::Conflict` | verified on reconcile → `Corrupt` / `Unknown` |
| `ix-loop` `Store` (`store.rs`) | loops propose (`proposals/<id>.json`), only a person's `accept` moves one into `library/` | local disk, `IX_LOOP_STORE` or `~/.cache/ix-loop` | none | **`accept` overwrites a library file of the same name** (`fs::write`, line 191): no conflict | error string |
| `ix-loop` `ClaudeCode` node (`node.rs`) | child Claude call | one process per ask, `--no-session-persistence` | n/a | n/a | n/a |
| IXQL `Host` (`host.rs:97`) | real filesystem confined to a root (canonicalized ancestors); `ix.io.write` passes the schema gate | the run's root | **no digest** on reads (no hash in `host.rs`) | not read | host error |
| `ix-duck` `ArtifactLens` (`source.rs`) | directory of artifact files → a DuckDB table | a derived view, rebuilt on read | none | n/a | **absent or empty dir → empty table** ("graceful degrade") |
| ixctl bookmarks / state dir | per-plane bookmarks (tmp + rename, 0600) and an OS lock per name | local disk | none | one instance per name | — (journal never read back: see the G1 grill map) |
| GitHub artifacts | — | — | no code fetches a pinned GitHub path (`git grep` for raw/contents URLs: nothing) | — | — |

## What this says about #485

- **The two storage modes already exist, with different contracts.** Memory: `ReadContext`, a pure
  projection pinned by ordinal. Durable local disk: `ix-recover`, with a manifest, SHA-256 per file
  and a single-writer conflict. Nothing joins them, and #485's own rule is to add a universal store
  only if two concrete modes require it.
- **Private vs shared is not a notion anywhere.** Every seam above is private to one process, run id
  or session. No seam offers a shared dossier with expected-version (CAS) updates: the only
  `Conflict` found (`ix-recover`) guards one run id's bindings, not a shared document.
- **The one shared-ish write path loses updates silently.** Two proposals with the same name: the
  second `accept` replaced the first library file with no conflict. It is the exact counterexample
  #485 names ("concurrent writers conflict visibly rather than overwrite"). **Tested**:
  `accepting_a_second_loop_under_a_taken_name_is_a_visible_conflict` was seen red ("the second accept
  replaced the first loop"); the fix (`create_new`, the proposal stays pending) is pull request #491.
  With it, replacing a loop on purpose needs removing the old file first: no explicit replace.
- **Proposals instead of writes already exist once.** `ix-loop`'s proposal / accept split is the
  "returned additions are proposals" rule, for loops only.
- **A missing artifact reads as empty in the DuckDB lens.** For a dashboard that is the intended
  degrade; for a task's context it would turn "missing file" into "no data" silently. #485's
  missing-file case must not go through that path unchanged.
- **Reproducible references are content-addressed in one place only.** `ix-recover` hashes with
  SHA-256; `ix-context` uses SipHash (a cache key, not an identity); IXQL reads carry no digest.

## Proposal for the operator (not a decision)

Explicit immutable artifact lists first, not a ContextStore: a task receives a list of items, each
`{id, source (memory | local path | github repo/path/commit), sha256, schema, classification,
bytes}`, the `ix-recover` digest format reused. A memory-only run says restart recovery is
unavailable. Shared updates stay out of the first slice: no seam has CAS, so any shared write would
have to add one.

Smallest discriminating check (offline, as #485 asks), with the seams above:

1. In-memory private snapshot: a `ReadContext` at ordinal *n* is unchanged after an event at *n+1*.
2. Exported/restored local snapshot: an `ix-recover` commit read back; one changed byte → refused.
3. Two conflicting shared proposals: two `ix-loop` proposals with one name, both accepted. Expected
   by #485: a visible conflict. On main the second overwrites. This first red test is done (#491).
4. Missing file, changed hash, incompatible schema, incomplete dump: through the artifact list, each
   an explicit outcome, never the lens's empty table. Revoked read: indeterminate (no seam models
   access rights).

**Checks 1, 2 and 4 done (tested), offline:**

- Check 1: `crates/ix-agent-core/tests/snapshot_isolation.rs`. A `ReadContext` projected at ordinal 0
  is equal to its earlier clone after a `BeliefChanged` is emitted at 1 through `WriteContext`;
  only a new `project_beliefs` sees the change. A pin, green from the start: the type system
  (owned data, shared reference) is the guard.
- Check 2: already covered on main by `ix-recover`'s `test_a_tampered_file_is_corrupt` (one changed
  byte → `Error::Corrupt`) and `test_publish_then_begin_replays_the_same_files`.
- Check 4: `crates/ix-router-spike/src/bin/context_list.rs` (binary `context-list`, synthetic
  fixtures, 9 tests, each of 10 rules seen red under a mutant). A manifest (version 0, declared
  count, items with id, local or GitHub source, `sha256` in `ix-recover`'s format, schema,
  classification, bytes) is checked whole before any read: wrong version, incomplete dump, duplicate
  id, a GitHub reference that is not a full commit sha, a classification outside `public` /
  `internal` (so `secret` is refused), or an item over 16 MiB refuses the list and reads nothing.
  Then each item gets one outcome: `ready`, `missing`, `read_denied` (distinct from missing: the
  spike's reader models a revoked read, IX itself still has no access-rights seam),
  `size_mismatch`, `changed_hash`, `incompatible_schema` (not read). One bad item leaves the
  others named; the task never sees an empty table.

French: [`docs/fr/research/agent-composition/2026-10-30-485-context-seams-inventory.md`](../../fr/research/agent-composition/2026-10-30-485-context-seams-inventory.md).
