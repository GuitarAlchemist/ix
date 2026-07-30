# ix-fleet — cross-session coordination

The **fleet / handoff layer** for concurrent Claude Code sessions across the
GuitarAlchemist ecosystem. The 2026-07-21 loop-engineering review
(`../hari/docs/research/2026-07-21-loop-engineering.md`) flagged this as ix's one
structural gap ("tier-1, no fleet/handoff layer"). This is it — deliberately
minimal.

## The ledger

`~/.agents/claims.jsonl` — append-only JSONL, one claim event per line,
**latest line per `(repo, lane)` wins**. Sessions append a claim before starting
a lane another session might touch, and a `done`/`released` line when finished.
It is a collision-avoidance channel, **not** a lock server or task queue.

Schema (see `~/.agents/README.md` for the canonical protocol):

```json
{"ts":"RFC3339 UTC","repo":"ix","lane":"slug","status":"claimed|in-progress|done|released",
 "session":"ix","evidence":"commit|PR|path|null","note":"one line, optional"}
```

## `claims.py`

Stdlib-only (no deps) so any session can run it anywhere. Path resolves from
`--file`, else `$AGENTS_CLAIMS_FILE`, else `~/.agents/claims.jsonl`.

```bash
python fleet/claims.py validate          # schema-check every line (exit 1 if any bad)
python fleet/claims.py status            # current owner per (repo, lane)
python fleet/claims.py status --open     # only claimed / in-progress lanes
python fleet/claims.py append --repo ix --lane my-lane --status claimed \
    --session ix --evidence PR#123 --note "starting work"
```

Prefer `append` over hand-editing: it stamps `ts`, enforces the schema, and
refuses to write a malformed line — the failure mode a hand-appended JSONL ledger
is prone to. `validate` is safe to wire as a pre-flight gate.

## Tests

```bash
python -m unittest fleet/test_claims.py   # hermetic (tempfiles, no real ledger)
```
