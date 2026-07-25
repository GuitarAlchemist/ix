---
title: "`streeling catalog` silently drops all ga records when run from a worktree outside repos/, and `streeling check` then blesses the result"
category: workflow-patterns
date: 2026-07-24
tags: [streeling, ix-registrar, worktree, federation, catalog, drift-gate, green-but-dead]
symptom: "state/streeling/catalog.jsonl loses every `\"repo\":\"ga\"` record after running `streeling catalog`; `streeling check` reports `catalog is fresh` on the damaged file"
root_cause: "default_roots resolves ga as <ix_root>/../ga. From a worktree outside repos/ that path does not exist, so ingest sees only ix. `Cmd::Catalog` writes whatever it ingested with no missing-root guard, while `drift()` deliberately ignores records whose repo was not seen — so the deletion is invisible to the gate that exists to catch it"
---

# `streeling catalog` drops ga's records when run from a worktree

## Problem

Add a learnings doc, dutifully follow CLAUDE.md ("Run the `ix-streeling`
indexer after document changes"), and commit. If you did that from a git
worktree that does **not** live next to the `ga` clone, the diff is not `+1`
record — it is `+1 / -36`:

```
committed:     36 "repo":"ga"   62 "repo":"ix"
regenerated:    0 "repo":"ga"   63 "repo":"ix"
```

Every `ga` record is gone. The command tells you, but only in passing, and
only *after* it has already written the file:

```
catalog: 63 records (23 skipped); roots seen ["ix"], missing ["ga"]
```

Exit code is **0**.

## Why the drift gate does not save you

This is the part worth internalising. `streeling check` — the staleness gate —
reports the damaged catalog as healthy:

```
streeling: catalog is fresh (99 records)
```

That is not a bug in `check`; it is an intentional guard working as designed,
in a situation it was not designed for. `drift()` in
`crates/ix-registrar/src/lib.rs` only counts a committed record as
stale/removed if its repo root was actually seen during ingest:

```rust
for (k, cr) in &cmap {
    if !fmap.contains_key(k) && seen(cr.repo()) {
        report.extra.push(cr.report_id());
    }
}
```

The `seen(cr.repo())` clause exists so an ix-only checkout (CI, where no `ga`
sibling is cloned) does not flag all 36 ga records as stale and demand their
deletion. Correct for `check`.

The asymmetry is that **`Cmd::Catalog` has no equivalent guard.** It writes
`rep.records` unconditionally:

```rust
Cmd::Catalog => {
    let rep = ingest(&roots);
    ...
    std::fs::write(&path, to_jsonl(&rep.records))?;   // no missing-root check
    eprintln!("catalog: {} records ...; roots seen {:?}, missing {:?}", ...);
}
```

So one command tolerates a missing root by *ignoring* it, and the other
tolerates it by *deleting from it*. Worse, the two compose into a trap: once
`catalog` has stripped ga's records, `check` sees a ga-less catalog in a
ga-less context, finds nothing to compare, and returns clean. The gate that
should catch the damage is structurally blind to it — precisely the
"green-but-dead" failure mode.

Verified 2026-07-24: `streeling check` in a detached worktree under
`scratchpad/` (no `ga` sibling) prints `catalog is fresh (99 records)` and
exits 0, even though `streeling catalog` in that same worktree emits 63.

## Fix (workflow)

Run the indexer from a checkout that is a **sibling of `ga`**:

```
C:/Users/spare/source/repos/
├── ga/
├── ix/                  ← ok
└── ix-<topic>-wt/       ← ok (worktree placed here)
```

Not from `C:/tmp/...` or a session scratchpad. Concretely:

```bash
git worktree add C:/Users/spare/source/repos/ix-<topic>-wt <branch>
cd C:/Users/spare/source/repos/ix-<topic>-wt
cargo run -q -p ix-streeling -- catalog
```

Then **always** confirm the blast radius before committing — the count is the
oracle, not the exit code:

```bash
grep -oE '"repo":"[a-zA-Z]+"' state/streeling/catalog.jsonl | sort | uniq -c
git diff --stat state/streeling/catalog.jsonl   # expect +N, never -36
```

If `catalog` printed a non-empty `missing [...]`, discard the result
(`git checkout -- state/streeling/catalog.jsonl`) and re-run from the right
location.

This bites the existing worktree fleet: `C:/tmp/ix-agent-deepen`,
`C:/tmp/ix-contract-followup`, and `C:/tmp/ix-pr232-20260718` are all outside
`repos/`, so any agent adding a doc from one of them and following CLAUDE.md
will strip ga's records with a green exit code.

## Open — not fixed here

The durable fix is code, not discipline: make `Cmd::Catalog` refuse to write a
partial catalog when `roots_missing` is non-empty, unless an explicit
`--allow-partial` (or `--repo ix`) flag is passed. That converts a silent
destructive default into a loud, opt-in one, and costs a few lines in
`crates/ix-streeling/src/main.rs`. Deliberately left undone so this note stays
a report rather than a change; see
[parallel-worktree-merge-pitfalls](parallel-worktree-merge-pitfalls.md) for the
adjacent worktree hazards.

## Related

- [ix-duck bundled build fails under a long CARGO_TARGET_DIR](../build-errors/duckdb-bundled-build-fails-under-long-target-path.md)
  — discovered in the same session; also a case of a tool reporting a
  misleading cause.
