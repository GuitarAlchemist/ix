# Mission — Learn IX & DuckDB

**Started:** 2026-06-14 · **Tutor:** Claude (`/teach`)

## Why
Learn the project + how DuckDB is used across GA and IX, to **steer and review** development
confidently (the "Learn" in the Learn→Ship→Steer loop, `docs/LEARNING.md`).

## Goal
**Steer & understand** — reason about the architecture and know *when/why* DuckDB is the right
tool. Light on writing code.

## Starting level
Know SQL (SELECT / JOIN / GROUP BY). New to DuckDB specifics.

## How I learn best
Worked examples on real ix data (Streeling `catalog.jsonl`, `quality-snapshots/`, the
`quality.duckdb` analytics layer).

## Success criteria
- [ ] Explain *why* ix uses DuckDB (SQL over on-disk state, zero ETL) and when not to.
- [ ] Read any `build-views.sql` in the repo and say what each table/view does.
- [ ] Judge whether a proposed "analytics over X" is a `.sql` file or a service.
- [ ] Know the cross-repo producer/consumer SQL-parity story (ix produces, ga consumes).

## Language
English (French translation available on request — `feedback_french_docs`).
