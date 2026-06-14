# Streeling Learning Record — cross-repo contract (v0.1, draft)

> **Status:** v0.1 **draft** (Phase 0–3). Not frozen. Locked fields require cross-repo
> coordination (ix + ga + tars + Demerzel); freeze only at a named Phase-4 milestone,
> per the CLAUDE.md contract-phasing rule. This is a **format** contract (JSON-on-disk),
> not runtime coupling.

## Purpose
Normalize each repo's learning artifacts into one shape so Streeling University can index,
query, and teach the whole corpus. Source-of-truth stays in each repo's `.md` files; this
is the derived record that lands in `state/streeling/catalog.jsonl` (one JSON object/line).

## Producer
`ix-streeling` (`crates/ix-streeling`) parses YAML frontmatter from each repo's learning
stores. v1 ingests **ix** (`docs/solutions`, `state/knowledge`, `docs/plans`,
`docs/brainstorms`) and **ga** (`docs/solutions`). **tars** (knowledge store) and
**Demerzel** (governance) need per-repo adapters — first fast-follow.

## Schema
See `streeling-learning.schema.json`. Record fields:

| field | type | required | notes |
|-------|------|----------|-------|
| `schema_version` | string | yes | `"0.1.0"` |
| `id` | string | yes | `"{repo}:{path}"` — stable key |
| `repo` | string | yes | `ix` \| `ga` \| `tars` \| `Demerzel` |
| `kind` | enum | yes | `solution` \| `knowledge` \| `plan` \| `brainstorm` |
| `category` | string | yes | faculty; solution category or the kind's default |
| `title` | string | yes | from frontmatter `title`/`topic`, else filename stem |
| `date` | string | no | ISO date as written in source |
| `tags` | string[] | no | omitted when empty |
| `symptom` | string | no | solutions |
| `root_cause` | string | no | solutions |
| `path` | string | yes | repo-relative, POSIX separators |

## Ingest rules (tolerant by design)
- A file with **no / invalid** YAML frontmatter is **skipped and counted**, never fatal.
- A **missing repo root** (sibling clone absent) is reported in `roots_missing`, not an error.
- Records are emitted **sorted by `id`** for stable diffs.

## Freshness
The committed catalog is gated by `streeling check` (CI: `.github/workflows/streeling-freshness.yml`),
which re-ingests and fails on drift. The check is **repo-scoped**: it validates only records
from repos it actually scanned, so a single-repo CI checkout won't false-positive on a sibling's
records. Regenerate with `cargo run -p ix-streeling -- catalog` (+ `campus`) and commit.

## Consumers
- **Campus** index — `docs/streeling/README.md` (generated).
- **Registrar** — DuckDB over the catalog (`docs/streeling/queries.sql`, `docs/DUCKDB.md`).
- **`learnings-researcher`** agent — consult `state/streeling/catalog.jsonl` first (structured, cross-repo) before globbing.

## Versioning
`links.supersedes` may introduce a non-breaking baseline shift without freezing the schema.
Field changes that break consumers are a one-way door pending sibling sign-off.
