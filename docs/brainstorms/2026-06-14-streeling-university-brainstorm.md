---
date: 2026-06-14
topic: streeling-university
---

# Streeling University — expose all our learnings

## What We're Building
A hub that **exposes all institutional learnings** across the GuitarAlchemist ecosystem
(ix + ga + tars + Demerzel) — the "expose" half of the Cherny `/learnings` loop and the
knowledge backbone of the Learn→Ship→Steer methodology (`docs/LEARNING.md`). Asimov frame:
Seldon's academic home. It is an **index / registrar layer over existing files**, never a
new source of truth.

Two tiers in v1:
- **Campus** (Tier 1) — a curated `docs/streeling/` index organizing artifacts like a
  university: Faculties (the solution categories), Library (`docs/solutions`), Archives
  (knowledge packages), Course Catalog (`/teach` courses), Constitution (Demerzel).
- **Registrar** (Tier 2) — a **DuckDB catalog** over the learnings' frontmatter so the whole
  corpus is queryable ("all CI root-causes", "voicing learnings last 60d"), reusing the new
  `ix-duck` / duckdb-skills work. It backs the existing `learnings-researcher` agent.

Tier 3 "Lecture Hall" (interactive teaching — NotebookLM audio-overviews / Prime Radiant /
NebulaChat) is **deferred / demand-gated**.

## Why This Approach
Compose existing infra (`learnings-researcher`, `/teach`, `/learnings`, knowledge packages,
NotebookLM MCP, Prime Radiant); don't rebuild. Keep source-of-truth in the existing `.md`
files; Streeling derives an index + a catalog from them. Align cross-repo on **formats, not
runtime coupling** (the ecosystem's JSON-on-disk contract pattern).

## Key Decisions
- **v1 scope = Tier 1 (Campus) + Tier 2 (Registrar); Tier 3 deferred.** Ships real value; demand-gates the heavy UI.
- **Federate from the start (ix + ga + tars + Demerzel).** Requires a cross-repo **learnings frontmatter contract** (normalized schema) as the linchpin v1 artifact.
  - Reality check (2026-06-14 inventory): **ix** 17 `docs/solutions` (+6 knowledge, 28 plans, 19 brainstorms, 91 memory); **ga** 29 `docs/solutions`; **tars** 0 solutions but ~90 *knowledge* files (different shape); **Demerzel** governance artifacts (no `docs/solutions`).
  - ⇒ ix + ga ingest directly (same `docs/solutions` + frontmatter shape, ~46 entries). **tars/Demerzel need light per-repo adapters** that map their native stores into the contract. Degrade gracefully when a sibling clone is absent (cf. the `governance/demerzel` submodule pattern).
- **Registrar ingest = generator → `state/streeling/catalog.jsonl` → DuckDB.** A small extractor parses YAML frontmatter (+ source path, repo, mtime) into a JSONL catalog; DuckDB `read_json_auto`s it. Source-of-truth stays in the `.md`; catalog is derived. Matches align-on-formats + DuckDB-reads-JSONL.
- **Freshness = CI regen + staleness gate.** CI regenerates the catalog and **fails if `catalog_ts < newest_learning_ts`** — the explicit green-but-dead guard. (Optionally also a `/learnings` post-step + local hook.)
- **Registrar backs `learnings-researcher`.** The agent queries the catalog (structured) instead of grepping — faster, cross-repo. (folded)
- **The DuckDB course capstone BUILDS the Registrar.** `/teach DuckDB` L5 capstone = generate the catalog + write 3 registrar queries — learning the tool produces Streeling Tier 2. (folded)

## Open Questions (for planning)
- Where does the generator live? (Recommend a small `ix-duck` bin or an `ix-skill` verb, behind the optional `duck`/a plain-Rust path so the catalog build doesn't need DuckDB — DuckDB only *reads* the catalog.)
- The contract schema fields (superset of ix/ga frontmatter: `repo, category, title, date, tags, symptom, root_cause, path`) + where it's versioned (`docs/contracts/streeling-learning.contract.md`).
- tars/Demerzel adapters: in v1 or fast-follow? (Recommend: contract + ix/ga in v1; tars/Demerzel adapters as the first fast-follow so v1 isn't blocked on 90 heterogeneous tars files.)
- Campus index: hand-curated vs generated from the catalog? (Recommend: generated sections from the catalog + a hand-written intro, so it never drifts.)
- Does Campus link into `/teach` to auto-suggest courses from gaps in the corpus? (later)

## Next Steps
→ `/ce-plan` for implementation details (contract schema, generator, catalog, DuckDB views, CI staleness gate, Campus index).
