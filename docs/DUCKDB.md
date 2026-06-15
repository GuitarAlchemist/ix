# DuckDB in IX — living hub

> Single capture point for everything DuckDB-related in IX. Decisions here come from the
> 2026-06-14 brainstorm. Formal implementation plan: `docs/plans/2026-06-14-ix-duck.md`
> (run `/ce-plan` to generate). Update this file as decisions change.

## TL;DR

DuckDB is IX's **analyst's bench** — an in-process, in-memory OLAP layer over the JSONL/Parquet
IX (and GA) already emit. It is **not** a production engine and **not** a source of truth.
`optick.index` stays the production voicing-search path; GA stays the source of truth for domain
data. Integration is by **format contract** (clean stable-schema JSONL/Parquet), not runtime coupling.

## Tier model

| Tier | What | Status |
|------|------|--------|
| **0** | Install `duckdb/duckdb-skills` (official, MIT, local) → ad-hoc CLI over our JSONL/Parquet | ✅ **DONE + PROVEN** (2026-06-14) |
| **1** | `ix-duck` crate: embed `duckdb-rs` in-process, register IX algorithms as SQL UDFs (in-memory, no server) | ✅ **DONE** (Phases 1–3: `open_bench`, `ix_cosine`/`ix_euclidean`, yield-split example) |
| **2** | GA emits an analyzable slice (voicing embeddings + metadata + search telemetry) as Parquet under a versioned on-disk contract | deferred (not in v1) |
| **3** | Real `ix_optick` DuckDB extension: OPTIC-K mmap as a table function + voicing-distance UDF | demand-gated, **one-way door, needs sign-off** |

## v1 scope (locked 2026-06-14)

- **MVP analytical question:** *thinking-machine yield before/after a fix* — IX-local, **no GA dependency**,
  directly fixes the documented `hits.jsonl` yield-split footgun. Ships on Tier 0 + Tier 1 only.

  ```sql
  -- the footgun, killed in one query (no blended pre/post mean)
  SELECT ts_ms < $fix_ts AS pre, avg(coverage_max) AS yield, count(*) AS n
  FROM 'state/thinking-machine/hits.jsonl'
  GROUP BY pre;
  ```

- **Initial UDF set** (the differentiated value — things DuckDB can't do natively; it already has avg/stddev/percentile):
  - `ix_cosine(a, b)` — vector similarity
  - `ix_kdist(v, k)` — kNN-distance (OOD signal)
  - `ix_pca_project(v, k)` — dimensionality reduction
  - `ix_silhouette(points, labels)` — clustering quality

- **Crate structure:** new `crates/ix-duck` (library only), `duckdb` as an **optional dep** behind a
  `duck` cargo feature. Mirrors the established `fastembed`/`embeddings` pattern in `ix-skill`
  (`crates/ix-skill/Cargo.toml`). **Not in the stable tier** — isolates the native (C++) DuckDB dep.

  ```toml
  # crates/ix-duck/Cargo.toml
  duckdb = { version = "...", optional = true }
  [features]
  duck = ["dep:duckdb"]
  ```

## Tier 0 — proven (2026-06-14, Windows 11)

- DuckDB CLI **v1.5.3** installed locally at `%USERPROFILE%\tools\duckdb\duckdb.exe`
  (official MIT release zip; **not** on system PATH, not committed — reversible). The `winget`
  route (`winget install DuckDB.cli`) also works; the `duckdb-skills` `install-duckdb` skill
  installs *extensions*, not the CLI.
- Ran the MVP yield-split over `state/thinking-machine/hits.jsonl` (86 rows) with zero setup —
  DuckDB read JSONL natively. Result: **blended yield 0.4215** (the footgun), `compiled` 0.4495
  vs `out_of_domain` 0.3883; pre/post-median split 0.4093→0.4334. Windows "incomplete support"
  caveat did **not** bite the raw CLI path.
- Working query (verbatim):
  ```sql
  SELECT outcome, count(*) AS n, round(avg(coverage_max),4) AS yield
  FROM read_json_auto('state/thinking-machine/hits.jsonl')
  GROUP BY outcome ORDER BY n DESC;
  ```

## Available IX telemetry (DuckDB-readable today)

`state/thinking-machine/{hits,gaps,provenance,coverage-probes}.jsonl`,
`state/adversarial/findings.jsonl`, `state/graph/nodes.jsonl`,
`state/voicings/raw/{bass,guitar,ukulele}.jsonl`.
(Voicing *search* telemetry lives in GA, not IX → that's the Tier-2 export.)

## Skills

- **Adopt:** `duckdb/duckdb-skills` (official Claude Code plugin, MIT, local-first; 6 skills:
  attach-db, query, read-file, duckdb-docs, read-memories, install-duckdb).
  - Install (you must type these — `/plugin` is interactive):
    ```
    /plugin marketplace add duckdb/duckdb-skills
    /plugin install duckdb-skills@duckdb-skills
    ```
  - ⚠️ "Windows support is incomplete" (we're on Windows 11). DuckDB CLI itself is cross-platform;
    skill shell scripts may have path friction. Prereq: DuckDB CLI (`winget install DuckDB.cli`).
- **Skip:** `motherduckdb/agent-skills` — cloud-first, requires a MotherDuck account + `MOTHERDUCK_TOKEN`,
  ships data off-box. Contradicts the in-memory/local goal. Revisit only on a deliberate cloud decision.

## Evaluated / parked

- **`acp` community extension** (Agent Client Protocol, maintainer `sidequery`) — adds a `claude()`
  table function + `CLAUDE` NL→SQL statement using Claude Code. **Parked, not on the critical path:**
  it's query-*authoring* (orthogonal to our compute-UDF plan), third-party with license unspecified,
  needs Node/bun + network + Anthropic API credentials, and **sends your schema to the API**. We already
  get NL→SQL from Claude Code directly and the `duckdb-skills` `query` skill, so it's largely redundant.
  Revisit only if in-DuckDB NL querying becomes a recurring need AND the Node/network/license caveats are acceptable.

## One-way-door log

- A **published** DuckDB extension (Tier 3 `ix_optick`) is a one-way door — loadable artifact,
  per-platform builds + signing, public API surface, community Rust-extension path still experimental.
  Requires a plan doc + explicit sign-off before building (per CLAUDE.md "log one-way doors").
- Keeping `duckdb` an **optional dep** (Tier 1) is a two-way door. Making it a core dep of a stable
  crate would not be — don't.

## Next steps

**Plan:** [`docs/plans/2026-06-14-001-feat-ix-duck-duckdb-udfs-plan.md`](plans/2026-06-14-001-feat-ix-duck-duckdb-udfs-plan.md)
(Tier 1: crate skeleton, `ix_cosine`/`ix_euclidean` scalar UDFs, yield-split example, tests, feature wiring).

> ⚠️ **Plan finding:** duckdb-rs `VScalar` is *row-wise*. Only `ix_cosine` is a true scalar UDF;
> `ix_pca_project`/`ix_silhouette`/`ix_kdist` are set-relative → table functions / SQL recipes (staged to
> Phase 4 / v1.1). The MVP yield-split needs **no** custom UDF (pure SQL). See the plan's "Key Technical Finding".

→ Next: `/ce-work docs/plans/2026-06-14-001-feat-ix-duck-duckdb-udfs-plan.md`

## Chatbot flight recorder (Slice A/B) — a worked application

`ix_duck::chatbot` (example `ix_chatbot_lens`) is the canonical "SQL over our files"
application: it reads GA's per-run golden-traces (`../ga/state/quality/chatbot-qa`) into a
`chatbot_traces` table (Slice A — weak-intent / ungrounded / latency / routing-share queries)
and runs a canonical-diff **regression gate** on the routed agent (Slice B), emitting the
`chatbot-trace-regression` contract.

```bash
cargo run -p ix-duck --features duck --example ix_chatbot_lens             # lens over live ../ga
cargo run -p ix-duck --features duck --example ix_chatbot_lens -- check    # the gate
```

The hard gate runs hermetically over vendored fixtures in CI (`ix-duck-chatbot.yml`); the live
corpus is checked nightly (advisory). See
[`docs/plans/2026-06-14-004-feat-chatbot-duckdb-flight-recorder-plan.md`](plans/2026-06-14-004-feat-chatbot-duckdb-flight-recorder-plan.md)
and [`docs/contracts/chatbot-trace-regression.contract.md`](contracts/chatbot-trace-regression.contract.md).
Lessons: [`docs/solutions/feature-implementations/2026-06-14-duckdb-signature-unnest-over-lambda.md`](solutions/feature-implementations/2026-06-14-duckdb-signature-unnest-over-lambda.md).
