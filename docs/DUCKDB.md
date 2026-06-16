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
| **2** | GA emits an analyzable slice (voicing embeddings + metadata + search telemetry) as Parquet under a versioned on-disk contract | 🟡 **contract drafted + lens** (2026-06-16): telemetry half live in `ix_voicing_lens`; embeddings/metadata Parquet pending GA export. See `docs/contracts/2026-06-16-ga-voicing-analysis-parquet.contract.md` |
| **3** | Real `ix_optick` DuckDB extension: OPTIC-K mmap as a table function + voicing-distance UDF | ✅ **BUILT** (2026-06-16, signed off): `crates/ix-duck-ext` LOAD-able `ix.duckdb_extension` exposes the full UDF surface + `ix_optick_scan(index_path)` over the production mmap (validated: 313k voicings, dim 124). Voicing-distance composes from `ix_euclidean`. Workspace-excluded (out-of-band). Signed cross-platform distribution = open follow-up. See `docs/plans/2026-06-16-001-feat-ix-optick-duckdb-extension-plan.md` |

## v1 scope (locked 2026-06-14)

- **MVP analytical question:** *thinking-machine yield before/after a fix* — IX-local, **no GA dependency**,
  directly fixes the documented `hits.jsonl` yield-split footgun. Ships on Tier 0 + Tier 1 only.

  ```sql
  -- the footgun, killed in one query (no blended pre/post mean)
  SELECT ts_ms < $fix_ts AS pre, avg(coverage_max) AS yield, count(*) AS n
  FROM 'state/thinking-machine/hits.jsonl'
  GROUP BY pre;
  ```

- **Initial UDF set** (the differentiated value — things DuckDB can't do natively; it already has avg/stddev/percentile) — **all shipped** (2026-06-16):
  - `ix_cosine(a, b)` — vector similarity ✅ (scalar UDF)
  - `ix_euclidean(a, b)` — distance ✅ (scalar UDF; the kNN-distance primitive)
  - `ix_kdist(json_vectors, k)` — kNN-distance / OOD signal ✅ (table fn; mean dist to `k` nearest neighbours, leave-one-out)
  - `ix_pca_project(json_vectors, k)` — dimensionality reduction ✅ (table fn)
  - `ix_silhouette(json_vectors, json_labels)` — clustering quality ✅ (table fn)
  - `ix_dbscan(json_vectors, eps, min_points)` — density clustering labels ✅ (table fn; `0` = noise/OOD; composes with `ix_silhouette`)

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

- **`duckle`** (`SouravRoy-ETL/duckle`, MIT/Apache-2.0, evaluated 2026-06-15) — a local-first **visual
  drag-and-drop ETL/ELT studio** (Tauri desktop app, Rust + React) that compiles a pipeline graph to SQL
  and runs it on DuckDB; bundles llama.cpp for an NL→pipeline assistant and ships its own MCP server.
  **Skip for IX's analyst-bench / CI / production paths.** Architecturally aligned with IX (Rust,
  local-first, no telemetry, git-able plain-text workspace) but alignment ≠ need — its three differentiated
  value props are each orthogonal or redundant here: (1) *visual pipeline authoring* contradicts
  pipelines-as-governed-code (`ix-pipeline::dag::Dag` under the Demerzel constitution) and can't run in a
  hermetic CI gate, which is the whole point of `ix-duck`'s committed SQL; (2) *290+ connectors* are
  irrelevant when our data is local JSONL/Parquet we already emit; (3) *Duckie NL→pipeline + MCP server*
  duplicate Claude Code + `duckdb-skills` + `ix-agent` (same reason `acp` is parked). As a *local
  exploration* tool it overlaps **DuckDB UI** (already adopted, far lighter — no Tauri app, no bundled
  65 MB LLM). Optional as a personal visual scratchpad only. Revisit trigger: a non-engineer needs to
  author ETL over GA/ix artifacts visually (none today), or a connector-breadth wall our own emitters
  can't cover. Two-way door.

## One-way-door log

- A **published** DuckDB extension (Tier 3 `ix_optick`) is a one-way door — loadable artifact,
  per-platform builds + signing, public API surface, community Rust-extension path still experimental.
  Requires a plan doc + explicit sign-off before building (per CLAUDE.md "log one-way doors").
  **Update 2026-06-16:** signed off + built *locally* (`crates/ix-duck-ext`, `-unsigned` LOAD); the
  still-open one-way-door parts are **signed cross-platform distribution** (version-matched engines +
  DuckDB signing infra) — do those only on a deliberate publish decision.
- Keeping `duckdb` an **optional dep** (Tier 1) is a two-way door. Making it a core dep of a stable
  crate would not be — don't.

## Next steps

**Plan:** [`docs/plans/2026-06-14-001-feat-ix-duck-duckdb-udfs-plan.md`](plans/2026-06-14-001-feat-ix-duck-duckdb-udfs-plan.md)
(Tier 1: crate skeleton, `ix_cosine`/`ix_euclidean` scalar UDFs, yield-split example, tests, feature wiring).

> ⚠️ **Plan finding (resolved):** duckdb-rs `VScalar` is *row-wise*, so `ix_cosine`/`ix_euclidean` are
> true scalar UDFs while the set-relative ones (`ix_pca_project`/`ix_silhouette`/`ix_kdist`) ship as
> **table functions** (`VTab`) — all now landed in `crates/ix-duck/src/{udf,tablefn}.rs`. The MVP
> yield-split needs **no** custom UDF (pure SQL). See the plan's "Key Technical Finding".

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

## Business-value scorecard — a second registrar payload

`ix-value` federates per-repo `state/value/manifest.json` (RICE) into
`state/value/catalog.jsonl` — the same federation shape as Streeling, a different payload.
DuckDB reads the catalog directly for the demo/repo leaderboards:

```bash
duckdb -c ".read docs/value/queries.sql"   # top demos, repo leaderboard, low-confidence/high-impact
```

See [`docs/contracts/business-value.contract.md`](contracts/business-value.contract.md),
`crates/ix-value`, and `docs/plans/2026-06-14-003-feat-business-value-scorecard-plan.md`.
The stars render on the GA dashboard via a separate ga PR (`/dev-data/value` + `<StarRating>`).

## Voicing analysis lens (Tier 2) — telemetry now, embeddings pending

`ix_voicing_lens` is the Tier-2 analyst's bench over GA's voicing artifacts. The
**search-telemetry** half runs today over `../ga/state/telemetry/voicing-search/*.jsonl`
(coverage gaps = zero-result queries, latency p50/p95, most-repeated misses). The
**embeddings + metadata** half reads `state/voicings/analysis/voicings.parquet` once GA
exports it (the production `optick.index` is a binary mmap, not DuckDB-readable) — until
then it degrades to a one-line hint. When the Parquet lands, the vector UDFs
(`ix_pca_project`/`ix_kdist`/`ix_silhouette`) compose over `embedding`.

```bash
cargo run -p ix-duck --features duck --example ix_voicing_lens   # GA_ROOT or ../ga
```

See [`docs/contracts/2026-06-16-ga-voicing-analysis-parquet.contract.md`](contracts/2026-06-16-ga-voicing-analysis-parquet.contract.md).

## Streeling registrar lens — learnings corpus

`docs/streeling/queries.sql` runs over `state/streeling/catalog.jsonl` (the federated
learnings/plans/brainstorms catalog from `ix-streeling`): enrollment by repo/kind,
CI root-cause search, faculties by size, recent learnings, free-text topic lookup.

```bash
duckdb -c ".read docs/streeling/queries.sql"
```
