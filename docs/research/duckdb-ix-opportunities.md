# DuckDB + IX for local analytics, traces, vectors and corpus mining

**Issue:** [#191](https://github.com/GuitarAlchemist/ix/issues/191) · **Companion:** [2026-09-07-agentic-sdlc-observability.md](2026-09-07-agentic-sdlc-observability.md) (#206)

**Status:** evaluation complete, measured. Every number below was produced by a command
reproduced in this document. Claims that could not be verified are listed in
[Limits and what was not reached](#limits-and-what-was-not-reached) rather than smoothed over.

## Measurement environment

| Item | Value |
|---|---|
| Repo commit | `4331cf7712a2fbd6b9c66d3580fa3d1b1509496e` |
| DuckDB | v1.5.3 (Variegata) `14eca11bd9` |
| IX extension | release `v0.3.1`, asset `ix-windows_amd64.zip` |
| Asset sha256 | `e85a9fc96e73e994007d7c7318a719933b65794484f9f03b72295474e3e5fb2f` |
| Platform | Windows 11 Pro 26200, x86_64 |
| Cost | $0 — everything runs locally, no cloud runner, no paid model |

Setup, exactly as run:

```bash
gh release download v0.3.1 --repo GuitarAlchemist/ix --pattern '*'
unzip -o ix-windows_amd64.zip          # yields ix.duckdb_extension
duckdb -unsigned -c "LOAD 'C:/tmp/ixext/ix.duckdb_extension';"
```

`-unsigned` is required: the extension is not signed by the DuckDB registry.

## Verification of prior measurements

Five claims were handed to this evaluation as already-measured. Three hold as stated, two
needed correction. This table is the honest result, not a restatement.

| Claim | Verdict | Measured |
|---|---|---|
| Zip ships `ix.duckdb_extension`; the **file name** derives the init symbol, so a renamed file fails at LOAD | **Confirmed** | Copying to `renamed_test.duckdb_extension` → `IO Error: ... did not contain function "renamed_test_init_c_api"` |
| 81 functions register: 55 scalar, 26 table | **Confirmed exactly** | 55 scalar + 26 table = 81, all `ix_`-prefixed |
| `read_text` + `ix_code_metrics` analysed 566 files / 132 230 SLOC in 0.9 s | **Confirmed, drifted** | 567 files / 132 582 SLOC in **1.028 s** at `4331cf7` |
| Second argument is a **path** used for language detection, **not** a language name | **Corrected** | It accepts *both*. `a.rs`, `rs`, `rust`, `Rust` all → `Rust`; `a.py`, `python` → `Python` |
| `maintainability_index` saturates at 0 above roughly **200 SLOC** | **Direction right, threshold wrong** | Saturation onset is **458 SLOC**. Zero files below 400 SLOC saturate; files up to 740 SLOC still report MI > 0 |

### The two corrections in detail

**Language argument.** The parameter is not path-only. Probing eight forms:

```sql
LOAD 'C:/tmp/ixext/ix.duckdb_extension';
SELECT ix_code_metrics('fn main(){ let x=1; }', 'rust')->>'language';   -- Rust
SELECT ix_code_metrics('fn main(){ let x=1; }', 'rs')->>'language';     -- Rust
SELECT ix_code_metrics('fn main(){ let x=1; }', 'a.py')->>'language';   -- Python
SELECT ix_code_metrics('fn main(){ let x=1; }', '')->>'language';
-- Invalid Input Error: ix_code_metrics: unrecognised language/extension ""
```

The error string itself says `language/extension`. Passing `filename` works because a path
ends in a recognised extension, not because a path is required. This matters for callers
that hold content without a filename — they can pass `'rust'` directly instead of
fabricating a path.

**Maintainability index saturation.** The limit is real and it does block ranking, but not
at 200 SLOC:

```sql
LOAD 'C:/tmp/ixext/ix.duckdb_extension';
CREATE OR REPLACE TABLE m AS
SELECT filename, CAST(ix_code_metrics(content, filename) AS JSON) AS j
FROM read_text('crates/*/src/**/*.rs');

CREATE OR REPLACE TABLE f AS
SELECT filename,
  CAST(j->'file_scope'->>'sloc' AS BIGINT)                  AS sloc,
  CAST(j->'file_scope'->>'maintainability_index' AS DOUBLE) AS mi,
  CAST(j->'file_scope'->>'cyclomatic' AS DOUBLE)            AS cc
FROM m;

SELECT CASE WHEN sloc<100 THEN 'a: <100'
            WHEN sloc<200 THEN 'b: 100-199'
            WHEN sloc<400 THEN 'c: 200-399'
            ELSE 'd: >=400' END AS sloc_band,
       count(*) AS files,
       round(min(mi),2) AS mi_min, round(max(mi),2) AS mi_max,
       sum(CASE WHEN mi=0 THEN 1 ELSE 0 END) AS mi_exactly_zero
FROM f GROUP BY 1 ORDER BY 1;
```

| sloc_band | files | mi_min | mi_max | mi_exactly_zero |
|---|---|---|---|---|
| a: <100 | 149 | 44.72 | 171.00 | 0 |
| b: 100-199 | 184 | 25.51 | 51.82 | 0 |
| c: 200-399 | 151 | 1.74 | 33.99 | **0** |
| d: >=400 | 83 | 0.00 | 17.24 | **36** |

Precise onset: minimum SLOC with `mi = 0` is **458**; maximum SLOC with `mi > 0` is **740**.
Saturation is therefore **not a pure function of SLOC** — there is a 458–740 overlap band
where Halstead volume and cyclomatic count decide. Overall 36 / 567 files (6.3%) are
saturated, not "everything above 200 SLOC".

**The practical consequence is unchanged and confirmed:** MI cannot rank the files you most
want ranked. Cyclomatic complexity can.

```sql
SELECT replace(filename, chr(92), '/') AS file, sloc, cc, mi
FROM f ORDER BY cc DESC LIMIT 8;
```

| file | sloc | cc | mi |
|---|---|---|---|
| `crates/ix-agent/src/handlers.rs` | 5316 | 822.0 | **0.0** |
| `crates/ga-chatbot/src/main.rs` | 1494 | 229.0 | **0.0** |
| `crates/ix-code/src/analyze.rs` | 802 | 226.0 | **0.0** |
| `crates/ix-voicings/src/lib.rs` | 1613 | 205.0 | **0.0** |
| `crates/ix-ixql/src/parser.rs` | 670 | 154.0 | **0.0** |
| `crates/ix-duck/src/chatbot.rs` | 672 | 127.0 | **0.0** |
| `crates/ix-skill/src/verbs/compile.rs` | 1505 | 124.0 | **0.0** |
| `crates/ix-ixql/src/eval.rs` | 904 | 122.0 | **0.0** |

Every file in the top-8 complexity list has MI = 0. MI ranks them all identically; CC
separates 822 from 122. **Rule: never order by `maintainability_index` in IX tooling — order
by `cyclomatic`, and use MI only as a below-400-SLOC signal.**

## What the extension actually gives you

### Corpus metrics at interactive speed

```sql
LOAD 'C:/tmp/ixext/ix.duckdb_extension';
CREATE OR REPLACE TABLE m AS
SELECT filename, CAST(ix_code_metrics(content, filename) AS JSON) AS j
FROM read_text('crates/*/src/**/*.rs');

SELECT count(*) AS files,
       sum(CAST(j->'file_scope'->>'sloc' AS BIGINT)) AS sloc,
       any_value(j->>'language') AS lang
FROM m;
```

→ **567 files, 132 582 SLOC, `Rust`, 1.028 s.** Whole-workspace structural analysis inside
one second is the single strongest result in this evaluation. It makes per-commit corpus
metrics affordable with no daemon and no index to maintain.

Note the JSON shape — the top level is **not** flat:

```json
{"path": "...", "language": "...", "file_scope": { "...22 metrics..." }, "functions": [ "..." ]}
```

`j->>'sloc'` silently returns NULL. The correct path is `j->'file_scope'->>'sloc'`. This cost
a wrong first measurement here and will cost every future caller the same, so it is recorded
prominently.

### `ix_ast_query` is a real tree-sitter query, not a regex

```sql
SELECT ix_ast_query('// fn commented_out(){}
fn real_one(){ let s = "fn in_a_string(){}"; }
/* fn block_comment(){} */', 'x.rs', '(function_item name: (identifier) @f)');
```

→ `[{"capture":"f","text":"real_one","start_line":2,"end_line":2,"start_col":3}]`

Three decoy `fn` tokens — line comment, string literal, block comment — are all correctly
excluded. This is a genuine parse. Returned fields are `capture`, `text`, `start_line`,
`end_line`, `start_col` (two more than previously documented).

This is the capability that distinguishes DuckDB+IX from `grep`-based corpus mining, and it
is the one worth building on.

### Retrieval and hexavalent primitives

```sql
SELECT ix_cosine([1.0,0.0,1.0]::DOUBLE[], [1.0,0.0,0.0]::DOUBLE[]) AS cos,      -- 0.7071067811865475
       ix_precision_at_k([1.0,0.0,1.0,1.0]::DOUBLE[], 3)           AS p_at_3,   -- 0.6666666666666666
       ix_ndcg([3.0,2.0,1.0]::DOUBLE[], 3)                         AS ndcg,     -- 1.0
       ix_reciprocal_rank([0.0,0.0,1.0]::DOUBLE[])                 AS mrr,      -- 0.3333333333333333
       ix_hex_consensus(['T','T','F'])                             AS consensus,-- F
       ix_hex_or(['T','F'])                                        AS disj,     -- T
       ix_hex_not('T')                                             AS neg;      -- F
```

Signature warning: these are **not** the shapes you would guess. `ix_precision_at_k` is
`(DOUBLE[], BIGINT)` — a relevance vector plus k, *not* `(retrieved, relevant, k)`.
`ix_hex_or` is `(VARCHAR[])`, a list, not two arguments. Both guesses failed with a binder
error before the correct form was found. Always check `duckdb_functions()` first:

```sql
SELECT function_name, parameter_types, return_type
FROM duckdb_functions() WHERE function_name LIKE 'ix\_%' ESCAPE '\';
```

Behavioural note worth flagging to governance: `ix_hex_consensus(['T','T','F'])` returns
**`F`** — a single dissent flips the consensus. It is fail-closed, which is the correct
default for a gate but wrong if a caller expects a majority vote.

## Use cases ranked by value and complexity

Ranking is by **measured available fuel**, because that turned out to be the binding
constraint — not by how interesting the query would be.

| # | Use case | Value | Complexity | Fuel measured here | Verdict |
|---|---|---|---|---|---|
| 1 | **Code corpus mining** (metrics + AST over `crates/**`) | High | Low | 567 files / 132 582 SLOC / 1.03 s | **Build.** Fuel is abundant and already local. |
| 2 | **Issue / PR / CI metadata analytics** | High | Low | 209 PRs, 400 runs, 71 issues | **Build.** See [#206 companion](2026-09-07-agentic-sdlc-observability.md). Needs *no* IX extension. |
| 3 | **Vector / retrieval benchmark scoring** | Medium | Low | primitives verified; no benchmark corpus on disk here | Ready, unfuelled. |
| 4 | **Local research corpus queries** (`state/**.jsonl`) | Low | Low | 102 / 147 / 184 rows, 35–50 KB | **Do not build.** See below. |
| 5 | **TARS trace / AIW episode analytics** | Unknown | Unknown | **zero artifacts present** | **Not reached.** |

### The fuel problem, measured

```bash
duckdb -c "SELECT count(*) FROM read_json_auto('state/streeling/catalog.jsonl', format='newline_delimited');"
```

| Artifact | Rows | Bytes |
|---|---|---|
| `state/streeling/catalog.jsonl` | 102 | 49 803 |
| `state/assumptions/belief-events.jsonl` | 147 | 37 294 |
| `state/thinking-machine/coverage-probes.jsonl` | 184 | 35 208 |

Hundreds of rows and tens of kilobytes. **DuckDB is several orders of magnitude oversized for
the IX `state/` corpus**, and any "local analytics warehouse" framed over these files is a
re-skin, not a capability. The honest conclusion for #191 is that the two use cases with real
fuel are the *code corpus* and *GitHub metadata* — both large enough to justify a columnar
engine, and neither of which is what the original schema sketch (appendix below) was aimed at.

## No-cloud prototype plan

Smallest end-to-end slice that touches every layer, per the tracer-bullet discipline:

1. **Extract** — `gh` JSON to disk + `read_text('crates/*/src/**/*.rs')`. No service.
2. **Load** — one DuckDB file, `duckdb ix-analytics.duckdb`. Created on demand, deletable.
3. **Transform** — the SQL views in this document and the companion, checked in as `.sql`.
4. **Emit** — `ix-agentic-sdlc-scorecard.json` via `COPY (...) TO ... (FORMAT JSON)`.
5. **Verify** — assert row counts and a known-value regression (e.g. CC of `handlers.rs`).

No always-on database. No daemon. The `.duckdb` file is a cache, never a source of truth —
every table is rebuildable from `gh` and the working tree in about a second.

## IX-owned vs TARS-owned

| Concern | Owner | Why |
|---|---|---|
| Code corpus metrics, AST queries, structural scoring | **IX** | `ix_code_metrics` / `ix_ast_query` live in the IX extension; offline catalog work is explicitly IX's per `CLAUDE.md`. |
| Retrieval metrics (`ndcg`, `precision_at_k`, `mrr`) | **IX** | Pure functions over vectors; no runtime state. |
| Hexavalent aggregation of verdicts | **IX** primitives, **Demerzel** policy | `ix_hex_*` computes; the constitution decides what a verdict means. |
| Trace / closure-run emission | **TARS** | TARS owns the runtime that produces `trace-events.jsonl`. IX must not grow a trace emitter. |
| Realtime structural quality | **sentrux** | Boundary already fixed in `CLAUDE.md`; DuckDB work here is offline by construction. |
| Analytics *over* TARS traces once they exist | **IX** | Reading is IX's; producing is TARS's. |

The boundary that matters: **IX reads, TARS emits.** Nothing in this evaluation requires IX
to run a service or hold state between invocations.

## Cost control

- **$0.** No cloud runner, no paid model, no network beyond `gh` API reads.
- Wall clock for the full corpus pass: **1.03 s**; the extension is 17 MB and loads per process.
- Process startup dominates one-shot use: the same query took 5.4 s via `duckdb -c` including
  load, vs 1.03 s measured inside the session. **Batch statements into one session**; do not
  invoke `duckdb -c` per file.
- No secrets and no raw private logs are read. `gh` metadata only — titles, timestamps,
  labels, logins. Issue/PR *bodies* are read for the companion's heuristics and must not be
  redistributed outside the repo.

## Limits and what was not reached

Stated explicitly, because an unverified claim is worse than an absent one.

1. **"Table functions take JSON in VARCHAR" is only spot-checked.** `ix_ast_query` and
   `ix_code_metrics` (both scalar) return JSON text. The argument convention of all 26 table
   functions was **not** systematically verified. Treat that claim as unverified.
2. **The 566 → 567 / 132 230 → 132 582 delta is unexplained.** It is consistent with the tree
   having moved between the two measurements, but this was not bisected. Do not treat either
   number as a fixed baseline; re-measure per commit.
3. **MI saturation onset (458 SLOC) is corpus-specific.** It is the minimum observed in *this*
   567-file Rust corpus, not a property of the formula. The 458–740 overlap proves SLOC alone
   does not determine it.
4. **No TARS traces, AIW episodes, or Demerzel decision logs exist in this worktree.** Use
   case #5 is entirely unevaluated — not "low value", *unmeasured*.
   `find . -name 'trace-events*.jsonl'` returned nothing.
5. **No vector benchmark corpus was present.** The retrieval primitives were verified on
   hand-built vectors only; end-to-end embedding benchmark scoring was not exercised.
6. **`ix_optick_scan` was not run.** The OPTIC-K index is not present in this worktree.
7. **Windows-only.** Only `ix-windows_amd64.zip` was tested. Linux/macOS assets untested.
8. **Deprecation, will break:** DuckDB 1.5.3 warns that the `->` lambda arrow is deprecated
   and removed next release. Use `lambda x: x.name`, not `x -> x.name`, in any checked-in SQL.
   (This affects list lambdas only, not the JSON `->` / `->>` operators used above.)

## Appendix — original schema sketch

The tables below were the pre-measurement draft. They are retained for provenance. Read them
against the fuel table above: `corpus_files`, `exploration_candidates`, `aiw_episodes`,
`budget_ledger_entries`, `trace_events`, `vector_benchmarks` and `algorithm_runs` all describe
artifacts that **do not currently exist at analytically useful volume**, which is precisely
the finding this evaluation adds.

### 1. `corpus_files`

| Column | Type | Description |
|--------|------|-------------|
| `file_path` | VARCHAR | Primary key; relative path to the artifact. |
| `corpus_type` | VARCHAR | Category: `voicing`, `thinking-machine`, `learning`, `contract`. |
| `format` | VARCHAR | `jsonl`, `parquet`, `json`, `md`. |
| `file_size_bytes` | BIGINT | Size on disk. |
| `row_count` | BIGINT | Number of records (for tabular formats). |
| `last_modified` | TIMESTAMP | Last write time. |
| `checksum` | VARCHAR | Content hash (BLAKE3). |

### 2. `exploration_candidates`

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | VARCHAR | Unique identifier for the candidate. |
| `source_id` | VARCHAR | Reference to a corpus file or specific record. |
| `candidate_type` | VARCHAR | e.g., `adversarial_probe`, `edge_case_voicing`. |
| `metadata` | JSON | Domain-specific context. |
| `score` | DOUBLE | Priority or heuristic score. |
| `status` | VARCHAR | `pending`, `explored`, `rejected`. |
| `created_at` | TIMESTAMP | Discovery timestamp. |

### 3. `aiw_episodes`

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | VARCHAR | Unique episode ID. |
| `session_id` | VARCHAR | Parent session identifier. |
| `start_time` | TIMESTAMP | Start of the episode. |
| `end_time` | TIMESTAMP | End of the episode. |
| `summary` | VARCHAR | Brief description of the work performed. |
| `verdict_count` | INTEGER | Number of governance/algorithmic verdicts emitted. |
| `token_usage` | INTEGER | Total tokens consumed. |
| `cost_estimate` | DOUBLE | Estimated USD cost. |

### 4. `budget_ledger_entries`

| Column | Type | Description |
|--------|------|-------------|
| `entry_id` | UUID | Unique ledger entry ID. |
| `timestamp` | TIMESTAMP | Time of the transaction/usage. |
| `account_id` | VARCHAR | Internal budget account. |
| `provider` | VARCHAR | e.g., `anthropic`, `openai`, `local-llama`. |
| `operation` | VARCHAR | e.g., `sampling`, `embedding`. |
| `amount` | DOUBLE | Transaction amount. |
| `currency` | VARCHAR | `USD`, `credits`. |
| `tags` | JSON | Project, Persona, or Task tags. |

### 5. `trace_events`

| Column | Type | Description |
|--------|------|-------------|
| `trace_id` | VARCHAR | Unique trace identifier. |
| `span_id` | VARCHAR | Specific operation identifier. |
| `parent_span_id` | VARCHAR | Hierarchy pointer. |
| `event_name` | VARCHAR | e.g., `ix_kmeans_run`, `tool_call`. |
| `timestamp` | TIMESTAMP | Event occurrence. |
| `duration_ms` | DOUBLE | Execution time. |
| `tool_name` | VARCHAR | The name of the IX tool invoked. |
| `input_json` | JSON | Tool arguments. |
| `output_json` | JSON | Tool results. |
| `outcome` | VARCHAR | `success`, `failure`, `timeout`. |

### 6. `vector_benchmarks`

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | VARCHAR | Unique benchmark run ID. |
| `timestamp` | TIMESTAMP | Run time. |
| `model_name` | VARCHAR | e.g., `bge-large-en-v1.5`. |
| `dimension` | INTEGER | Embedding size. |
| `metric` | VARCHAR | `cosine`, `euclidean`, `dot_product`. |
| `k` | INTEGER | Top-K depth evaluated. |
| `recall_at_k` | DOUBLE | Evaluation metric. |
| `latency_ms_p50` | DOUBLE | Median latency. |
| `latency_ms_p95` | DOUBLE | Tail latency. |

### 7. `algorithm_runs`

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | VARCHAR | Unique execution ID. |
| `algorithm_name` | VARCHAR | e.g., `pso`, `dbscan`, `viterbi`. |
| `parameters` | JSON | Hyperparameters used. |
| `input_size` | BIGINT | Number of input data points/dimensions. |
| `duration_ms` | DOUBLE | Total execution time. |
| `result_summary` | JSON | High-level results (e.g., loss, cluster count). |
| `timestamp` | TIMESTAMP | Execution start time. |
