# ix-duck-ext

A **loadable DuckDB extension** exposing IX algorithms as SQL UDFs. The
`LOAD`-able sibling of the in-process [`ix-duck`](../ix-duck) bench: it reuses the
exact same registration (`ix_duck::udf::register_all`) but ships as a standalone
`ix.duckdb_extension` that **any `duckdb.exe` can load** — no embedding host.

## Functions

### Scalar

| UDF | Signature | Notes |
|---|---|---|
| `ix_cosine` | `(DOUBLE[], DOUBLE[]) -> DOUBLE` | Cosine similarity in [-1, 1]. Dimension mismatch → SQL error. |
| `ix_euclidean` | `(DOUBLE[], DOUBLE[]) -> DOUBLE` | L2 distance. Primitive for kNN / OOD: `ORDER BY ix_euclidean(q, r) LIMIT k`. |
| `ix_forte_number` | `(BIGINT[]) -> VARCHAR` | Forte set-class of the notes (mod 12), e.g. `"3-11"`; NULL if unclassifiable. Wraps `ix-bracelet`. |
| `ix_icv` | `(BIGINT[]) -> VARCHAR` | Interval-class vector, e.g. `"<0,0,1,1,1,0>"`. |
| `ix_prime_form` | `(BIGINT[]) -> VARCHAR` | Bracelet prime form, e.g. `"[0,3,7]"`. |
| `ix_classify_triad` | `(BIGINT[]) -> VARCHAR` | `"<root> major\|minor"`, or NULL if not a consonant triad. |
| `ix_ndcg` | `(DOUBLE[], BIGINT) -> DOUBLE` | nDCG@k of a ranked relevance list. |
| `ix_reciprocal_rank` | `(DOUBLE[]) -> DOUBLE` | 1/(rank of first relevant); `avg(...)` ⇒ MRR. |
| `ix_precision_at_k` | `(DOUBLE[], BIGINT) -> DOUBLE` | relevant in top-k / k. |
| `ix_recall_at_k` | `(DOUBLE[], BIGINT, BIGINT) -> DOUBLE` | relevant in top-k / total_relevant. |

### Table

| UDF | Signature | Notes |
|---|---|---|
| `ix_pca_project` | `(json_vectors VARCHAR, n_components BIGINT) -> TABLE(row BIGINT, coords DOUBLE[])` | Fits PCA over the input set, returns each vector's projection. Wraps `ix_unsupervised::pca::PCA`. |
| `ix_silhouette` | `(json_vectors VARCHAR, json_labels VARCHAR) -> TABLE(row BIGINT, label BIGINT, silhouette DOUBLE)` | Per-point silhouette for a clustering. Mean: `SELECT avg(silhouette) FROM ix_silhouette(...)`. |
| `ix_kdist` | `(json_vectors VARCHAR, k BIGINT) -> TABLE(row BIGINT, kdist DOUBLE)` | Mean distance to the `k` nearest neighbours (leave-one-out) — the OOD / local-outlier signal. |
| `ix_dbscan` | `(json_vectors VARCHAR, eps DOUBLE, min_points BIGINT) -> TABLE(row BIGINT, cluster BIGINT)` | Density clustering labels; `0` = noise. Composes with `ix_silhouette`. |
| `ix_kmeans` | `(json_vectors VARCHAR, k BIGINT) -> TABLE(row BIGINT, cluster BIGINT)` | Centroid (k-means) labels `0..k-1`, deterministic (`k` capped at sample count). |
| `ix_gmm` | `(json_vectors VARCHAR, k BIGINT) -> TABLE(row BIGINT, cluster BIGINT)` | Gaussian-mixture component labels `0..k-1` — soft-assignment counterpart to `ix_kmeans`. |
| `ix_optick_scan` | `(index_path VARCHAR) -> TABLE(voicing BIGINT, instrument VARCHAR, embedding DOUBLE[])` | **Tier 3:** the production OPTIC-K `optick.index` mmap as a table — no Parquet export. Wraps `ix_optick::OptickIndex`. Voicing distance = `ix_euclidean` over two rows. |
| `ix_pagerank` | `(edges_json VARCHAR, damping DOUBLE, iterations BIGINT) -> TABLE(node BIGINT, rank DOUBLE)` | PageRank over a JSON edge list `[[from,to(,w)],…]`. Wraps `ix-graph`. |
| `ix_shortest_path` | `(edges_json VARCHAR, src BIGINT, dst BIGINT) -> TABLE(step BIGINT, node BIGINT)` | Dijkstra path src→dst (empty if unreachable). |
| `ix_rfft` | `(series_json VARCHAR) -> TABLE(bin BIGINT, magnitude DOUBLE)` | Real-FFT magnitude spectrum of a JSON number series. Wraps `ix-signal`. |
| `ix_autocorrelation` | `(series_json VARCHAR) -> TABLE(lag BIGINT, value DOUBLE)` | Two-sided normalized autocorrelation; lag 0 = 1.0 peak. |
| `ix_classification_report` | `(predicted_json VARCHAR, actual_json VARCHAR) -> TABLE(label BIGINT, precision DOUBLE, recall DOUBLE, f1 DOUBLE, support BIGINT)` | Per-class one-vs-rest metrics. Wraps `ix-supervised::metrics`. |
| `ix_knn_leakage` | `(vectors_json VARCHAR, labels_json VARCHAR, k BIGINT) -> TABLE(leakage DOUBLE, random_baseline DOUBLE)` | Mean k-NN label agreement — embedding separability/leakage vs the random baseline. |

Scalars wrap `ix_math::distance` directly; the PCA table function wraps
`ix_unsupervised` — no reimplementation, identical numbers to the in-process bench.
Table functions take a whole set as a JSON param (2-D number array; labels a 1-D int
array), so one SQL call processes the set:
`SELECT * FROM ix_pca_project('[[1,2,3],[4,5,6]]', 2);`

### Probabilistic sketches

DuckDB has `approx_count_distinct` (a HyperLogLog) and uses Bloom filters
internally for joins, but exposes none of them as **queryable, persistable,
mergeable** objects. These do: each sketch is a portable column value — build once
(scalar over a `BIGINT[]`, paired with `list()`), store the JSON blob, then probe
cheaply or merge across partitions / repos. (duckdb-rs has no aggregate-UDF API, so
each structure is a **build → probe/merge** triad of scalars. Wraps `ix-probabilistic`.)

| structure | build | query | combine |
|---|---|---|---|
| **Bloom** (set membership) | `ix_bloom_build(items BIGINT[], capacity BIGINT, fp_rate DOUBLE) -> VARCHAR` | `ix_bloom_contains(sketch VARCHAR, item BIGINT) -> BOOLEAN` | `ix_bloom_union(a VARCHAR, b VARCHAR) -> VARCHAR` |
| **HyperLogLog** (cardinality) | `ix_hll_build(items BIGINT[], precision BIGINT) -> VARCHAR` | `ix_hll_count(sketch VARCHAR) -> BIGINT` | `ix_hll_merge(a VARCHAR, b VARCHAR) -> VARCHAR` |
| **Count-Min** (frequency) | `ix_cms_build(items BIGINT[], epsilon DOUBLE, delta DOUBLE) -> VARCHAR` | `ix_cms_estimate(sketch VARCHAR, item BIGINT) -> BIGINT` | `ix_cms_merge(a VARCHAR, b VARCHAR) -> VARCHAR` |
| **Cuckoo** (membership + delete) | `ix_cuckoo_build(items BIGINT[], capacity BIGINT) -> VARCHAR` | `ix_cuckoo_contains(sketch VARCHAR, item BIGINT) -> BOOLEAN` | `ix_cuckoo_remove(sketch VARCHAR, item BIGINT) -> VARCHAR` |

```sql
-- Build a Bloom filter over a column, then probe a whole other column against it:
SELECT q.id
FROM queries q, (SELECT ix_bloom_build(list(seen_id), 100000, 0.01) AS bf FROM history) h
WHERE ix_bloom_contains(h.bf, q.id);

-- Distinct-count without a GROUP BY DISTINCT scan; merge two partitions' sketches:
SELECT ix_hll_count(ix_hll_merge(part_a.s, part_b.s)) AS approx_distinct ...;
```

Items are `BIGINT` so build-vs-probe hashing is identical. **Text keys** bridge
through DuckDB's deterministic `hash()` (mask to 63 bits — `hash()` is `UBIGINT` and
overflows `BIGINT`): `ix_bloom_contains(bf, (hash(q) & 9223372036854775807)::BIGINT)`.
Blobs are JSON and portable: `ix-probabilistic` hashes with a fixed-seed
`DefaultHasher`, so a blob probes identically on any machine (same Rust std version).

Two caveats:
- **Blob trust.** A non-JSON blob is a SQL error; a structurally-degenerate blob
  (e.g. mismatched array lengths) probes to a safe empty result rather than
  panicking, so a corrupt column never crashes the query.
- **`ix_cuckoo_remove` is safe only for keys you inserted.** Cuckoo deletion works
  on fingerprints, so removing a never-inserted key that collides (bucket +
  fingerprint) with a real entry can evict that entry — an inherent property of
  Cuckoo filters, not a bug. Don't run deletes on keys outside the inserted set.

### Code analysis (SQL over a codebase, incl. tree-sitter)

Pair with DuckDB's `read_text('crates/**/*.rs')` (rows of `filename, content`) and
these scalar UDFs give you SQL over an entire codebase — wraps `ix-code`.

| tier | UDF | returns |
|---|---|---|
| **A** | `ix_code_complexity(source VARCHAR, language VARCHAR) -> DOUBLE` | file-scope cyclomatic complexity |
| **A** | `ix_code_metrics(source VARCHAR, language VARCHAR) -> VARCHAR` | JSON `FileMetrics` (complexity + Halstead + SLOC, per function) |
| **A** | `ix_code_smells(source VARCHAR, language VARCHAR) -> VARCHAR` | JSON `[{name,line,severity,message}]` |
| **B** | `ix_semantic_metrics(source VARCHAR, language VARCHAR) -> VARCHAR` | JSON — parse_quality, AST node count, nesting, error-handling density, unsafe blocks, call graph |
| **B** | `ix_ast_query(source VARCHAR, language VARCHAR, query VARCHAR) -> VARCHAR` | JSON `[{capture,text,start_line,end_line,start_col}]` for a tree-sitter S-expression query |

```sql
-- Rank a codebase by complexity (Tier A — no tree-sitter):
SELECT filename, ix_code_complexity(content, filename) AS cc
FROM read_text('crates/**/*.rs') ORDER BY cc DESC LIMIT 20;

-- Find every function definition via a tree-sitter query (Tier B):
SELECT filename, ix_ast_query(content, 'rust', '(function_item name:(identifier) @fn)') AS fns
FROM read_text('crates/ix-duck/src/*.rs');
```

The `language` arg is flexible — an extension (`'rs'`), a path/filename (so
`read_text`'s `filename` column works directly), or a name (`'rust'`); an
unrecognised value is a SQL error. Tier A is keyword-based and always present.
**Tier B** (`ix_ast_query`, `ix_semantic_metrics`) needs the `code-semantic`
cargo feature (enabled in this extension's build) — it pulls C-compiled
tree-sitter grammars (Rust/C#/TS/JS/F#); other languages return `parse_quality 0`.

## Build

```powershell
pwsh crates/ix-duck-ext/build.ps1              # → ix.duckdb_extension
pwsh crates/ix-duck-ext/build.ps1 -SmokeTest   # also LOAD into duckdb.exe and assert
```

Requires `cargo`, `python` (for the metadata footer), `duckdb.exe` on PATH, and a
C compiler (the `code-semantic` tree-sitter grammars build from C).
This crate is **excluded** from the ix workspace, so `cargo build --workspace`
never compiles it (preserving the "default/CI build never pulls DuckDB" invariant).

## Use

```sql
-- Unsigned local extensions need the flag (startup config):
--   duckdb -unsigned     OR     SET allow_unsigned_extensions=true; at launch
LOAD 'C:/path/to/ix.duckdb_extension';
SELECT ix_cosine([1,0]::DOUBLE[], [1,0]::DOUBLE[]);    -- 1.0
SELECT ix_euclidean([0,0]::DOUBLE[], [3,4]::DOUBLE[]); -- 5.0

-- Tier 3: query the production OPTIC-K index (mmap) directly.
SELECT count(*), any_value(len(embedding)) AS dim
FROM ix_optick_scan('C:/path/to/ga/state/voicings/optick.index');   -- 313047 | 124
-- Voicing distance composes from ix_euclidean (no dedicated UDF):
SELECT ix_euclidean(a.embedding, b.embedding)
FROM ix_optick_scan('…/optick.index') a, ix_optick_scan('…/optick.index') b
WHERE a.voicing = 0 AND b.voicing = 1;
```

### One-word launch (`ix-duck.ps1`)

Skips the `-unsigned` flag and manual `LOAD` (and builds the extension if missing):

```powershell
pwsh ix-duck.ps1                       # interactive CLI, ix_* ready
pwsh ix-duck.ps1 -Ui                   # browser UI at http://localhost:4213
pwsh ix-duck.ps1 -Database work.duckdb # open/persist a database file
pwsh ix-duck.ps1 -Sql "SELECT ix_cosine([1,0]::DOUBLE[],[1,0]::DOUBLE[])"
```

`examples.sql` has ready-made demos (kNN over a table, cosine ranking, PCA→2-D
scatter, silhouette) — open it in the UI and run cells top to bottom. It also
shows the `SET VARIABLE` + `getvariable()` bridge for feeding a table-built set
into the table functions (whose args must be constant).

## How it works (and why it's not version-locked)

Built via DuckDB's **C Extension API** (`C_STRUCT` ABI) through duckdb-rs's
`loadable-extension` feature + `#[duckdb_entrypoint_c_api]`. DuckDB passes a struct
of function pointers at `LOAD` time rather than the extension linking a specific
engine, so one artifact loads into any engine **≥ the declared `min_duckdb_version`**
(`v1.0.0` here) — verified loading into the v1.5.3 CLI. No bundled DuckDB, no C++
build.

A raw cdylib will not `LOAD` ("metadata at the end of the file is invalid"); DuckDB
requires a 512-byte footer, appended by `append_extension_metadata.py` (vendored
from `duckdb/extension-ci-tools`, MIT). `build.ps1` wires the whole chain.

Provenance / design: GA `docs/plans/2026-06-15-tools-ix-duckdb-loadable-extension-plan.md`.
