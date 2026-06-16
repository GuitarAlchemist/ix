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

Scalars wrap `ix_math::distance` directly; the PCA table function wraps
`ix_unsupervised` — no reimplementation, identical numbers to the in-process bench.
Table functions take a whole set as a JSON param (2-D number array; labels a 1-D int
array), so one SQL call processes the set:
`SELECT * FROM ix_pca_project('[[1,2,3],[4,5,6]]', 2);`

## Build

```powershell
pwsh crates/ix-duck-ext/build.ps1              # → ix.duckdb_extension
pwsh crates/ix-duck-ext/build.ps1 -SmokeTest   # also LOAD into duckdb.exe and assert
```

Requires `cargo`, `python` (for the metadata footer), and `duckdb.exe` on PATH.
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
