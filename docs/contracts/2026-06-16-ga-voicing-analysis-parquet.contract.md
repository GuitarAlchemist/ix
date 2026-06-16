# Contract: `ga-voicing-analysis` — analyzable voicing slice (Parquet + JSONL)

- **Status:** v0.1 **draft** (Phase 0 — not frozen; freeze at the named Phase 4 milestone)
- **Producer:** GA (`../ga`) — exports the slice from `optick.index` + telemetry
- **Consumer:** IX `ix-duck` (`examples/ix_voicing_lens.rs`), DuckDB analyst's bench
- **Date:** 2026-06-16
- **Relates to:** `docs/DUCKDB.md` Tier 2; `docs/contracts/chatbot-trace-regression.contract.md` (same JSON-on-disk handoff pattern)

## Why

The production voicing index `state/voicings/optick.index` is a **binary mmap** — fast for search, but **not** DuckDB-readable, so the analyst's bench can't see embeddings or per-voicing metadata. GA already emits one analyzable component (search telemetry, JSONL); this contract adds the missing two (embeddings + metadata) as **Parquet**, so IX can run distribution/OOD/cluster analytics over the corpus without touching the production path. `optick.index` stays the source of truth; this is a derived, regenerable slice.

## Components

### 1. Search telemetry — `state/telemetry/voicing-search/{YYYY-MM-DD}.jsonl` (EXISTS today)

One object per query. Already emitted by GA; the lens reads it now.

| field | type | notes |
|-------|------|-------|
| `ts` | string (ISO-8601) | query time |
| `src` | string | `mcp` / `api` / … |
| `q` | string | raw query |
| `chord` | string? | parsed chord, if any |
| `results` | int | hit count |
| `top` | double? | top similarity score (absent when empty) |
| `instr` | string? | instrument filter |
| `ms` | double | latency |
| `empty` | bool | true ⇒ zero results (a coverage gap) |

### 2. Voicing metadata — `state/voicings/analysis/voicings.parquet` (Tier-2, to export)

One row per indexed voicing. Stable join key `id` links to component 3.

| column | type | notes |
|--------|------|-------|
| `id` | VARCHAR | stable voicing id (join key) |
| `instrument` | VARCHAR | guitar / bass / ukulele |
| `diagram` | VARCHAR | e.g. `7-7-x-x-x-x` |
| `midi_notes` | LIST<INT> | sounding pitches |
| `min_fret` / `max_fret` / `fret_span` | INT | shape stats |
| `chord` | VARCHAR? | canonical chord name, if labelled |
| `partition` | VARCHAR | OPTIC-K partition tag (e.g. `optk-v4-pp-r`) |

### 3. Embeddings — `embedding LIST<FLOAT>` column on the same Parquet (Tier-2, to export)

The OPTIC-K vector per voicing, same row as its metadata (one Parquet file, columns 2+3 together).

| column | type | notes |
|--------|------|-------|
| `embedding` | LIST<FLOAT> | fixed dim (declare `schema_dim`, e.g. 124 for `optk-v4-pp-r` v1.8) |
| `schema_dim` | INT | embedding dimensionality (lets the consumer validate) |
| `schema_version` | VARCHAR | OPTIC-K schema id, e.g. `optk-v4-pp-r` |

## Consumer behaviour (IX)

- `ix_voicing_lens` reads component 1 today; components 2+3 via `read_parquet(...)` when present, else a one-line hint (graceful-degrade, never an error — hermetic).
- Once the Parquet exists, the IX vector UDFs compose over `embedding`:
  `ix_pca_project` (2-D map of the corpus), `ix_kdist` (per-voicing OOD / sparsity), `ix_silhouette` over a partition labelling.

## Versioning / one-way-door notes

- **v0.1 is a draft** — column names/types may change until Phase 4. Breaking shifts use the `links.supersedes` pattern (cf. `optick-sae-artifact`) to introduce a new baseline without freezing the schema.
- `embedding` dim + `schema_version` are **locked fields** once frozen — changing them needs cross-repo coordination (CLAUDE.md "Locked-field changes").
- This contract does **not** make IX depend on GA at runtime: the slice is read offline, the consumer degrades when it's absent.

## Open (Phase 0 → 4)

1. GA picks the export trigger (corpus rebuild hook vs. a `FretboardVoicingsCLI --export-parquet` step).
2. Confirm `id` stability across rebuilds (needed as the telemetry↔metadata join key).
3. Decide whether telemetry also moves to Parquet (v0.1 keeps it JSONL — it's append-only and small).
