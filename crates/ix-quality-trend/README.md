# ix-quality-trend

Aggregates timestamped quality snapshots from the GuitarAlchemist ecosystem
(embedding diagnostics, voicing analysis audits, chatbot QA) and emits an
executive-readable markdown trend report with deltas, regression flags, and
sparklines.

## Layout

```text
<snapshots-dir>/
  embeddings/YYYY-MM-DD.json
  voicing-analysis/YYYY-MM-DD.json
  chatbot-qa/YYYY-MM-DD.json
```

## Loader behaviour

The loader is **schema-tolerant** at the field level (every typed field is
`Option`) and **filename-tolerant** at the directory-walk level. Concretely:

1. **Filename prefix match** — a stem starting with `YYYY-MM-DD` loads with
   that date. Trailing suffixes are allowed: `2026-05-15-soak.json` loads
   with date `2026-05-15`. (Before 2026-05-17 the loader rejected anything
   that wasn't *exactly* `YYYY-MM-DD.json`, silently. That was the bug.)
2. **`timestamp` field fallback** — if the filename has no date prefix, the
   loader parses the JSON's top-level `timestamp` field (RFC3339 or
   date-only).
3. **mtime fallback** — last resort. Records `date_source = "mtime"` in the
   manifest so the consumer knows the date is approximate.
4. **Skipped only if all three fail** — and even then, it is a **warning on
   stderr plus a manifest entry**, never a silent drop.

### Modes

| Flag | What it does |
|------|--------------|
| _(default)_ | Warns on undatable/empty/malformed files. Falls back to `timestamp` field or mtime when possible. Backward compatible. |
| `--strict` | Promotes undatable-with-no-fallback **and** JSON parse failures to hard errors (exit 1). Empty files still warn-and-skip — they aren't an authoring mistake. |
| `--quiet` | Suppresses per-file warnings on stderr. Manifest still records them. |
| `--manifest <path>` | Writes a JSON audit trail with one entry per file: `{path, category, status, date, date_source, note}`. Use this in CI to verify producers wrote what you expect. |

`status` is one of:

- `loaded` — happy path, dated from filename.
- `loaded-fallback-date` — loaded via `timestamp` field or mtime; investigate
  if a stable filename was expected.
- `skipped-date-unparseable` — no filename date, no `timestamp` field, no
  mtime. Hard error in `--strict`.
- `skipped-empty` — zero-byte file.
- `failed-parse` — JSON parse error. Hard error in `--strict`.

### Library API

```rust
use ix_quality_trend::{load_with, LoadOptions};

let (set, manifest) = load_with(
    snapshots_dir,
    LoadOptions { strict: false, quiet: false },
)?;
```

The historical `load_all(dir)` entry point is preserved and is exactly
equivalent to `load_with(dir, LoadOptions::default())` followed by
discarding the manifest.

### Why this matters

Two production consumers were burned by the original silent skip:

- **ga-chatbot-qa** — its producer landed on 2026-05-16 (ga#218); before
  that the absence was hidden because the loader silently dropped
  `baseline.json` and `last.json` snapshots.
- **embeddings** — the producer wrote `2026-04-17-postrefactor.json`, which
  the strict-prefix matcher rejected. The snapshot was on disk and visible,
  yet missing from every trend report.

After this change both files surface in the manifest and contribute to the
report. Pointing the loader at `state/quality/` in ga now produces:

```text
ix-quality-trend: loaded …/embeddings/baseline.json using fallback date source mtime
ix-quality-trend: loaded …/chatbot-qa/baseline.json using fallback date source mtime
ix-quality-trend: loaded …/chatbot-qa/last.json   using fallback date source timestamp-field
ix-quality-trend: loader audit — 0 skipped, 3 loaded-via-fallback.
```
