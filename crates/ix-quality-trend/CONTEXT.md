# ix-quality-trend Context

> Fresh-session orientation for this crate. Read BEFORE touching it.

## What this crate is

Aggregates timestamped JSON quality snapshots from the GuitarAlchemist ecosystem and emits an executive-readable markdown trend report (+ optional `out_json` health artifact). Three categories supported today: `embeddings/` (from `ix-embedding-diagnostics`), `voicing-analysis/` (from .NET `Demos/VoicingAnalysisAudit`, PascalCase JSON), `chatbot-qa/` (from `ga-chatbot qa --benchmark`). Schema-tolerant: every snapshot field is `Option<T>` so older files missing newer metrics still load. Computes 7-day / 30-day averages, regression flags vs configurable threshold, sparklines, and drift detection via `ix-signal::timeseries::page_hinkley`. The CLI binary `ix-quality-trend` is what CI calls; three helper binaries (`bootstrap`, `build-ci-index`, `seed-history`) cover repo setup and CI fan-in.

## Key invariants (DO NOT VIOLATE)

- Filename stem MUST parse as `YYYY-MM-DD`. Files that don't match are SILENTLY skipped — this has burned the ecosystem twice (chatbot-qa producer + embeddings producer filename drift, see GA project memory). Don't add tolerant filename matching; instead make the producer conform.
- Schema tolerance is at the FIELD level (`Option<T>`), NOT the document level. A malformed JSON file is a hard error, not silently ignored.
- Series are sorted ascending by date inside `load_all`. Downstream `trend` code relies on this invariant; don't push points after the sort.
- `TrendDirection` is part of the contract surface — `HigherIsBetter` (pass rates, consistency, Forte coverage) vs `LowerIsBetter` (leak accuracy, unknown-chord rate, invariant failures) drive arrow/checkmark rendering. Misclassifying a metric inverts the regression flag.
- `RetrievalConsistency::match_pct()` prefers `avg_pc_set_match_pct_top10` over `avg_pc_set_match_pct` — both names ship in the wild; do not collapse them.
- `CrossInstrumentConsistency::consistency_pct()` prefers an explicit `pct` field over the derived `consistent/shared_sets*100` ratio.
- `InvariantFailures::total()` sums the four named buckets only (`midi_notes_mismatch`, `null_pitch_class_set`, `negative_physical_layout`, `interval_spread_invariant`). Adding a new failure kind requires updating both the struct and `total()`.
- CLI `--baseline` is INFORMATIONAL only; the report always anchors on the most recent snapshot. Don't change this without revisiting the health-artifact contract.

## The 5-10 files that matter

- `src/lib.rs` — public surface (`build_health_artifact`, `QualityAlert`, `MetricSeries`, etc.) and the layout contract docstring.
- `src/snapshot.rs` — typed Deserialize structs for all three categories + `load_all` (the filename-date gate lives here).
- `src/trend.rs` — `MetricSeries`, `MetricTrend`, `TrendDirection`, regression/drift flagging.
- `src/report.rs` — markdown renderer + `QualityHealthArtifact` JSON shape consumed by Demerzel/ga-react dashboards.
- `src/main.rs` — `ix-quality-trend` CLI; arg validation (especially the `--baseline` date check) and exit codes.
- `src/bin/bootstrap.rs` — first-time repo scaffold (creates `embeddings/`, `voicing-analysis/`, `chatbot-qa/` subdirs).
- `src/bin/build_ci_index.rs` — fan-in of per-PR snapshots into the trend index (used by `karpathy-cherny-discipline.yml`).
- `src/bin/seed_history.rs` — backfill from existing producers when adding a new category.

## How to add a new snapshot category

1. Add a variant to `SnapshotCategory` with `dir_name()` and `display()` strings. Update `SnapshotCategory::all()`.
2. Define a `<Name>Snapshot` struct in `snapshot.rs` with `#[derive(Default, Deserialize)] #[serde(default)]`. Every field MUST be `Option<T>` for schema tolerance. Match the producer's case convention via `rename_all`.
3. Add a `Vec<DatedSnapshot<...>>` field to `SnapshotSet` and load it in `load_all`. Sort by date after load.
4. Wire metrics into `report::summarize` and the markdown renderer in `report::render`. Decide `TrendDirection` per metric.
5. Add to `is_key_metric_name` if the metric should drive `QualityHealthStatus`.
6. Producer side: write `<root>/<dir-name>/YYYY-MM-DD.json`. **Filename stem must be ISO date** or the loader will silently skip it.

## What NOT to do here

- Don't make the filename-date parse tolerant ("YYYY-MM-DD-foo.json" etc.). The silent skip is the contract, not a bug — but fix it at the producer.
- Don't surface JSON parse errors as soft warnings. Document-level malformed JSON is a hard error per the schema-tolerance discipline doc.
- Don't make `TrendDirection::Neutral` the default. Forces every metric to declare its direction; missing direction = silent regression-flag inversion.
- Don't pull in `rmp-serde` for snapshot loading — it's there for the health artifact only. Snapshots stay JSON for diffability.
- Don't add HashMap-keyed series; trend code assumes deterministic iteration order (`BTreeMap` already in `cardinality_distribution`).
- Don't merge a producer change that bumps a metric name without keeping the old name as a fallback (see `avg_pc_set_match_pct` vs `_top10`).

## Where to look for related context

- Top-level `README.md` — workspace layout, Stable-tier maturity table.
- `docs/MANUAL.md` — pipeline format + how trend reports feed Demerzel governance.
- GA project memory (`reference_quality_trend_pipeline.md`) — the two filename-mismatch incidents that shaped these invariants.
- `.github/workflows/karpathy-cherny-discipline.yml` (in GA repo) — CI consumer.
- `ix-signal::timeseries::page_hinkley_detect` — drift detector this crate composes.
