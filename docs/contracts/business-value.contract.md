# Contract: `business-value` (v0.1 — draft)

**Status:** v0.1 **draft** (not frozen — freeze at the named Phase-4 milestone).
**Producer:** `ix-value catalog` (`crates/ix-value`).
**Consumers:** GA dashboard `/dev-data/value` middleware + `<StarRating>` (separate ga PR);
DuckDB (`docs/value/queries.sql`).
**Schema:** [`business-value.schema.json`](business-value.schema.json) (validates the
hand-authored **manifest**).

A **declared, decomposable, honest** business-value signal — RICE→stars — for each demo and
each ecosystem repo, federated like Streeling (one federation shape, a second payload). Value
can't be auto-derived and we refuse to fake it: every score is traceable to a rubric and a
visible rationale, and a low **Confidence** axis caps the stars (no invented precision).

## Rubric (RICE → stars)

Each item scored on **Reach, Impact, Confidence**, each `1..=5`:

```
stars   = round( geomean(R, I, C) )  clamped to [1,5]      where geomean = (R·I·C)^(1/3)
score01 = geomean(R, I, C) / 5  ∈ (0,1]                    (continuous; sorting/rollup/DuckDB)
```

- **Geometric** mean (not arithmetic) so the weakest axis dominates — `certainty := strength
  of live binding`, applied to value. Effort is omitted (value ≠ priority).
- **Confidence IS the binding:** `5` = usage telemetry confirms; `3` = reasoned but unmeasured
  (`P:assumed`); `1` = speculative.
- Worked: `(4,5,3)` → geomean 3.91 → **4★**. `(5,5,1)` → geomean 2.92 → **3★** (arithmetic mean
  would give 4 — the geomean cap is the point). *(An earlier draft mis-stated `(5,5,1)→2`; that
  is the harmonic-mean result, not the locked geometric rubric.)*

## Manifest (hand-authored, per repo)

`state/value/manifest.json` — the source of truth a human edits:

```jsonc
{
  "schema_version": "0.1.0",
  "repo": "ix",
  "items": [
    { "id": "crate/ix-optick", "kind": "demo", "title": "OPTIC-K voicing search",
      "reach": 4, "impact": 5, "confidence": 4,
      "rationale": "production voicing-search path consumed by ga" }
  ],
  "repo_score": { "reach": 4, "impact": 5, "confidence": 4, "rationale": "…" }  // optional
}
```

- `items[].kind ∈ {demo, epic}` (default `demo`); `epic` is reserved (not produced in v1).
- Axes out of `1..=5` are **skipped + counted** (with a stderr note), never silently coerced.
- `repo_score` optional — if absent, the repo rollup is the **plain mean** of item `score01`
  (reach-weighting is a revisit trigger).

## Catalog record (generated)

`state/value/catalog.jsonl` — one JSON object per line, sorted by `id`:

```jsonc
{ "schema_version": "0.1.0", "id": "crate/ix-optick", "repo": "ix", "kind": "demo",
  "title": "OPTIC-K voicing search", "reach": 4, "impact": 5, "confidence": 4,
  "stars": 4, "score01": 0.872, "rationale": "…" }
```

- Item rows keep their authored `id`; **repo rollups** are emitted as `kind:"repo"` rows with
  `id = "<repo>"`. (Drift keys on `repo`+`id`, so bare item ids can't collide across repos.)

## Federation & freshness

`ix-value catalog` reads ix + sibling `ga` (tars/Demerzel fast-follow; absent = graceful,
reported in `roots_missing`). `.github/workflows/business-value-freshness.yml` re-ingests
**ix-scoped** and fails if committed ix records drift — the green-but-dead guard. The catalog is
a derived, regenerable artifact.

## Locked surface (at Phase 4) / one-way-ish door

`items[]` axis names + range, `stars`/`score01` formulas, catalog record shape, `kind` enum.
These are the GA-readable surface — a change needs sibling coordination, logged as a one-way
door. Everything else is additive. Consumers MUST ignore unknown keys.

## Out of scope (fast-follow)
tars/Demerzel manifests (absent → "unset") · declared×derived evidence guardrail (v2) ·
epic-level scoring · reach-weighted rollups.

## Reference
- Plan: [`docs/plans/2026-06-14-003-feat-business-value-scorecard-plan.md`](../plans/2026-06-14-003-feat-business-value-scorecard-plan.md)
- Federation prior art: `crates/ix-streeling`, `docs/contracts/streeling-learning.contract.md`.
- Producer: `crates/ix-value`.
