---
title: "Business-Value Scorecard — RICE stars for demos + repos"
type: feat
status: completed
date: 2026-06-14
origin: docs/brainstorms/2026-06-14-business-value-scorecard-brainstorm.md
---

# Business-Value Scorecard — RICE stars for demos + repos

## Overview

Render a **declared** business-value score (1–5 ★) for each GA demo and each
ecosystem repo (ix / ga / tars / Demerzel) in the GA dashboard
(`demos.guitaralchemist.com/test#dev/summary`). Source-of-truth is a per-repo
`state/value/manifest.json`; a new plain-Rust ix generator (`ix-value`)
federates them into `state/value/catalog.jsonl`, drift-gated in CI, read by GA's
`/dev-data/value` middleware and DuckDB. Mirrors the just-shipped Streeling
federation (`docs/contracts/streeling-learning.contract.md`,
`crates/ix-streeling`) — one federation shape, a second payload.

(see brainstorm: `docs/brainstorms/2026-06-14-business-value-scorecard-brainstorm.md`)

## Problem Statement / Motivation

Business value can't be auto-derived, and the GA dashboard deliberately refuses
to fake it (`OverviewSection.tsx` `MetadataConventionCard`: *"ETA, business
value, and priority are not tracked yet"*). The demos are unranked title cards;
there is no cross-repo way to ask "which surfaces actually matter." We need an
**honest, declared, decomposable** value signal — traceable to a rubric, not
invented precision (Sentinel's-Void guard).

## Proposed Solution

**RICE/ICE → stars**, federated like Streeling.

- **Rubric:** each demo/repo scored on **Reach, Impact, Confidence**, each `1–5`.
  `stars = round( geomean(R,I,C) ) = round( (R·I·C)^(1/3) )`, clamped `[1,5]`.
  Geometric mean (weakest axis dominates) so low **Confidence** caps the score —
  `certainty := strength of live binding` applied to value. Effort omitted (value
  ≠ priority). Also emit `score01 = geomean/5 ∈ [0,1]` for sorting/DuckDB.
- **Confidence axis IS the binding:** `5` = usage telemetry confirms; `3` =
  reasoned but unmeasured (`P:assumed`); `1` = speculative.
- **Repo rollup:** if a repo declares a `repo` entry, use it; else mean of its
  item `score01` → stars. (Plain mean v1; reach-weighted is a revisit trigger.)
- **Federation:** per-repo `state/value/manifest.json` → `ix-value catalog`
  (reads ix + sibling ga; tars/Demerzel fast-follow, absent = graceful) →
  `state/value/catalog.jsonl`. NO duckdb dep in the generator.

## Technical Approach

### Contract (Phase 0)

- `docs/contracts/business-value.contract.md` (v0.1 **draft**, not frozen) +
  `docs/contracts/business-value.schema.json`.
- **Manifest** (`state/value/manifest.json`, hand-authored per repo):
  ```json
  {
    "schema_version": "0.1.0",
    "repo": "ga",
    "items": [
      { "id": "demo/prime-radiant", "kind": "demo", "title": "Prime Radiant",
        "reach": 4, "impact": 5, "confidence": 3,
        "rationale": "flagship governance viz; no usage telemetry yet" }
    ],
    "repo_score": { "reach": 4, "impact": 5, "confidence": 3,
                    "rationale": "..." }   // optional; else rolled up from items
  }
  ```
- **Catalog record** (`state/value/catalog.jsonl`, generated, one obj/line):
  `{ schema_version, id, repo, kind, title, reach, impact, confidence, stars,
     score01, rationale }` where `kind ∈ {demo, repo, epic}`. Repo rollups are
  emitted as `kind:"repo"` rows (`id = "<repo>"`). Sorted by `id`.
- Logged as a **one-way-ish door** (cross-repo format contract; sibling
  coordination to change locked fields).

### ix-value crate (Phases 1–2)

`crates/ix-value` — plain Rust, workspace member, **NOT** an MCP tool/skill
(no `tools.rs`/`parity.rs` changes). Deps: serde / serde_json / clap / chrono
(workspace) + `walkdir`, `anyhow`; dev-dep `tempfile`. NO duckdb. Mirror
`ix-streeling` module layout:

- `src/model.rs` — `SCHEMA_VERSION`, `Manifest`, `Item`, `RepoScore`,
  `ValueRecord`; `fn stars(r,i,c) -> u8` (geomean, round, clamp 1..=5);
  `fn score01(r,i,c) -> f64`; `make_id`.
- `src/ingest.rs` — locate per-repo `state/value/manifest.json`; parse; compute
  per-item records + repo rollup record; tolerant (missing manifest → repo
  reported in `roots_missing`, not fatal; malformed → skip+count).
- `src/check.rs` — `DriftReport { missing, extra, changed }`; **repo-scoped**
  (`extra` filtered to scanned repos), copied from `ix-streeling::check`.
- `src/lib.rs` — `default_roots(ix_root)` (canonicalize ix_root, add sibling
  `ga` from parent); `to_jsonl` / `from_jsonl`.
- `src/main.rs` — clap `Cmd::{Catalog, Check}` (no `campus`; UI is GA-side).
- Validation: clamp/validate axes to `1..=5` at ingest; out-of-range → skip+count
  + stderr note (don't silently coerce).

### Freshness gate (Phase 2)

`.github/workflows/business-value-freshness.yml`, repo-scoped, modeled on
`.github/workflows/streeling-freshness.yml` (which we just validated catches real
drift): re-ingest ix-only, fail if committed ix records ≠ fresh. `@ai:invariant`
on catalog completeness + staleness, bound to tests.

### Seed manifests (Phase 3)

- `ix/state/value/manifest.json` — `repo:"ix"` + a few high-value items
  (e.g. governance, ix-optick, ix-streeling) + `repo_score`.
- `ga/state/value/manifest.json` — the 4 demo items (Ecosystem Roadmap, Prime
  Radiant, GA Chat, Grothendieck DSL) with **real** RICE declarations +
  `repo_score`. (Committed to the ga repo on its own branch/PR.)
- Run `ix-value catalog` → commit `state/value/catalog.jsonl`.

### GA UI slice (Phase 4) — separate ga PR

- **`/dev-data/value` middleware** in `vite.config.ts` — copy the
  `/dev-data/assumption` cross-repo pattern verbatim:
  `const ixRoot = process.env.IX_ROOT || path.resolve(repoRoot, '../ix')`,
  read `state/value/catalog.jsonl`, 404 gracefully if absent. Returns
  `{ generated_at, records }`.
- **Parser + unit test** in `src/dev-data/parsers.ts` (`parseValueCatalog`,
  rollup helpers) — tested like `parseBacklog`.
- **`<StarRating value={stars} />`** small presentational component (MUI
  `Rating` readOnly, or ★/☆ Typography) + tooltip showing `R·I·C` + rationale.
- **Demo cards:** extend `devLinks` rendering in `DevelopmentSection.tsx`
  `DashboardLinks` to look up each card's record by `id` and render stars.
  Cards with no record show "unset" (no fake zero).
- **Repo Scorecard row:** new card on `#dev/summary` (in `OverviewSection` or a
  sibling) rendering `kind:"repo"` records as a row of `repo ★★★★☆`.
- **Crossover-skip Playwright** spec (`waitFor + test.skip` idiom) per the
  ga dashboard convention — don't fail first push on live deploy.

### DuckDB + docs (Phase 5)

- `docs/streeling/queries.sql` sibling or `docs/value/queries.sql` — ≥2 DuckDB
  queries over `state/value/catalog.jsonl` (top demos by score01; repo
  leaderboard). Note in `docs/DUCKDB.md` as a second registrar payload.
- Cross-link `docs/LEARNING.md` (value = the "steer" signal) + the Streeling
  campus.

## System-Wide Impact

- **Interaction graph:** generator is offline (CI + manual); GA middleware is
  dev-server-only (`vite build` strips it — same caveat as all `/dev-data/*`,
  already logged in `DevelopmentSection` Operational TODO).
- **Error propagation:** missing sibling manifest → `roots_missing` (generator)
  / 404 (middleware), never fatal. Malformed → skip+count.
- **State lifecycle:** catalog is a derived, regenerable artifact; no orphan risk.
- **API surface parity:** the generator CLI (`catalog`/`check`) is the only
  entry; no MCP/skill surface to keep in parity (explicitly out of scope).
- **Integration test:** ingest fixture dir (well-formed + malformed + absent
  sibling) → assert records + rollup + drift detection.

## Acceptance Criteria

- [ ] `docs/contracts/business-value.contract.md` + `.schema.json` (v0.1 draft).
- [ ] `crates/ix-value` builds; `cargo clippy -p ix-value --all-targets -- -D warnings` clean.
- [ ] `stars()` = `round(geomean(R,I,C))` clamped `[1,5]`; unit-tested incl. the
      (4,5,3)→4 case and a low-confidence cap case (e.g. (5,5,1)→2).
- [ ] `ix-value catalog` federates ix+ga (graceful when `../ga` absent) →
      `state/value/catalog.jsonl`, sorted by id, with `kind:"repo"` rollups.
- [ ] `ix-value check` drift gate is repo-scoped; CI workflow added.
- [ ] Seed manifests for ix + ga committed; catalog regenerated + committed.
- [ ] GA `/dev-data/value` endpoint returns the federated catalog (cross-repo
      read), 404s gracefully; `parseValueCatalog` unit-tested.
- [ ] Stars render on the 4 demo cards + a Repo Scorecard row on `#dev/summary`;
      "unset" shown for records-absent (no fake zero).
- [ ] Tests: ingest (fixture), malformed-skip, absent-sibling, drift, stars math.
- [ ] `@ai:invariant` on catalog completeness + staleness bound to tests.

## Dependencies & Risks

- **Cross-repo PRs:** ix (crate + contract + ix manifest + catalog) and ga (UI +
  ga manifest) are **two PRs**. The ga PR's stars need the ix catalog present at
  `../ix/state/value/catalog.jsonl` — but the middleware degrades gracefully, so
  ga can merge independently and light up once ix's catalog lands.
- **Subjectivity:** RICE inputs are operator estimates. Mitigated by the
  Confidence axis (low conf ⇒ low stars) + visible rationale; not a truth oracle.
- **green-but-dead:** the freshness gate + a real seed manifest (not empty)
  guard against shipping an inert catalog.

## Out of Scope (fast-follow)

- tars/Demerzel manifests (v1 = ix + ga; absent rolls up to "unset").
- Declared × derived **evidence guardrail** (hybrid v2 — reconcile declared
  value against usage telemetry / completeness / QA).
- Epic-level scoring (wiring the `MetadataConventionCard` HTML-comment
  convention) — `kind:"epic"` is reserved in the schema but not produced in v1.

## Sources & References

- **Origin brainstorm:** `docs/brainstorms/2026-06-14-business-value-scorecard-brainstorm.md`
  — decisions carried forward: RICE→stars (geomean), demos+repo-rollup scope,
  per-repo-manifest federation.
- Federation prior art: `crates/ix-streeling`, `docs/contracts/streeling-learning.contract.md`,
  `.github/workflows/streeling-freshness.yml`.
- Value prior art: `ga/.../components/AiAnnotations/ValueComplexityHeatmap.tsx`
  (`@ai:business-value` scoring), `ga/.../pages/OverviewSection.tsx:406`
  (`MetadataConventionCard`).
- Cross-repo `/dev-data` read pattern: `ga/.../vite.config.ts:1484`
  (`/dev-data/assumption` reads IX `IX_ROOT || ../ix`).
- Demo list: `ga/.../pages/DevelopmentSection.tsx:68` (`devLinks`).

## Implementation notes (ix federation PR shipped 2026-06-15)

Branch `feat/business-value-scorecard`. This PR ships the **ix-side federation** (Phases 0–3 +
5 ix-side); the **GA UI (Phase 4)** is the immediate follow-on ga PR (it reads this catalog).
- **`ix-value` crate** built mirroring `ix-streeling` (model/ingest/check/lib/main); 9 tests,
  clippy clean. CLI `catalog | check`.
- **Math correction:** the locked rubric is `round(geomean(R,I,C))`. The plan's acceptance
  example `(5,5,1)→2` was a miscalculation — geomean(5,5,1)=2.92 → **3** (2 is the harmonic-mean
  result). Implemented + tested as geometric; `(4,5,3)→4`, `(5,5,1)→3`.
- **Schema included** (`business-value.schema.json`) — it validates the hand-authored manifest
  (a real consumer: the human author), unlike the chatbot contract which had no consumer yet.
- **Seed manifests:** ix committed here (5 items + rollup); ga authored locally for federation,
  **committed in the ga PR**. Catalog regenerated (11 records), DuckDB queries verified live.
- **`@ai:invariant`** on `stars()` + `ingest()`, test-bound; freshness workflow added (ix-scoped).
- ix-side only (no ix-streeling change needed; the value generator is self-contained).
