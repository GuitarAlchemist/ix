---
date: 2026-06-14
topic: business-value-scorecard
---

# Business-Value Scorecard — RICE stars for demos + repos

## What We're Building

A **declared** business-value score (rendered as 1–5 ★) for each GA demo and
each ecosystem repo (ix / ga / tars / Demerzel), surfaced in the GA dashboard
at `demos.guitaralchemist.com/test#dev/summary`. Source-of-truth is a
per-repo `state/value/manifest.json`; an ix generator federates them into
`state/value/catalog.jsonl` (DuckDB-queryable, drift-gated) — the **same
pattern as Streeling University** (`docs/contracts/streeling-learning.contract.md`).

This is the "value" sibling of the Streeling "learnings" catalog: both are
per-repo declared JSON → ix generator → catalog.jsonl → `/dev-data` + DuckDB →
dashboard. One federation shape, two payloads.

## Why This Approach

Business value **cannot be auto-derived** — and the GA dashboard deliberately
refuses to fake it: `OverviewSection.tsx`'s `MetadataConventionCard` states
*"ETA, business value, and priority are not tracked yet"* and resists inventing
it. The ecosystem's established answer is **operator-declared with a binding**
(`@ai:business-value` annotations, schema v2, already scored file-level by the
`ValueComplexityHeatmap` on `#dev/annotations`).

So stars must trace to a declared rubric, never invented precision (else it's a
Sentinel's-Void metric over nothing). We chose **RICE/ICE → stars** over
operator-direct-stars (no decomposition), derived-only (usage ≠ value,
cold-start), and declared×derived-hybrid (deferred — the evidence-binding
guardrail is the v2 fast-follow).

## Key Decisions

- **Rubric (locked):** each demo/repo scored on **Reach, Impact, Confidence**,
  each `1–5`. `stars = round( geomean(R, I, C) )` = `round( (R·I·C)^(1/3) )`,
  clamped `[1,5]`. **Geometric mean, not additive** — the weakest axis pulls the
  score down, so a high-reach/high-impact demo with low Confidence can't claim
  4★ on assumption alone. This is `certainty := strength of live binding`
  applied to value. (Effort is omitted — this is *value*, not *priority*.)
- **Confidence axis = the binding.** `Confidence` is not "how excited are we" —
  it is *how well-evidenced the value claim is*: 5 = usage telemetry confirms;
  3 = reasoned but unmeasured (`P:assumed`); 1 = speculative. It is the
  `@ai:business-value` `conf:` term, made explicit as an axis.
- **Scope (locked):** the 4 GA demo cards (`DevelopmentSection.tsx` `devLinks`)
  **+ a per-repo rollup** for ix/ga/tars/Demerzel. Repo score = mean of its
  declared item scores (on the 0–1 scale, then → stars); a repo with a directly
  declared `repo` entry uses that instead.
- **Source-of-truth (locked):** per-repo `state/value/manifest.json`, federated
  by a new ix generator (`ix-value`, plain Rust, NO duckdb dep — DuckDB only
  *reads* the catalog, mirroring `ix-streeling`) → `state/value/catalog.jsonl`.
- **Contract:** `docs/contracts/business-value.contract.md` (v0.1 draft) +
  `business-value.schema.json`. Cross-repo, JSON-on-disk, format-not-runtime —
  a one-way-ish door needing sibling coordination (log in `docs/plans`).
- **Freshness:** `ix-value check` staleness/drift gate + CI workflow, repo-scoped
  (mirrors `streeling-freshness.yml` — `extra` filtered to scanned repos so a
  single-repo checkout won't false-flag a sibling).
- **UI:** stars on the demo cards in `#dev/summary` + a cross-repo "Repo
  Scorecard" row. GA reads `state/value/catalog.jsonl` via a `/dev-data/value`
  middleware endpoint (same mechanism as `/dev-data/ai-annotations`).
- **Compose, don't rebuild:** reuse the Streeling generator scaffolding, the
  `@ai:business-value` grammar/semantics, the `/dev-data` middleware pattern,
  and the existing star/Chip UI vocabulary.

## Open Questions

- **Repo rollup weighting** — plain mean vs reach-weighted mean of item scores?
  (Plan default: plain mean; revisit if it misranks.)
- **tars/Demerzel manifests** — ship ix+ga manifests in v1; tars/Demerzel
  manifests are the first fast-follow (they can be empty/stub and still rollup
  to "unset"), same as the Streeling adapter sequencing.
- **Should demo `value` reconcile with `@ai:business-value` file annotations?**
  Deferred to the hybrid v2 (declared × derived evidence guardrail).

## Next Steps
→ `/ce:plan` for implementation details (ix-value crate + contract + GA UI slice).
Tracer-bullet: one demo card with a real manifest entry → generator → catalog →
`/dev-data/value` → stars rendered, end-to-end, before scaling to all demos+repos.
