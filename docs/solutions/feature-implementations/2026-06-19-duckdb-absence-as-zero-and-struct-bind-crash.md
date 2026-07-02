---
title: "DuckDB over read_json_auto: absence-as-zero and struct-field bind-crashes (the lens defect class)"
category: feature-implementations
date: 2026-06-19
tags: [duckdb, ix-duck, read_json_auto, json_extract, null, coalesce, struct, union_by_name, defect-class, profiling, type_complexity]
symptom: "ix-duck lenses over GA JSON artifacts (1) reported 0 for metrics that were never measured (absence read as a real zero), and (2) hard-crashed at bind time with 'Binder Error: Could not find key X' when a nested struct field was absent from every file in the union."
root_cause: "coalesce(optional_metric, 0) conflates 'absent' with 'zero'; and nested struct access (overall.Field, response.subfield) over read_json_auto resolves the field name against the inferred STRUCT type at bind time — union_by_name masks per-file absence but NOT whole-corpus absence, so an old-only / pre-field corpus errors before a single row is read."
---

# DuckDB over `read_json_auto`: absence-as-zero & struct-field bind-crashes

A defect class that recurred across **four** ix-duck lenses (routing, chatbot, loops,
and the scalar distance UDFs) while dogfooding the DuckDB analyst bench against real
`../ga` artifacts. Two distinct failure modes, same root: **JSON is ragged over time,
and DuckDB's two ways of reaching into it both have an absence trap.**

## Failure mode 1 — `coalesce(optional_metric, 0)` reads absence as a real zero

A lens that does `coalesce(sum(metric_delta), 0)` or `coalesce(overall.accuracy, 0)`
cannot tell *"this loop made no measured change"* from *"this loop measured a delta of
exactly 0.0."* Both surface as `0`. Downstream (trend lines, net-delta summaries,
oscillation detection) then treats unmeasured runs as genuine zeros — a silent
data-quality bug that profiling on real data exposes immediately (a column that is "all
zeros" is usually "all absent").

**Fix:** let absence be `NULL`. Drop the `coalesce`; type the Rust side as `Option<f64>`.

```sql
-- before: absent and measured-zero collapse together
SELECT loop_id, coalesce(sum(metric_delta), 0) AS net_delta FROM ...

-- after: absent stays NULL, measured-zero stays 0.0 — distinguishable
SELECT loop_id, sum(metric_delta) AS net_delta FROM ...
```

```rust
// before: pub net_delta: f64        // 0.0 means... which?
pub net_delta: Option<f64>;          // None = unmeasured, Some(0.0) = measured zero
```

The same principle applies to scalar UDFs: `ix_cosine`/`ix_euclidean` were filling a
value for NULL input rows; they now read `input.flat_vector(col).row_is_null(i)` and call
`out.set_null(i)` so SQL `NULL` passes through instead of being computed as if it were a
real vector (PR #124).

## Failure mode 2 — nested struct access crashes at **bind** time on whole-corpus absence

This is the subtle one. `read_json_auto(paths, union_by_name=true, sample_size=-1)` infers
a single `STRUCT` type for a column from the union of all sampled files. Accessing a nested
field with dot syntax — `overall.inScopeAccuracy`, `response.grounding`, `step.agentId` —
resolves that field name **against the inferred STRUCT type, at bind time, before any row is
scanned**.

- `union_by_name` makes a field that is missing from *some* files NULL-fill fine.
- But if the field is missing from **every** file (an older corpus written before the field
  existed, or a schema rename), the inferred STRUCT simply has no such key, and the query
  dies with `Binder Error: Could not find key <field> in struct` — a hard crash, not a NULL.

So a lens that works on today's corpus silently becomes a crash on any pre-field / archived
corpus. We hit it as a *test* failure first (`Could not find key inscopeaccuracy`), which is
what pointed at the latent production crash.

**Fix:** reach the field through JSON instead of through the STRUCT type. `json_extract`
returns `NULL` for a missing key rather than failing to bind:

```sql
-- before: binds against the STRUCT type → crashes if the key is absent corpus-wide
SELECT overall.inScopeAccuracy AS acc FROM read_json_auto(...);

-- after: missing key → NULL, never a bind error
SELECT json_extract(to_json(overall), '$.inScopeAccuracy') AS acc FROM read_json_auto(...);
-- for the chatbot lens, materialize to_json(response) once in a CTE, then json_extract /
-- json_extract_string each subfield off the rj column.
```

Verified **byte-identical** output on the present corpus (45 real chatbot traces) before/after
the rewrite — `json_extract` is a safe swap on the happy path and only changes behavior on the
absent-field path (crash → NULL).

## Where it lived (the signature sweep)

| Lens | Mode 1 (absence=0) | Mode 2 (struct bind-crash) | Status |
|---|---|---|---|
| `routing.rs` | `coalesce(...,0)` on trend metrics | `overall.Field` struct access | fixed, PR #126 (found by profiling real data) |
| `chatbot.rs` | — | `response.{...}` + `step.{...}` | fixed, PR #127 (validated byte-identical, 45 traces) |
| `loops.rs` | `coalesce(sum(metric_delta),0)` | — | fixed, PR #127 (`net_delta: Option<f64>`) |
| `udf.rs` (scalar) | NULL row computed as data | — | fixed, PR #124 (NULL passthrough) |
| `maintain.rs`, `ood.rs` | clean | clean | swept, no occurrence |

Bonus snag fixed in the same pass: returning an `Option`-4-tuple from a public fn trips
`clippy::type_complexity` under CI `-D warnings` — extract a `pub type TrendRow = (...)` alias.

## Lessons (grep-worthy)

- **Over `read_json_auto`, `coalesce(x, 0)` is almost always a bug.** Absence is not zero.
  Let it be `NULL` / `Option<T>` and decide what zero-vs-absent means *downstream*, explicitly.
- **`union_by_name=true` protects per-file absence, not whole-corpus absence.** Dot-access on a
  nested field that no file has is a **bind-time** crash. Prefer `json_extract(to_json(obj),
  '$.field')` for any *optional* nested field; reserve struct dot-access for fields you can prove
  are present in every input forever (a one-way door).
- **The crash is invisible until you run an old corpus.** A green test + a working run on
  today's data says nothing about a pre-field archive. Add a fixture missing the field
  (`response_missing_subfields_builds_as_null`, `missing_metric_reads_as_none_not_zero`).
- **Profile the real data before optimizing or trusting a lens.** Three of the four fixes were
  found by actually running the lens over `../ga` and reading the columns — not by code review.
- **`json_extract` is a safe refactor: prove it byte-identical on the present corpus**, then it
  only diverges on the path you're trying to harden (crash → NULL).

## Related

- `[[2026-06-14-duckdb-signature-unnest-over-lambda]]` — sibling DuckDB-over-GA-JSON doc:
  flat `_signature.json` + `UNNEST` over nested list-lambda; explicit file LIST over glob for
  graceful-degrade. Same producer/consumer seam, complementary traps.
- `[[reference_dogfood_yield_measurement_gotcha]]` — profile real data; don't read cumulative
  blended means.
- `[[feedback_tests_must_use_real_data_shapes]]` — synthetic inputs the pipeline never emits
  camouflage end-to-end bugs (here: a uniform-schema fixture hid the absent-field branch).
- `[[feedback_green_but_dead]]` — green CI on the happy path ≠ correct on the ragged path.
- **Files:** `crates/ix-duck/src/{routing,chatbot,loops,udf}.rs`; PRs #124, #126, #127.
