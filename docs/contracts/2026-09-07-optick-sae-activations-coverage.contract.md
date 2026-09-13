---
title: OPTIC-K SAE activation coverage — split semantics for feature_activations.parquet
type: contract
status: active
version: 0.1.0
date: 2026-09-07
issue: GuitarAlchemist/ix#248
producer: ix-optick-sae (crates/ix-optick-sae)
consumers: ga (state/quality/optick-sae/<date>/), any DuckDB/pandas join on optick_row
reversibility: two-way (an additive optional field plus assertions; no index schema change)
revisit-trigger: a deliberate held_out_pct > 0.10, or a second split becoming publishable
---

# OPTIC-K SAE activation coverage

Addendum to `ga/docs/contracts/2026-05-02-optick-sae-artifact.contract.md`. It
declares one thing the base contract never stated: **what fraction of the corpus
`feature_activations.parquet` actually contains, and how a consumer asserts it.**

This document changes no OPTIC-K index schema, no partition layout, and no
existing artifact field. It adds one optional block and the checks that keep it
honest.

## 1. The gap being closed

`feature_activations.parquet` carries one row per **training** voicing, keyed by
`optick_row` — that row's stable position in the full OPTIC-K index. The
held-out validation rows are absent **by design**.

Measured on this repository, 2026-09-07:

| Fact | Value | Source |
| --- | --- | --- |
| Live OPTIC-K corpus | 313,047 voicings | `ga_voicing_index_info` → `totalVoicings` |
| `feature_activations.parquet` rows (2026-07-20) | 297,395 | parquet footer, pyarrow |
| `optick_row` range | `[0, 313046]`, unique, no NULLs | pyarrow scan of the key column |
| Rows a full-corpus join drops | **15,652** | 313,047 − 297,395 |
| **Coverage** | **95.0%** — a **5.0% gap** | 297,395 / 313,047 |

The gap reported in ix#248 is unchanged: still 15,652 rows, still 5.0%.

The design is fine. The *silence* is the bug. A consumer joining `optick_row`
against the whole index gets 297,395 matched rows and no error, and 297,395
looks plausible next to 313,047. Nothing in the artifact said otherwise.

`optick_row` itself is correct — `max(optick_row) = 313046 ≥ n_train`, so the
key holds corpus positions, not split positions. This is a coverage gap, not a
keying bug.

## 2. The declaration

`optick-sae-artifact.json` gains one **required** top-level block:

```json
"activations_coverage": {
  "optick_row_split": "train",
  "n_train": 297395,
  "n_val": 15652,
  "corpus_n": 313047,
  "coverage_pct": 95.0
}
```

| Field | Meaning |
| --- | --- |
| `optick_row_split` | Which split `optick_row` enumerates. `"train"` is the only publishable value today. |
| `n_train` | Rows present in the parquet. |
| `n_val` | Held-out rows, absent from the parquet by design. |
| `corpus_n` | Corpus the split partitions. Must equal `input.corpus_size`. |
| `coverage_pct` | `round(100 * n_train / corpus_n, 2)`. |

**Required for every artifact produced from 2026-09-07 onward.** An artifact
without it is rejected by `validate_artifact`, because "how much of the corpus is
in here" has no safe default — guessing it is what ix#248 is.

The two pre-existing snapshots (2026-06-14, 2026-07-20) predate the block. They
still *parse* so tooling can report on them, and both fail verification. That is
the correct verdict, not a regression: neither declares its coverage.

## 3. Expected coverage, and what falls below it

**Floor: `coverage_pct` ≥ 90.0.**

The trainer holds out 5% by default (`--held-out-pct 0.05`) and is capped at 10%
by policy, so a compliant train-split artifact lands at 90–95%. Below 90% the
split has changed materially and every consumer's "the parquet is approximately
the corpus" assumption breaks.

Declared once in `python/optick_coverage.py::MIN_COVERAGE_PCT` and mirrored in
`src/lib.rs::MIN_COVERAGE_PCT`; `coverage_floor_matches_python_producer` reads
the Python literal and fails if the two drift.

To hold out more, raise both constants in one PR with the reason. **Do not widen
a consumer join to close the gap** — a join that silently absorbs missing rows
is the original bug wearing a fix's clothes.

## 4. Consumer assertions

Before joining `optick_row` against the OPTIC-K index, a consumer should assert:

```sql
-- 1. the artifact declares its coverage at all
activations_coverage IS NOT NULL
-- 2. the split partitions the corpus
n_train + n_val = corpus_n
-- 3. the declaration is about the corpus you are joining against
corpus_n = <live optick.index row count>
-- 4. the parquet is the size it claims
(SELECT count(*) FROM read_parquet('feature_activations.parquet')) = n_train
```

A consumer that wants full-corpus semantics must `LEFT JOIN` and handle the
`n_val` misses explicitly. There is no artifact from which the val activations
can be recovered; they were never computed.

## 5. Enforcement

Three layers, each catching what the previous one structurally cannot.

| Layer | Where | Catches | Blind to |
| --- | --- | --- | --- |
| Pre-write guard | `train.py` main, before `save_outputs` | non-additive split, before anything reaches disk | everything about the bytes |
| Declaration validation | `src/lib.rs::validate_coverage` | missing block, unknown split, non-additive counts, stale `corpus_n`, inconsistent `coverage_pct`, sub-floor coverage | whether the parquet matches |
| Physical reconciliation | `python/optick_coverage.py::reconcile` | short write, NULL key, duplicate key, out-of-range key, split-position key, rows ≠ declared | nothing on disk to read |

The first two are **self-referential** — they compare the producer's own numbers
to each other. Only the third opens the parquet. Demonstrated: truncating a
verified snapshot's parquet by 100 rows leaves the declaration green and turns
`rows_match_declared` red.

### Reconciliation assertions

Run in this order; each is reported separately so one run shows every failure.

| Assertion | Red when |
| --- | --- |
| `coverage_declared` | no `activations_coverage` block |
| `split_is_known` | `optick_row_split` is not `"train"` |
| `split_additivity` | `n_train + n_val ≠ corpus_n` |
| `coverage_pct_consistent` | declared pct ≠ recomputed pct |
| `coverage_floor` | coverage below `MIN_COVERAGE_PCT` |
| `key_present` | parquet has no `optick_row` column |
| `rows_match_declared` | parquet rows ≠ `n_train` |
| `key_non_null` | any NULL `optick_row` |
| `key_unique` | duplicate `optick_row` |
| `key_in_corpus_range` | `optick_row` outside `[0, corpus_n)` |
| `key_is_corpus_positions` | `max(optick_row) < n_train` (see caveat below) |

`key_non_null` is separate on purpose: `COUNT(DISTINCT)` and `bool_and` both
ignore NULLs, so an all-NULL key column passes uniqueness and range checks.

`key_is_corpus_positions` is the discriminator for the keying bug class. A key
of `{0..n_train-1}` is unique, in range, and exactly the declared row count — it
passes every other assertion. A seeded random train split must reach past
`n_train`.

It is a **probabilistic** argument, not a proof: a correct producer could
legitimately hold out exactly the corpus suffix, which happens for one of the
`C(corpus_n, n_val)` equally likely val sets. Negligible at production scale
(`C(313047, 15652)`), but not at toy scale — a 10-row corpus with a 1-row
holdout hits it one run in ten. Since this is a hard produce-time failure, the
assertion **stands down when a legitimate prefix is more likely than 1 in a
million**. A gate that rejects valid artifacts gets ignored, which costs more
than the case it would have caught.

## 6. Running it

Produce time is automatic: the trainer reconciles after writing the parquet and
**exits 4 without writing the artifact** if any assertion is red. The parquet and
weights stay on disk as evidence; with no artifact JSON there is nothing to
federate, so a bad snapshot cannot reach a consumer.

Any artifact JSON already in the output directory is **deleted before**
`save_outputs` overwrites the parquet and weights. Runs re-use dated
directories, so a previous run's artifact would otherwise survive a failed
reconciliation and sit there describing bytes that are no longer present — a
federatable artifact that lies. Between that unlink and a green reconciliation
the snapshot is explicitly undeclared, and an undeclared snapshot is one a
consumer refuses rather than misreads.

Auditing an existing snapshot:

```bash
ix-optick-sae verify --snapshot <ga>/state/quality/optick-sae/2026-07-20 \
                     --corpus-n 313047
```

| Exit | Meaning |
| --- | --- |
| 0 | declaration valid **and** parquet reconciles |
| 1 | declaration invalid or absent |
| 4 | parquet contradicts the declaration |
| 5 | physical check could not run — **not verified, not a pass** |

Exit 5 is deliberately distinct. `feature_activations.parquet` is gitignored
(56 MB), so it is structurally absent on a fresh checkout; "I could not check
this" must never render as "this is fine".

Because of that absence, **do not wire `verify` into a scheduled CI job over the
ga tree** — ix runners have no ga checkout, and it inverts the federation
direction (ga pulls ix outputs). The gate belongs at the artifact's birth.

## 7. Cross-repo action — NOT done here

`ga/docs/contracts/optick-sae-artifact.schema.json` sets
`"additionalProperties": false` at the root and does not list
`activations_coverage`. **Every artifact the current trainer emits therefore
fails that schema.** This was already true before this PR — PR #250 started
emitting the block without updating the consumer-side schema.

The ga-side patch (additive, non-breaking to existing readers):

```jsonc
// properties:
"activations_coverage": {
  "type": "object",
  "additionalProperties": false,
  "required": ["optick_row_split", "n_train", "n_val", "corpus_n", "coverage_pct"],
  "properties": {
    "optick_row_split": { "type": "string", "enum": ["train"] },
    "n_train":     { "type": "integer", "minimum": 0 },
    "n_val":       { "type": "integer", "minimum": 0 },
    "corpus_n":    { "type": "integer", "minimum": 1 },
    "coverage_pct": { "type": "number", "minimum": 90.0, "maximum": 100.0 }
  }
}
```

Add it to `properties`, and to `required` once the pre-#248 snapshots are
superseded. It is a ga-repo change and needs the contract owner; it is
deliberately out of scope for this ix PR.

Until it lands, ix is the authoritative declaration of these semantics.

## 8. Not covered

- **Weighted valuations.** Coverage is row counts only. Reconciling activation
  *mass* across shards is a follow-up.
- **Recovering the val split.** Publishing val activations is a producer change,
  not a contract one. If it ever ships, `optick_row_split` gains a value and
  §3's floor needs rethinking.
- **The 2026-06-14 snapshot's missing key column.** Its parquet has no
  `optick_row` at all, so it cannot be joined correctly by any means. `verify`
  reports this; regenerating it is a separate call.
