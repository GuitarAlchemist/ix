# Human Attention Budget — Weekly Report (EXAMPLE)

**Repo:** GuitarAlchemist/ix · **Generated:** 2026-09-08 · **Status:** advisory

> **This is a format example.** The numbers are real (measured 2026-09-08 against this
> repository) so the format can be judged against real data rather than plausible-looking
> filler. Nothing here is automated — see [ix#216](https://github.com/GuitarAlchemist/ix/issues/216),
> "no runtime automation in first pass". Generated from
> [`human-attention-snapshot.example.json`](human-attention-snapshot.example.json).

## Summary

One person merged all 120 PRs in the sample. 85.3% of review activity was one bot. Three of
five rubber-stamp indicators are tripped, and two of them are substantive.

The repository's attention *allocation* is healthy — review coverage rises with diff size. Its
attention *timing* is not: 43% of measurable merges did not wait for their own checks.

## 1. Where the attention went

| | |
|---|---|
| Merges in sample | 120 |
| Distinct merge actors | **1** |
| Merge authority concentration | **100%** |
| Reviews total | 68 |
| Bot share of reviews | **85.3%** (was 93.5% on 2026-09-07) |
| Approvals | **0** |

There are no approvals in this repository. The attention gate is the **merge**. Every metric
below is anchored there; a design assuming an approval event would report nothing.

Bot share fell 8.2 points in one day because the human added six reviews (#304, #309). The
metric is live — do not quote last week's figure as a standing fact.

## 2. Allocation — is attention going where the risk is?

| churn band | PRs | median churn | merged unreviewed | had human review |
|---|---|---|---|---|
| < 50 | 22 | 13 | 68.2% | 0.0% |
| 50–199 | 31 | 107 | 58.1% | 0.0% |
| 200–599 | 38 | 372 | 52.6% | 7.9% |
| ≥ 600 | 29 | 1188 | **37.9%** | **10.3%** |

**Verdict: healthy.** Coverage rises monotonically with diff size and human review appears only
in the two largest bands. Attention tracks blast radius.

> The aggregate — "53.3% of merged PRs had no review" — is true and misleading. The unreviewed
> PRs are disproportionately the small ones. Report the bands; never the total.

## 3. Timing — did merges wait for their evidence?

Measured on the 21 PRs whose head SHA matched a check run.

| | |
|---|---|
| Merged with ≥1 check still running | **9 / 21 (42.9%)** |
| …of which a check later **failed** | **2** — PRs #295, #297 |
| Post-merge re-run confound | **0 runs** — signal confirmed genuine |

Median dwell (last check finished → merge):

| churn band | median dwell |
|---|---|
| < 50 | −9 min |
| 50–199 | −2.5 min |
| 200–599 | −9 min |
| ≥ 600 | **+26 min** |

Negative dwell = merged before the last check finished. Only the largest band waits.

**Verdict: this is the finding of the week.** Two PRs were merged while a check was running
that then reported failure.

## 4. Threshold status

| id | indicator | threshold | observed | n | status |
|---|---|---|---|---|---|
| RS1 | in-flight merge rate | > 0.20 | **0.429** | 21 | 🔴 tripped |
| RS2 | in-flight check later failed | > 0.05 | **0.095** | 21 | 🔴 tripped |
| RS3 | unreviewed merge, any size | > 0.60 | 0.533 | 120 | 🟢 |
| RS4 | unreviewed merge, churn ≥ 600 | > 0.15 | 0.092 | 120 | 🟢 |
| RS5 | no human review, churn ≥ 600 | > 0.15 | **0.217** | 120 | 🟡 expected |
| NC-TRUE | negative control | = 1.00 | 1.000 | 120 | ✅ harness ok |
| NC-FALSE | negative control | = 0.00 | 0.000 | 120 | ✅ harness ok |

RS5 is expected in a single-maintainer repository and is shown for visibility, not as a defect.
The negative controls are reported every week so that "all green" can be distinguished from "the
evaluator returned nothing".

## 5. Not measurable this week

Reported as `null`, never as `0`:

- **No approval event** → `approval_without_change_rate`, `outcome_changed_by_human_rate`,
  `fast_review_time_to_confidence`, `focused_review_resolution_rate`.
- **No prompt ledger** → `prompts_by_review_mode`, `prompts_later_judged_unnecessary`,
  `batch_digest_success_rate`, `escalation_due_to_missing_evidence`, `decision_gate_quality_notes`.

**Weakest number in this report:** RS1/RS2 rest on 21 of 120 PRs, because check-run history is
truncated at 500 runs. Over the full sample the same rates read 7.5% and 1.7%. Both framings are
shown deliberately; quoting either alone misleads.

## 6. Recommended reading order for the maintainer

1. §3 — the two PRs merged against a failing in-flight check.
2. §2 — confirm allocation is still monotone; a **flat** profile is the alarm, not a high value
   in the small band.
3. §1 — bot share direction.

Everything else is context.

## Interpretation notes

- Every metric in this report improves if the human is simply never asked. None may be
  optimised alone.
- Do not score engagement on review body length: the human's 0-char reviews carry 513–1676 char
  inline comments.
- `CHANGES_REQUESTED = 0` is structural here (author ≈ merger), not evidence of rubber-stamping.
- Verdict is **P (Probable)**, not T: one repository, two sampling days, 21-PR denominator on the
  substantive indicators.
- Advisory only. IX measures; Demerzel governs; the human decides. This report never gates a merge.
