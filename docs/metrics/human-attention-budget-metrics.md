# Human Attention Budget Metrics and Rubber-Stamp Detection

Date: 2026-09-08

Status: draft / advisory — **no runtime automation** (per #216 acceptance criteria)

Refs: [ix#216](https://github.com/GuitarAlchemist/ix/issues/216)

## Purpose

Define IX-owned metrics for whether human review prompts are *earning* the attention they
cost, and for detecting when they decay into mechanical confirmation.

Companion to [`methodology-guard-usefulness-metrics.md`](methodology-guard-usefulness-metrics.md),
which already defines `useful_signal_rate` and `review_friction_delta`. **Those are not
redefined here.** This document covers the attention-allocation layer only.

## Principle

Unchanged from the companion document — reused verbatim, not re-invented:

```text
IX measures.
Demerzel governs.
.github executes.
TARS reasons.
Human decides.
```

## Headline finding: the issue's central metric is undefined in this repository

`approval_without_change_rate` cannot be computed here. Measured 2026-09-08 across the 120
most recent merged PRs:

| review state | count |
|---|---|
| `COMMENTED` | **68** |
| `APPROVED` | **0** |
| `CHANGES_REQUESTED` | **0** |

There are **zero approvals in this repository**, so the denominator of
`approval_without_change_rate` is zero, and "time between PR ready and approval" has no event
to measure. Confirmed through two independent API paths (`gh pr list --json reviews` and
`GET /repos/{o}/{r}/pulls/{n}/reviews`).

This is not a data-collection failure. The repository does not use GitHub's approval
mechanism: `main` has no required status checks, and PRs are merged directly.

**Therefore the human attention gate in this repository is the MERGE, not the approval.**
Every metric below is re-anchored on the merge event. Any design that assumes an approval
event will measure nothing here.

```text
merges in sample:                    120
distinct merge actors:                 1   (spareilleux)
human merge rate:                   100%
```

One person is the sole merge authority for every PR. That is the attention budget, in full.

## The bot question, stated plainly

85.3% of all review activity is one bot.

| reviewer | reviews | share | avg. review body |
|---|---|---|---|
| `chatgpt-codex-connector` | 58 | **85.3%** | 621 chars |
| `spareilleux` (human) | 10 | 14.7% | 0 chars |

So "human attention budget" is, in this repository, mostly a question about **a bot**: how much
of one person's attention is spent triaging one bot's output, and whether that triage is real.

Two things follow, and both matter more than the headline:

1. **The number moved.** The companion SDLC note measured this at **93.5%** on 2026-09-07
   (58 bot / 62 total). One day later it is **85.3%** (58 bot / 68 total) — the human added six
   reviews on #304 and #309. The metric is live and human engagement went *up*. A weekly report
   that quoted 93.5% as a standing fact would already be wrong.
2. **The human's average body length of 0 is a trap, not a finding.** See below.

## Metric validity — the part that matters most

*(This section is the assigned Claude lane for #216: critique metric validity and Goodhart
risk. It is placed before the metric definitions deliberately — three of the "obvious"
rubber-stamp signals are wrong, and shipping them would have made the repository look worse
the more carefully it was reviewed.)*

### V1. `review.body` length is a broken rubber-stamp signal — it inverts the truth

The human reviewer's average review body is **0 characters** across all 10 reviews. Read
naively, that is a perfect rubber-stamp signature: ten silent sign-offs.

It is the opposite. Those reviews carry substantial **inline** comments:

| PR | reviewer | inline comment lengths |
|---|---|---|
| #304 | `spareilleux` | 910, 667, 718, 976 chars |
| #309 | `spareilleux` | 1676, 966 chars |
| #291 | `spareilleux` | 513 chars |

The human's inline comments are **longer than the bot's** (bot range 564–820 on the same PRs).
GitHub records a review-with-inline-comments as a review whose top-level `body` is empty, and
files each inline comment thread as a separate review submission — which is also why #304 shows
four "reviews" 13 seconds apart.

> **Rule: never score rubber-stamping on `review.body` length.** Use
> `pulls/{n}/comments` (inline) plus `issues/{n}/comments`, or the signal ranks the most
> engaged reviews as the most mechanical.

This one defect would have been invisible without pulling the inline comments; the aggregate
looked clean.

### V2. Counting review *submissions* rewards volume, not attention

#304 produced four review records in 13 seconds — one per inline thread. A single
carefully-written review scores 1; four replies to a bot score 4. Any per-review counter must
deduplicate by `(pr, reviewer, minute)` or it measures UI mechanics.

### V3. Fast merge on a small diff is correct behaviour, and this repo already does it right

Normalising by diff size is not a refinement here; without it the headline is simply false.
Measured across 120 merged PRs, by churn (`additions + deletions`):

| churn band | PRs | merged with **no review** | no-review rate | had a **human** review |
|---|---|---|---|---|
| < 50 | 22 | 15 | **68.2%** | 0.0% |
| 50–199 | 31 | 18 | 58.1% | 0.0% |
| 200–599 | 38 | 20 | 52.6% | 7.9% |
| ≥ 600 | 29 | 11 | **37.9%** | 10.3% |

Review coverage rises **monotonically** with diff size, and human review appears only in the
two largest bands. That is correct triage: attention tracks blast radius.

The aggregate figure — "53.3% of merged PRs had no review at all" — is technically true and
substantively misleading. The unreviewed PRs are disproportionately the small ones. **A metric
that reported the total would have punished the repository for behaving correctly.**

> **Methodological rule, inherited from a mistake made elsewhere in this repo today: measure
> boundaries, not totals.** Every metric below is banded by churn. A single aggregate number is
> not an acceptable output of this schema.

### V4. Structural confound: the merger is usually the author

One person authors, reviews and merges. `CHANGES_REQUESTED` is structurally near-impossible —
you do not request changes from yourself — so its absence is *not* evidence of rubber-stamping.
Any metric keyed on `CHANGES_REQUESTED` is dead in a single-maintainer repository, and should
be reported as `null`, never as `0`.

### V5. Goodhart risks in the issue's own metric list

| metric | how it gets gamed | mitigation |
|---|---|---|
| `human_prompts_per_week` ↓ | route real decisions to auto-approve; prompts fall, misses rise | pair with `inflight_merge_rate`; never report alone |
| `approval_without_change_rate` ↓ | request trivial cosmetic changes to avoid a clean approval | require the change to alter a non-doc file |
| `bot_review_share` ↓ | human posts empty reviews to dilute the ratio — **which is exactly what an empty `body` looks like (V1)** | weight by inline-comment volume, not review count |
| `prompts_later_judged_unnecessary` ↑ | judged by the person who wants fewer prompts | judge from outcomes (was a defect found later?), not opinion |
| `batch_digest_success_rate` | batching hides urgent items in a digest | cap digest latency for Tier-3 items |

The general failure mode: **every one of these metrics improves if the human simply stops being
asked.** None may be optimised alone. See [Demerzel#696 — Anti-Goodhart Guardrails for AFK
Metrics](https://github.com/GuitarAlchemist/Demerzel/issues/696), which is the open epic for
this class of problem.

## Candidate metrics

Formula + purpose, matching the companion document's format. All are **banded by churn** per V3.

### `inflight_merge_rate`

```text
count(PRs merged while >=1 check run was still executing) / count(PRs with matched check runs)
```

Purpose:

```text
Detect merges that did not wait for their own evidence.
This is the strongest rubber-stamp signal available in a repo with no approvals.
```

### `inflight_merge_failed_rate`

```text
count(PRs where an in-flight-at-merge check later concluded 'failure') / count(PRs with matched check runs)
```

Purpose:

```text
Escalate inflight_merge_rate from "impatient" to "demonstrably wrong".
```

### `unreviewed_merge_rate_by_band`

```text
for each churn band: count(merged PRs with zero reviews) / count(merged PRs in band)
```

Purpose:

```text
Measure whether attention is allocated in proportion to blast radius.
A FLAT profile across bands is the alarm, not a high value in the small band.
```

### `human_engagement_volume`

```text
sum(length of inline review comments by humans) / count(PRs reviewed by a human)
```

Purpose:

```text
Replace review-count and body-length proxies (V1, V2) with something that
tracks actual reading. Deduplicate submissions by (pr, reviewer, minute).
```

### `bot_review_share`

```text
count(reviews by bot accounts) / count(all reviews)
```

Purpose:

```text
State plainly what fraction of "review" is machine output.
Lower is NOT automatically better - see V5 for the dilution attack.
```

### `merge_authority_concentration`

```text
count(merges by the single most frequent merger) / count(all merges)
```

Purpose:

```text
Bus-factor and attention-load on one person. Currently 100%.
```

## Rubber-stamp warning thresholds

Provisional, and — per the discipline that a check which cannot fire is not a check — **each
one was run against real repository data to confirm it both fires and can fail.** A threshold
that fires on 0% or 100% of a real corpus is not yet useful.

| id | indicator | threshold | fires on real data | verdict |
|---|---|---|---|---|
| RS1 | `inflight_merge_rate` | > 0.20 | 9/21 measurable = **42.9%** (9/120 all = 7.5%) | **fires — TRIPPED** |
| RS2 | `inflight_merge_failed_rate` | > 0.05 | 2/21 measurable = **9.5%** (2/120 all = 1.7%) | **fires — TRIPPED** |
| RS3 | unreviewed merge, any size | > 0.60 | 64/120 = **53.3%** | fires, below threshold |
| RS4 | unreviewed merge, churn ≥ 600 | > 0.15 | 11/120 = **9.2%** | fires, below threshold |
| RS5 | no *human* review, churn ≥ 600 | > 0.15 | 26/120 = **21.7%** | **fires — TRIPPED** |
| — | negative control: always-true | — | 120/120 = 100.0% | harness works |
| — | negative control: impossible | — | 0/120 = 0.0% | harness works |

The two negative controls exist so that a future "all thresholds green" result can be
distinguished from "the evaluator silently returned nothing".

**Three of five indicators are currently tripped.** RS1/RS2 are the substantive ones and are
discussed below; RS5 is expected in a single-maintainer repository (V4) and is reported for
visibility rather than as a defect.

### Threshold evaluator validation — seeded violation

Firing on real data proves a threshold is *reachable*; it does not prove the evaluator would
*catch* a violation that was not already there. So the RS4 evaluator was run against the real
corpus with synthetic rows appended, in both directions:

| scenario | observed | tripped (> 0.15)? |
|---|---|---|
| baseline (real data, 120 PRs) | 0.0917 | no |
| **+30 seeded unreviewed PRs at churn 5000** | **0.2733** | **yes — caught** |
| +200 seeded well-reviewed PRs at churn 10 | 0.0344 | no |

The evaluator catches the injected violation and moves *down* when clean evidence is added, so
it is responsive in both directions rather than merely monotone in row count. Combined with the
two negative controls, this is the minimum bar for any threshold promoted out of draft:
**fires on real data, catches a seeded violation, and can return to green.**

RS1/RS2 have not been seed-tested — their denominator is a join against check-run history, so a
synthetic row would need a fabricated SHA-to-run mapping. Recorded as a gap, not as done.

### RS1/RS2 — merges that outran their own evidence

Of the 21 merged PRs whose exact head SHA could be matched to check runs:

- **9 (42.9%) were merged while at least one check was still running.**
- **2 of those (PRs #295 and #297) had an in-flight check that subsequently FAILED.**

The obvious confound — a workflow triggered *after* the merge — was measured and **excluded**:

```text
check runs that started AFTER the merge timestamp: 0
```

Every one of the 23 in-flight run records started at or before the merge and finished after it.
The signal is real.

Merge dwell (last check completion → merge) by churn band, on the same 21 PRs:

| churn band | PRs | median dwell |
|---|---|---|
| < 50 | 2 | −9 min |
| 50–199 | 2 | −2.5 min |
| 200–599 | 5 | −9 min |
| ≥ 600 | 12 | **+26 min** |

Negative dwell means the merge preceded the last check finishing. Only the largest band waits.

## Feeding the AFK scorecard

Emit one snapshot per week to `state/quality/analytics/` following the existing snapshot
convention. The scorecard consumes the banded table, never a single score — collapsing this to
one number reintroduces exactly the error V3 documents. See the example snapshot at
[`examples/human-attention-snapshot.example.json`](examples/human-attention-snapshot.example.json).

## Feeding Demerzel review-mode routing

**IX already ships the review-mode taxonomy; do not invent a second one.**
[`crates/ix-approval`](../../crates/ix-approval) defines `Tier::{One, Two, Three}` with
`is_auto_approved()` and `requires_approval()`, where Tier Three is precisely "requires explicit
human approval". The attention budget *is* the Tier-3 volume.

| Demerzel review mode | existing IX type | who pays attention |
|---|---|---|
| auto | `Tier::One` | nobody |
| auto + audit | `Tier::Two` | nobody live; auditable after |
| decision gate | `Tier::Three` | the human |

Routing effectiveness is then measurable without new vocabulary: if Tier-3 items are being
merged with checks in flight (RS1), the gate is nominal rather than real.

Relatedly, [`crates/ix-loop-detect`](../../crates/ix-loop-detect) already implements sliding-window
repeat detection over agent actions, which is the mechanism #216's "the same prompt type repeats
frequently" indicator needs. It is keyed on `AgentAction::loop_key`, not on review prompts, so it
would need a new key — but the algorithm is not missing.

## Non-goals

```text
Do not optimize for fewer prompts. Every metric here improves if the human is simply
  never asked; that is the failure mode, not the goal.
Do not reduce the banded tables to a single attention score.
Do not score rubber-stamping on review body length (it inverts the truth).
Do not treat CHANGES_REQUESTED = 0 as evidence in a single-maintainer repo.
Do not add runtime automation in this pass (per #216).
Do not let these metrics override human judgment.
```

## Limits and what was NOT verified

1. **`approval_without_change_rate`, `outcome_changed_by_human_rate`,
   `fast_review_time_to_confidence` and `focused_review_resolution_rate` are UNMEASURABLE here** —
   all four need an approval event, and there are none. Reported as a measured negative, not
   estimated.
2. **RS1/RS2 rest on 21 PRs, not 120.** Only 21 merged PRs had a head SHA matching a run in the
   500-run window. 42.9% is a rate over the *measurable subset*; over all 120 it is 7.5%. Both
   are given above because quoting either alone misleads. **This denominator gap is the weakest
   part of the document.**
3. **`gh run list --limit 500` truncates history**, so check-derived metrics cover only recent
   PRs while review-derived metrics cover 120 and the PR total is 223.
4. **Review sample capped at 120 PRs** by the GraphQL 500k-node limit (requesting `reviews` and
   `commits` together at limit 300 fails outright).
5. **`updatedAt` is used as a proxy for check completion.** Not verified against the jobs API.
6. **"PR ready for review" time was never obtained** — `gh pr list` exposes `isDraft` but no
   ready-for-review timestamp, so `createdAt` would have to stand in. Not used, because in a
   repo with no approvals it has nothing to pair with.
7. **`prompts_later_judged_unnecessary`, `batch_digest_success_rate`,
   `escalation_due_to_missing_evidence` and `decision_gate_quality_notes` have no data source at
   all** — they require a prompt ledger that does not exist in this repository. Schema fields are
   defined; no value was computed. Not reached.
8. **Demerzel#632 and #634 are CLOSED.** They are linked as #216 requires, but the design they
   define may be settled or superseded; this was not investigated. TARS#163 and Demerzel#696 are
   open.
9. **`governance/demerzel` is an uninitialized submodule in this worktree**, so no Demerzel policy
   document was read directly. All Demerzel alignment here is inferred from issue text.
10. **Bot identification is a hardcoded login match** (`chatgpt-codex-connector`). No general
    bot-detection; a new bot would be counted as human.
11. **Single repository, two sampling days.** No cross-repo comparison and no trend series.

## Related

- [ix#216](https://github.com/GuitarAlchemist/ix/issues/216) — this work
- [Demerzel#632](https://github.com/GuitarAlchemist/Demerzel/issues/632) — Anti-Rubber-Stamp Human Review Design (**closed**)
- [Demerzel#634](https://github.com/GuitarAlchemist/Demerzel/issues/634) — Review Mode Router and Human Attention Budget (**closed**)
- [Demerzel#696](https://github.com/GuitarAlchemist/Demerzel/issues/696) — Anti-Goodhart Guardrails for AFK Metrics (open)
- [tars#163](https://github.com/GuitarAlchemist/tars/issues/163) — Review Mode Verdict for AFK Human Attention Routing (open)
- [`docs/metrics/methodology-guard-usefulness-metrics.md`](methodology-guard-usefulness-metrics.md) — companion; owns `useful_signal_rate`
- [`docs/research/2026-09-07-agentic-sdlc-observability.md`](../research/2026-09-07-agentic-sdlc-observability.md) — data path and the original 93.5% measurement
