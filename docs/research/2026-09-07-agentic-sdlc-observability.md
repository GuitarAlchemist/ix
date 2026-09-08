# Agentic SDLC observability over issues, PRs, CI and traces

**Issue:** [#206](https://github.com/GuitarAlchemist/ix/issues/206) · **Companion:** [duckdb-ix-opportunities.md](duckdb-ix-opportunities.md) (#191)

**Status:** design note with a working prototype. Every metric below was computed from this
repository's real `gh` output on 2026-09-07. Nothing here is illustrative or projected.

The purpose, per #206, is to measure whether agentic work is becoming **safer, clearer,
faster and more reproducible** — not whether agents produce more code. The findings below
suggest the answer is currently *faster, but not clearer and not obviously safer*, and the
document says why with numbers.

## Headline result

`gh` emits JSON that DuckDB reads with zero preprocessing. That single fact makes the whole
use case reachable at $0 with no service, no schema migration and no ingestion pipeline:

```bash
gh pr list --repo GuitarAlchemist/ix --state merged --limit 300 \
  --json number,title,createdAt,mergedAt,author,additions,deletions,changedFiles,labels,headRefName,baseRefName \
  > prs.json

duckdb -c "SELECT count(*) FROM read_json_auto('prs.json');"   -- 209
```

**The IX DuckDB extension is not required for any metric in this document.** That is a
deliberate finding, not an omission: the SDLC observability use case is pure stock DuckDB
over `gh` JSON. The extension earns its place on the *code corpus* side (#191), not here.

## Measurement environment

| Item | Value |
|---|---|
| Repo commit | `4331cf7712a2fbd6b9c66d3580fa3d1b1509496e` |
| DuckDB | v1.5.3 (Variegata) |
| `gh` | 2.83.2 |
| Captured | 2026-09-07 |
| PR sample | 209 merged PRs, 2026-03-14 → 2026-09-08 (complete) |
| CI sample | 400 workflow runs, 2026-08-09 → 2026-09-08 (**API-capped, see limits**) |
| Issue sample | 71 issues, all states |
| Review sample | 120 most recent merged PRs (**GraphQL node-capped, see limits**) |
| Cost | $0 |

## Data source map

| Source | Command | Reachable? | Rows |
|---|---|---|---|
| GitHub PR metadata | `gh pr list --json ...` | Yes | 209 |
| GitHub PR reviews | `gh pr list --json number,reviews` | Yes, capped | 62 reviews / 120 PRs |
| GitHub issues + bodies | `gh issue list --json ...,body` | Yes | 71 |
| GitHub Actions runs | `gh run list --json ...` | Yes, capped | 400 |
| IX `state/**.jsonl` | `read_json_auto(..., format='newline_delimited')` | Yes, tiny | 102–184 rows/file |
| TARS `trace-events.jsonl` | — | **No — absent** | 0 |
| TARS closure-run artifacts | — | **No — absent** | 0 |
| AIW episode ledgers | — | **No — absent** | 0 |
| Demerzel governance decisions | — | **Not reached** | — |
| IX doctor / registry checks | — | **Not reached** | — |

Four of the ten candidate sources named in #206 produced no artifacts in this worktree. The
design below is therefore built on the six that exist, and the trace-side views are marked
as speculative rather than presented as working.

### A GraphQL gotcha that will bite the next implementer

Requesting `reviews` and `commits` together at `--limit 300` fails:

```
GraphQL: By the time this query traverses to the authors connection, it is requesting
up to 1,000,000 possible nodes which exceeds the maximum limit of 500,000.
```

Nested connections multiply. **Split the fetch**: one light query for PR scalars at
`--limit 300`, a second for `number,reviews` at `--limit 120`. This is why the review sample
is smaller than the PR sample, and why review-derived percentages below are explicitly scoped
to 120 PRs.

## Metrics, measured

### 1. Cycle time — fast, with a heavy tail

```sql
CREATE OR REPLACE VIEW pr AS
SELECT number, title, author.login AS author,
       createdAt::TIMESTAMP AS created,
       mergedAt::TIMESTAMP  AS merged,
       date_diff('minute', createdAt::TIMESTAMP, mergedAt::TIMESTAMP) AS cycle_min
FROM read_json_auto('prs.json');

SELECT count(*) AS merged_prs,
       round(median(cycle_min),1)             AS median_min,
       round(avg(cycle_min),1)                AS mean_min,
       round(quantile_cont(cycle_min,0.90),1) AS p90_min,
       max(cycle_min)                         AS max_min,
       sum(CASE WHEN cycle_min < 60 THEN 1 ELSE 0 END) AS under_1h,
       sum(CASE WHEN cycle_min < 10 THEN 1 ELSE 0 END) AS under_10min
FROM pr;
```

| merged_prs | median_min | mean_min | p90_min | max_min | under_1h | under_10min |
|---|---|---|---|---|---|---|
| 209 | **49.0** | 3056.7 | 3609.0 | 87189 | 110 | 25 |

Median 49 minutes from open to merge; 110 of 209 PRs (52.6%) merge inside an hour. That is
agent-speed and it is the strongest evidence that the "faster" axis is real.

But the **mean is 62× the median**, and the worst PR took 87 189 minutes (60.5 days). Reporting
mean cycle time here would be actively misleading. **Use the median and p90; never the mean.**
The p90 of 3 609 min (2.5 days) is the number that describes the tail worth fixing.

### 2. Review is almost entirely a bot

```sql
CREATE OR REPLACE VIEW rv AS
SELECT r.number, u.submittedAt::TIMESTAMP AS submitted, u.author.login AS reviewer
FROM read_json_auto('reviews.json') r, UNNEST(r.reviews) AS t(u);

SELECT reviewer, count(*) AS reviews, count(DISTINCT number) AS prs
FROM rv GROUP BY 1 ORDER BY reviews DESC;
```

| reviewer | reviews | prs |
|---|---|---|
| `chatgpt-codex-connector` | **58** | 56 |
| `spareilleux` | 4 | 4 |

Of 62 reviews across the 120 most recent merged PRs, **93.5% come from the Codex bot**. Human
review touched **4 PRs (3.3%)**. Review coverage overall is 56/120 = 46.7% — over half of
merged PRs received no review of any kind.

Time to first review:

```sql
SELECT round(median(lat),1) AS median_min, round(avg(lat),1) AS mean_min, count(*) AS n
FROM (SELECT p.number, date_diff('minute', p.created, min(v.submitted)) AS lat
      FROM pr p JOIN rv v ON v.number = p.number
      GROUP BY p.number, p.created) s;
```

→ median **5.0 min**, mean 108.8 min, n=56.

This is the single most important governance finding in the document, and it directly
substantiates the existing repo rule that *Codex bot comments must be read before merge*: the
bot is not a supplementary reviewer, it is **the** reviewer. `human_intervention_rate`
(a candidate metric in #206) currently measures **3.3%**.

### 3. CI failure rate by workflow — and a green-but-dead workflow

```sql
CREATE OR REPLACE VIEW runs AS
SELECT databaseId AS run_id, workflowName, headBranch, headSha, conclusion,
       createdAt::TIMESTAMP AS created,
       updatedAt::TIMESTAMP AS finished,
       date_diff('second', startedAt::TIMESTAMP, updatedAt::TIMESTAMP) AS dur_s
FROM read_json_auto('runs.json');

SELECT workflowName, count(*) AS runs,
       sum(CASE WHEN conclusion='success' THEN 1 ELSE 0 END) AS ok,
       sum(CASE WHEN conclusion='failure' THEN 1 ELSE 0 END) AS fail,
       round(100.0*sum(CASE WHEN conclusion='failure' THEN 1 ELSE 0 END)/count(*),1) AS fail_pct,
       round(median(dur_s)) AS median_s
FROM runs GROUP BY 1 HAVING count(*) >= 5
ORDER BY fail_pct DESC, runs DESC;
```

| workflowName | runs | ok | fail | fail_pct | median_s |
|---|---|---|---|---|---|
| GA Nightly Quality | 31 | 1 | 30 | **96.8** | 89 |
| CI | 62 | 37 | 25 | **40.3** | 1004 |
| Agent Blackbox | 20 | 16 | 4 | 20.0 | 553 |
| Auto-Update Demerzel Submodule | 114 | 114 | 0 | 0.0 | 13 |
| Claude Code | 32 | 0 | 0 | **0.0** | 1 |
| chatbot trace regression (nightly, advisory) | 30 | 30 | 0 | 0.0 | 722 |
| maintain gate (nightly, advisory) | 30 | 30 | 0 | 0.0 | 715 |
| Claude Code Review | 18 | 13 | 0 | 0.0 | 80 |
| QA Verdict Dispatch | 18 | 18 | 0 | 0.0 | 8 |
| Assumption Drift | 14 | 14 | 0 | 0.0 | 44 |
| Wiki Sync | 12 | 12 | 0 | 0.0 | 12 |
| Stable Surface Guard | 10 | 10 | 0 | 0.0 | 492 |

Two findings jump out, and neither is visible from a PR page.

**`GA Nightly Quality` fails 96.8% of the time** (30 of 31 runs). A nightly that has failed
every night for a month is not a signal, it is furniture.

**`Claude Code` shows 0 failures *and* 0 successes.** Drilling in:

```sql
SELECT workflowName, coalesce(conclusion,'(null/in-progress)') AS conclusion, count(*) AS n
FROM runs WHERE workflowName LIKE 'Claude Code%' GROUP BY 1,2 ORDER BY 1, n DESC;
```

| workflowName | conclusion | n |
|---|---|---|
| Claude Code | **skipped** | **32** |
| Claude Code Review | success | 13 |
| Claude Code Review | skipped | 5 |

**32 of 32 `Claude Code` runs were skipped**, with a median duration of 1 second. This is the
exact "green-but-dead" signature: a gate that opens, skips instantly, and reports no failure.
A `fail_pct = 0.0` column makes it look like the healthiest workflow in the repo. It has never
run.

**Therefore: any CI scorecard must count `skipped` as its own class.** Success rate computed
as `1 - failures/total` is wrong; it scores a workflow that never executes as perfect. Across
all 400 runs there are **37 skipped runs**.

### 4. The twelve-day red streak, recovered from data

The repository carries a known constraint: the nightly toolchain leg is pinned in `ci.yml`
because a floating nightly drifted and broke every run for twelve days. That is independently
reproducible from the run data:

```sql
SELECT created::DATE AS day, count(*) AS ci_runs,
       sum(CASE WHEN conclusion='failure' THEN 1 ELSE 0 END) AS fail,
       sum(CASE WHEN conclusion='success' THEN 1 ELSE 0 END) AS ok
FROM runs WHERE workflowName='CI' GROUP BY 1 ORDER BY 1;
```

Abridged output:

| day | ci_runs | fail | ok |
|---|---|---|---|
| 2026-08-23 | 2 | 0 | 2 |
| 2026-08-24 | 2 | 0 | 2 |
| 2026-08-25 | 1 | 1 | **0** |
| … every day … | | | |
| 2026-09-06 | 1 | 1 | **0** |
| 2026-09-07 | 13 | 4 | **9** |
| 2026-09-08 | 4 | 0 | 4 |

```sql
SELECT min(created) AS streak_start, max(created) AS streak_end,
       count(*) AS consecutive_failing_runs,
       date_diff('day', min(created), max(created)) AS span_days
FROM runs
WHERE workflowName='CI' AND conclusion='failure'
  AND created BETWEEN TIMESTAMP '2026-08-25' AND TIMESTAMP '2026-09-07';
```

→ **21 consecutive failing runs, 2026-08-25 12:54:18 → 2026-09-06 15:13:42, span = 12 days**,
with **zero successes** in that window. Recovery lands on 2026-09-07, the day the pin commit
`06cb4da` ("ci: pin the nightly matrix leg to nightly-2026-08-23", 2026-09-07T16:30:04-04:00)
merged.

The measurement matches the institutional memory exactly — twelve days. This is the
strongest available evidence that these metrics track something real, and it argues for
**`days_since_last_green` per workflow** as a first-class scorecard dimension. A single
failing run is noise; a 12-day unbroken red streak is a broken repository, and nothing in the
current GitHub UI surfaces the difference.

### 5. Stale-green detection

A check whose run predates a toolchain or base change is green about a world that no longer
exists. Operationalised as: *the branch's last successful CI run finished before another PR
merged into `main`*.

```sql
CREATE OR REPLACE VIEW last_green AS
SELECT headBranch AS branch, max(finished) AS last_green_at
FROM runs WHERE workflowName='CI' AND conclusion='success'
GROUP BY 1;

CREATE OR REPLACE VIEW base_drift AS
SELECT p.number, p.branch, p.merged, g.last_green_at,
       (SELECT max(q.merged) FROM pr q WHERE q.merged < p.merged) AS prev_merge_at
FROM pr p JOIN last_green g ON g.branch = p.branch
WHERE p.merged >= TIMESTAMP '2026-08-09';   -- runs.json coverage window

SELECT number, branch, last_green_at, prev_merge_at, merged,
       date_diff('minute', last_green_at, merged) AS green_to_merge_min
FROM base_drift WHERE last_green_at < prev_merge_at
ORDER BY green_to_merge_min DESC;
```

| number | branch | last_green_at | prev_merge_at | merged | green_to_merge_min |
|---|---|---|---|---|---|
| 283 | `codex/ix-190-tars-inventory` | 2026-09-07 21:22:56 | 2026-09-07 23:42:16 | 2026-09-08 00:31:31 | 189 |
| 286 | `codex/ix-203-fractal-duckdb` | 2026-09-07 21:25:09 | 2026-09-07 23:18:25 | 2026-09-07 23:42:16 | 137 |

**2 of the 6 PRs with CI runs in the window (33.3%) merged on a stale green.** Both are recent
agent-authored PRs, both merged 2–3 hours after their last green while `main` moved underneath
them. Neither was re-tested against the base it actually landed on.

**This sample is 6 PRs. It is directional, not significant** — see limits. The value is that
the detector demonstrably fires on real merges, not that 33.3% is a stable rate.

A second, complementary variant worth building (not computed here, because all greens in the
window postdate the pin): *last green predates the most recent change to the toolchain pin or
to `.github/workflows/**`*. That variant would have caught the entire 12-day window above.

### 6. Issue quality — the `ready-for-agent` label is not earned

```sql
CREATE OR REPLACE VIEW iss AS
SELECT number, state, body,
       list_transform(labels, lambda x: x.name) AS label_names
FROM read_json_auto('issues_body.json');

SELECT CASE WHEN list_contains(label_names,'ready-for-agent')
            THEN 'ready-for-agent' ELSE 'other' END AS bucket,
       count(*) AS n,
       round(100.0*sum(CASE WHEN body ILIKE '%acceptance criteria%' THEN 1 ELSE 0 END)/count(*),1) AS ac_pct,
       round(100.0*sum(CASE WHEN body ILIKE '%cargo %' OR body ILIKE '%pwsh %' THEN 1 ELSE 0 END)/count(*),1) AS test_cmd_pct,
       round(100.0*sum(CASE WHEN body ILIKE '%evidence_required%' THEN 1 ELSE 0 END)/count(*),1) AS evidence_pct
FROM iss GROUP BY 1 ORDER BY 1;
```

| bucket | n | ac_pct | test_cmd_pct | evidence_pct |
|---|---|---|---|---|
| other | 52 | **55.8** | 0.0 | 38.5 |
| ready-for-agent | 19 | **42.1** | 26.3 | 31.6 |

Issues labelled `ready-for-agent` have a **lower** acceptance-criteria rate (42.1%) than
unlabelled issues (55.8%), and a *lower* `evidence_required` rate. The label is applied by
intent, not gated on the properties that make an issue agent-executable.

Repo-wide, `missing_test_command_rate` is **93.0%** — only 5 of 71 issues name a command that
would verify the work.

This is the "clearer" axis, and it is the one going backwards. It also yields the cheapest
available intervention: **a label gate that refuses `ready-for-agent` unless the body contains
acceptance criteria and a verification command.**

## Scorecard

Emitted for real, not proposed:

```sql
COPY (
  SELECT 'ix' AS repo, now()::TIMESTAMP AS generated_at,
    (SELECT count(*) FROM pr)                             AS merged_prs,
    (SELECT round(median(cycle_min),1) FROM pr)           AS cycle_time_median_min,
    (SELECT round(quantile_cont(cycle_min,0.90),1) FROM pr) AS cycle_time_p90_min,
    (SELECT round(100.0*count(DISTINCT number)/120,1) FROM rv) AS review_coverage_pct,
    (SELECT round(100.0*sum(CASE WHEN reviewer='chatgpt-codex-connector' THEN 1 ELSE 0 END)/count(*),1) FROM rv) AS bot_review_share_pct,
    (SELECT round(100.0*sum(CASE WHEN conclusion='failure' THEN 1 ELSE 0 END)/count(*),1) FROM runs WHERE workflowName='CI') AS ci_failure_pct,
    (SELECT count(*) FROM runs WHERE conclusion='skipped') AS skipped_runs,
    (SELECT round(100.0*sum(CASE WHEN list_contains(label_names,'ready-for-agent') THEN 1 ELSE 0 END)/count(*),1) FROM iss) AS agent_ready_issue_rate_pct,
    (SELECT round(100.0*sum(CASE WHEN body NOT ILIKE '%acceptance criteria%' THEN 1 ELSE 0 END)/count(*),1)
       FROM iss WHERE list_contains(label_names,'ready-for-agent')) AS missing_ac_rate_agent_ready_pct,
    (SELECT round(100.0*sum(CASE WHEN NOT (body ILIKE '%cargo %' OR body ILIKE '%pwsh %') THEN 1 ELSE 0 END)/count(*),1) FROM iss) AS missing_test_command_rate_pct
) TO 'ix-agentic-sdlc-scorecard.json' (FORMAT JSON, ARRAY true);
```

Actual output, 2026-09-07:

```json
[
  {
    "repo": "ix",
    "generated_at": "2026-09-07 21:06:04.508494",
    "merged_prs": 209,
    "cycle_time_median_min": 49.0,
    "cycle_time_p90_min": 3609.0,
    "review_coverage_pct": 46.7,
    "bot_review_share_pct": 93.5,
    "ci_failure_pct": 40.3,
    "skipped_runs": 37,
    "agent_ready_issue_rate_pct": 26.8,
    "missing_ac_rate_agent_ready_pct": 57.9,
    "missing_test_command_rate_pct": 93.0
  }
]
```

### Scorecard dimensions and their direction

| Dimension | Metric | Now | Direction | Guardrail |
|---|---|---|---|---|
| **Faster** | `cycle_time_median_min` | 49.0 | lower | not at the cost of `review_coverage_pct` |
| **Faster** | `cycle_time_p90_min` | 3609.0 | lower | — |
| **Safer** | `review_coverage_pct` | 46.7 | higher | — |
| **Safer** | `bot_review_share_pct` | 93.5 | **lower** | a rise means humans disengaged further |
| **Safer** | `stale_green_rate_pct` | 33.3 (n=6) | lower | — |
| **Safer** | `skipped_runs` | 37 | lower | must never be counted as success |
| **Clearer** | `missing_ac_rate_agent_ready_pct` | 57.9 | lower | the cheapest fix available |
| **Clearer** | `missing_test_command_rate_pct` | 93.0 | lower | — |
| **Reproducible** | `days_since_last_green` per workflow | 12-day streak observed | lower | alert on ≥2 |
| **Reproducible** | `ci_failure_pct` | 40.3 | lower | exclude advisory nightlies |

Per the repo's "instrument before you ship" discipline, each dimension above has a baseline
(the "Now" column, measured 2026-09-07), an expected direction, and where meaningful a
guardrail. `bot_review_share_pct` is deliberately a *lower-is-better* metric with no floor:
optimising it upward would mean celebrating the removal of humans from the loop.

## Local-only prototype plan

Tracer bullet: one thin vertical slice through every layer, already demonstrated end to end
above.

1. **Capture** — three `gh` calls to a scratch dir. Split the reviews query (GraphQL cap).
2. **Query** — one DuckDB session; views as written above. No extension, no daemon.
3. **Emit** — `COPY ... TO 'ix-agentic-sdlc-scorecard.json' (FORMAT JSON, ARRAY true)`.
4. **Report** — render `ix-agentic-sdlc-report.md` from the scorecard.
5. **Trend** — append the scorecard to `state/quality/` per the existing snapshot convention,
   so the direction column becomes a series rather than a single reading.

Deliberately **not** in the prototype: a dashboard (#206 non-goal), any always-on service
(#206 non-goal), and any CI workflow. `.github/workflows/**` is in `blocked_paths` of
`agent-blackbox.policy.json`; automating this collection would require a workflow edit and a
human override label. **That step is intentionally left to a human** — this document changes
no CI configuration.

## How this feeds TARS AX and Demerzel

- **Demerzel** consumes `stale_green_rate_pct`, `skipped_runs` and `bot_review_share_pct` as
  governance inputs. The stale-green detector is a risk-gate candidate: a PR whose green
  predates the current base is a merge the constitution should be able to question.
  Aggregation across those signals is a natural fit for `ix_hex_consensus`, noting the
  fail-closed behaviour documented in the companion (`['T','T','F'] → F`).
- **TARS AX** consumes `missing_ac_rate` and `missing_test_command_rate` as the measurable
  form of "handoff quality" — the properties an issue needs before an agent can act on it
  unattended.
- **IX** owns the reading and the SQL. TARS owns emitting trace artifacts. Nothing here asks
  IX to run a service.

## Limits and what was not reached

Stated explicitly. Several numbers above are directional only, and saying which is the point.

1. **The stale-green sample is 6 PRs.** Only 6 merged PRs in the run-coverage window have a
   matching `CI` run keyed by `headBranch`. **33.3% is not a rate**; it is "the detector fires
   on 2 real merges". Do not quote it as a repository statistic.
2. **`gh run list --limit 400` truncates history** to 2026-08-09 → 2026-09-08. Every CI number
   is scoped to that month. The 209 PRs span six months, so PR and CI metrics are **not over
   the same period** and must not be divided into each other.
3. **Review coverage is capped at 120 PRs** by the GraphQL node limit, not chosen. 46.7% is
   coverage within that sample, not across all 209.
4. **Acceptance-criteria and test-command detection are substring heuristics**
   (`ILIKE '%acceptance criteria%'`, `ILIKE '%cargo %' OR '%pwsh %'`). They undercount issues
   that express the same thing differently. Treat 93.0% as an upper bound on the problem, and
   replace with a structured `issue_meta` parse before acting on it.
5. **Branch-to-run matching is by `headBranch`, which is not unique over time.** A recycled
   branch name would mis-join. A correct implementation should join on `headSha`.
6. **Stale-green here is a proxy.** It shows the last green predates a newer merge; it does
   **not** confirm whether the merge commit itself was re-tested. GitHub's merge-queue and
   `required_status_checks` state was not consulted.
7. **No TARS traces, closure-run artifacts or AIW episode ledgers exist in this worktree** —
   `find . -name 'trace-events*.jsonl'` returned nothing. Four of #206's ten candidate sources
   are **unmeasured, not low-value**. Every trace-derived metric in the original issue
   (`artifact_completeness`, `handoff_quality` in its trace sense) is **not reached**.
8. **Demerzel governance decisions and IX doctor/registry checks were not read.**
   `risk_gate_hit_rate` is **not reached**.
9. **`rework_rate` and `ci_failure_by_task_type` were not computed.** Both need a task-type
   classification of PRs that does not currently exist in the label set.
10. **Single repository, single point in time.** No cross-repo comparison against `ga`, `tars`
    or `Demerzel`, and no historical series — every "direction" column is an intent, not yet a
    trend.
11. **The 12-day streak reconstruction is bounded by the same 400-run window.** It happens to
    fall inside it; an older streak would be invisible.
