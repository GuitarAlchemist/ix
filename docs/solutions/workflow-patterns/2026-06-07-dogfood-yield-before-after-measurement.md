---
title: "Measuring a yield improvement from an append-only ledger (the cumulative-blend trap)"
category: workflow-patterns
date: 2026-06-07
tags: [thinking-machine, dogfood, telemetry, hits-jsonl, measurement, before-after, xargs-race]
symptom: "After shipping catalog + data-binding fixes, `ix pipeline hits` reported yield_rate 0.43 — looked like a weak gain, but the real post-fix yield was 0.66. The aggregate blended pre-fix and post-fix runs."
root_cause: "hits.jsonl is an append-only cumulative ledger; the `hits` aggregator averages ALL entries since the file began, so a large pre-fix tail drags the mean down and hides the post-fix delta. A parallel `xargs -P` batch harness also corrupted the raw results (append race)."
---

# Measuring a yield improvement from an append-only ledger

## Problem

The IX thinking machine (`ix pipeline compile`: NL → PipelineSpec) logs every
translation attempt to `state/thinking-machine/hits.jsonl` (gitignored yield
ledger) with `{outcome, kind, ts_ms, ...}`. After a session that shipped three
catalog skills (`pca`/`dbscan`/`eigen`, registry 52→55) and run-time
`{param}` data-binding — the fixes that were *supposed* to close the dominant
"no inline data" refusal — I re-ran a 25-request dogfood batch and read the
instrument to confirm the gain.

`ix pipeline hits` (the built-in aggregator) reported:

```
yield_rate 0.432 · coverage_refusal 0.541 · governance_refusal 0.0135 · translate_fail 0.0 · n=74
```

`0.432` looked like a tepid result — barely up from the 0.281 pre-fix baseline,
and not the structural win the fixes promised. **It was an artifact of how the
ledger aggregates.**

Two traps hid the real number:

1. **The cumulative-blend trap (the important one).** `hits.jsonl` is
   *append-only and never reset*; the `hits` verb averages **every entry since
   the file began**. The 42 pre-fix attempts (many correct refusals before the
   catalog existed) are still in the denominator, dragging the mean toward the
   old regime. A single cumulative `yield_rate` cannot answer "did the fix
   help?" — it answers "what is the lifetime average," which is a different,
   much less useful question after an intervention.

2. **The parallel-append race (the embarrassing one).** My batch harness
   (`.ix/run_dogfood.sh`) fanned the 25 requests out with `xargs -d '\n' -P 5`,
   each worker `>>`-appending a TSV row. Concurrent appends interleaved: 2 rows
   were lost and one status was mispaired ("standardize→PCA→classifier" logged
   as `out_of_domain` when it compiles cleanly on a serial re-run). The raw
   per-request file was untrustworthy.

## Solution

**Split the ledger by timestamp, don't read the cumulative mean.** The
intervention has a known wall-clock boundary (the merge of the fixes), so
partition `hits.jsonl` on `ts_ms` and compute yield on each side:

```python
import json, time
cut = (time.time() - 60*60) * 1000          # 60-min boundary = "this batch"
rows = [json.loads(l) for l in open("state/thinking-machine/hits.jsonl")]
pre  = [r for r in rows if r["ts_ms"] <  cut]
post = [r for r in rows if r["ts_ms"] >= cut]
for name, g in (("pre", pre), ("post", post)):
    n = len(g); c = sum(r["outcome"] == "compiled" for r in g)
    print(f"{name}: {c}/{n} = {c/n:.0%}")
```

Result — the real story the cumulative mean buried:

```
pre-fix  (older ledger):   11/42 = 26%
post-fix (last 60 min):    21/32 = 66%
```

Yield **more than doubled**; on *in-domain* requests (excluding the ~10
intentionally out-of-domain / honest-boundary prompts in the batch) it's ~87%.

**For the harness race: run the batch serially, or write one file per request
and concatenate after.** Don't `>>`-append from parallel workers to one file.
The executable re-runs (`ix pipeline compile "<req>"` one at a time) are the
ground truth; the parallel TSV was only ever a convenience index.

## Lessons (grep-worthy)

- **An append-only ledger's cumulative aggregate is the wrong instrument for
  before/after.** It measures lifetime average, not the effect of an
  intervention. Partition on the intervention's timestamp boundary. (Reset /
  rotate the ledger at an intervention if you want the built-in mean to mean
  something — but partitioning is non-destructive and keeps the history.)
- **`translate_fail_rate: 0.0` is the no-confabulation guardrail.** It stayed
  0% across the batch, which is what lets you trust the yield gain as real
  compiles rather than fabricated specs. Always read it *paired* with yield: a
  yield rise with translate-fail or coverage-refusal also moving is the
  gate-loosening / confabulation signature.
- **Verify suspicious "free" compiles against real handler ops before calling
  them confabulation.** Two batch successes — `topological sort` and
  `shortest path` — weren't in the planned catalog work, so they looked like
  hallucinations. They map to **real** `ix_graph` operations
  (`topological_sort` at `handlers.rs:2026`, `shortest_path` at `:1989`):
  legitimate bonus yield, not false-accepts.
- **Never `>>` from `xargs -P` workers to a shared file.** Append races lose
  and mispair rows; the corruption is silent and looks like real data.

## Outcome of this measurement

Structural bottlenecks (catalog breadth, run-time data binding) are
**resolved**. The remaining ~34% of refusals are predominantly *correct*:
out-of-domain (haiku, restaurant booking, news scraping) and honest capability
boundaries (NN training, arbitrary-objective gradient descent), plus governance
correctly rejecting "delete the production database" (Article 3, reversibility).
The only two *genuine* remaining catalog gaps the batch surfaced are
**random-forest feature-importances** (the `random_forest` skill exists; needs
an importance output) and a **silhouette** clustering-quality score. The next
lever is incremental skill additions driven by demand — not another structural
fix.

## Related

- `state/thinking-machine/dogfood-2026-06-07-findings.md` — the increment log
  (findings #1–#4 closed).
- `[[project_thinking_machine_catalog_bottleneck]]` (memory) — the durable
  outcome fact.
- `[[feedback_telemetry_sweep_before_design]]` — drive realistic queries +
  read telemetry before picking the next feature.
- `[[feedback_green_but_dead]]` — the guardrail this measurement honors (yield
  gain must be real executing compiles, not a loosened gate).
