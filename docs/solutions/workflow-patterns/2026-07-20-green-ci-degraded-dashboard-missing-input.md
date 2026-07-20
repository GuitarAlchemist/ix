---
title: "Green CI + DEGRADED dashboard = a missing input, not a metric regression"
category: workflow-patterns
date: 2026-07-20
tags: [green-but-dead, ci, gitignore, quality-scorecard, oracle-status, maintain-gate, readme-drift, diagnosis]
symptom: "A quality domain reads DEGRADED on the ecosystem scorecard (metric_value 0, oracle_status warn/error) while its producer workflow reports success on every recent run — or reports failure for weeks with nobody noticing, because the data kept flowing."
root_cause: "A load-bearing input never reaches the environment the producer runs in (usually because it is gitignored), so the gate is structurally undecidable there. The job still exits 0 and still publishes a snapshot, so a structural gap is rendered as a metric regression."
---

# Green CI + DEGRADED dashboard = a missing input

## Problem

Two domains on the ecosystem scorecard showed as DEGRADED with active
regressions. Neither was a metric regression. Both were diagnosed in one
session (2026-07-20) and both turned out to be the same disease.

**Symptom A — `maintain-gate`.** Scorecard: `metric_value: 0`,
`oracle_status: warn`, summary `"metric evidence missing — cannot decide"`.
Workflow `maintain-gate-nightly.yml`: **success on 8 consecutive runs.**

**Symptom B — `readme-drift`.** Scorecard: `metric_value: 0.3333`,
`oracle_status: error`, regressing `0.4444 → 0.3333`. Workflow
`readme-drift-sensor.yml`: **failure on every run for five weeks**
(2026-06-15, 06-22, 06-29, 07-06, 07-13) — while still committing snapshots the
dashboard consumed.

The pair is instructive: in A the job was green and the gate was dead; in B the
job was red and nobody looked, because the *data* was fresh. Both produce a
plausible-looking number on the dashboard.

## Investigation attempts that did NOT find it

- **Reading the metric value and the trend.** Both look like ordinary
  regressions. `readme-drift` even has a coherent story (READMEs age, ratio
  falls), which is exactly why it survived.
- **Reading the verdict code.** `ix-duck`'s `fuse()` in
  `crates/ix-duck/src/maintain/verdict_fusion.rs:50-55` is correct — it returns
  `("U", "escalate", "metric evidence missing — cannot decide")` when
  `metric_up` is `None`. Failing safe, not failing.
- **Reasoning from a plausible bug.** In B, a real PowerShell typo was found
  first — `${($stale.Count)}` instead of `$($stale.Count)` — and it was
  convincing enough to nearly ship as *the* fix. It is a genuine bug, but it
  lives in a branch that had never executed. Reading the actual job log is what
  corrected this.

## Root cause

**A load-bearing input never reaches the environment the producer runs in.**

For `maintain-gate`, `.gitignore:79` excludes the metric ledger deliberately:

```
# ix "thinking machine" translation ledger — high-volume runtime instrumentation
# (every NL→pipeline outcome). … the raw rows are per-environment runtime data,
# not a committed artifact.
state/thinking-machine/hits.jsonl
```

It is also the *only* source of the metric lens. So on a CI checkout:

1. `metric_delta()` — `crates/ix-duck/src/maintain/lens_runner.rs:44-47` —
   hits `if !hits_path.exists() { return Ok(None) }`
2. `metric_up = None` (`lens_runner.rs:78`)
3. `fuse()` returns `U / escalate / "metric evidence missing"`
4. The job exits 0, because producing that verdict is not an error
5. `build_snapshot` writes `metric_value: verdict.metric_delta.unwrap_or(0.0)`
   — `crates/ix-duck/src/maintain/ledger.rs:193` — so **0.0 lands on the
   dashboard**, indistinguishable from "the gate ran and the metric fell"

For `readme-drift` the missing input was an *authorization* rather than a file:
the `readme-drift` label did not exist in the repo, so
`gh issue create --label readme-drift` hard-failed with
`could not add label: 'readme-drift' not found`. Only the tracking-issue step
died; survey, snapshot-commit and artifact-upload all succeeded, so the data
kept flowing and the red went unnoticed.

## Solution

**Do not publish a verdict the environment cannot earn.** Skip instead.

`crates/ix-duck/examples/ix_maintain_snapshot.rs` already had the right
precedent one guard above — `absent-ga → skip (exit 0)`. The fix extends the
same rule to the other load-bearing input:

```rust
// absent-hits → skip (exit 0), mirroring absent-ga above. hits.jsonl is
// gitignored by design, so a CI checkout never has it. Publishing the
// resulting "cannot decide" as a SNAPSHOT puts metric_value 0.0 on the
// dashboard, where a structural gap is indistinguishable from a regression.
if !hits.exists() {
    eprintln!(
        "ix_maintain_snapshot: no metric evidence at {} — skipping (exit 0). \
         hits.jsonl is per-environment runtime data; the maintain gate is only \
         decidable where the harness has written it.",
        hits.display()
    );
    return Ok(());
}
```

Verified both paths — present writes a snapshot and the gate decides; absent
(clean cwd + `GA_ROOT` set, i.e. the CI shape) skips with exit 0 and writes
nothing. 36 maintain tests pass. Shipped as ix#238.

For label-shaped missing inputs, self-heal before use — the idiom already in
`jules-auto-delegate.yml`, which hit the identical failure in its run #2:

```bash
gh label create readme-drift --color 5319E7 \
  --description "Tracking issue for cross-repo README staleness" --force
```

## Prevention

**Diagnostic order for any DEGRADED quality domain — do this before reading the
metric:**

1. `gh run list --workflow <producer>.yml --limit 8` — is the producer even
   passing? A wall of `failure` means the metric is not what is broken.
2. If green: check per-step conclusions, not just the job. `gh run view <id>
   --json jobs` reveals a single failing step inside a "successful" job.
3. Trace every input the gate reads and ask **"does this file exist on a fresh
   checkout?"** Run `git check-ignore -v <path>` on each one. A gitignored
   input is structurally absent in CI, always.
4. Only then read the metric.

**Design rule:** a producer must distinguish *"I could not run here"* from
*"I ran and the result is bad."* Collapsing both into
`metric_value: 0 + oracle_status: warn` guarantees a permanent false regression
that everyone learns to ignore. Prefer skip-with-a-reason over publishing an
unearned verdict.

**Corollary:** a gate whose inputs are per-environment runtime data is only
decidable in that environment. That is a legitimate design, but it means the
scheduled CI run can never be the gate — at best it is a liveness check.

## Related

- [Federating an ix producer snapshot into ga/state/quality](../ecosystem-integration/2026-06-21-federate-ix-snapshot-into-ga-state-quality.md)
  — the GA-pull federation channel these snapshots travel through.
- [Dogfood yield before/after measurement](./2026-06-07-dogfood-yield-before-after-measurement.md)
  — the `hits.jsonl` ts_ms-split measurement that `metric_delta` implements.
- Third instance of the same class, for pattern confirmation: the embedding
  topology baseline was "dark since May" because a 175MB gitignored index never
  reached CI — same shape, different artifact.

## Surfaced, not fixed

With `hits.jsonl` present the maintain gate does decide, and returns
`C / reject / +0.0184 — "REWARD-HACK: metric improved while a held capability
regressed"`. **Do not act on that verdict as it stands.** It fuses a metric from
`hits.jsonl` (every row timestamped 2026-06-07) with a guardrail from the live
`ga/state/quality/chatbot-qa` corpus (current) — disjoint time windows. A
reward-hack signature manufactured from two unrelated measurement periods is not
evidence. The deeper fix is for `fuse()` to refuse when lens windows do not
overlap, rather than silently comparing June to July.
