---
name: assumption-graph-loop
description: Autonomous belief-revision loop — fuse @ai: annotations + /deep-research claims into the temporal assumption graph and revise the persistent belief log over time
---

# Assumption-Graph Loop

One turn of the longitudinal belief loop for the temporal assumption graph
(`crates/ix-assumption-graph`, contract `docs/contracts/2026-05-31-assumption-graph.contract.md`):

```
@ai: annotations ─┐
                  ├─► fuse (ix-fuzzy, Belnap C synthesis) ─► revise ─► belief-events.jsonl
research claims ──┘        cross-source only                 (append on verdict change)
```

It ingests the workspace's `@ai:` annotations plus any research-domain claims,
fuses each claim's evidence, and appends a `BeliefEvent` for every claim whose
verdict changed since the last run. Re-running on unchanged evidence is a no-op
— so it is safe to schedule.

## When to Use

- On a schedule (daily / weekly), to keep the belief log current as code and
  research evolve — the "semantic story over a long period".
- After a `/deep-research` run, to fold its verified findings into the graph and
  detect where new evidence contradicts a standing assumption.
- Before a review, to surface `escalated` (Contradictory) claims.

## Run It

```bash
cargo run -p ix-assumption-graph --bin ix-assumption-graph-loop -- \
  --workspace .                                   \
  --research state/assumptions/research-claims.json   \  # optional
  --log     state/assumptions/belief-events.jsonl     \
  --trigger deep-research-reverify
```

Defaults: `--workspace .`, `--log state/assumptions/belief-events.jsonl`,
`--trigger loop`, no research file. The log is created (with parent dirs) if
absent. Output reports node count, total/appended belief events, and each flip
(`T -> C`, etc.).

## The `research-claims.json` shape

A JSON array. The **truth-value judgment** — mapping a verified finding to a
hexavalent value — is the adapter's responsibility, NOT the crate's, so this
file is the seam between `/deep-research`'s output and the graph:

```json
[
  {
    "claim": "p99 latency under 5ms",
    "truth_value": "F",
    "confidence": 0.85,
    "source": "deep-research",
    "evidence": "arxiv:2510.11822"
  }
]
```

- `truth_value`: one of `T P U D F C` (hexavalent).
- `confidence`: `0.0`–`1.0`.
- `source` (optional, default `"deep-research"`): the **independence class**.
  Distinct sources are what let fusion synthesize a contradiction — give
  genuinely independent runs distinct source labels; do NOT relabel correlated
  re-runs as independent (contract §7.1 — over-counting).
- `evidence` (optional): a confirmed-real citation.

## Converting a `/deep-research` run → `research-claims.json`

`/deep-research` returns `result.findings[]` (with `claim`, `confidence`
high/medium/low, `vote`, `sources`) and `result.refuted[]`. Suggested mapping:

| deep-research | truth_value | confidence |
|---|---|---|
| confirmed finding, confidence high | `T` | 0.85–0.95 |
| confirmed finding, confidence medium | `P` | 0.6–0.75 |
| confirmed finding, confidence low | `U` | ~0.4 |
| `refuted[]` entry | `F` | by vote margin |

**Fail-closed (contract §7.2):** a multi-judge panel confirms well but catches
invalidity poorly (~96% TPR / <25% TNR). Treat panel agreement as a gate to
*promote*, and let any credible dissent push to `U`/`D`/`C` — never relabel
correlated judges as independent sources. Verify every cited source resolves
before writing it as `evidence` (deep-research has hallucinated arXiv ids).

## Scheduling

Wrap the command in your scheduled-runner of choice (cron, the
`octo:schedule` skill, or a `state/`-driven local runner like the adversarial-qa
runner). Commit `state/assumptions/belief-events.jsonl` if you want the belief
story versioned; otherwise treat it as regenerable local state.

## What it does NOT do

- It does not invoke `/deep-research` itself — produce `research-claims.json`
  first (a `/deep-research` run + the mapping above), then run the loop.
- It does not run tests to verify code assumptions — feed test outcomes as
  additional `@ai:` annotation reconciliation or research claims.
