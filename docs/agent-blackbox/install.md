# Agent Blackbox — install notes for ix

This document records how Agent Blackbox is wired up in `ix`, what the install audit checks, and how the supervised-loop guardrails compose with the Rust workspace.

## Files at a glance

| File | Purpose |
|---|---|
| `.github/workflows/agent-blackbox.yml` | PR risk report, harness audit, response quality, comment, enforce. Pins `AGENT_BLACKBOX_REF` to a 40-char SHA. |
| `agent-blackbox.policy.json` | PR risk policy: blocked paths, one-way doors (`Cargo.toml`, `Cargo.lock`, governance schemas), required evidence. |
| `agent-blackbox.loop-policy.json` | Supervised-loop edit scope. `crates/`, `src/`, and Cargo manifests are protected; docs, scripts, tests, and observability state are loop-eligible. |
| `scripts/verify.ps1` | Local verification gate; same command that `VERIFY_COMMAND` runs in CI. |
| `scripts/risk_report_binding.py` | Holds the risk report to the PR's changeset before `enforce` reads it (ix#316); tested by `crates/ix-harness-github-actions/tests/risk_report_binding.rs`. |
| `scripts/dev-process-overseer.ps1` | Emits `state/governance/dev-process-overseer.json` summarising loop readiness. |
| `scripts/supervised-loop-preflight.ps1` | Deterministic `LOOP_READY=true|false` gate before `/loop` or `/goal` runs. |
| `state/quality/ix-harness/baseline.json` | Loop baseline: oracle command, kill switches, lock + history file paths, scope boundary. |
| `state/governance/dev-process-overseer.json` | Last overseer rollup. |

## Rewrite budget defaults

Supervised loops in ix run with these defaults; per-cycle overrides go through `state/quality/ix-harness/baseline.json`.

- `max_lines_per_fix`: 200 (hard halt when exceeded without a passing oracle)
- `max_files_per_fix`: 10
- `max_cycles_per_session`: 6 (rewrite budget for the whole session)
- `oracle_command`: `pwsh scripts/verify.ps1` (mirrored in `baseline.json._harness.oracle_command`)

When the rewrite budget is exhausted without `LOOP_READY=true`, the loop must stop and surface a human-review request rather than continue thrashing.

## Review independence

ix uses a producer-reviewer split for autonomous changes:

- **Producer:** the compound-engineering plugin's `ce-work` skill writes the diff in the active context.
- **Fresh evaluator:** its `ce-compound` skill runs in a separate session with no shared state and reviews the artifact. The fresh evaluator cannot self-certify its own author work.
- **Cross-vendor review:** any change touching `Cargo.toml`, `Cargo.lock`, `schemas/**`, `governance/demerzel/schemas/**`, or other one-way-door paths requires a second model (Codex, Gemini, or another vendor) to confirm the diff before the `agent-blackbox-reviewed` override label is applied.

## CI flow

1. `actions/checkout@v4` for the target PR, plus a pinned checkout of `GuitarAlchemist/agent-blackbox@${AGENT_BLACKBOX_REF}`.
2. Collect evidence: `dist/changed-files.txt`, `dist/diff.patch`, `dist/test-output.txt`, `dist/agent-trace.jsonl`.
3. `analyze` → `dist/risk-report.json` + `dist/risk-report.md`.
4. `harness-audit --fail-below 60` → `dist/harness-audit.json` + `.md`.
5. `response-quality` (skipped when no `dist/agent-response.txt` is captured).
6. `scripts/risk_report_binding.py --dist dist` — fails closed unless `risk-report.json` names exactly the files of `dist/changed-files.txt`, records the digests of `dist/changed-files.txt` and `dist/diff.patch` as supplied evidence, and `risk-report.md` shows the same digests (ix#316: on #313 the two reports described different changesets). It runs before the report is shown or enforced, whether or not the label is present, since the label attests a review of this PR's report.
7. Comment the merged risk + harness + response-quality body on the PR.
8. Upload `dist/` as the `agent-blackbox-risk-report` artifact, also when an earlier step failed.
9. `enforce --report dist/risk-report.json` — exits nonzero on a failing verdict unless the `agent-blackbox-reviewed` label is present. If the PR is a draft when the job runs, a failing verdict is a warning instead (a crash or a missing verdict still fails); marking the PR ready for review runs the job again and enforces it. The label and the draft flag are read from the PR when the step runs, not from the triggering event, so re-running an old run cannot replay them. Converting a PR back to a draft does not run the job; its last result stands until the next push or label change.

## Pinning policy

`AGENT_BLACKBOX_REF` is pinned to a 40-char commit SHA, not `main`. Bump the pin in a dedicated PR that explicitly justifies the new commit (and the cross-vendor reviewer signs off on the bump).

## Manual install audit

```powershell
python -m cli.agent_blackbox install-audit `
  --repo C:\Users\spare\source\repos\ix `
  --out-dir dist
```

Target score: 110/110. See `docs/agent-blackbox/install-audit-2026-05-17.md` for the closure record from the 71 → 110 PR.
