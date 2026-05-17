# Agent Blackbox — install notes for ix

This document records how Agent Blackbox is wired up in `ix`, what the install audit checks, and how the supervised-loop guardrails compose with the Rust workspace.

## Files at a glance

| File | Purpose |
|---|---|
| `.github/workflows/agent-blackbox.yml` | PR risk report, harness audit, response quality, comment, enforce. Pins `AGENT_BLACKBOX_REF` to a 40-char SHA. |
| `agent-blackbox.policy.json` | PR risk policy: blocked paths, one-way doors (`Cargo.toml`, `Cargo.lock`, governance schemas), required evidence. |
| `agent-blackbox.loop-policy.json` | Supervised-loop edit scope. `crates/`, `src/`, and Cargo manifests are protected; docs, scripts, tests, and observability state are loop-eligible. |
| `scripts/verify.ps1` | Local verification gate; same command that `VERIFY_COMMAND` runs in CI. |
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

- **Producer:** `.claude/skills/ce-work/SKILL.md` writes the diff in the active context.
- **Fresh evaluator:** `.claude/skills/ce-compound/SKILL.md` runs in a separate session with no shared state and reviews the artifact. The fresh evaluator cannot self-certify its own author work.
- **Cross-vendor review:** any change touching `Cargo.toml`, `Cargo.lock`, `schemas/**`, `governance/demerzel/schemas/**`, or other one-way-door paths requires a second model (Codex, Gemini, or another vendor) to confirm the diff before the `agent-blackbox-reviewed` override label is applied.

## CI flow

1. `actions/checkout@v4` for the target PR, plus a pinned checkout of `GuitarAlchemist/agent-blackbox@${AGENT_BLACKBOX_REF}`.
2. Collect evidence: `dist/changed-files.txt`, `dist/diff.patch`, `dist/test-output.txt`, `dist/agent-trace.jsonl`.
3. `analyze` → `dist/risk-report.json` + `dist/risk-report.md`.
4. `harness-audit --fail-below 60` → `dist/harness-audit.json` + `.md`.
5. `response-quality` (skipped when no `dist/agent-response.txt` is captured).
6. Comment the merged risk + harness + response-quality body on the PR.
7. Upload `dist/` as the `agent-blackbox-risk-report` artifact.
8. `enforce --report dist/risk-report.json` — exits nonzero on a failing verdict unless the `agent-blackbox-reviewed` label is present.

## Pinning policy

`AGENT_BLACKBOX_REF` is pinned to a 40-char commit SHA, not `main`. Bump the pin in a dedicated PR that explicitly justifies the new commit (and the cross-vendor reviewer signs off on the bump).

## Manual install audit

```powershell
python -m cli.agent_blackbox install-audit `
  --repo C:\Users\spare\source\repos\ix `
  --out-dir dist
```

Target score: 110/110. See `docs/agent-blackbox/install-audit-2026-05-17.md` for the closure record from the 71 → 110 PR.
