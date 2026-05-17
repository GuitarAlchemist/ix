# Agent Blackbox install audit — ix, 71 → 110 closure (2026-05-17)

This is the closure record for the seven install-audit deductions on `ix` as of 2026-05-17.

## Before

Generated `2026-05-17T20:15:59Z` via `python -m cli.agent_blackbox install-audit --repo C:\Users\spare\source\repos\ix --out-dir C:\tmp\ix-audit` against `agent-blackbox` WIP at `052db74`.

| Check | Status | Score | Why deducted |
|---|---|---:|---|
| Agent instructions | warn | 6/10 | `AGENTS.md` existed but contained no `verify/test/build/lint` keyword. |
| Agent Blackbox policy | pass | 15/15 | `agent-blackbox.policy.json` already valid. |
| GitHub Action workflow | warn | 16/20 | Missing literal `risk-report.json` reference; no `createComment`/`updateComment` step. |
| Agent Blackbox version pin | fail | 0/10 | No `AGENT_BLACKBOX_REF` env var; workflow checked out `main`. |
| Verification command | pass | 10/10 | `VERIFY_COMMAND=pwsh scripts/verify.ps1` already set. |
| Review override enforcement | fail | 0/10 | No `enforce --report` step; no `agent-blackbox-reviewed` label check. |
| Harness and quality observability | warn | 11/15 | `response-quality` step missing. |
| Loop controls | warn | 4/10 | No `agent-blackbox.loop-policy.json`, no supervised-loop preflight, no skill. Only baseline files present. |
| Review independence | warn | 9/10 | `producer-reviewer`, `fresh-evaluator`, `cross-vendor-review` patterns matched; `rewrite-budget` pattern absent. |

**Total: 71 / 110. Verdict: fail.**

## After

Generated `2026-05-17T20:18:11Z` via the same CLI invocation, after the changes in this PR.

| Check | Status | Score |
|---|---|---:|
| Agent instructions | pass | 10/10 |
| Agent Blackbox policy | pass | 15/15 |
| GitHub Action workflow | pass | 20/20 |
| Agent Blackbox version pin | pass | 10/10 |
| Verification command | pass | 10/10 |
| Review override enforcement | pass | 10/10 |
| Harness and quality observability | pass | 15/15 |
| Loop controls | pass | 10/10 |
| Review independence | pass | 10/10 |

**Total: 110 / 110. Verdict: pass.**

## Closure mapping

| Deduction | Closed by |
|---|---|
| Agent instructions `+4` | `AGENTS.md` now describes the verify/test/build gate (`pwsh scripts/verify.ps1`). |
| Workflow `+4` | `.github/workflows/agent-blackbox.yml` now references `risk-report.json` literally, adds `createComment`/`updateComment` step, and uploads artifacts. |
| Pinning `+10` | `AGENT_BLACKBOX_REF` pinned to commit SHA `942f808d3290441ab7d47e78a6b05e0e5587441d` (latest agent-blackbox `main` as of 2026-05-17). |
| Enforcement `+10` | New `enforce --report` step + `agent-blackbox-reviewed` label override check. |
| Observability `+4` | New `response-quality` step emits `dist/response-quality.{json,md}` when `dist/agent-response.txt` is captured. |
| Loop controls `+6` | New `agent-blackbox.loop-policy.json` (top-level) + `scripts/supervised-loop-preflight.ps1` (deterministic `LOOP_READY=` gate). |
| Review independence `+1` | `AGENTS.md` now includes the rewrite-budget paragraph (line budget + lines-per-fix cap) so all four review-independence patterns hit. |

## Out of scope (deliberate)

- No Rust source touched (`crates/`, `src/`, `Cargo.toml`, `Cargo.lock` are listed under `agent-blackbox.loop-policy.json::protected_paths`).
- No service restarts.
- No override labels applied.
- `governance/demerzel` submodule pointer is unchanged.

## Reproduction

```powershell
python -m cli.agent_blackbox install-audit `
  --repo C:\Users\spare\source\repos\ix `
  --out-dir C:\tmp\ix-audit
```

`C:\tmp\ix-audit\install-audit.json` will show `score: 110, verdict: pass`.
