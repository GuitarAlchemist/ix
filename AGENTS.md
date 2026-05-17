# ix — Agent Notes

Read [CLAUDE.md](CLAUDE.md) first. It is the working agreement for this Rust workspace and applies to all agents (Claude, Codex, Gemini, Conductor, OpenCode).

This repo participates in cross-repo JSON-on-disk handoffs with sibling repos `ga` and `Demerzel`. See the **Cross-repo contracts** section in CLAUDE.md before changing any field listed in `governance/demerzel/schemas/` or any artifact shape consumed by another repo.

## Verification

Every agent-driven PR MUST run the local verification gate before requesting review:

```powershell
pwsh scripts/verify.ps1
```

`verify.ps1` runs `cargo fmt --all --check` and `cargo test --workspace` (build + lint + test in one shot) and is the same command Agent Blackbox invokes in CI via `VERIFY_COMMAND`. If it fails locally, do not push.

Agent Blackbox additionally emits a `harness-audit` and (when an agent response is captured) a `response-quality` report against every PR. Both artifacts are uploaded under `agent-blackbox-risk-report` for durable review evidence.

## Agent Blackbox operating boundaries

- The PR risk policy lives in [`agent-blackbox.policy.json`](agent-blackbox.policy.json). One-way-door paths (Rust manifests, governance schemas, migrations) force escalation when touched.
- The supervised-loop edit scope lives in [`agent-blackbox.loop-policy.json`](agent-blackbox.loop-policy.json). `crates/`, `src/`, `Cargo.toml`, and `Cargo.lock` are protected from autonomous loops; only docs, scripts, tests, and observability state are loop-eligible.
- The supervised-loop preflight ([`scripts/supervised-loop-preflight.ps1`](scripts/supervised-loop-preflight.ps1)) is the deterministic gate that must print `LOOP_READY=true` before any `/loop` or `/goal` automation runs in this repo.
- Halt markers: a global `$HOME/.demerzel/HALT-ALL` or a repo-local `.STOP` immediately stops the loop.

## Review independence

Autonomous changes in ix follow a producer-reviewer split: the author skill (for example `.claude/skills/ce-work/SKILL.md`) generates the diff, and a fresh evaluator session (for example `.claude/skills/ce-compound/SKILL.md`, running in a different context with no shared state) signs off before merge. The fresh evaluator cannot self-certify its own author work — this is enforced at the harness layer by separating the writer skill from the reviewer skill and by routing risk reports through Agent Blackbox before the final approval.

Cross-vendor review is mandatory for any change touching governance schemas or one-way doors: at least one of Codex, Gemini, or a different vendor's model must independently confirm the diff is correct before the `agent-blackbox-reviewed` override label is applied.

Each supervised-loop cycle additionally honours a hard rewrite budget (also called a line budget or lines-per-fix cap): if the agent exceeds the configured max lines per fix without a passing oracle, the loop halts and surfaces a human-review request rather than continuing to thrash. The current rewrite-budget defaults are documented in [`docs/agent-blackbox/install.md`](docs/agent-blackbox/install.md) and override values are read from `state/quality/ix-harness/baseline.json`.
