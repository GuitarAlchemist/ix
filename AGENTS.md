# ix — Agent Notes

Read [CLAUDE.md](CLAUDE.md) first. It is the working agreement for this Rust workspace and applies to all agents (Claude, Codex, Gemini, Conductor, OpenCode).

This repo participates in cross-repo JSON-on-disk handoffs with sibling repos `ga` and `Demerzel`. See the **Cross-repo contracts** section in CLAUDE.md before changing any field listed in `governance/demerzel/schemas/` or any artifact shape consumed by another repo.

## Verification

Every agent-driven PR MUST run the local verification gate before requesting review:

```powershell
pwsh scripts/verify.ps1
```

`verify.ps1` is the same command Agent Blackbox invokes in CI via `VERIFY_COMMAND`. If it fails locally, do not push. It runs four steps, and three of them are gates:

| Step | Invocation | Blocking? |
| --- | --- | --- |
| Format | `cargo fmt --all --check` | **No — advisory.** Several crates intentionally use a terser hand style (e.g. `ix-duck`, see CLAUDE.md), so this reports diffs repo-wide. It prints a warning and continues, by design. |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` | **Yes.** Byte-identical to the `Clippy lint` step in `.github/workflows/ci.yml`. |
| Test | `cargo test --workspace` | **Yes.** Also builds the workspace. |
| Doctor | `cargo run -p ix-skill -- check doctor` | **Yes.** Single pre-PR health command (ix#185) — see below. |

Plus the supervised-loop preflight regression harness (`scripts/test-supervised-loop-preflight.ps1`), which is blocking.

So "verified" means: **clippy clean at CI's exact flags, tests green, doctor green, formatting merely reported.** `cargo fmt` is deliberately *not* mandatory in this repo — do not "fix" the fmt warning by reformatting unrelated crates. If you change the clippy flags in either `verify.ps1` or `ci.yml`, change both.

### `ix check doctor` — the one pre-PR health command

`cargo run -p ix-skill -- check doctor` is the single command a contributor or agent runs before opening or merging a PR (it also runs as part of `scripts/verify.ps1`, so a green `verify.ps1` already implies a green doctor). It checks:

- **Skill/tool inventory drift** — compares the live `#[ix_skill]` registry against the committed snapshot at [`state/registry/skills.snapshot.json`](state/registry/skills.snapshot.json). No hand-edited magic number: the snapshot is *generated* from the registry, not typed. If a PR adds or removes a skill without updating the snapshot, doctor fails and names exactly which skill names were added/removed and the fix command.
- Demerzel governance submodule + default constitution presence.
- `state/` directory presence.

If you intentionally add or remove an `#[ix_skill]`-annotated function, regenerate the snapshot and commit it as part of the same change:

```bash
cargo run -p ix-skill -- check doctor --write-snapshot
```

Exit code is hexavalent (0=T ok, 1=P warnings-only, 4=F blocking failure) — `--format json` output names the failing check under `checks[].message`.

See `docs/MANUAL.md` §8 for checklists on adding an MCP skill or a DuckDB UDF.

Agent Blackbox additionally emits a `harness-audit` and (when an agent response is captured) a `response-quality` report against every PR. Both artifacts are uploaded under `agent-blackbox-risk-report` for durable review evidence.

## Agent Blackbox operating boundaries

- The PR risk policy lives in [`agent-blackbox.policy.json`](agent-blackbox.policy.json). One-way-door paths (Rust manifests, governance schemas, migrations) force escalation when touched.
- The supervised-loop edit scope lives in [`agent-blackbox.loop-policy.json`](agent-blackbox.loop-policy.json). `crates/`, `src/`, `Cargo.toml`, and `Cargo.lock` are protected from autonomous loops; only docs, scripts, tests, and observability state are loop-eligible.
- The supervised-loop preflight ([`scripts/supervised-loop-preflight.ps1`](scripts/supervised-loop-preflight.ps1)) is the deterministic gate that must print `LOOP_READY=true` before any `/loop` or `/goal` automation runs in this repo.
- Halt markers (canonical set, defined in [`agent-blackbox.loop-policy.json`](agent-blackbox.loop-policy.json) `halt_markers`): any one of them immediately stops the loop —
  - global: `$HOME/.demerzel/HALT-ALL` or `state/.loop-halted`
  - repo: `state/quality/{domain}/.STOP` or a repo-root `.STOP`

## Review independence

Autonomous changes in ix follow a producer-reviewer split: the author skill (for example `.claude/skills/ce-work/SKILL.md`) generates the diff, and a fresh evaluator session (for example `.claude/skills/ce-compound/SKILL.md`, running in a different context with no shared state) signs off before merge. The fresh evaluator cannot self-certify its own author work — this is enforced at the harness layer by separating the writer skill from the reviewer skill and by routing risk reports through Agent Blackbox before the final approval.

Cross-vendor review is mandatory for any change touching governance schemas or one-way doors: at least one of Codex, Gemini, or a different vendor's model must independently confirm the diff is correct before the `agent-blackbox-reviewed` override label is applied.

Each supervised-loop cycle additionally honours a hard rewrite budget (also called a line budget or lines-per-fix cap): if the agent exceeds the configured max lines per fix without a passing oracle, the loop halts and surfaces a human-review request rather than continuing to thrash. The current rewrite-budget defaults are documented in [`docs/agent-blackbox/install.md`](docs/agent-blackbox/install.md) and override values are read from `state/quality/ix-harness/baseline.json`.
