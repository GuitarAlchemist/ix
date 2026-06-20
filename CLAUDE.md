# ix — ML Algorithms + Governance for Claude Code Skills

Rust workspace (32 crates) implementing foundational ML/math algorithms and AI governance as composable crates, exposed via MCP server (`ix-agent`) and CLI (`ix-skill`). Part of the GuitarAlchemist ecosystem (ix + tars + ga + Demerzel).

**Crate map**: see `README.md` for the full list of 32 crates grouped by domain.

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

Repo harness verification:

```powershell
pwsh scripts/verify.ps1
```

MSRV: Rust 1.80+ (due to wgpu 28).

## Conventions

- Pure Rust; no external ML frameworks except `wgpu` for GPU compute.
- CPU algorithms use `f64` and `ndarray::Array{1,2}<f64>`; GPU uses `f32` via WGPU shaders.
- Each crate defines traits (`Regressor`, `Classifier`, `Clusterer`, `Optimizer`, etc.).
- Builder pattern for algorithm configuration; seeded RNG for reproducibility.
- Governance: all agent actions subject to Demerzel constitution (see `governance/demerzel`).
- Before adding a new graph primitive, check `docs/guides/graph-theory-in-ix.md` — IX already has 10 graph-theory modules.
- Do NOT add `petgraph`/`daggy`/`graph-rs` as new dependencies. Use `ix-graph`, `ix-pipeline::dag::Dag<N>`, `ix-search`, or `ix-topo`.

## MCP Federation

MCP peers in `.mcp.json`: **ix** (Rust, algorithms + governance), **tars** (F#, grammar + metacognition), **ga** (C#, music theory), **sentrux** (Rust, realtime structural-quality sensor), **hari** (Rust, belief-state substrate — batch tools `hari_query_belief` / `hari_snapshot` / `hari_diff` / `hari_record_observation` / `hari_consensus` plus live Phase-6 multiplexer `hari_session_open` / `hari_session_event` / `hari_session_close`). Capability registry: `governance/demerzel/schemas/capability-registry.json`.

**Realtime vs offline code analysis boundary.** Sentrux owns *realtime* structural quality (live treemap, regression gate, file-watcher feedback for in-progress agent edits). The `ix-code-*` crates own *offline* catalog work (large-corpus AST analysis, code-smell mining, code catalog snapshots that feed governance reports). When in doubt: if it's "what changed in the last 30 seconds," call sentrux; if it's "snapshot the whole codebase for the daily quality trend," use `ix-code-analyze`.

## Belief State

State lives in `state/` — beliefs, PDCA cycles, knowledge packages, snapshots. File naming: `{date}-{short-description}.{type}.json`.

For governance details, Demerzel policies, Galactic Protocol contracts, or agent persona requirements, use the `demerzel-*` skills.

## Cross-repo contracts

ix collaborates with sibling repos via JSON-on-disk contracts (the canonical handoff pattern across the GuitarAlchemist ecosystem). Sibling clones are typically peers under the same parent directory:

- **ga** (`../ga/`, .NET / C# / F# / React, music theory + RAG): consumes `state/voicings/optick.index` produced by `ix-voicings` / `ix-optick`; consumes SAE artifacts produced by `ix-optick-sae` per `ga/docs/contracts/2026-05-02-optick-sae-artifact.contract.md` (schema: `ga/docs/contracts/optick-sae-artifact.schema.json`).
- **Demerzel** (`../Demerzel/`, governance + IXQL): defines constitutions, policies, and the Galactic Protocol; orchestrates the QA Architect tribunal per `ga/docs/contracts/2026-05-02-qa-verdict.contract.md`.
- **tars** (`../tars/`, F# grammar + metacognition): cross-model theory validator.

Locked-field changes need cross-repo coordination. The `links.supersedes` pattern in `optick-sae-artifact` is how to introduce a non-breaking baseline shift without freezing the schema. Contracts marked v0.1.x are still drafts (Phase 0–3 of their respective plans); only freeze at the explicitly named Phase 4 milestone.

## Collaboration discipline

Drawn from Karpathy's skill + sohaibt/product-mode (merged, not installed). These apply to non-trivial work only — typos and one-liners skip this.

- **Surface, don't guess.** If a request has multiple plausible interpretations, list them with tradeoffs — don't pick silently. Mark each assumption as *validated / assumed / unknown*.
- **Frame problem before solution.** State who is in pain and what changes for them before proposing code. Check prior art in the workspace first — IX already has ≥10 graph modules, most ML primitives, and full governance.
- **Instrument before you ship.** Metric-moving changes declare baseline + expected direction + guardrail. Baselines live in `ga/state/quality/`, aggregated by `ix-quality-trend` → `ga/docs/quality/README.md`. Never "we'll add analytics later."
- **Log one-way doors.** Non-trivial decisions go in `docs/plans/YYYY-MM-DD-*.md` with reversibility (one-way / two-way door) and revisit trigger (metric / date / condition). One-way doors — schema hashes, public crate APIs, OPTIC-K partition layout, Galactic Protocol contracts — require explicit sign-off.

## Karpathy 4 Rules — AI coding discipline

These rules complement (don't replace) the Collaboration discipline above. They apply to every Claude proposal that touches code:

1. **Think before coding.** State your interpretation of the request + assumptions; ask one clarifying question if anything is ambiguous; wait for confirmation before writing code.
2. **Simplicity first.** Write minimum code that solves the exact problem. No speculative features, no future-proofing.
3. **Surgical changes only.** Only modify code directly related to the request. Don't refactor adjacent code, don't fix unrelated style issues.
4. **Goal-driven execution.** Transform every task into verifiable success criteria. Loop until each is demonstrably met. "Task completed" ≠ "goal achieved." Use native `/goal <condition>` (Claude Code v2.1.139+) to mechanize this — Claude keeps working across turns until an evaluator confirms the condition holds. `/digest`'s `success_criteria` field is the **declared** form; `/goal` is the **operational** driver.

Before-merge addendum: scan Codex bot comments before any `gh pr merge` — see **Session-learned rules** below for the exact command and policy.

## Tracer-bullets + vertical slices (aihero delta, 2026-06-14)

Adopted from aihero.dev as the genuine *delta* over the rules above (complements Karpathy r2 simplicity + r4 goal-driven). Counters AI's "build the whole thing at once" failure mode:

- **Tracer-bullet first.** For any non-trivial feature, build the smallest **end-to-end** slice that touches *every* layer, test it, get feedback, then expand — never build layers in isolation. aihero: "Context-window constraints make the discipline non-negotiable." (This session's spike was exactly this: held-out→train→head→eval→cross-validate as one thin slice before scaling.)
- **Vertical, not horizontal, decomposition.** Each task/PR is a thin slice cutting through all integration layers (surfacing unknowns early), not a horizontal layer.

**2026-06-14 (reversal — operator decision):** the actual aihero skills are now INSTALLED (project-scoped, `.claude/skills/`, via `npx skills@latest add mattpocock/skills --copy`, MIT, Socket/Snyk-clean): `grill-with-docs`, `grill-me`, `to-prd`, `to-issues`, `tdd`, `improve-codebase-architecture` (+ `setup-matt-pocock-skills`). The earlier "do NOT add duplicate skills — already covered by internal machinery" stance is **superseded**: the operator wants the named, documented aihero workflow (a methodology a human can read and follow), not an opaque mapping onto internal tools. Use the aihero skills directly; the internal equivalents (`brainstorming`, `ce-plan`, `ce-work`, `simplify`/`code-review`, sentrux) still stand as the IX-native path. Run `/setup-matt-pocock-skills` once to wire the issue tracker (GitHub) + `CONTEXT.md` + docs location that `grill-with-docs`/`to-prd`/`to-issues` depend on.

Self-improvement reflex: when the user corrects you, invoke `/correct` so the rule lands in this file's **Session-learned rules** section — Cherny's "most important loop" from the 2026 Sequoia talk.

## Session continuity (Cherny pattern)

- `/digest` — captures meaningful session state (cursor, in-flight, hypotheses, success criteria) to `state/digests/latest.md`. Auto-fallback via `.claude/hooks/precompact-digest.sh`; auto-injected on next session via `.claude/hooks/sessionstart-digest.sh`. See `.claude/skills/digest/SKILL.md`.
- `/learnings` — captures surprises (non-obvious facts worth grep-finding later) into `docs/solutions/<category>/<date>-<topic>.md`.
- `/correct` — turns user corrections into permanent rules in this CLAUDE.md.

The hooks are validated in CI by `.github/workflows/karpathy-cherny-discipline.yml`.

## Session-learned rules

_Appended by `/correct` when the user corrects an approach. Persists across sessions. **Full rationale + evidence for each rule:** [docs/session-learned-rules.md](docs/session-learned-rules.md) (read-on-demand — kept out of this always-loaded file per the context-hygiene discipline)._

- **Always read Codex bot comments before merging a PR.** Before any `gh pr merge`, run `gh api repos/$REPO/pulls/$PR/comments --jq '.[] | select(.user.login == "chatgpt-codex-connector[bot]")'` and block on any unresolved P0/P1 (P2/P3 advisory). Codex comments are invisible in the default `gh pr view` flow.
- **2026-05-24 — `@ai:` annotations.** Before claiming an invariant/assumption/hypothesis in code, attach an `@ai:` annotation with hexavalent truth_value + certainty per `docs/contracts/2026-05-24-ai-annotation.contract.md` (e.g. `// @ai:invariant arr is sorted ascending [T:test conf:0.95 src:test_search.rs:42]`).
- **2026-06-01 — `certainty := strength of live binding`.** Don't write `[T]` without a binding (test/compiler/sentrux); cap human-only claims at `P:assumed`; surface unenforced assumptions as `@ai:assumption [U:uncertain]`; drift-gated by `.github/workflows/assumption-drift.yml`. Annotate one module at a time, each pass driving ≥1 real fix; **never** mass-generate or use an LLM panel as the drift oracle.

## Agent skills

Per-repo config for the installed aihero/mattpocock engineering skills (see the aihero-delta note above). Configured 2026-06-14 via `/setup-matt-pocock-skills`.

### Issue tracker

GitHub Issues on `GuitarAlchemist/ix`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical defaults (`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root (`/grill-with-docs` grows them lazily). See `docs/agents/domain.md`.
