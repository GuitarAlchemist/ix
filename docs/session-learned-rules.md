# Session-learned rules — full rationale

> The actionable directives live in [`CLAUDE.md`](../CLAUDE.md) (§ Session-learned rules, always
> loaded). This file holds the **why / evidence / detailed discipline** for each rule — read on
> demand. Appended by `/correct`. Moved out of CLAUDE.md (2026-06-20) to keep the always-loaded
> context surface lean, per `docs/methodology/agentic-engineering.md` (context hygiene).

## Always read Codex bot comments before merging a PR

Before any `gh pr merge`, run:

```bash
gh api repos/$REPO/pulls/$PR/comments --jq '.[] | select(.user.login == "chatgpt-codex-connector[bot]")'
```

Address or explicitly dismiss every P0/P1 finding. P2/P3 are advisory.

**Why:** PR #308 (ga, merged 2026-05-23) shipped with an unresolved Codex P2 that broke the README's
setup instruction on fresh checkout (the `cp` step assumed `.claude/local/` existed, but the directory
is gitignored). Codex comments are not surfaced in the standard `gh pr view` merge flow — Claude must
opt-in to see them. A 2026-05-24 sweep of the last 30 days of merged `ga` PRs found ~30 outstanding
Codex findings across 20+ PRs, including multiple P1s — silent drift, not isolated.

**How to apply:** insert into the standard "ready to merge" checklist. If Codex P0/P1 are unresolved,
do NOT merge; surface to the operator with the comment body and propose a fix. Priority parses from the
`![P{0,1,2,3} Badge]` markdown shield at the start of `body`.

## 2026-05-24 — `@ai:` annotations on claims

Before claiming an invariant, assumption, or hypothesis in code, attach an `@ai:` annotation with
truth_value + certainty per `docs/contracts/2026-05-24-ai-annotation.contract.md`. Hexavalent
T/P/U/D/F/C only; confidence per Demerzel thresholds. Example:
`// @ai:invariant arr is sorted ascending [T:test conf:0.95 src:test_search.rs:42]`.

## 2026-06-01 — `certainty := strength of live binding` (maintainability discipline)

This is how we keep code maintainable / understandable / explainable across all repos, and it makes
the 2026-05-24 rule *enforceable* rather than aspirational. A claim is only as true as what it is
bound to:

- test-bound → `T:test`; compiler/type-enforced → `T:formal-proof`; sentrux rule →
  `detected-by-sentrux`; **human-only with no executable check → cap at `P:assumed` and treat as
  perishable.** Don't write `[T]` without a binding.
- **Surface the holes.** Unenforced preconditions/assumptions are the maintainability killers — make
  them explicit as `@ai:assumption [U:uncertain]` rather than leaving them implicit (sorted-input,
  A\* admissible-vs-consistent, mmap UB, `HashTable::new(0)` panic were all invisible until annotated).
- **Drift is the lint that makes it last.** `ix-assumption-graph-drift --check` (CI:
  `.github/workflows/assumption-drift.yml`) fails a PR when a claim's anchored code changed
  (`span_drifted`) or a cited test vanished (`broken_bindings`). Re-snapshot
  `state/assumptions/annotations.snapshot.json` when a claim legitimately changes. Agents check their
  own edits via the `ix_assumption_drift` / `ix_assumption_claims` MCP tools.
- **Process discipline:** annotate **one module at a time**, each pass must drive ≥1 real fix (or it
  doesn't ship — no green-but-dead inventory); declare intent + expectations before coding (IDSD);
  **never** mass-generate annotations or use an LLM panel as the drift oracle (≈96% TPR / <25% TNR —
  it rubber-stamps drift). `@ai:` markers are plain comments, so this applies to ga/tars/Demerzel too
  via the same extractor + the JSON-on-disk contract.
