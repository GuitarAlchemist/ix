# Audit: workflows green-but-dead (ix + Demerzel)

- **Date:** 2026-05-24
- **Author:** Claude (Opus 4.7, 1M context)
- **Type:** Audit only (no code changes in this PR)
- **Reversibility:** Two-way door (advisory findings)
- **Revisit trigger:** Demerzel submodule bump that touches `seldon-plan.yml`

## Why this audit

ga's session memory `feedback_green_but_dead` (2026-05-24) recorded
that Demerzel Seldon produced **50 days of green CI runs while
emitting zero artifacts**, because the agent invocation had been
commented out and replaced with a `echo "skipped"` placeholder.

This audit swept every workflow in ix and in the Demerzel submodule
for the same anti-pattern: workflows that pass CI while doing nothing
useful. Smells looked for:

- Commented-out agent invocations with surrounding scaffold intact
- Silent mocks (`echo "would-have-run"` then `echo "success"`)
- `continue-on-error: true` swallowing real failures
- No-op summary steps after every other step was skipped
- Dead schedules where the body is permanently `if: false`

## Workflows enumerated

### ix (`.github/workflows/`, 11 files)

| File | Verdict |
|---|---|
| `adversarial-qa.yml` | Honest — see note below |
| `agent-blackbox.yml` | Honest — real evidence collection + risk-report + enforce |
| `ci.yml` | Honest — full workspace build/test/clippy/showcase smoke |
| `claude-code-review.yml` | Honest — real `anthropics/claude-code-action@v1` |
| `claude.yml` | Honest — real `anthropics/claude-code-action@v1` |
| `ga-nightly-quality.yml` | Honest — real artifacts + health gate |
| `karpathy-cherny-discipline.yml` | Honest — schema/hook/skill validation |
| `qa-verdict-dispatch.yml` | Honest — real `repository_dispatch` to Demerzel |
| `stable-surface.yml` | Honest — real diff/enforce on Stable-tier API hashes |
| `submodule-auto-update.yml` | Honest — real submodule pointer bump or PR |
| `wiki-sync.yml` | Honest — degrades gracefully + documents PAT requirement |

**ix verdict:** zero genuine green-but-dead findings.

**Borderline case (no change applied):** `adversarial-qa.yml` lines
94–96 emit a `::notice::` "LLM judge panel not yet wired" as a step.
This is project-tracking, not deception — the deterministic regression
gauge is real and fails loudly on mismatches (lines 81–84). Per the
spec's "surgical changes only" constraint, no change applied.

### Demerzel (`governance/demerzel/.github/workflows/`, 17 files)

| File | Verdict |
|---|---|
| `cross-model-review.yml` | Honest — real Claude/Gemini/Codex API calls |
| `demerzel-autofix.yml` | Honest — real Claude API triage |
| `demerzel-capability-expansion.yml` | Honest — real Claude API (degrades on missing key) |
| `demerzel-cross-repo-issues.yml` | Honest — real `gh` cross-repo scans |
| `demerzel-discussion-lifecycle.yml` | Honest — real Claude API + graphql |
| `demerzel-discussion-responder.yml` | Honest — real Claude API |
| `demerzel-driver-triggers.yml` | Honest — writes real trigger files |
| `demerzel-ideation.yml` | Honest — real Claude API + issue creation |
| `demerzel-self-improvement.yml` | Honest — real gap detection + Claude prioritization |
| `demerzel-showcase.yml` | Honest — static templated content (no agent claim) |
| `demerzel-wiki-cross-repo.yml` | Honest — real wiki regeneration |
| `ga-chatbot-discussions.yml` | Honest — three-tier resolution (Claude/MCP/static) |
| `governance-validate.yml` | Honest — real schema validator |
| `project-automation.yml` | Honest — real project board automation |
| **`seldon-plan.yml`** | **GREEN-BUT-DEAD** — see below |
| `streeling-daily.yml` | Honest — real artifact counting + discussion posting |
| `submodule-notify.yml` | Honest — real dispatch + issue creation |
| `wiki-sync.yml` | Honest — real wiki regeneration |

## Demerzel finding (audit only — fix in companion PR)

**File:** `governance/demerzel/.github/workflows/seldon-plan.yml`
**Lines:** 112–128 (step `Run Seldon Plan cycle`)
**Smell:** commented-out agent invocation replaced with `echo`

```yaml
- name: Run Seldon Plan cycle
  if: |
    steps.kill-check.outputs.kill_active != 'true' &&
    steps.cap-check.outputs.cap_reached != 'true' &&
    (github.event_name != 'workflow_dispatch' || github.event.inputs.action == 'run')
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SELDON_FORCE_DEPARTMENT: ${{ github.event.inputs.department }}
  run: |
    echo "Starting Seldon Plan research cycle"
    echo "Department override: ${SELDON_FORCE_DEPARTMENT:-auto}"
    # Invoke Claude Code with the seldon-plan skill
    # claude -p "/seldon plan${SELDON_FORCE_DEPARTMENT:+ $SELDON_FORCE_DEPARTMENT}" --output-format json
    # Note: actual invocation depends on Claude Code CLI availability in CI
    echo "Cycle invocation placeholder — see policies/seldon-plan-policy.yaml for full spec"
```

The cron schedule fires **5×/day** (`0 6,10,14,18,22 * * *`), so
roughly **150 green runs per month** are emitted by this step doing
nothing — exactly the pattern that produced 50 days of dead Seldon
runs.

### Proposed fix (apply in a separate Demerzel PR)

Replace the `echo` placeholder with an explicit failure that surfaces
the disabled invocation:

```yaml
run: |
  echo "ERROR: Seldon Plan agent invocation is disabled." >&2
  echo "The 'claude -p \"/seldon plan ...\"' call was commented out." >&2
  echo "Restore the invocation or remove this workflow." >&2
  echo "A scheduled workflow that does nothing is worse than red — it hides reality." >&2
  exit 1
```

If the pause is intentional pending a budget/auth decision, gate the
failure on an explicit environment override so the silent state cannot
return:

```yaml
run: |
  if [ "${SELDON_PLANNER_INTENTIONALLY_DISABLED:-false}" = "true" ]; then
    echo "::notice::Seldon Plan intentionally disabled via SELDON_PLANNER_INTENTIONALLY_DISABLED=true"
    exit 0
  fi
  # ... the rest of the failure block above
```

Required separately in the Demerzel repo (cannot land via a submodule
pointer bump alone — needs the actual workflow change in
`GuitarAlchemist/Demerzel`):

1. Open a PR in Demerzel against the workflow file directly.
2. Either restore the `claude -p` invocation (preferred) or apply the
   loud-fail variant above so the silent drift stops.
3. After Demerzel merges, the submodule auto-update workflow in ix
   will pull the change naturally.

This ix PR does **not** modify the submodule pointer.

## What this PR ships

This PR ships only the audit record (this file). No workflow changes
in ix — all ix workflows passed the audit. The Demerzel finding is
documented here for a follow-up Demerzel-side PR.
