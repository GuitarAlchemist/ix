---
title: "Federating an ix producer snapshot into ga/state/quality (GA-pull, not ix-push)"
category: ecosystem-integration
date: 2026-06-21
tags: [federation, ix, ga, state-quality, snapshot, github-actions, cross-repo, duckdb, streeling]
symptom: "Need to land an ix-produced JSON snapshot in ga/state/quality/<domain>/ so the GA health-scorecard / ecosystem manifest surfaces it — but it's unclear whether ix pushes to ga or ga pulls from ix."
root_cause: "The ix→ga quality federation is GA-PULL: a GA-side scheduled workflow checks out the ix sibling, builds + runs the ix producer against the live GA corpus, and commits the snapshot to ga itself. No ix workflow pushes to ga."
---

# Federating an ix producer snapshot into ga/state/quality

## Problem

Phase B of the maintain-verdict tile (ix#146) needed IX's fused verdict snapshot
to appear in `ga/state/quality/maintain-gate/<date>.json` (+ `last.json`) so GA's
health-scorecard / ecosystem manifest could read it. The issue explicitly said
**reuse the existing federation channel, do not invent a new one** — but the
direction (does ix push, or does ga pull?) was not obvious from the ix side.

A red herring: the ix repo has `.github/workflows/ga-nightly-quality.yml`, which
writes to a local `state/quality/` and **uploads a GitHub artifact**. That artifact
never lands in the ga repo — so reading only the ix side suggests "ix produces but
nothing federates."

## Root cause / the actual mechanism

**Federation is GA-PULL.** Every quality domain (`embeddings`, `invariants`,
`chatbot-qa`, `readme-drift`, `voicing-analysis`) has a **GA-side** scheduled
workflow that:

1. Checks out **ga** (`permissions: contents: write`, `token: PAT_TOKEN || GITHUB_TOKEN`).
2. Checks out the **ix sibling** (`repository: GuitarAlchemist/ix`, `path: ix`).
3. Builds + runs the **ix producer** against the **live GA corpus**.
4. Commits the snapshot to ga: `chore(quality): <domain> snapshot <date> [skip ci]`
   by `github-actions[bot]`.

No ix workflow checks out ga and pushes. The canonical template is
`ga/.github/workflows/embeddings-snapshot.yml` (ix-produced data → ga).

How to confirm the mechanism quickly (do this first, before building):

```bash
# Who commits the snapshots? → github-actions[bot], from a GA-side workflow
git -C ../ga log --oneline -5 -- state/quality/embeddings
# Which GA workflow produces them?
grep -rl "state/quality" ../ga/.github/workflows/   # → *-snapshot.yml per domain
```

## Solution

Add a GA-side `<domain>-snapshot.yml` mirroring `embeddings-snapshot.yml`. Core steps:

```yaml
on:
  schedule: [{ cron: '30 7 * * *' }]   # offset from other snapshot crons
  workflow_dispatch:
permissions: { contents: write }
concurrency: { group: <domain>-snapshot, cancel-in-progress: false }
jobs:
  snapshot:
    steps:
      - uses: actions/checkout@v4                     # ga (write)
        with: { token: ${{ secrets.PAT_TOKEN || secrets.GITHUB_TOKEN }} }
      - uses: actions/checkout@v4                     # ix sibling
        with: { repository: GuitarAlchemist/ix, path: ix }
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with: { workspaces: ix -> target }
      - working-directory: ix                         # build the producer
        run: cargo build --release -p ix-duck --features duck --example ix_maintain_snapshot
      - env: { GA_ROOT: ${{ github.workspace }} }     # producer reads ga corpus
        run: |
          DATE="$(date -u +%F)"; DST="state/quality/<domain>/${DATE}.json"
          mkdir -p state/quality/<domain>
          ix/target/release/examples/<producer> "${DST}"
          [ -f "${DST}" ] || exit 0                   # absent-corpus skip → no fake entry
          cp "${DST}" state/quality/<domain>/last.json
      - run: |                                         # commit-if-new
          git config user.name github-actions[bot]
          git config user.email github-actions[bot]@users.noreply.github.com
          git add state/quality/<domain>/
          git diff --cached --quiet && exit 0          # identical day → no-op
          git commit -m "chore(quality): <domain> snapshot ${DATE} [skip ci]"; git push
```

Pair it with a registry entry + envelope contract so it validates:

- Add the domain to `ga/state/quality/.snapshot-registry.json` (`dir`, dated
  `snapshot_glob`). `Scripts/validate-quality-snapshots.ps1` validates ONLY
  registered dated snapshots against `docs/contracts/quality-snapshot.schema.json`.
- The GA dashboard envelope **requires** `domain`, `emitted_at`, `metric_name`,
  `metric_value`, `oracle_status`, `summary` (additive fields pass through). A
  producer that emits `timestamp` instead of `emitted_at`, or omits `domain`/
  `metric_name`, will register green-but-dead (validates a hand-written sample the
  real producer never matches). Align the producer to the envelope, not the reverse.

### Verify end-to-end before trusting CI

A workflow-file-only PR does **not** trigger GA's `paths`-filtered build jobs, so it
merges with no executed validation. Prove it two ways:

1. Locally: build the release producer from current ix `main`, run the workflow's
   exact produce step against a real ga checkout (`GA_ROOT=$(pwd)`), then run
   `Scripts/validate-quality-snapshots.ps1` — the new dated snapshot must move the
   `pass` count up by 1 (the 87 pre-existing failures are unrelated domains).
2. After merge: `gh workflow run <domain>-snapshot.yml --repo GuitarAlchemist/ga`,
   then confirm `gh api repos/GuitarAlchemist/ga/contents/state/quality/<domain>`
   lists `<today>.json` + `last.json` and a `chore(quality)` commit landed.

## Prevention / related gotchas (same session)

- **Dropped `synchronize` event.** A second push to a PR branch produced **zero**
  CI runs (no `concurrency` block, trigger config fine — a GitHub event-delivery
  hiccup). Diagnosis: `gh pr view <n> --json statusCheckRollup` had 0 checks on the
  new head. Fix: force a fresh `synchronize` with an empty commit
  (`git commit --allow-empty -m "ci: re-trigger"`). Don't assume "mergeable/CLEAN"
  means CI ran — CLEAN with 0 checks just means no required checks block it.

- **Streeling catalog drift on a stale branch.** Adding a doc trips the
  `freshness-check` (`streeling check`: "uncatalogued (1)"). Regenerate with
  `cargo run -p ix-streeling --bin streeling -- catalog`. **Gotcha:** if you
  regenerate on a branch behind main, the new catalog *drops* entries for docs that
  only exist on main (a `-27/+1` diff), which re-drifts after merge. Always
  `git merge origin/main` into the branch **first**, then regenerate.

- **Codex rate-limit ≠ clean.** When `gh api .../comments` for
  `chatgpt-codex-connector[bot]` returns "You have reached your Codex usage limits",
  there is no review — report "no findings *because the bot didn't run*", not
  "Codex-clean".
