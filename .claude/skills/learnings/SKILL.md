---
name: learnings
description: Capture a surprise as a solution note in docs/solutions/<category>/<date>-<topic>.md, the store the learnings-researcher agent and `streeling search` read. Use at the commit that closes a problem which took more than one hypothesis, a CI red on one platform only, a tool that behaved against its documentation, or when the user says "note ça" / "learnings". Distinct from /correct (behavioural rules) and /digest (session state).
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
last_verified: 2026-10-22
karpathy_rule: R4-goal-driven-execution (a surprise that is not written down is re-discovered)
---

# /learnings

Turns a resolved surprise into a note under `docs/solutions/` that a later session finds by
symptom. CLAUDE.md named this command since 2026-06-20; the skill itself did not exist until
2026-10-22, so the notes before that date were written by hand.

## When to run

- **At the commit that closes the problem**, not during the diagnosis: the note needs the root
  cause and the fix, both confirmed by the test or gate that went green.
- Triggers: a fix that needed more than one hypothesis; a CI row red on one platform only; a
  command or tool that behaved against its own documentation; a silent failure (green but dead).
- Claude fires it without being asked, then ends the reply with the teaching line (step 6).

**Bar:** would a future session, hitting the symptom cold, save an hour by grepping this note?
A routine fix whose commit message already explains it is not a learning.

## How to run

1. **Pick the category** from the existing directories (`ls docs/solutions`): `build-errors`,
   `ecosystem-integration`, `feature-implementations`, `integration-issues`, `math-correctness`,
   `test-failures`, `workflow-patterns`. Add a directory only when none fits, and say so in the
   report.

2. **Name the file** `docs/solutions/<category>/<YYYY-MM-DD>-<topic>.md`, the topic in kebab-case
   naming the symptom, not the fix (`go-suite-green-on-windows-red-on-linux`, not `writedelay-knob`).

3. **Write the frontmatter** the researcher greps. Every field, one line each, quoted when it
   holds a colon:

   ```yaml
   ---
   title: "<symptom as a sentence, the way the user would describe it>"
   category: <directory name>
   date: <YYYY-MM-DD>
   tags: [<tool>, <crate or module>, <platform>, <failure class>]
   symptom: "<what was observed: the exact error text or the exact green-that-should-be-red>"
   root_cause: "<the mechanism, one or two sentences>"
   ---
   ```

4. **Write the body** in four sections, short, English, no secrets, no local paths:
   `## Problem` (what was tried and what it showed), `## Root cause`, `## Fix` (the commit, the
   test or gate that now binds it), `## How to find it again` (the command that reproduces the
   symptom in under a minute).

5. **Index it.** Run `cargo run -q -p ix-streeling -- catalog` from a checkout whose parent
   directory also holds the `ga` sibling (from a worktree elsewhere the catalog silently drops
   ga's records: see `docs/solutions/workflow-patterns/streeling-catalog-drops-sibling-repo-records-from-a-worktree.md`).
   Commit `state/streeling/catalog.jsonl` with the note when it changed. If the build is not
   affordable right now (disk-bound), say so in the report and leave the catalog to the next
   indexer run; the note is still found by the researcher's grep fallback.

6. **Report, then teach.** One line naming the file, then one teaching line for the user:
   "🎓 Réflexe : j'ai lancé /learnings parce que <the surprise>, au commit qui la ferme. Tu peux
   le taper toi-même à ce moment-là." The user asked (2026-10-22) to be told each time a reflex
   fired, so they learn the moment.

## Anti-patterns

- A note written mid-diagnosis: the root cause changes, the note lies.
- A note that restates the commit message: no symptom, no reproduction, nothing to grep.
- A behavioural rule ("always run the Linux suite before a push") filed here: that is `/correct`,
  which writes CLAUDE.md; the note may link to the rule.
- Session state ("next step is X"): that is `/digest`.

## Related

- `/correct` — behavioural rules into CLAUDE.md.
- `/digest` — session state into `state/digests/latest.md`.
- `.claude/agents/research/learnings-researcher.md` — the reader; it greps the frontmatter
  fields above, then falls back to full-text search.
- `docs/contracts/streeling-learning.contract.md` — the catalog record these notes become.
