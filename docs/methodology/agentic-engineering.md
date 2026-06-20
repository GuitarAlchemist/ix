# Agentic Engineering — the harness is the work

> A read-on-demand reference, **not** an always-loaded instruction block. Distilled from Matt
> Pocock's "Agentic Engineering Workflow" (aihero.dev) + Ousterhout's *A Philosophy of Software
> Design*, and mapped to **this repo's** existing machinery. Read it when you're deciding *how* to
> direct AI on a non-trivial change — not on every turn. (Mirrors `ga/docs/methodology/agentic-engineering.md`,
> adapted to ix.)

## The one idea

**Optimise the harness, not the model.** The model is the engine; the *harness* — prompts, skills,
the codebase itself, the environment the agent runs in — is roughly half the system and the half you
fully control. The load-bearing consequence:

> *"How do you optimise token spend? Have a codebase that's easier to make changes in."*

A deeper, lower-duplication, better-documented codebase lets a **cheaper model** do the same work with
fewer tokens, because the guardrails are tighter and there's less head-banging. Hamstring the codebase
and you'll need an expensive model just to cope. ix takes this literally: it *is* the ecosystem's
harness substrate — the `ix-harness-{cargo,clippy,github-actions,signing,tars,ga}` crates, `ix-approval`,
`ix-sentinel`, and the `ix-duck::maintain` gate exist to make the harness measurable and self-correcting.

## Strategic over tactical

AI ate **tactical** programming (writing syntax, chasing bugs, making commits) — it's cheaper and
faster than you at it. Your leverage is **strategic** programming (Ousterhout):

- **Design the hard parts up front.** Decide the consequential things before delegating. In ix that's
  the IDSD intent gate (declare intent + expectations, get approval, *then* code) and logging one-way
  doors in `docs/plans/` / `docs/adr/`.
- **Scope tasks tightly.** A well-scoped task is one an AFK agent can finish with no further context.
- **Own the interfaces / seams between modules.** This is where bugs and rework concentrate.
- **Keep just-enough docs that point agents to the right place** — not exhaustive, navigational.

"Your skills are the ceiling on what AI can do." Delegate the tactical; keep the strategic mindset.

## DX ≈ AX

Agent experience ≈ developer experience. What makes a codebase pleasant for a senior human makes it
tractable for an agent: **deep modules** (a lot of behaviour behind a small interface), **low
duplication**, **clear seams**, **guardrails** (types, tests, invariants). Improving the codebase
*is* improving the harness — the most overlooked lever. In ix that's the
[`/improve-codebase-architecture`](../../.claude/skills/improve-codebase-architecture/SKILL.md)
vocabulary (module / interface / depth / seam / **deletion test**), read against [`CONTEXT.md`](../../CONTEXT.md).
Recent worked examples: the `ix-duck::telemetry` ingestion seam (PR #138 — one graceful-degrade
contract behind a small interface, replacing four copies), `ToneRow`/`PcSet` in `ix-bracelet`, and the
`maintain::evaluate` hexavalent verdict (a deep module: four lenses fused into one `MaintainVerdict`).

## Procedures vs abilities (and context hygiene)

- **Procedure** — a skill *you* invoke to stay in the driver's seat (`/grill-me`, `/grill-with-docs`,
  `/to-prd`, `/to-issues`, `/improve-codebase-architecture` — all installed under `.claude/skills/`).
  Prefer these; keep the thinking in the human.
- **Ability** — a skill the *model* self-invokes (coding standards it pulls in mid-task). Every ability
  leaks its description into the context window. Too many = bloat. (ix already marks ~29 skills
  `disable-model-invocation: true` so they don't leak.)

Matt's blank-slate test: periodically strip skills / MCP / CLAUDE.md back toward nothing, watch what
the agent does unaided, then **layer back only the procedures you deliberately choose**. Treat a long
CLAUDE.md as a smell — push detail into read-on-demand docs (like this one) and keep the always-loaded
surface lean. **ix's weakest point on this axis:** the always-loaded surface is heavy (a dense
`CLAUDE.md`, a ~90-entry auto-memory index, and the GA manifest bootstrap injected each session).
`/demerzel-context-budget` audits exactly this — run it when the surface feels bloated.

## Queues, not loops

The unit of AFK work is a **queue** of well-scoped tasks, not an infinite prompt loop (that just burns
tokens). Tasks flow **triage → explore → implement → review → merge**, pulled off by labelled agents.
ix already speaks this: GitHub Issues + the canonical triage labels
(`needs-triage`/`needs-info`/`ready-for-agent`/`ready-for-human`/`wontfix`, see
[docs/agents/triage-labels.md](../agents/triage-labels.md)), plus `supervised-loop` / `/auto-optimize`
and the `ralph-loop` machinery. Keep **human-in-the-loop checkpoints**, but push them as far toward the
final output as the work safely allows.

## Build self-improving systems

When a model finds a deep bug, the lesson is **not** "the model is great" — it's *"I should have a
system that catches this."* Prefer a cheap, scheduled review (an Action/cron that sweeps a rotating
slice of the repo) over waiting for a smarter model. *"If someone keeps stealing your bike, buy a
lock."* You're reviewing **the system that produces the code**, not just the code. ix is dense with
these: the `@ai:` drift gate (`assumption-drift.yml`), the `stable-surface.yml` API-hash guard,
`karpathy-cherny-discipline.yml`, the nightly `chatbot-trace-regression-nightly.yml` + `ga-nightly-quality.yml`,
the `state/quality/` baselines — and, as the principle made *executable*, the `ix-duck::maintain` RSI
oracle (metric↑ ∧ guardrail held ∧ converging ∧ in-distribution, never an average). Extend these
rather than one-shotting fixes.

## Make review seamless

The bottleneck is human review, so spend the harness on making review *fast*: rich PR context, AI-
assisted review passes (multi-LLM review repeatedly catches real bugs across this ecosystem), and
structured diffs over raw GitHub. ix has `claude-code-review.yml`, the `/code-review ultra` cloud
review, and `delegate-cli` (Codex/Gemini/Mistral) for cross-model passes. **Before any merge**, scan
Codex bot comments (they're invisible in the default `gh pr merge` flow — see CLAUDE.md). You stay the
gate on security and on "did the system do a good job," but you make that gate one click, not a
debugging session.

## You own the product

AI is weak at original ideas and at deciding *what* to build. Choose the features; ask "what can I
**remove**, how do I make this **simpler**." The classic product-design fundamentals still hold — AI
just implements them faster.

## The two action steps Matt actually recommends

1. **Strip to a blank slate, then layer deliberately.** Remove the bloat; re-add only procedures you
   choose and can customise.
2. **Move work AFK.** Scope a task tightly, hand it to a sandboxed agent (a git worktree off `main`),
   review the result. Two of you, then three, then five — then you review.

---

*Pointer, not gospel: this doc is read when you're deciding how to direct a non-trivial change. It is
deliberately **not** wired into the always-loaded instruction set — that would contradict its own
context-hygiene advice.*
