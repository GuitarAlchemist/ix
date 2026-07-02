---
date: 2026-06-14
topic: lifecycle-fsm-rules-engine
---

# Governance lifecycles as explicit state machines + an upgradeable rules engine

## What We're Building

A **thin facade** that makes the ecosystem's currently-implicit governance/methodology
lifecycles **explicit, inspectable, and versioned**: state machines (FSMs) for the
workflow lifecycles, and a rules engine for the guards/policies that gate transitions —
both **composed from existing ix machinery**, not a new heavyweight engine.

The point is the pain the operator named this session: *"the development process is
untractable… I don't understand the methodology or direction."* Today every lifecycle —
brainstorm→plan→work→review, the aihero `triage` flow, PDCA, contract phases (v0.1→Phase-4),
the `@ai:` annotation lifecycle — lives in prose. Modelling them as data you can validate,
**visualize (Prime Radiant)**, and **upgrade safely** turns methodology into something you
can see and query.

## Why This Approach

Grounded prior-art scan (2026-06-14): ix has **no FSM primitive** (only incidental
`enum State`), and rules logic is **scattered** (`ix-governance` constitution = the de-facto
rules engine; `ix-approval`, `ix-pipeline::gate`, `ix-fuzzy`, `ix-autoresearch::policy`,
sentrux `check_rules`). The **upgrade** half already exists too: the `@ai:` drift-gate, the
`links.supersedes` contract pattern, and tars `grammar_evolve`. So the primitives are absent
but the demand is real and recurring — and a generic engine built speculatively would be
YAGNI. Composing wins: smallest surface, reuses the *already-validated* safe-upgrade machinery.

## Key Decisions (locked via operator)

- **Driver = governance & methodology** (not general primitives, not GA-domain, not defer).
- **Approach = compose existing**, a thin facade:
  - **States/transitions** — a small serde FSM spec (states, transitions = {from, to, event,
    guard}, initial, current, history). Reuse **`ix-graph`** for reachability/validation
    (unreachable states, dead-ends, missing terminal). FSMs have cycles, so this is *not*
    `ix-pipeline::dag` (acyclic) — but the DAG engine stays the model for acyclic pipelines.
  - **Guards/rules** — generalize `ix-governance`'s constitution check into a versioned
    `RuleSet` (ordered condition→verdict/effect); hexavalent guards via **`ix-fuzzy`**,
    belief-dependent guards via **hari**.
  - **Upgrade** — a committed spec snapshot + the **drift-gate** pattern (live machine must
    match committed spec) + **`links.supersedes`** for versioned rule-set evolution;
    optional learned weighting via tars **`grammar_evolve`**.
  - **Management/visibility** — CLI/MCP to list/validate/render; FSMs are graphs → render to
    **Prime Radiant** (the methodology becomes a picture).

## Tracer-bullet (the thin end-to-end slice) — the WIP board over in-flight features

**Operator decision (2026-06-14):** model the lifecycle that *actually hurt this session*
first — the **work-item flow with a WIP limit** — not the triage flow. This session opened
3 features at once and the operator felt it as "untractable"; that is a missing WIP cap, and
kanban's entire value here collapses to **limit-WIP + make-flow-visible**. (Kaizen is already
covered — `/correct`, `/learnings`, PDCA, drift gate, safe-RSI — so we add *no* kaizen surface.)

The board IS the methodology FSM made visible:
- **States (columns)** = `brainstorm → plan → build → review → merged` (the existing cadence).
- **Cards** = in-flight work items, *derived* from artifacts we already produce —
  `docs/brainstorms/*`, `docs/plans/*` (status frontmatter), and open PRs/issues (`gh`).
  No new store; same compose-from-state shape as Streeling.
- **First guard rule** = **WIP limit** — count cards in `build`, warn/fail if `> N` (e.g. 2).
  This is the rules-engine's first real rule and the whole point of the slice.
- **Render** = on `#dev/summary` (it already shows BACKLOG epics) and/or a CLI `lifecycle board`.

Thin end-to-end path: derive cards from brainstorms/plans/PRs → place in columns →
evaluate the WIP-limit guard → render + flag breach. If it earns its keep, generalize to
the triage flow, contract-phase, and `@ai:` annotation lifecycles.

**The WIP-limit rule also lands in `CLAUDE.md` via `/correct`** (deferred until the working
tree is clear — the DuckDB background agent is live) so the discipline holds even before the
board is built. It is exactly the convergence already enacted this session: cap WIP → finish
DuckDB → then one feature at a time.

## Open Questions

- **Crate home** — one new thin facade crate (`ix-lifecycle`?) vs a module inside
  `ix-governance`? (Lean: a small `ix-lifecycle` so governance stays focused; decide in plan.)
- **MCP/parity** — expose as tools (bumps `ix-agent/tests/parity.rs`) or CLI-only first?
  (Lean: CLI-only tracer-bullet, MCP later — avoids the parity cascade mid-spike.)
- **Relationship to the value scorecard + Streeling federation** — same JSON-on-disk +
  drift-gate shape; should lifecycle specs live under `state/lifecycle/` and be Streeling-indexed?

## Next Steps
→ `/ce:plan` (or `/grill-with-docs` first, to sharpen against `CONTEXT.md`).
**Sequencing:** implementation is DEFERRED until the working tree is clear — a background
agent is currently live on the DuckDB analytics task in this same tree (no concurrent crate
work). Plan now; build when clear.
