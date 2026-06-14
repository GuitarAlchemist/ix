# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase. **ix is single-context**: one `CONTEXT.md` + `docs/adr/` at the repo root.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the domain glossary for ix (ML algorithms + governance).
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If any of these files don't exist yet, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates and grows them lazily when terms or decisions actually get resolved.

> Note: ix also has rich existing institutional docs the skills should treat as
> first-class context alongside `CONTEXT.md`/ADRs: `docs/solutions/` (learnings),
> `docs/plans/` + `docs/brainstorms/` (decisions/rationale), `state/streeling/catalog.jsonl`
> (the federated learnings registrar), and `state/assumptions/` (`@ai:` annotation graph).

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-....md
│   └── 0002-....md
└── crates/
```

## Use the glossary's vocabulary

When your output names a domain concept (an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 — but worth reopening because…_
