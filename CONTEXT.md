# CONTEXT — ix domain glossary

> The shared language of this repo. `/grill-with-docs` grows this file lazily as
> terms get resolved during planning; `/improve-codebase-architecture`,
> `/diagnose`, and `/tdd` read it so their output uses **our** words, not synonyms.
> This is a **seed** — add terms when a real ambiguity is resolved, not speculatively.

## What ix is

A Rust workspace (77 crates) implementing foundational **ML/math algorithms** and
**AI governance** as composable crates, exposed via an MCP server (`ix-agent`) and
CLI (`ix-skill`). Part of the GuitarAlchemist ecosystem (**ix** + **tars** + **ga**
+ **Demerzel**). Source-of-truth for cross-repo collaboration is JSON-on-disk
contracts (see `docs/contracts/`), not runtime coupling.

## Core terms

- **Crate** — a unit of capability (`ix-<domain>`). Each defines traits
  (`Regressor`, `Classifier`, `Clusterer`, `Optimizer`, …) and uses the builder
  pattern + seeded RNG for reproducibility. CPU = `f64` + `ndarray`; GPU = `f32`
  via WGPU shaders. See `README.md` for the full crate map.
- **Skill** — a capability exposed to Claude Code / agents (a `.claude/skills/<name>/SKILL.md`).
  Distinct from a **crate** (Rust library) and a **tool** (MCP-callable function).
- **Tool** — an MCP-callable function (registered in `ix-agent`; the count is
  asserted by `crates/ix-agent/tests/parity.rs` — every tool-adding PR bumps it).
- **Governance / Demerzel constitution** — all agent actions are subject to the
  Demerzel constitution (`governance/demerzel`). The **Galactic Protocol** is the
  cross-repo contract layer; **Prime Radiant** is the 3D governance-graph viz.
- **Hexavalent logic** — truth values are **T / P / U / D / F / C** (not just
  true/false/unknown/contradictory). Used in `@ai:` annotations and belief state.
- **`@ai:` annotation** — an inline claim marker (`@ai:invariant`/`assumption`/…)
  with a truth_value + certainty, where **`certainty := strength of live binding`**
  (test-bound → `T:test`; human-only → cap at `P:assumed`). Drift-gated in CI.
- **Federation / registrar** — the JSON-on-disk pattern where each repo declares a
  per-repo source (frontmatter / manifest) that an ix generator federates into a
  `state/<x>/catalog.jsonl`, DuckDB-queryable + drift-gated. Instances: **Streeling**
  (learnings), and the in-flight **business-value scorecard**.
- **OPTIC-K** — the mmap voicing index schema (`ix-voicings`/`ix-optick`) consumed
  by `ga`. **Voicing** = a fingered chord shape (music-domain, lives in `ga`).

## Conventions

See `CLAUDE.md` for the authoritative build/convention/discipline rules
(Karpathy 4 Rules, Cherny loops, tracer-bullets, the certainty-binding rule).
