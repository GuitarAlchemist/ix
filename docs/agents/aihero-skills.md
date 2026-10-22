# The aihero / mattpocock engineering skills in ix

The methodology rules themselves (tracer-bullet first, vertical slices) stay in
`CLAUDE.md`. This page holds the history and the wiring, which no session needs
on every turn.

## History

- **2026-06-14, adoption by name (operator decision).** The aihero skills are
  adopted as a named, documented workflow a human can read and follow. The
  earlier stance "do NOT add duplicate skills, the internal machinery already
  covers them" was superseded that day: the operator wanted the methodology
  visible, not an opaque mapping onto internal tools. The internal equivalents
  (compound-engineering's `ce-brainstorm`, `ce-plan`, `ce-work`,
  `simplify` / `code-review`, sentrux) still stand as the IX-native path.
- **2026-06-14, wiring.** `/setup-matt-pocock-skills` configured the issue
  tracker (GitHub), `CONTEXT.md` and the docs location that `grill-with-docs`,
  `to-spec` and `to-tickets` depend on. The result is the `## Agent skills`
  section of `CLAUDE.md` plus `docs/agents/issue-tracker.md`,
  `docs/agents/triage-labels.md` and `docs/agents/domain.md`.
- **2026-10-09, source.** The skills come from the versioned `mattpocock-skills`
  plugin (marketplace `mattpocock`, enabled at user level), no longer from
  copies under `.claude/skills/`: see
  [ADR-0008](../adr/0008-skills-from-versioned-plugins.md). Renames since the
  copies: `to-prd` is `to-spec`, `to-issues` is `to-tickets`.

## Names

Plugin skills are namespaced: `/mattpocock-skills:grill-with-docs` always works.
The bare `/grill-with-docs` works too, because each skill sets `name:` in its
frontmatter, unless another command takes that name (`/teach` is ix's own).

Skills in use: `grill-with-docs`, `grill-me`, `to-spec`, `to-tickets`, `tdd`,
`improve-codebase-architecture`, `retro`, and `setup-matt-pocock-skills`.

## Vocabulary

<https://github.com/mattpocock/dictionary-of-ai-coding> is the plain-English
glossary behind the methodology (smart-zone, tracer-bullets, context windows,
handoffs, failure modes). Referenced, not vendored, so it tracks upstream.

Version française : la méthode (tracer-bullet, tranches verticales) reste dans
`CLAUDE.md` ; cette page ne porte que l'historique et le câblage des skills
aihero, qui ne sont utiles à aucune session à chaque tour.
