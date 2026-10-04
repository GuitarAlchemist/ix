# ADR-0008: `.knot` files — IX owns the grammar and checks the meaning; TARS drafts

Status: proposed (2026-10-04)

## Context

Knots now reach IX four ways: a braid word (`ix_braid`), a catalogue name, ropes
drawn through control points, and a Gauss code (`ix_knot`). Each is a tool
argument, written by whoever calls the tool. Nothing says, in one place a person
can read and keep, *which* knot a drawing is meant to be: the catalogue holds
that for its own entries in Rust, and the sailor-knot drawings of the ComfyUI
tracer held it in JSON plus a notebook of results.

The wish is a small language for talking about knots: a file a person can read,
diff and review, that a model (TARS) can draft, and that IX can check. The
question is who owns what.

## Decision

**1. A `.knot` file states a knot and what it must be.** One statement per line:
`knot <name>`, optional names and family (`fr`, `en`, `family`, `abok`), the knot
itself — `rope open|closed` followed by its control points and `over`, or
`gauss <code>` with an optional `closure` — and `expect` lines: `crossings`,
`components`, `writhe`, `jones` (a Rolfsen name or the polynomial's text),
`clearance >=` (at the file's `radius`) and `slips <outcome> <count>`.
`crates/ix-knot/knots/` holds two examples, the bowline drawn and the overhand
spelled; both are tests.

**2. IX owns the grammar and the meaning.** The parser is
`ix_knot::knot_file`; the grammar, in EBNF, is `knot_file::GRAMMAR`, and
`ix_knot` returns it with `grammar: true`. Given `knot: <text>`, `ix_knot`
parses the file, draws the knot, checks every expectation and reports each one
with what it found (`holds`, `expectations[{line, expect, got, holds}]`). A file
that does not parse is refused with its line; a file that parses but whose
expectations fail is *answered*, not refused, so a writer can see what to fix.

**3. TARS drafts under constraint; it never decides.** TARS (F#, grammars and
metacognition) may write `.knot` drafts from `GRAMMAR`, for example from a
tying description. A draft is only a proposal until `ix_knot` says every
expectation holds. The handoff is the file's text: no shared runtime, the same
JSON-on-disk pattern as the other cross-repo contracts.

**4. An expectation is a claim IX can check, not a name it can prove.**
`expect jones 6_3` says the closure's polynomial is that of 6_3; it does not say
the closure *is* 6_3, which the Jones polynomial cannot decide. Files and their
comments say so where it matters (the bowline's does).

**Rejected:**

- *JSON or YAML specs* — already used in the tracer; fine for machines,
  unpleasant to read and review, and no better checked.
- *A grammar owned by TARS* — TARS would then be both writer and judge. The
  point of the split is that the judge is deterministic and tested.
- *Expectations as hard refusals* — a draft loop needs to see which claim
  failed and what IX found instead.

## Reversibility

**Two-way door while proposed**, one-way once files exist outside this repo:
the statement words and the `expect` forms are what `.knot` files depend on.
Adding a statement or an expectation is backward compatible; renaming or
removing one breaks files. Until this ADR is accepted, the format may change.

## Revisit triggers

- **TARS actually drafts files.** Then decide how a draft travels (a file in a
  shared folder, or the text through MCP) and how many repair rounds TARS gets.
- **A knot needs what a line cannot say** — a post or ring as part of the knot,
  a pull direction (rolling versus Magnus hitch), a mechanical claim (`expect
  holds`) once a test bench exists.
- **Files are written by hand at scale.** Then error messages and an `ix knot
  check` command matter more than the MCP tool.
