# `.origami` files state a folding sequence; IX replays it and checks the result

Status: accepted (2026-10-16)

## Decision

A **`.origami` file** states a folding sequence of a square sheet, the way a diagram does, with
`expect` lines saying what the result must be. IX owns the grammar (`ix_origami::sequence::GRAMMAR`)
and its meaning. A person, or a model such as TARS drafting under that grammar, writes the file;
IX replays it into a stated folded state, runs every check of `ix-origami` on that state, and
answers each expectation. The writer proposes; IX decides.

1. **A file states a folding sequence, the high level of the origami grammar.** Replaying it gives
   a stated folded state (a FOLD crease pattern, one folded frame and its `faceOrders`), the low
   level, which is what gets checked. The lowering is deterministic.
2. **Fold lines come from landmarks.** A step names points of the paper (the corners `A` `B` `C`
   `D`, counterclockwise from the bottom left; the side midpoints `mAB` `mBC` `mCD` `mDA`; the
   centre `O`), taken where they lie at that step, and builds its fold line by a Huzita–Justin
   axiom:
   - `through P Q`: the line through two points (O1);
   - `P to Q`: the fold that puts P onto Q (O2);
   - `L1 to L2`: the fold that puts line L1 onto line L2, direction onto direction (O3). A line is
     two corners (`AC`) or two points joined by `-` (`mAB-mCD`), and its letter order is its
     direction, so `AB to AC` is the bisector at A, and `AB to CD` is refused with the hint
     "write AB to DC".

   Coordinates `(x, y)`, positions in the folded state at that step, are a fallback for lines no
   landmark gives (thirds).
3. **Simple folds only.** A step carries every layer on its moving side over the fold line: in
   front for a valley, behind for a mountain. Each step then has exactly one result, so the
   replay derives the layer order the steps dictate and never searches for one.
4. **The moving side is named, never defaulted.** It is the side of the point or line a step
   names first (`A to C`: A's side). `through P Q` names nothing that moves, so it must say
   `moving R`. A step is refused, with its line number, when its points coincide, its moving
   side is ambiguous (a line that crosses the fold line, or a point on it), or its fold line
   does not cross the paper.
5. **The replay is a separate module, `ix_origami::sequence`, and its result is always checked.**
   It writes FOLD JSON and reads it back with `Fold::from_value`; it shares no code with the
   layer checks, and every result goes through `analyse`. The crate's doc line "Nothing here
   computes a layer order" becomes "searches for".
6. **The language.**
   - The extension is `.origami`. One statement per line; `#` starts a comment at the line's
     start or after a space.
   - `origami <name>` comes first, once; `sheet square`, optional, is the only sheet.
   - `expect ok`, `expect faces N` and `expect ply N` (the most faces stacked at one point). A
     failed expectation is **answered, not refused**: the result says what IX found.
7. **The MCP tool `ix_origami_check` takes `sequence: <text>`** in place of `fold`, exactly one
   of the two. With `sequence`, the report is the one for the replayed fold, plus a `sequence`
   block: `{ name, holds, expectations: [{ line, expect, got, holds }], fold }`. `census` and
   `thickness` apply to the replayed fold. No tool is added.
8. **Limits.** A file has at most 64 KiB, a replay at most 1000 faces (a simple fold can double
   the faces), a coordinate at most 10⁶ in absolute value; the tool's `LIMITS` then apply to the
   replayed fold as to any other.
9. This record.

## Context

- **The interview** (`/grill-with-docs`, 2026-10-13 to 2026-10-16) settled the five design
  questions and the language questions; the terms are in `CONTEXT.md` (folding sequence,
  landmark, simple fold, moving side, line, `.origami` file).
- **The prototype** (Python, 2026-10-16) replayed seven sequences of simple folds into FOLD
  files, and the crate's checks passed all seven, from 2 faces (half) to 49 faces and 945 stated
  pairs (six oblique folds). Its swap census rejected every single swap except one pair of the
  letter fold: the two flaps of a letter fold can lie in either order, and both orders fold.
  The Rust replay reproduces the faces, the stated pairs and the census of all seven
  (`tests/sequences.rs`).
- **Why no default side.** Both sides of a fold line give a valid model, and they are different
  models. A default would pick one silently.
- **Why positions in the folded state.** A diagram shows the paper as it is at each step, and a
  folder names what they see. A crease-pattern coordinate would ask the writer to unfold the
  model in their head.
- **Why simple folds first.** Reverse, squash, petal and sink folds move some layers and not
  others, so each needs a way to say which; that is more grammar, and the next slice. A simple
  fold needs no choice, which is what lets the replay derive the order with no search.

## Consequences

- `analyse`, `Report`, `Rule`, the census and the thickness reading are unchanged. The tool's
  output has a new `sequence` key, present only for a `sequence` input. No tool is added, so the
  registry snapshot and the parity count do not move. The tool's input schema no longer lists
  `fold` as required; the handler requires exactly one of `fold` and `sequence`.
- `ix-origami` is tier "experimental", so its Rust API (`Sequence`, `Checked`, `Verdict`,
  `SequenceError`, `GRAMMAR`) is a two-way door. The one-way doors this record fixes are the
  language (its statement words, the corner letters and their positions, letter order as
  direction, the moving-side rule) and the tool's `sequence` input and output.
- A replayed fold is checked like a fold from a file: the replay can be wrong, and the checks
  would say so.

## Revisit trigger

Reopen if any of these happens:
- a sequence that a consumer needs cannot be written with simple folds (a reverse, squash or sink
  fold): that is the next slice of the grammar;
- landmarks run out often enough that coordinates become the common case (add intersections, or
  O4 to O7);
- the replay and a reference model (Rabbit Ear, a folded sheet) disagree on a sequence.
