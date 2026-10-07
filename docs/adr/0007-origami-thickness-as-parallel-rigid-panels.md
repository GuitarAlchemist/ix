# ix-origami reads a thickness as parallel rigid panels, reported beside the zero-thickness verdict

Status: accepted (2026-10-07)

## Decision

`ix-origami` checks one stated flat folded state, and its checks treat the sheet as having no
thickness. A material thickness is an **opt-in reading reported beside those checks**
(`ix_origami::thickness::stack`, and `thickness: {t}` on the `ix_origami_check` MCP tool). It
never changes the top-level verdict.

1. **`t` is the full thickness, in the crease pattern's units**, the units of `vertices_coords`.
   It is not the half-thickness, and it is not in millimetres. The caller converts, for
   example `t = 0.4 / 150` for 0.4 mm board on a 150 mm sheet whose crease pattern is the unit
   square.
2. **Two readings are reported:**
   - **ply**: the most faces in one overlay cell. Any stack at the stated positions is at least
     `ply · t` thick.
   - **parallel rigid panels**: a panel is a set of faces joined by flat joints (`F`, `J`, or
     `U` edges whose faces keep one orientation). Each panel keeps one height, parallel to the
     sheet. Such a stack exists if and only if the stated "above" over overlapping panels has
     no cycle. When every overlapping pair is stated, the least height is the longest chain,
     times `t`. The JSON name is `rigid`.
3. **A cycle is refused only for `t > 0`.** The refusal stays out of the top-level `ok`, out of
   `rejected_by` and out of `Rule`, so the census tallies do not change. `rigid.ok` is the
   top-level `ok` **and** no refusal, so a state the checks reject is never reported as
   stackable. There is no thickness-wide `ok`.
4. **The rigid reading is always reported**, for paper too. It is information, and the caller
   decides what it means for their material.
5. **Out of scope:**
   - computing or optimising a layering (LP-optimal levels, crease widths);
   - refusals based on bend radius or creasing grooves;
   - folding motion;
   - tilted-panel models.
6. This record.

## Context

A multi-agent research (2026-10-07) compared models of a thick flat folded state against the
literature and against the crane fixture. It used four sweeps, three adversarial reviews and
one integration pass.

- **No tool checks thickness.**
  - No flat-folding tool inspected checks thickness: Rabbit Ear, Origami Simulator and
    Flat-Folder.
  - Ku and Demaine ("Folding Flat Crease Patterns With Thick Materials", *J. Mechanisms
    Robotics* 8(3), 2016) start from a flat folded state, but they *modify* the crease
    pattern: they double the creases and cut holes at vertices. Their method therefore cannot
    certify the stated one.
  - Their non-wrapping precondition also fails on the crane: 45 of 102 creases wrap another.
- **No check can say "makeable".** The two faces of a folded crease touch along it, and no
  exact thick state allows that. So a thickness check can only refuse, never accept.
- **Three models were compared:**
  - **Cellwise**: a face may sit at a different height in each overlay cell. Its least height
    is `ply · t`, and its only refusals for `t > 0` are those of today's Cells rule. It adds
    nothing to refuse.
  - **Parallel rigid panels**: the one new refusal.
  - **Bending**: deferred, because no sufficient condition is known.
- **The crane** (59 faces, 838 stated pairs):
  - ply is 28, but parallel rigid panels need 36 levels, with forced air gaps in 54 of 83
    cells;
  - a chain of 36 faces, each stated above the previous one, proves that 36 is also a lower
    bound;
  - at 7 mm corrugated board the stack is 196 mm (any model) or 252 mm (rigid panels).
- **Why the refusal is gated on `t > 0`.** The FOLD spec allows a stated order with cycles,
  and the per-cell Cells rule allows a cycle across cells. Four strips woven
  A > B > C > D > A pass every current check, yet parallel panels cannot stack them for any
  `t > 0`. The reduction to the current theory is exact at `t = 0` but not continuous.
- **Why the refusal says "no parallel stack", not "cannot be made".** The same weave can
  probably be built from boards tilted by about `t` (a hand derivation in the review, not
  built).
- **Why flat joints are merged into panels.** Without merging, a strip stated over one face of
  a flat joint and under the other would get levels 1, 2, 3, although one rigid panel cannot
  lie both under and over it. The base checks accept that fold today: it is the known
  flat-joint gap in `lib.rs`.

## Consequences

- `analyse`, `Report`, `Rule` and the census are unchanged. The thickness output is a new key,
  present only when `thickness` is passed. No tool is added, so the registry snapshot and the
  parity count do not move.
- `ix-origami` is tier "experimental", so its Rust API (`stack`, `Stack`, `Rigid`) is a
  two-way door. The MCP input `thickness: {t}`, the meaning of `t` and the output keys are the
  one-way doors this record fixes.
- `face_levels` is one witness: the longest path from the bottom. Other valid stacks exist, so
  no consumer should treat per-face levels as canonical.

## Revisit trigger

Reopen if any of these happens:
- a consumer asks for bend or groove refusals, and the refusal can be proved, or sourced, under
  a stated assumption;
- a single-sheet fold with a cross-cell cycle turns up that boards demonstrably build;
- FOLD gains a thickness or unit field that should replace the explicit `t`.
