//! Checks of a flat folded state, read from a FOLD file.
//!
//! A [`Fold`] holds a crease pattern and one folded frame that inherits from it, with the
//! frame's stated layer order (`faceOrders`). [`analyse`] runs every check on it:
//!
//! - **structure**: indices in range, faces counterclockwise, every face side an edge;
//! - **face isometry**: each face keeps its shape in the folded frame, up to one global scale;
//! - **crease orientation**: a mountain or valley turns one face over relative to the other;
//! - **Kawasaki and Maekawa** at interior vertices whose creases are all mountains or valleys;
//! - **the layer order**: the adjacency rule across every crease, no cycle in any overlay cell,
//!   taco-tortilla and taco-taco.
//!
//! [`swap_census`] flips each stated pair alone and records which rules reject it: a check
//! that cannot fail proves nothing.
//!
//! Conventions, from the FOLD spec (`doc/spec.md` at
//! `edemaine/fold@824f9fa6f944248787b0b2077ef622761489201e`):
//!
//! - `faces_vertices` run counterclockwise, and that order fixes each face's normal;
//! - in `faceOrders`, `[f, g, s]` has `s = +1` when `f` lies on the side `g`'s normal points to,
//!   `-1` on the other side, and `0` (or an omitted triple) when unknown. If `[g, f, t]` is also
//!   given, `t = -s` when `f` and `g` face the same way and `t = s` when they face opposite ways;
//! - "a valley crease points the two face normals into each other, while a mountain crease makes
//!   them point away from each other". Hence the adjacency rule: two faces that share a crease,
//!   folded flat, have `s = +1` both ways across a valley and `s = -1` both ways across a
//!   mountain.
//!
//! These are local checks of one stated folded state. They do not prove that the sheet folds
//! flat: testing global flat-foldability is NP-complete (Bern and Hayes 1996). Nothing here
//! computes a layer order; the checks only test the one the file states.
//!
//! Known gaps, so `ok` is not proof:
//!
//! - the tolerances are absolute, set for a crease pattern about 1 across (the unit square):
//!   scale a file to that size before checking it;
//! - only mountain and valley creases enter the layer rules: a folded unassigned (`U`) edge is
//!   not checked, nor is a flat (`F`, `J`) edge that lies along a crease, and the
//!   tortilla-tortilla rule is not implemented;
//! - keys the folded frame overrides (`edges_assignment`, `faces_vertices`) are ignored, and so
//!   are `faceOrders` placed on the crease pattern.
//!
//! For a fold from outside the process, [`analyse_within`], [`Context::within`] and
//! [`swap_census_within`] take [`Limits`]: a fold over any of them is refused, never checked in
//! part.
//!
//! FOLD 1.1 files that keep the folded state at the top level and the crease pattern in frame 1
//! (Rabbit Ear's layout before 2024) are rearranged into the 1.2 layout on load: places change,
//! and `file_spec` becomes 1.2.
//!
//! The test fixture is the traditional crane from Rabbit Ear (Robby Kraft), MIT, taken at a
//! commit from before the project moved to GPL-3.0; see `tests/fixtures/NOTICE.md`.

pub mod controls;
pub mod fold;
pub mod geometry;
pub mod layers;
pub mod limits;
pub mod local;

pub use fold::{Assignment, Fold, FoldError, FoldedFrame, Point};
pub use layers::{
    analyse, analyse_within, first_swap, swap, swap_census, swap_census_within, CensusRow, Context,
    LayerChecks, Report, Rule, Unchecked,
};
pub use limits::{Limits, OverLimit};
