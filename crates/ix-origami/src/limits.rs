//! Caps on the work one check may do, for folds that come from outside the process.
//!
//! The overlay, the tortillas and tacos, and the census grow faster than the file: a request of
//! a few kilobytes can ask for gigabytes. A fold over a limit is refused, before or while the
//! work is done, never checked in part. [`Limits::NONE`] checks everything.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Faces: the stated relations hold faces² entries.
    pub faces: usize,
    /// Face-vertex incidences, summed over the faces. Edges, creases, the isometry's vertex
    /// pairs and the tortilla scan all grow with it.
    pub face_vertices: usize,
    /// Overlay cells, covered or not, at any point of the overlay.
    pub cells: usize,
    /// The sum over the overlay cells of (faces in the cell)²: the overlapping pairs and the
    /// cycle checks visit each.
    pub cell_pairs: usize,
    /// Tortillas plus tacos.
    pub tacos: usize,
    /// An upper bound on the steps of [`crate::swap_census`], checked by
    /// [`crate::swap_census_within`] before any flip.
    pub census_steps: usize,
}

impl Limits {
    /// No limit.
    pub const NONE: Self = Self {
        faces: usize::MAX,
        face_vertices: usize::MAX,
        cells: usize::MAX,
        cell_pairs: usize::MAX,
        tacos: usize::MAX,
        census_steps: usize::MAX,
    };
}

/// A fold refused because one count went over its limit.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{what}: more than {limit}")]
pub struct OverLimit {
    pub what: &'static str,
    pub limit: usize,
}

/// `Err` when `count` is over `limit`.
pub(crate) fn within(what: &'static str, count: usize, limit: usize) -> Result<(), OverLimit> {
    if count > limit {
        return Err(OverLimit { what, limit });
    }
    Ok(())
}
