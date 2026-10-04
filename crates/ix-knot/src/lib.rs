//! # ix-knot — braids, the knots they close into, and how to draw them
//!
//! A braid on `n` strands is a word in the generators σ₁ … σₙ₋₁ and their
//! inverses; joining the bottom of each strand back to its top (the *closure*)
//! gives a knot or a link. This crate answers three questions about a braid
//! word:
//!
//! - **What does it close into, coarsely?** The permutation of the strands,
//!   the number of components of the closure, and the writhe.
//! - **Which knot is it?** The Jones polynomial of the closure, computed as the
//!   Kauffman bracket over the Temperley–Lieb algebra: the state space is the
//!   Catalan(n) planar diagrams, not the 2^crossings smoothings, so a long word
//!   on few strands is cheap.
//! - **How is it drawn?** A 3D layout of each strand, from which the word can be
//!   read back, so a picture of the braid is checked against the braid.
//!
//! ```
//! use ix_knot::{jones, Braid};
//!
//! // The trefoil is the closure of σ₁³; the figure-eight that of (σ₁σ₂⁻¹)².
//! let trefoil = Braid::parse(None, "s1^3").unwrap();
//! assert_eq!(jones(&trefoil).to_string(), "t + t^3 - t^4");
//! let figure_eight = Braid::parse(None, "s1 s2^-1").unwrap().repeat(2).unwrap();
//! assert_eq!(jones(&figure_eight).to_string(), "t^-2 - t^-1 + 1 - t + t^2");
//! ```
//!
//! A sailor's three-strand plait is σ₁σ₂⁻¹ repeated; laid rope twists one way,
//! σ₁σ₂ repeated. Their closures are the figure-eight knot, the Borromean
//! rings, and on through the family, which is how a decorative plait and a knot
//! table meet.
//!
//! Most knots tied in rope are not closed braids: they have ends, and many are
//! tied around something. Those are drawn rather than spelled: a
//! [`RopeDiagram`] is ropes through control points and, at each crossing, which
//! passage is in front. It answers the same three questions, and the
//! [`catalog`] names rope knots, each tested against the knot its closure must
//! be.
//!
//! ```
//! use ix_knot::catalog::find;
//!
//! let eight = find("figure-eight").unwrap().diagram().unwrap();
//! assert_eq!(eight.drawn_crossings(), 4);
//! assert_eq!(eight.jones().to_string(), "t^-2 - t^-1 + 1 - t + t^2");
//! ```

pub mod braid;
pub mod catalog;
pub mod diagram;
pub mod gauss;
pub mod jones;
pub mod knot_file;
pub mod layout;
pub mod mechanics;
pub mod mistakes;
mod poly;
#[cfg(test)]
mod testing;

pub use braid::{Braid, BraidError, MAX_CROSSINGS, MAX_STRANDS};
pub use diagram::{
    Crossing, DiagramError, Geometry, Rope, RopeDiagram, RopePath, MAX_CONTROL_POINTS,
    MAX_DIAGRAM_CROSSINGS, MAX_ROPES,
};
pub use gauss::{GaussCode, GaussError, MAX_GAUSS_CROSSINGS};
pub use jones::{jones, Jones};
pub use layout::{layout, LayoutError, StrandPath, MAX_POINTS};
pub use mechanics::Mechanics;
pub use mistakes::{Mistake, Outcome};
