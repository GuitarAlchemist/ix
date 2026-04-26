//! # ix-bracelet
//!
//! Dihedral group D₁₂ operator algebra on 12-bit pitch-class sets.
//!
//! Unifies transposition (Tₙ) and inversion (I) under a single
//! [`DihedralElement`] type. Composition lives on the group, not on sets — fold
//! a pipeline once, apply the result to many sets. [`bracelet_prime_form`]
//! returns the canonical D₁₂-orbit representative (Forte set class); the
//! transposition-only analogue is [`necklace_prime_form`].

pub mod action;
pub mod dihedral;
pub mod forte;
pub mod grothendieck;
pub mod neo_riemannian;
pub mod orbit;
pub mod pc_set;
pub mod prime_form;

pub use action::Action;
pub use dihedral::{DihedralElement, Group};
pub use forte::{all_forte_numbers, forte_number, ForteNumber};
pub use grothendieck::{find_nearby, grothendieck_delta, icv, Delta, Icv};
pub use neo_riemannian::{classify_triad, h, l, n, p, r, s, TriadKind};
pub use orbit::{all_prime_forms, orbit, orbit_unique};
pub use pc_set::PcSet;
pub use prime_form::{bracelet_prime_form, necklace_prime_form};
