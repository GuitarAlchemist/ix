//! Sedenion session-product signature.
//!
//! # The idea
//!
//! Sedenions (S^16) are 16-dimensional hypercomplex numbers built by
//! the Cayley-Dickson construction from octonion pairs. They have
//! two properties that make them unusual among hypercomplex
//! algebras:
//!
//! 1. **Non-associative:** `(a * b) * c ≠ a * (b * c)`. Unlike
//!    quaternions (associative) and octonions (alternative), the
//!    order of operations in a sedenion product is load-bearing.
//! 2. **Zero divisors:** there exist non-zero `a, b` with `a * b = 0`.
//!    Sedenions form a *non-division* algebra.
//!
//! For normal use these are bugs. For session fingerprinting, they
//! are features:
//!
//! - Non-associativity means reordering events changes the product.
//!   A naive left-fold is sensitive to the exact event sequence at
//!   the algebra level — stronger than a hash, which only flags
//!   *any* change.
//! - Zero divisors let us construct intentional "cancellation"
//!   patterns for cryptographic curiosities, if we ever want that.
//!   Not used here but the option is open.
//!
//! # What this module computes
//!
//! Given a sequence of [`SignatureAtom`]s (each a sedenion), the
//! session signature is their left-fold product:
//!
//! ```text
//! sig = ((((e_1 * e_2) * e_3) * e_4) * ... * e_n)
//! ```
//!
//! The fold order is fixed (left) so the same input sequence always
//! produces the same sedenion. Because sedenion multiplication is
//! non-associative, a permutation of the inputs produces a
//! measurably different output.
//!
//! # How to use it for session signing
//!
//! Map each SessionEvent to a sedenion [`SignatureAtom`]. Fold the
//! stream. The resulting 16 × f64 = 128-byte signature is your
//! tamper-evidence marker. To verify, replay the events on a
//! reference implementation and compare.
//!
//! Swapping two events anywhere in the middle changes the
//! signature. Dropping an event changes it. Inserting an event
//! changes it. Reordering a run of events changes it.

use ix_sedenion::Sedenion;
use serde::{Deserialize, Serialize};

/// One "atom" of the session signature — a sedenion derived from a
/// SessionEvent. The mapping from event to atom lives outside this
/// module because it depends on the SessionEvent shape; this crate
/// only needs to know how to fold them.
#[derive(Debug, Clone, Copy)]
pub struct SignatureAtom(pub Sedenion);

impl SignatureAtom {
    /// Build an atom from a seed integer. Maps each seed to a
    /// unique sedenion by placing the seed's bits into a basis
    /// combination. Not a cryptographic construction — just a
    /// stable deterministic function from `u64` to `Sedenion`.
    pub fn from_seed(seed: u64) -> Self {
        // Build a sedenion whose components are bit-derived from
        // the seed. We lay out 16 components as bytes derived from
        // the seed, each mapped to a small signed f64 so products
        // stay bounded.
        let mut components = [0.0_f64; 16];
        let bytes = seed.to_le_bytes(); // 8 bytes
        for i in 0..16 {
            // Cycle through the 8 seed bytes; mix in the slot
            // index to differentiate repeated bytes.
            let byte = bytes[i % 8] ^ (i as u8).wrapping_mul(17);
            components[i] = (byte as f64) / 255.0 - 0.5;
        }
        // Ensure the atom is nonzero in the real slot so the
        // identity element (zero) is never accidentally produced.
        if components[0].abs() < 1e-12 {
            components[0] = 0.1;
        }
        Self(Sedenion::new(components))
    }

    /// Build an atom from a pair of indices, where `basis_a` and
    /// `basis_b` are `0..16` indices into the sedenion basis. The
    /// resulting atom is `e_{basis_a} + 0.1 * e_{basis_b}`, giving
    /// distinct non-commuting atoms for distinct index pairs.
    ///
    /// This is the construction used when each SessionEvent variant
    /// is mapped to a fixed basis element.
    pub fn from_basis_pair(basis_a: usize, basis_b: usize) -> Self {
        let mut components = [0.0_f64; 16];
        components[basis_a % 16] = 1.0;
        components[basis_b % 16] += 0.1;
        Self(Sedenion::new(components))
    }

    /// Access the underlying sedenion.
    pub fn sedenion(&self) -> &Sedenion {
        &self.0
    }
}

/// A 16-component sedenion signature of a session. Serialized as 16
/// little-endian f64 values = 128 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SessionSignature {
    /// The 16 components of the resulting sedenion.
    pub components: [f64; 16],
}

impl SessionSignature {
    /// The zero signature — used as the identity for empty sessions.
    /// Note: sedenion multiplication with a zero factor yields
    /// zero, so this cannot be used as a multiplicative identity
    /// mid-session. Only valid for an empty input sequence.
    pub fn zero() -> Self {
        Self {
            components: [0.0; 16],
        }
    }

    /// Compute the session signature by left-folding sedenion
    /// multiplication over the atom stream.
    ///
    /// For an empty sequence returns [`Self::zero`]. For a single
    /// atom returns that atom's sedenion. For longer sequences
    /// returns `((a[0] * a[1]) * a[2]) * ...`.
    pub fn fold(atoms: &[SignatureAtom]) -> Self {
        if atoms.is_empty() {
            return Self::zero();
        }
        let mut acc = *atoms[0].sedenion();
        for atom in atoms.iter().skip(1) {
            acc = acc.mul(atom.sedenion());
        }
        Self {
            components: acc.components,
        }
    }

    /// Serialize to 128 bytes: 16 × f64 in little-endian order.
    pub fn encode(&self) -> [u8; 128] {
        let mut out = [0u8; 128];
        for (i, c) in self.components.iter().enumerate() {
            let bytes = c.to_le_bytes();
            out[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
        out
    }

    /// Deserialize from 128 bytes produced by [`Self::encode`].
    pub fn decode(bytes: &[u8; 128]) -> Self {
        let mut components = [0.0_f64; 16];
        for i in 0..16 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
            components[i] = f64::from_le_bytes(buf);
        }
        Self { components }
    }

    /// L2 distance between two signatures. Used to compare signatures
    /// in tests and to quantify "how different" two sessions are.
    pub fn distance(&self, other: &Self) -> f64 {
        self.components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn atoms_from_seeds(seeds: &[u64]) -> Vec<SignatureAtom> {
        seeds.iter().map(|&s| SignatureAtom::from_seed(s)).collect()
    }

    #[test]
    fn empty_sequence_yields_zero_signature() {
        let sig = SessionSignature::fold(&[]);
        assert_eq!(sig, SessionSignature::zero());
    }

    #[test]
    fn single_atom_sequence_equals_that_atom() {
        let atoms = atoms_from_seeds(&[42]);
        let sig = SessionSignature::fold(&atoms);
        assert_eq!(sig.components, atoms[0].sedenion().components);
    }

    #[test]
    fn same_sequence_yields_same_signature() {
        let a = atoms_from_seeds(&[1, 2, 3, 4, 5]);
        let b = atoms_from_seeds(&[1, 2, 3, 4, 5]);
        let sig_a = SessionSignature::fold(&a);
        let sig_b = SessionSignature::fold(&b);
        assert_eq!(sig_a, sig_b);
    }

    #[test]
    fn permutation_yields_different_signature() {
        // Non-associativity + non-commutativity means swapping two
        // events changes the product.
        let original = atoms_from_seeds(&[10, 20, 30, 40, 50]);
        let swapped = atoms_from_seeds(&[10, 30, 20, 40, 50]); // swap 2 & 3

        let sig_orig = SessionSignature::fold(&original);
        let sig_swap = SessionSignature::fold(&swapped);

        assert_ne!(sig_orig, sig_swap, "permutation must change the signature");
        // Distance should be meaningful (above floating-point noise).
        let d = sig_orig.distance(&sig_swap);
        assert!(d > 1e-6, "permutation distance too small: {d}");
    }

    #[test]
    fn insertion_yields_different_signature() {
        let short = atoms_from_seeds(&[1, 2, 3]);
        let long = atoms_from_seeds(&[1, 2, 99, 3]);
        let sig_s = SessionSignature::fold(&short);
        let sig_l = SessionSignature::fold(&long);
        assert_ne!(sig_s, sig_l);
    }

    #[test]
    fn deletion_yields_different_signature() {
        let full = atoms_from_seeds(&[1, 2, 3, 4, 5]);
        let partial = atoms_from_seeds(&[1, 2, 4, 5]); // dropped 3
        let sig_f = SessionSignature::fold(&full);
        let sig_p = SessionSignature::fold(&partial);
        assert_ne!(sig_f, sig_p);
    }

    #[test]
    fn basis_pair_atoms_are_distinct_for_distinct_inputs() {
        let a = SignatureAtom::from_basis_pair(1, 2);
        let b = SignatureAtom::from_basis_pair(2, 1);
        // (1, 2) and (2, 1) have swapped dominant/minor bases, so
        // the resulting atoms should differ.
        assert_ne!(a.sedenion().components, b.sedenion().components);
    }

    #[test]
    fn encode_decode_round_trip() {
        let atoms = atoms_from_seeds(&[100, 200, 300]);
        let sig = SessionSignature::fold(&atoms);
        let bytes = sig.encode();
        assert_eq!(bytes.len(), 128);
        let decoded = SessionSignature::decode(&bytes);
        assert_eq!(sig, decoded);
    }

    #[test]
    fn long_session_produces_stable_signature() {
        // A 100-event session folds without panicking and produces
        // a non-zero signature.
        let seeds: Vec<u64> = (0..100).collect();
        let atoms = atoms_from_seeds(&seeds);
        let sig = SessionSignature::fold(&atoms);
        let magnitude: f64 = sig.components.iter().map(|c| c * c).sum::<f64>().sqrt();
        // Non-trivially non-zero — sedenion products of unit-ish
        // atoms don't usually cancel to zero.
        assert!(
            magnitude > 1e-6,
            "100-event signature magnitude suspiciously small: {magnitude}"
        );
    }
}
