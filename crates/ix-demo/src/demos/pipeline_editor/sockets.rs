//! Socket type system + color map for the visual pipeline editor.
//!
//! Mirrors `ix_types::SocketType` but stays a thin local enum so the viewer
//! can expand variants / tweak colors without touching the upstream crate.

use eframe::egui::Color32;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SocketType {
    Any,
    Scalar,
    Vector,
    Matrix,
    Dataset,
    Model,
    Belief,
    Text,
}

impl SocketType {
    /// Per-type color used for pin fill and wire tint.
    pub fn color(self) -> Color32 {
        match self {
            SocketType::Any => Color32::from_gray(150),
            SocketType::Scalar => Color32::from_rgb(0xE8, 0xC4, 0x40), // yellow
            SocketType::Vector => Color32::from_rgb(0x2A, 0xB8, 0xB0), // teal
            SocketType::Matrix => Color32::from_rgb(0x3A, 0x80, 0xE0), // blue
            SocketType::Dataset => Color32::from_rgb(0x4C, 0xAF, 0x50), // green
            SocketType::Model => Color32::from_rgb(0x9B, 0x59, 0xB6),  // purple
            SocketType::Belief => Color32::from_rgb(0xE0, 0x3A, 0x9C), // magenta
            SocketType::Text => Color32::from_rgb(0xCC, 0xCC, 0xCC),   // grey-white
        }
    }

    /// Can a socket of type `self` feed into a socket of type `other`?
    ///
    /// `Any` matches everything; exact matches pass; allowed widenings
    /// are `Scalar → Vector` (broadcast) and `Vector → Matrix` (single row).
    pub const fn compatible_with(self, other: SocketType) -> bool {
        use SocketType::*;
        if matches!(self, Any) || matches!(other, Any) {
            return true;
        }
        if self as u8 == other as u8 {
            return true;
        }
        matches!((self, other), (Scalar, Vector) | (Vector, Matrix))
    }
}

#[cfg(test)]
mod tests {
    use super::SocketType::*;

    #[test]
    fn any_matches_everything() {
        for t in [Scalar, Vector, Matrix, Dataset, Model, Belief, Text] {
            assert!(Any.compatible_with(t));
            assert!(t.compatible_with(Any));
        }
    }

    #[test]
    fn exact_matches_pass() {
        for t in [Scalar, Vector, Matrix, Dataset, Model, Belief, Text] {
            assert!(t.compatible_with(t));
        }
    }

    #[test]
    fn widening_allowed() {
        assert!(Scalar.compatible_with(Vector));
        assert!(Vector.compatible_with(Matrix));
    }

    #[test]
    fn narrowing_refused() {
        assert!(!Matrix.compatible_with(Vector));
        assert!(!Vector.compatible_with(Scalar));
    }

    #[test]
    fn incompatible_rejected() {
        assert!(!Text.compatible_with(Scalar));
        assert!(!Dataset.compatible_with(Model));
        assert!(!Belief.compatible_with(Vector));
    }

    #[test]
    fn colors_are_distinct_for_real_types() {
        let types = [Scalar, Vector, Matrix, Dataset, Model, Belief, Text];
        for (i, a) in types.iter().enumerate() {
            for b in &types[i + 1..] {
                assert_ne!(a.color(), b.color(), "{:?} and {:?} share a color", a, b);
            }
        }
    }
}
