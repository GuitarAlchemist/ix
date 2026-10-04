//! Knots tied in rope, by name: each entry a [`RopeDiagram`] drawn by hand and
//! the knot its closure must be.
//!
//! Every entry is held to two oracles by the tests: its closure's Jones
//! polynomial equals that of the knot named in `closure`, computed from a
//! braid word for it, and its rope, drawn at `radius`, does not pass through
//! itself.

use crate::braid::Braid;
use crate::diagram::{DiagramError, Rope, RopeDiagram};
use crate::jones::{jones, Jones};

/// The most knots one connected sum may name.
pub const MAX_SUMMANDS: usize = 8;

/// A knot of the catalogue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry {
    /// Stable identifier, e.g. `figure-eight`.
    pub id: &'static str,
    pub en: &'static str,
    pub fr: &'static str,
    /// `stopper`, `loop`, `bend`, `hitch`, …
    pub family: &'static str,
    /// Number in Ashley's *Book of Knots*, when it has one.
    pub abok: Option<u16>,
    /// The closure's knot in Rolfsen's table; `m` in front for its mirror.
    pub closure: &'static str,
    /// Rope radius, in the drawing's units.
    pub radius: f64,
    ropes: &'static [Drawn],
    over: &'static str,
}

/// One rope of an entry: its control points, and whether it is closed.
type Drawn = (&'static [[f64; 2]], bool);

impl Entry {
    /// The entry's drawing.
    pub fn diagram(&self) -> Result<RopeDiagram, DiagramError> {
        let ropes: Vec<Rope> = self
            .ropes
            .iter()
            .map(|&(points, closed)| Rope {
                points: points.iter().map(|p| [p[0], p[1], 0.0]).collect(),
                closed,
            })
            .collect();
        RopeDiagram::new(&ropes, self.over)
    }
}

/// Every entry, in a fixed order.
pub fn catalog() -> &'static [Entry] {
    CATALOG
}

/// The entry with this `id`.
pub fn find(id: &str) -> Option<&'static Entry> {
    CATALOG.iter().find(|e| e.id == id)
}

/// A braid word whose closure is the knot `name` in Rolfsen's table, `m` in
/// front for its mirror image: KnotInfo's braid representatives, for the
/// knots of up to seven crossings and for 8_20. `None` for any other name.
///
/// From 7_1 on, KnotInfo and the Knot Atlas may draw a knot as each other's
/// mirror image; the braid here is KnotInfo's, whose positive generator is a
/// positive crossing (σ₁³ closes into the trefoil with V = t + t³ − t⁴).
pub fn closure_braid(name: &str) -> Option<Braid> {
    let (knot, mirror) = match name.strip_prefix('m') {
        Some(knot) => (knot, true),
        None => (name, false),
    };
    let (strands, word): (usize, &[i32]) = match knot {
        "0_1" => (2, &[1]),
        "3_1" => (2, &[1, 1, 1]),
        "4_1" => (3, &[1, -2, 1, -2]),
        "5_1" => (2, &[1, 1, 1, 1, 1]),
        "5_2" => (3, &[1, 1, 1, 2, -1, 2]),
        "6_1" => (4, &[1, 1, 2, -1, -3, 2, -3]),
        "6_2" => (3, &[1, 1, 1, -2, 1, -2]),
        "6_3" => (3, &[1, 1, -2, 1, -2, -2]),
        "7_1" => (2, &[1, 1, 1, 1, 1, 1, 1]),
        "7_2" => (4, &[1, 1, 1, 2, -1, 2, 3, -2, 3]),
        "7_3" => (3, &[1, 1, 1, 1, 1, 2, -1, 2]),
        "7_4" => (4, &[1, 1, 2, -1, 2, 2, 3, -2, 3]),
        "7_5" => (3, &[1, 1, 1, 1, 2, -1, 2, 2]),
        "7_6" => (4, &[1, 1, -2, 1, 3, -2, 3]),
        "7_7" => (4, &[-1, 2, -1, 2, -3, 2, -3]),
        "8_20" => (3, &[1, 1, 1, -2, -1, -1, -1, -2]),
        _ => return None,
    };
    let b = Braid::new(strands, word.to_vec()).ok()?;
    Some(if mirror { b.mirror() } else { b })
}

/// The Jones polynomial of the knot `name` (see [`closure_braid`]), or of a
/// connected sum of them written with `#`, e.g. `3_1#m3_1`, the closure of the
/// reef knot: a sum's polynomial is the product of its summands'. `None` when a
/// summand is not in the table or there are more than [`MAX_SUMMANDS`].
// @ai:invariant closure_jones of a sum a#b is the Jones polynomial of a braid whose closure is that sum: a's word and b's word side by side on disjoint strands, joined by one crossing [T:test conf:0.9 src:catalog::tests::a_sum_is_the_knot_of_the_braids_side_by_side]
pub fn closure_jones(name: &str) -> Option<Jones> {
    let summands: Vec<&str> = name.split('#').collect();
    if summands.len() > MAX_SUMMANDS {
        return None;
    }
    let mut v = Jones::one();
    for summand in summands {
        v = v.times(&jones(&closure_braid(summand)?))?;
    }
    Some(v)
}

// Coordinates, not π: the overhand's lobes reach x = ±3.14.
#[allow(clippy::approx_constant)]
const CATALOG: &[Entry] = &[
    // A loose trefoil opened at its lowest lobe, both ends hanging down.
    Entry {
        id: "overhand",
        en: "Overhand knot",
        fr: "Nœud simple",
        family: "stopper",
        abok: Some(514),
        closure: "3_1",
        radius: 0.28,
        ropes: &[(
            &[
                [1.42, -9.70],
                [1.42, -5.20],
                [1.42, -3.70],
                [1.83, -0.55],
                [-0.27, 2.44],
                [-3.14, 2.92],
                [-4.03, 0.94],
                [-1.83, -1.17],
                [1.83, -1.17],
                [4.03, 0.94],
                [3.14, 2.92],
                [0.27, 2.44],
                [-1.83, -0.55],
                [-1.42, -3.70],
                [-1.42, -5.20],
                [-1.42, -9.70],
            ],
            false,
        )],
        over: "UOUOUO",
    },
    // The standing part comes down from the top, the rope makes the two lobes
    // of the eight, and the working end leaves at the bottom right.
    Entry {
        id: "figure-eight",
        en: "Figure-eight knot",
        fr: "Nœud en huit",
        family: "stopper",
        abok: Some(570),
        closure: "4_1",
        radius: 0.28,
        ropes: &[(
            &[
                [0.0, 7.0],
                [0.0, 3.6],
                [0.3, 1.0],
                [1.6, -1.2],
                [1.2, -3.0],
                [-0.6, -3.2],
                [-1.5, -1.6],
                [0.2, 0.3],
                [1.6, 2.0],
                [0.8, 3.6],
                [-1.0, 3.2],
                [-1.6, 1.6],
                [-0.4, -0.6],
                [1.0, -1.8],
                [2.6, -2.4],
                [4.5, -3.0],
            ],
            false,
        )],
        over: "alternating",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// |V(-1)| and the span of V: for an alternating knot, its determinant and
    /// its crossing number.
    fn determinant_and_span(v: &Jones) -> (i128, i32) {
        let terms = v.terms();
        assert!(
            terms.iter().all(|(h, _)| h % 2 == 0),
            "a knot: whole powers"
        );
        let det = terms
            .iter()
            .map(|&(h, c)| if (h / 2) % 2 == 0 { c } else { -c })
            .sum::<i128>()
            .abs();
        (det, (terms[terms.len() - 1].0 - terms[0].0) / 2)
    }

    #[test]
    fn closure_braids_are_the_knots_they_name() {
        // Every knot here is alternating, so the span of V is its crossing
        // number; with the determinant it tells them all apart.
        for (name, det, span) in [
            ("0_1", 1, 0),
            ("3_1", 3, 3),
            ("4_1", 5, 4),
            ("5_1", 5, 5),
            ("5_2", 7, 5),
            ("6_1", 9, 6),
            ("6_2", 11, 6),
            ("6_3", 13, 6),
        ] {
            let b = closure_braid(name).unwrap();
            assert_eq!(b.components(), 1, "{name}");
            let v = jones(&b);
            assert_eq!(determinant_and_span(&v), (det, span), "{name}: {v}");
            let m = jones(&closure_braid(&format!("m{name}")).unwrap());
            assert_eq!(m, v.mirror(), "{name}");
        }
        // Two polynomials from the knot tables, up to mirror image.
        for (name, text) in [
            ("5_2", "-t^-6 + t^-5 - t^-4 + 2t^-3 - t^-2 + t^-1"),
            ("6_1", "t^-4 - t^-3 + t^-2 - 2t^-1 + 2 - t + t^2"),
        ] {
            let v = jones(&closure_braid(name).unwrap());
            assert!(
                v.to_string() == text || v.mirror().to_string() == text,
                "{name}: {v}"
            );
        }
        assert!(closure_braid("7_8").is_none());
        assert!(closure_braid("m").is_none());
    }

    /// The seven-crossing knots and 8_20 against KnotInfo's data file: each
    /// braid's closure has the Jones polynomial and determinant KnotInfo lists
    /// for it, in KnotInfo's handedness.
    #[test]
    fn knotinfo_braids_close_into_knotinfo_polynomials() {
        for (name, det, text) in [
            ("7_1", 7, "t^3 + t^5 - t^6 + t^7 - t^8 + t^9 - t^10"),
            ("7_2", 11, "t - t^2 + 2t^3 - 2t^4 + 2t^5 - t^6 + t^7 - t^8"),
            (
                "7_3",
                13,
                "t^2 - t^3 + 2t^4 - 2t^5 + 3t^6 - 2t^7 + t^8 - t^9",
            ),
            (
                "7_4",
                15,
                "t - 2t^2 + 3t^3 - 2t^4 + 3t^5 - 2t^6 + t^7 - t^8",
            ),
            (
                "7_5",
                17,
                "t^2 - t^3 + 3t^4 - 3t^5 + 3t^6 - 3t^7 + 2t^8 - t^9",
            ),
            ("7_6", 19, "t^-1 - 2 + 3t - 3t^2 + 4t^3 - 3t^4 + 2t^5 - t^6"),
            (
                "7_7",
                21,
                "t^-4 - 2t^-3 + 3t^-2 - 4t^-1 + 4 - 3t + 3t^2 - t^3",
            ),
            ("8_20", 9, "-t^-5 + t^-4 - t^-3 + 2t^-2 - t^-1 + 2 - t"),
        ] {
            let b = closure_braid(name).unwrap();
            assert_eq!(b.components(), 1, "{name}");
            let v = jones(&b);
            assert_eq!(v.to_string(), text, "{name}");
            assert_eq!(determinant_and_span(&v).0, det, "{name}");
        }
        // Alternating, so the span is the crossing number; 8_20 is not, and
        // its span of 6 falls short of its 8 crossings.
        for k in 1..=7 {
            let v = jones(&closure_braid(&format!("7_{k}")).unwrap());
            assert_eq!(determinant_and_span(&v).1, 7, "7_{k}");
        }
    }

    #[test]
    fn a_sum_is_the_knot_of_the_braids_side_by_side() {
        // The granny and the reef: two trefoils, alike or mirrored, on strands
        // 1-2 and 3-4 of one braid, joined by a crossing between strands 2 and 3.
        let granny = Braid::new(4, vec![1, 1, 1, 2, 3, 3, 3]).unwrap();
        let reef = Braid::new(4, vec![1, 1, 1, 2, -3, -3, -3]).unwrap();
        assert_eq!(closure_jones("3_1#3_1"), Some(jones(&granny)));
        assert_eq!(closure_jones("3_1#m3_1"), Some(jones(&reef)));
        assert!(closure_jones("3_1#m3_1").unwrap().is_symmetric());
        let eight_and_five = Braid::new(5, vec![1, -2, 1, -2, 3, 4, 4, 4, 4, 4]).unwrap();
        assert_eq!(closure_jones("4_1#5_1"), Some(jones(&eight_and_five)));
        assert_eq!(
            closure_jones("6_2"),
            Some(jones(&closure_braid("6_2").unwrap()))
        );
        assert_eq!(closure_jones("3_1#7_8"), None);
        assert_eq!(closure_jones("3_1#"), None);
        let nine = ["3_1"; MAX_SUMMANDS + 1].join("#");
        assert_eq!(closure_jones(&nine), None);
        assert!(closure_jones(&nine[4..]).is_some());
    }

    #[test]
    fn every_entry_closes_into_its_knot_and_clears_itself() {
        for e in catalog() {
            let d = e.diagram().unwrap_or_else(|err| panic!("{}: {err}", e.id));
            assert_eq!(d.components(), e.ropes.len(), "{}", e.id);
            assert_eq!(
                *d.jones(),
                jones(&closure_braid(e.closure).unwrap()),
                "{} should close into {}",
                e.id,
                e.closure
            );
            let g = d.geometry(e.radius).unwrap();
            let clear = g.min_clearance.unwrap();
            assert!(
                clear >= 1.0,
                "{}: the rope passes through itself ({clear:.3})",
                e.id
            );
        }
    }

    #[test]
    fn ids_are_unique_and_found() {
        for (i, e) in catalog().iter().enumerate() {
            assert!(catalog()[..i].iter().all(|o| o.id != e.id), "{}", e.id);
            assert_eq!(find(e.id), Some(e));
        }
        assert_eq!(find("no-such-knot"), None);
    }
}
