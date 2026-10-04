//! One tying mistake at a time: each crossing of a drawing passed the wrong
//! way in turn, and what that does to the closure.
//!
//! The commonest slip when tying a knot is taking a rope over where it should
//! go under. [`RopeDiagram::mistakes`] makes that slip at every crossing of the
//! drawing, one at a time, and says whether the closure is unchanged, untied
//! (one rope, now the unknot), apart (several ropes, now lying separate), or
//! something else. A bend that comes apart on many single slips is one that
//! forgives little.
//!
//! **Apart** is read from the Jones polynomial and the linking numbers: ropes
//! lying apart have the product of their own polynomials, times
//! -t^(-1/2) - t^(1/2) for each rope after the first, and no two of them link.
//! A different polynomial, or two ropes whose linking number is not zero,
//! proves the ropes are still caught; otherwise they are taken as apart,
//! though a few links that look split to both tests are not. Only the drawing's own crossings are flipped: those of the
//! closing arcs belong to the closure convention, not to the knot.

use crate::diagram::{DiagramError, RopeDiagram};
use crate::jones::Jones;

/// What one slip does to the closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The closure's Jones polynomial is unchanged.
    Same,
    /// One rope, and its closure is now the unknot.
    Untied,
    /// Several ropes, and their closures now lie apart.
    Apart,
    /// Anything else: another knot or link, the ropes still caught.
    Other,
}

impl Outcome {
    /// `same`, `untied`, `apart` or `other`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Same => "same",
            Self::Untied => "untied",
            Self::Apart => "apart",
            Self::Other => "other",
        }
    }
}

/// One crossing passed the wrong way.
#[derive(Debug, Clone, PartialEq)]
pub struct Mistake {
    /// Where the crossing is, in the plane.
    pub at: [f64; 2],
    /// The two ropes that cross there; the same rope twice when it crosses
    /// itself.
    pub ropes: [usize; 2],
    /// The `over` letters that draw the knot with this slip.
    pub over: String,
    pub writhe: i64,
    pub jones: Jones,
    pub outcome: Outcome,
}

impl RopeDiagram {
    /// Each crossing of the drawing passed the wrong way, one at a time, in the
    /// order the ropes first reach them.
    // @ai:invariant each mistake redraws the same ropes with exactly one crossing's two passages swapped, and the unchanged letters redraw the drawing itself [T:test conf:0.9 src:mistakes::tests::interlocked_bights_come_apart_on_four_slips_of_six]
    // @ai:assumption a Jones polynomial equal to the product of the ropes' own polynomials, with every linking number zero, means the ropes lie apart; a non-split link passing both would be misread as apart [U:uncertain conf:0.6 src:no-check]
    pub fn mistakes(&self) -> Result<Vec<Mistake>, DiagramError> {
        let passes = self.passes();
        let mut order: Vec<usize> = Vec::new();
        for &(k, _) in passes.iter().flatten() {
            if !order.contains(&k) {
                order.push(k);
            }
        }
        let single = self.components() == 1;
        order
            .into_iter()
            .map(|k| {
                let flipped: Vec<Vec<(usize, bool)>> = passes
                    .iter()
                    .map(|rope| {
                        rope.iter()
                            .map(|&(c, front)| (c, front != (c == k)))
                            .collect()
                    })
                    .collect();
                let over = letters(flipped.iter().flatten());
                let diagram = RopeDiagram::new(self.ropes(), &over)?;
                let jones = diagram.jones().clone();
                let outcome = if jones == *self.jones() {
                    Outcome::Same
                } else if single && jones == Jones::one() {
                    Outcome::Untied
                } else if !single
                    && diagram.linking_numbers().iter().all(|&(_, lk)| lk == 0)
                    && jones == self.apart(&flipped)?
                {
                    Outcome::Apart
                } else {
                    Outcome::Other
                };
                let mut ropes = flipped
                    .iter()
                    .enumerate()
                    .flat_map(|(r, rope)| rope.iter().filter(|p| p.0 == k).map(move |_| r));
                let ropes = [ropes.next().unwrap_or(0), ropes.next().unwrap_or(0)];
                Ok(Mistake {
                    at: self.crossings()[k].at,
                    ropes,
                    over,
                    writhe: diagram.writhe(),
                    jones,
                    outcome,
                })
            })
            .collect()
    }

    /// The polynomial the ropes would have lying apart, each drawn alone with
    /// the passages `passes` gives it at the crossings it makes with itself.
    fn apart(&self, passes: &[Vec<(usize, bool)>]) -> Result<Jones, DiagramError> {
        let mut product = Jones::one();
        for (r, rope) in passes.iter().enumerate() {
            let own = rope
                .iter()
                .filter(|(c, _)| rope.iter().filter(|(d, _)| d == c).count() == 2);
            let alone = RopeDiagram::new(&self.ropes()[r..=r], &letters(own))?;
            product = product.times(alone.jones()).ok_or(DiagramError::Overflow)?;
            if r > 0 {
                product = product
                    .times(&Jones::apart_factor())
                    .ok_or(DiagramError::Overflow)?;
            }
        }
        Ok(product)
    }
}

/// `O` for a passage in front, `U` for one behind.
fn letters<'a>(passes: impl Iterator<Item = &'a (usize, bool)>) -> String {
    passes
        .map(|&(_, front)| if front { 'O' } else { 'U' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::find;
    use crate::diagram::Rope;

    fn outcomes(d: &RopeDiagram) -> Vec<Outcome> {
        d.mistakes().unwrap().iter().map(|m| m.outcome).collect()
    }

    #[test]
    fn every_slip_unties_the_overhand_and_the_figure_eight() {
        for (id, crossings) in [("overhand", 3), ("figure-eight", 4)] {
            let d = find(id).unwrap().diagram().unwrap();
            assert_eq!(outcomes(&d), vec![Outcome::Untied; crossings], "{id}");
        }
    }

    /// Two rings, one passing over the other at the top and under it at the
    /// bottom: the Hopf link. Either slip lets them apart.
    #[test]
    fn either_slip_lets_the_hopf_rings_apart() {
        let ring = |cx: f64, lift: bool| Rope {
            points: (0..12)
                .map(|i| {
                    let a = f64::from(i) * std::f64::consts::TAU / 12.0;
                    let z = if lift { a.sin().signum() } else { 0.0 };
                    [cx + a.cos(), a.sin(), z]
                })
                .collect(),
            closed: true,
        };
        let d = RopeDiagram::new(&[ring(0.0, true), ring(1.2, false)], "height").unwrap();
        assert_eq!(d.components(), 2);
        assert_ne!(
            d.jones(),
            &Jones::one().times(&Jones::apart_factor()).unwrap()
        );
        let mistakes = d.mistakes().unwrap();
        assert_eq!(mistakes.len(), 2);
        for m in &mistakes {
            assert_eq!(m.outcome, Outcome::Apart);
            assert_eq!(m.ropes, [0, 1]);
            assert_eq!(m.jones.to_string(), "-t^(-1/2) - t^(1/2)");
        }
    }

    /// Two interlocked bights, heights given (the sailor-knot series' drawing
    /// of the reef knot, which its heights do not make one): an independent
    /// count, made by flipping crossings from the outside and asking `ix_knot`
    /// each time, found that 4 of its 6 slips let the ropes apart.
    #[test]
    fn interlocked_bights_come_apart_on_four_slips_of_six() {
        let bights = RopeDiagram::new(&crate::testing::bights(), "height").unwrap();
        assert_eq!(bights.drawn_crossings(), 6);
        assert_eq!(bights.jones().to_string(), "-t^(-5/2) - t^(-1/2)");

        // The letters the passages give redraw the drawing itself.
        let drawn = letters(bights.passes().iter().flatten());
        let again = RopeDiagram::new(bights.ropes(), &drawn).unwrap();
        assert_eq!(again.jones(), bights.jones());

        let mistakes = bights.mistakes().unwrap();
        assert_eq!(mistakes.len(), 6);
        let count = |o| mistakes.iter().filter(|m| m.outcome == o).count();
        assert_eq!((count(Outcome::Apart), count(Outcome::Other)), (4, 2));
        for m in &mistakes {
            let changed = drawn.chars().zip(m.over.chars()).filter(|(a, b)| a != b);
            assert_eq!(changed.count(), 2, "one crossing, two passages");
            assert_eq!(m.ropes, [0, 1], "every crossing joins the ropes");
        }
    }
}
