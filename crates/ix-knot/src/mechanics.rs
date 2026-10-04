//! How a knot holds, read from its drawing: the three counts of Patil, Sandt,
//! Kolle and Dunkel, "Topological mechanics of knots and tangles", Science
//! 367, 71 (2020).
//!
//! They orient each rope the way it is pulled and count, on the drawing:
//! - **N**, the crossings;
//! - **τ**, the twist fluctuation: the spread of the crossings' signs, which
//!   for signs ±1 is 1 − (Wr/N)² with Wr their sum: 1 when as many crossings
//!   turn one way as the other, 0 when all turn alike;
//! - **Γ**, the circulation: around each face, +1 for each edge the ropes run
//!   along anticlockwise and −1 for each clockwise, and |that| over the face's
//!   edges, summed over the faces.
//!
//! In their measurements a bend with more twist fluctuation, then more
//! circulation, held better: grief (6, 0, 1) < thief (6, 1, 1) < granny
//! (6, 0, 4) < reef (6, 1, 4), as (N, τ, Γ). They give no formula combining the
//! three, and neither does this module.
//!
//! Only the drawing's own crossings count, not those the arcs closing open
//! ropes add. Only bounded faces count, and an edge with the face on both of
//! its sides (a rope's end lying inside the face) is not one of the face's
//! edges: the paper's supplement, which would settle these, could not be read.
//! With those choices the four bends above come out as the paper gives them.

use crate::diagram::{DiagramError, RopeDiagram};

/// Patil et al.'s counts for a knot pulled from given ends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mechanics {
    /// N, the crossings of the drawing.
    pub crossings: usize,
    /// Wr, their signs summed with each rope oriented the way it is pulled.
    pub writhe: i64,
    /// τ = 1 − (Wr/N)²; 0 with no crossings.
    pub twist: f64,
    /// Γ, summed over the bounded faces.
    pub circulation: f64,
}

/// One edge of the drawing: a stretch of rope between two crossings, or
/// between a crossing and a rope's end, as drawn.
struct Edge {
    rope: usize,
    from: usize,
    to: usize,
    points: Vec<[f64; 2]>,
}

impl RopeDiagram {
    /// The counts with each rope pulled from its last control point, or from
    /// its first where `from_start` says so: a rope is oriented toward the end
    /// it is pulled from. Reversing every rope changes nothing; reversing one
    /// of two changes the signs of the crossings between them.
    // @ai:invariant mechanics() gives Patil et al.'s (N, τ, Γ) for grief, thief, granny and reef from one drawing of the reef, its crossings and pulls changed [T:test conf:0.8 src:mechanics::tests::the_four_bends_come_out_as_patil_et_al_give_them]
    pub fn mechanics(&self, from_start: &[bool]) -> Result<Mechanics, DiagramError> {
        let ropes = self.components();
        if from_start.len() != ropes {
            return Err(DiagramError::Pulls {
                ropes,
                got: from_start.len(),
            });
        }
        let crossings = self.drawn_crossings();
        let mut at: Vec<Vec<usize>> = vec![Vec::new(); self.crossings().len()];
        for (rope, list) in self.passages.iter().enumerate() {
            for p in list {
                at[p.crossing].push(rope);
            }
        }
        let writhe: i64 = self
            .crossings()
            .iter()
            .zip(&at)
            .filter(|(c, _)| !c.closure)
            .map(|(c, ropes)| {
                let flip = from_start[ropes[0]] != from_start[ropes[1]];
                i64::from(c.sign) * if flip { -1 } else { 1 }
            })
            .sum();
        let twist = if crossings == 0 {
            0.0
        } else {
            1.0 - (writhe as f64 / crossings as f64).powi(2)
        };
        Ok(Mechanics {
            crossings,
            writhe,
            twist,
            circulation: self.circulation(from_start),
        })
    }

    fn circulation(&self, from_start: &[bool]) -> f64 {
        let edges = self.edges();
        let vertices = self.crossings().len() + 2 * self.components();
        // The darts leaving each vertex, counterclockwise: dart 2e runs along
        // edge e as drawn, 2e + 1 back.
        let origin = |d: usize| {
            let e = &edges[d / 2];
            if d % 2 == 0 {
                e.from
            } else {
                e.to
            }
        };
        let path = |d: usize| -> Vec<[f64; 2]> {
            let mut p = edges[d / 2].points.clone();
            if d % 2 == 1 {
                p.reverse();
            }
            p
        };
        let heading = |d: usize| {
            let p = path(d);
            let q = p
                .iter()
                .find(|q| (q[0] - p[0][0]).hypot(q[1] - p[0][1]) > 1e-12);
            q.map_or(0.0, |q| (q[1] - p[0][1]).atan2(q[0] - p[0][0]))
        };
        let mut around: Vec<Vec<usize>> = vec![Vec::new(); vertices];
        for d in 0..2 * edges.len() {
            around[origin(d)].push(d);
        }
        let mut place = vec![0; 2 * edges.len()];
        for list in &mut around {
            list.sort_by(|&a, &b| heading(a).total_cmp(&heading(b)));
            for (i, &d) in list.iter().enumerate() {
                place[d] = i;
            }
        }
        // Each face to the left of its darts: a bounded one is walked
        // anticlockwise, the outside clockwise.
        let mut seen = vec![false; 2 * edges.len()];
        let mut total = 0.0;
        for start in 0..2 * edges.len() {
            if seen[start] {
                continue;
            }
            let mut face = Vec::new();
            let mut d = start;
            while !seen[d] {
                seen[d] = true;
                face.push(d);
                let back = d ^ 1;
                let list = &around[origin(back)];
                d = list[(place[back] + list.len() - 1) % list.len()];
            }
            let ring: Vec<[f64; 2]> = face.iter().flat_map(|&d| path(d)).collect();
            if area(&ring) <= 1e-9 {
                continue;
            }
            let (mut sum, mut count) = (0i64, 0usize);
            for &d in &face {
                if face.contains(&(d ^ 1)) {
                    continue;
                }
                let e = &edges[d / 2];
                let along = (d % 2 == 0) != from_start[e.rope];
                sum += if along { 1 } else { -1 };
                count += 1;
            }
            if count > 0 {
                total += sum.unsigned_abs() as f64 / count as f64;
            }
        }
        total
    }

    /// The drawing cut at its crossings and ends. Crossing `k` is vertex `k`;
    /// rope `r`'s first and last points are vertices `C + 2r` and `C + 2r + 1`,
    /// `C` the number of crossings, the closure's included.
    fn edges(&self) -> Vec<Edge> {
        let base = self.crossings().len();
        let mut edges = Vec::new();
        for (rope, list) in self.passages.iter().enumerate() {
            let pts = &self.loops[rope];
            let n = pts.len();
            let point = |seg: usize, t: f64| {
                let (a, b) = (pts[seg], pts[(seg + 1) % n]);
                [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
            };
            // (vertex, segment, parameter) of each stop along the drawn rope.
            let mut stops: Vec<(usize, usize, f64)> = list
                .iter()
                .filter(|p| !self.crossings()[p.crossing].closure)
                .map(|p| (p.crossing, p.seg, p.t))
                .collect();
            let closed = self.closed[rope];
            if !closed {
                let last = self.drawn[rope] - 1;
                stops.insert(0, (base + 2 * rope, 0, 0.0));
                stops.push((base + 2 * rope + 1, last - 1, 1.0));
            }
            let legs = if closed { stops.len() } else { stops.len() - 1 };
            for i in 0..legs {
                let (from, sa, ta) = stops[i];
                let (to, sb, tb) = stops[(i + 1) % stops.len()];
                let mut steps = (sb + n - sa) % n;
                if closed && steps == 0 && tb <= ta {
                    steps = n;
                }
                let mut points = vec![point(sa, ta)];
                points.extend((1..=steps).map(|k| {
                    let p = pts[(sa + k) % n];
                    [p[0], p[1]]
                }));
                points.push(point(sb, tb));
                edges.push(Edge {
                    rope,
                    from,
                    to,
                    points,
                });
            }
        }
        edges
    }
}

/// Twice the signed area of a closed polygon, positive anticlockwise.
fn area(ring: &[[f64; 2]]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::closure_jones;
    use crate::diagram::Rope;
    use crate::testing::bights;
    use crate::Jones;

    /// The interlocked bights' crossings, rope 0's six passages then rope 1's,
    /// that make the reef knot and the granny.
    const REEF: &str = "OUOOUO UOUUOU";
    const GRANNY: &str = "OUOUOU UOUOUO";

    /// The two tails joined over the top and the two standing parts round the
    /// bottom, far from the knot: one closed rope, which a reef closes into
    /// 3_1#m3_1 and a granny into 3_1#3_1.
    fn joined(ropes: &[Rope], letters: &str) -> Jones {
        let mut points = ropes[0].points.clone();
        points.extend(
            [
                [-5.6, -2.5],
                [-3.0, -4.5],
                [0.0, -5.0],
                [3.0, -4.5],
                [5.6, -2.5],
            ]
            .map(|[x, y]| [x, y, 0.0]),
        );
        points.extend(ropes[1].points.iter().rev());
        points.extend(
            [[3.9, 2.5], [2.0, 4.0], [0.0, 4.3], [-2.0, 4.0], [-3.9, 2.5]]
                .map(|[x, y]| [x, y, 0.0]),
        );
        let (a, b) = letters.split_once(' ').unwrap();
        let along: String = a.chars().chain(b.chars().rev()).collect();
        let rope = Rope {
            points,
            closed: true,
        };
        RopeDiagram::new(&[rope], &along).unwrap().jones().clone()
    }

    fn counts(d: &RopeDiagram, from_start: &[bool]) -> (usize, f64, f64) {
        let m = d.mechanics(from_start).unwrap();
        let round = |x: f64| (x * 100.0).round() / 100.0;
        (m.crossings, round(m.twist), round(m.circulation))
    }

    /// The reef and the granny on one drawing of two interlocked bights, each
    /// rope's tail at its first point and its standing part at its last, both
    /// tails on one side. Pulling one rope from its tail instead puts the tails
    /// on opposite sides: the thief, and from the granny the grief knot.
    /// Patil et al.'s figure 3 gives them as (N, τ, Γ): reef (6, 1, 4), thief
    /// (6, 1, 1), granny (6, 0, 4), grief (6, 0, 1).
    #[test]
    fn the_four_bends_come_out_as_patil_et_al_give_them() {
        let ropes = bights();
        assert_eq!(joined(&ropes, REEF), closure_jones("3_1#m3_1").unwrap());
        let granny = joined(&ropes, GRANNY);
        assert!(
            granny == closure_jones("3_1#3_1").unwrap()
                || granny == closure_jones("m3_1#m3_1").unwrap()
        );

        let reef = RopeDiagram::new(&ropes, REEF).unwrap();
        assert_eq!(counts(&reef, &[false, false]), (6, 1.0, 4.0), "reef");
        assert_eq!(counts(&reef, &[false, true]), (6, 1.0, 1.0), "thief");
        let granny = RopeDiagram::new(&ropes, GRANNY).unwrap();
        assert_eq!(counts(&granny, &[false, false]), (6, 0.0, 4.0), "granny");
        assert_eq!(counts(&granny, &[false, true]), (6, 0.0, 1.0), "grief");
        // Reversing both ropes changes nothing.
        assert_eq!(counts(&reef, &[true, true]), counts(&reef, &[false, false]));
        // The series' heights make neither: joined, the unknot.
        let drawn = RopeDiagram::new(&ropes, "height").unwrap();
        assert_eq!(counts(&drawn, &[false, false]).1, 0.89);
    }

    #[test]
    fn one_pull_per_rope() {
        let d = RopeDiagram::new(&bights(), REEF).unwrap();
        assert_eq!(
            d.mechanics(&[false]),
            Err(DiagramError::Pulls { ropes: 2, got: 1 })
        );
    }
}
