//! A rope lying on a surface: heights for a renderer that puts the knot on a
//! deck, in place of [`RopeDiagram::geometry`]'s.
//!
//! `geometry` lifts the rope 1.1 radii where it passes in front and lowers it
//! as much where it passes behind, adding where crossings come close, so a
//! rope dips to −2.2 radii: a deck laid under its lowest point leaves most of
//! the knot floating. Here a rope rests at height 0, its centre one radius
//! above the surface, and where it passes over another it lies on top of it,
//! its centre one diameter above the one beneath. It stays there for 1.5 radii
//! of its length either side of the crossing and comes down over the next 4,
//! or over what room there is before a lower plateau. A rigid rope (a spar, a
//! post) never bends: it lies level, one diameter up when anything passes
//! under it.
//!
//! A passage behind lies at its own rope's level, so the one in front is
//! always one diameter above it at the crossing: which passage is in front is
//! never changed, and the path read by height gives back the drawing's
//! crossings.

use crate::diagram::{arclength, clearance, dist2, DiagramError, Geometry, RopeDiagram, RopePath};
use std::f64::consts::PI;

impl RopeDiagram {
    /// Each rope in 3D as a tube of `radius` resting on a surface, `rigid[r]`
    /// saying whether rope `r` is a spar that never bends.
    // @ai:invariant resting() keeps every rope on or above the surface, rigid ropes level, and the passage in front higher at every crossing [T:test conf:0.85 src:resting::tests::a_rope_rests_on_the_surface_and_keeps_its_crossings]
    pub fn resting(&self, radius: f64, rigid: &[bool]) -> Result<Geometry, DiagramError> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(DiagramError::Radius(radius));
        }
        let ropes = self.components();
        if rigid.len() != ropes {
            return Err(DiagramError::Rigid {
                ropes,
                got: rigid.len(),
            });
        }
        let (diameter, hold, ramp) = (2.0 * radius, 1.5 * radius, 4.0 * radius);
        let mut arcs = Vec::with_capacity(ropes);
        let mut lengths = Vec::with_capacity(ropes);
        // Each rope's drawn passages, in order: (position along it, crossing, in front).
        let mut stops: Vec<Vec<(f64, usize, bool)>> = Vec::with_capacity(ropes);
        for rope in 0..ropes {
            let pts = &self.loops[rope][..self.drawn[rope]];
            let s = arclength(pts);
            let length = *s.last().unwrap();
            let length = if self.closed[rope] {
                length + dist2(pts[pts.len() - 1], pts[0]).sqrt()
            } else {
                length
            };
            stops.push(
                self.passages[rope]
                    .iter()
                    .filter(|p| !self.crossings()[p.crossing].closure)
                    .map(|p| {
                        let next = if p.seg + 1 < pts.len() {
                            s[p.seg + 1]
                        } else {
                            length
                        };
                        (s[p.seg] + p.t * (next - s[p.seg]), p.crossing, p.over)
                    })
                    .collect(),
            );
            arcs.push(s);
            lengths.push(length);
        }
        // A spar lies one diameter up when something passes under it.
        let level: Vec<f64> = (0..ropes)
            .map(|r| {
                if rigid[r] && stops[r].iter().any(|&(_, _, over)| over) {
                    diameter
                } else {
                    0.0
                }
            })
            .collect();
        // The level of the rope behind at each crossing.
        let mut behind = vec![0.0; self.crossings().len()];
        for (r, list) in stops.iter().enumerate() {
            for &(_, c, over) in list {
                if !over {
                    behind[c] = level[r];
                }
            }
        }
        // Behind, a passage lies at its rope's level; in front, on top of what it crosses.
        let height: Vec<Vec<f64>> = stops
            .iter()
            .enumerate()
            .map(|(r, list)| {
                list.iter()
                    .map(|&(_, c, over)| {
                        if over && !rigid[r] {
                            behind[c] + diameter
                        } else {
                            level[r]
                        }
                    })
                    .collect()
            })
            .collect();
        let paths: Vec<RopePath> = (0..ropes)
            .map(|rope| {
                let plateaus: Vec<(f64, f64)> = stops[rope]
                    .iter()
                    .zip(&height[rope])
                    .map(|(&(at, _, _), &h)| (at, h))
                    .collect();
                let points = self.loops[rope][..self.drawn[rope]]
                    .iter()
                    .zip(&arcs[rope])
                    .map(|(p, &s)| {
                        let z = if rigid[rope] {
                            level[rope]
                        } else {
                            let closed = self.closed[rope];
                            profile(s, &plateaus, lengths[rope], closed, hold, ramp)
                        };
                        [p[0], p[1], z]
                    })
                    .collect();
                RopePath {
                    closed: self.closed[rope],
                    points,
                }
            })
            .collect();
        let min_clearance = clearance(&paths, &arcs, &lengths, 3.0 * radius).map(|d| d / diameter);
        Ok(Geometry {
            radius,
            ropes: paths,
            min_clearance,
        })
    }
}

/// The height at `s` along a rope of `length` with a plateau at each
/// `(position, height)`, in order along it: flat for `hold` either side, then
/// easing down over `ramp`, or over the room left when it comes down to a
/// lower plateau; where two plateaus are closer than `2 * hold`, straight from
/// one to the other.
fn profile(
    s: f64,
    plateaus: &[(f64, f64)],
    length: f64,
    closed: bool,
    hold: f64,
    ramp: f64,
) -> f64 {
    let ease = |u: f64| (1.0 - (PI * u.clamp(0.0, 1.0)).cos()) / 2.0;
    let fall = |h: f64, x: f64, reach: f64| h * ease(1.0 - x / reach);
    let k = plateaus.partition_point(|&(at, _)| at <= s);
    let before = match k {
        0 if closed => plateaus.last().map(|&(at, h)| (at - length, h)),
        0 => None,
        _ => Some(plateaus[k - 1]),
    };
    let after = match plateaus.get(k) {
        Some(&p) => Some(p),
        None if closed => plateaus.first().map(|&(at, h)| (at + length, h)),
        None => None,
    };
    match (before, after) {
        (Some((a, ha)), Some((b, hb))) => {
            let room = b - a - 2.0 * hold;
            if room <= 0.0 {
                return ha + (hb - ha) * ease((s - a) / (b - a).max(f64::EPSILON));
            }
            let x = s - a - hold;
            if x <= 0.0 {
                ha
            } else if x >= room {
                hb
            } else {
                let reach_a = if hb < ha { ramp.min(room) } else { ramp };
                let reach_b = if ha < hb { ramp.min(room) } else { ramp };
                fall(ha, x, reach_a).max(fall(hb, room - x, reach_b))
            }
        }
        (Some((a, ha)), None) => fall(ha, (s - a - hold).max(0.0), ramp),
        (None, Some((b, hb))) => fall(hb, (b - s - hold).max(0.0), ramp),
        (None, None) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::catalog;
    use crate::diagram::Rope;
    use crate::testing::bights;

    /// For each drawn crossing, how far the passage in front is above the one
    /// behind on the resting path: what reading it by height would decide.
    fn separations(d: &RopeDiagram, g: &Geometry) -> Vec<f64> {
        let mut at = vec![(f64::NAN, f64::NAN); d.crossings().len()];
        for (rope, path) in g.ropes.iter().enumerate() {
            let pts = &path.points;
            for p in &d.passages[rope] {
                if d.crossings()[p.crossing].closure {
                    continue;
                }
                let (a, b) = (pts[p.seg][2], pts[(p.seg + 1) % pts.len()][2]);
                let z = a + (b - a) * p.t;
                if p.over {
                    at[p.crossing].0 = z;
                } else {
                    at[p.crossing].1 = z;
                }
            }
        }
        at.into_iter()
            .filter(|(o, u)| !o.is_nan() && !u.is_nan())
            .map(|(o, u)| o - u)
            .collect()
    }

    fn lowest(g: &Geometry) -> f64 {
        g.ropes
            .iter()
            .flat_map(|r| &r.points)
            .map(|p| p[2])
            .fold(f64::MAX, f64::min)
    }

    #[test]
    fn a_rope_rests_on_the_surface_and_keeps_its_crossings() {
        let mut drawings: Vec<(String, RopeDiagram, f64)> = catalog()
            .iter()
            .map(|e| (e.id.to_string(), e.diagram().unwrap(), e.radius))
            .collect();
        let reef = RopeDiagram::new(&bights(), "OUOOUO UOUUOU").unwrap();
        drawings.push(("bights".into(), reef, 0.16));
        for (id, d, radius) in drawings {
            let g = d.resting(radius, &vec![false; d.components()]).unwrap();
            assert!(lowest(&g) >= 0.0, "{id}: below the surface");
            // Where the old heights hang a rope under the deck.
            assert!(lowest(&d.geometry(radius).unwrap()) < -radius, "{id}");
            let gaps = separations(&d, &g);
            assert_eq!(gaps.len(), d.drawn_crossings(), "{id}");
            assert!(gaps.iter().all(|&h| h > radius), "{id}: {gaps:?}");
            let clear = g.min_clearance.unwrap();
            assert!(
                clear >= 1.0 - 1e-9,
                "{id}: the rope passes through itself ({clear:.3})"
            );
        }
    }

    /// A rope wound over and under a spar: the spar stays level, one diameter
    /// up, and the rope lies on top of it where it passes over.
    #[test]
    fn a_rigid_spar_stays_level() {
        let spar = Rope {
            points: vec![[-3.6, 0.1, 0.0], [3.6, 0.1, 0.0]],
            closed: false,
        };
        let turns = Rope {
            points: [
                [-2.0, -2.5],
                [-2.0, 1.2],
                [-1.3, 1.8],
                [-0.6, 1.2],
                [-0.6, -1.2],
                [0.1, -1.8],
                [0.8, -1.2],
                [0.8, 1.2],
                [1.5, 1.8],
                [2.2, 1.2],
                [2.2, -2.5],
            ]
            .iter()
            .map(|p| [p[0], p[1], 0.0])
            .collect(),
            closed: false,
        };
        let d = RopeDiagram::new(&[turns, spar], "UOUO OUOU").unwrap();
        assert_eq!(d.drawn_crossings(), 4);
        let radius = 0.16;
        let g = d.resting(radius, &[false, true]).unwrap();
        let level = g.ropes[1].points.iter().map(|p| p[2]);
        assert!(
            level.clone().all(|z| z == 2.0 * radius),
            "{:?}",
            level.collect::<Vec<_>>()
        );
        let top = g.ropes[0]
            .points
            .iter()
            .map(|p| p[2])
            .fold(f64::MIN, f64::max);
        assert!((top - 4.0 * radius).abs() < 1e-12, "{top}");
        assert!(lowest(&g) >= 0.0);
        assert!(separations(&d, &g).iter().all(|&h| h > radius));
        assert!(g.min_clearance.unwrap() >= 1.0 - 1e-9);
        assert_eq!(
            d.resting(radius, &[true]),
            Err(DiagramError::Rigid { ropes: 2, got: 1 })
        );
    }

    #[test]
    fn the_rope_holds_on_top_then_comes_down_over_four_radii() {
        let (hold, ramp) = (1.5, 4.0);
        let one = [(10.0, 2.0)];
        let z = |s| profile(s, &one, 30.0, false, hold, ramp);
        assert_eq!(z(10.0), 2.0);
        assert_eq!(z(11.5), 2.0);
        assert!((z(13.5) - 1.0).abs() < 1e-12);
        assert_eq!(z(15.5), 0.0);
        assert_eq!(z(0.0), 0.0);
        // Closed, the plateau near the end reaches round to the start.
        assert_eq!(profile(0.5, &[(29.5, 2.0)], 30.0, true, hold, ramp), 2.0);
        // Over two strands close together it barely sags between them.
        let twice = [(10.0, 2.0), (14.0, 2.0)];
        assert!(profile(12.0, &twice, 30.0, false, hold, ramp) > 1.9);
        // From the surface up to a plateau 4 on, it climbs in the 1 left.
        let step = [(10.0, 0.0), (14.0, 2.0)];
        assert!((profile(12.0, &step, 30.0, false, hold, ramp) - 1.0).abs() < 1e-12);
        assert_eq!(profile(11.5, &step, 30.0, false, hold, ramp), 0.0);
    }
}
