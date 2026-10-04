//! A rope lying on a surface: heights for a renderer that puts the knot on a
//! deck, in place of [`RopeDiagram::geometry`]'s.
//!
//! `geometry` lifts the rope 1.1 radii where it passes in front and lowers it
//! as much where it passes behind, adding where crossings come close, so a
//! rope dips to −2.2 radii: a deck laid under its lowest point leaves most of
//! the knot floating. Here a rope rests at height 0, its centre one radius
//! above the surface, and where it passes in front it lies on top of the rope
//! it crosses: its centre one diameter above the highest that rope gets
//! nearby. It stays up either side while the two are within a diameter of
//! each other in plan (two radii over the sine of the angle they cross at, at
//! least 1.5 radii and at most 8), then comes down over 4 radii for a
//! diameter of height, longer when higher, so that it never bends
//! tighter coming down than it does climbing. Where two of these meet in a
//! dip, the dip is filled as a rope would bridge it: no curve turns upward
//! with a radius under 1.5 rope radii. A rope passing behind is not held down:
//! it goes where its own crossings take it, and the rope in front is raised
//! to clear it, round after round until nothing moves. A rigid rope (a spar, a
//! post) never bends: it lies level, high enough to clear whatever passes
//! under it.
//!
//! Which passage is in front is never changed, so the path read by height
//! gives back the drawing's crossings.

use crate::diagram::{
    along, arclength, clearance, dist2, DiagramError, Geometry, RopeDiagram, RopePath,
};
use std::f64::consts::PI;

/// Rounds of raising the passage in front onto the one behind.
const ROUNDS: usize = 32;

/// In rope radii: the least and the most the rope in front stays on top either
/// side of a crossing, how long it takes to come down a diameter, and the least
/// radius of a dip it bridges.
const HOLD: f64 = 1.5;
const MAX_HOLD: f64 = 8.0;
const RAMP: f64 = 4.0;
const BRIDGE: f64 = 1.5;

/// A drawn passage: (position along the rope, segment, fraction of the
/// segment, crossing, in front).
type Stop = (f64, usize, f64, usize, bool);

impl RopeDiagram {
    /// Each rope in 3D as a tube of `radius` resting on a surface, `rigid[r]`
    /// saying whether rope `r` is a spar that never bends.
    // @ai:invariant resting() keeps every rope on or above the surface, rigid ropes level, the passage in front a diameter above the one behind, and no bend in height tighter than one radius [T:test conf:0.85 src:resting::tests::a_rope_rests_on_the_surface_and_keeps_its_crossings]
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
        let diameter = 2.0 * radius;
        let mut arcs = Vec::with_capacity(ropes);
        let mut lengths = Vec::with_capacity(ropes);
        // Each rope's drawn passages, in order along it.
        let mut stops: Vec<Vec<Stop>> = Vec::with_capacity(ropes);
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
                        let at = s[p.seg] + p.t * (next - s[p.seg]);
                        (at, p.seg, p.t, p.crossing, p.over)
                    })
                    .collect(),
            );
            arcs.push(s);
            lengths.push(length);
        }
        // The passage behind at each drawn crossing, as (rope, index in its stops).
        let mut behind = vec![(0, 0); self.crossings().len()];
        for (rope, list) in stops.iter().enumerate() {
            for (i, &(.., c, over)) in list.iter().enumerate() {
                if !over {
                    behind[c] = (rope, i);
                }
            }
        }
        // How long the rope in front holds over each crossing: either side, the
        // stretch where the two ropes are within a diameter of each other in
        // plan, two radii over the sine of the angle they cross at.
        let mut tangents = vec![Vec::new(); self.crossings().len()];
        for (rope, list) in stops.iter().enumerate() {
            let pts = &self.loops[rope];
            for &(_, seg, _, c, _) in list {
                let (a, b) = (pts[seg], pts[(seg + 1) % pts.len()]);
                tangents[c].push([b[0] - a[0], b[1] - a[1]]);
            }
        }
        let hold: Vec<f64> = tangents
            .iter()
            .map(|t| match t[..] {
                [u, v] => {
                    let sin =
                        (u[0] * v[1] - u[1] * v[0]).abs() / (u[0].hypot(u[1]) * v[0].hypot(v[1]));
                    (2.0 / sin).clamp(HOLD, MAX_HOLD) * radius
                }
                _ => HOLD * radius,
            })
            .collect();
        // The height asked of each passage in front, and of each rigid rope.
        let mut lift: Vec<Vec<f64>> = stops.iter().map(|l| vec![diameter; l.len()]).collect();
        let mut level = vec![0.0; ropes];
        let profiles = |lift: &[Vec<f64>], level: &[f64]| -> Vec<Vec<f64>> {
            (0..ropes)
                .map(|r| {
                    if rigid[r] {
                        return vec![level[r]; arcs[r].len()];
                    }
                    let plateaus: Vec<(f64, f64, f64)> = stops[r]
                        .iter()
                        .zip(&lift[r])
                        .filter(|((.., over), _)| *over)
                        .map(|(&(at, _, _, c, _), &h)| (at, h, hold[c]))
                        .collect();
                    heights(&arcs[r], &plateaus, lengths[r], self.closed[r], radius)
                })
                .collect()
        };
        let mut round = 0;
        let z = loop {
            let z = profiles(&lift, &level);
            // The highest the rope behind gets while the one in front holds over it.
            let peak = |u: usize, j: usize| {
                let (at, seg, t, c, _) = stops[u][j];
                let n = z[u].len();
                let here = z[u][seg] + t * (z[u][(seg + 1) % n] - z[u][seg]);
                arcs[u]
                    .iter()
                    .zip(&z[u])
                    .filter(|&(&s, _)| along(s, at, lengths[u], self.closed[u]) <= hold[c])
                    .fold(here, |m, (_, &h)| m.max(h))
            };
            let mut raised = false;
            for (r, list) in stops.iter().enumerate() {
                for (i, &(.., c, over)) in list.iter().enumerate() {
                    if !over {
                        continue;
                    }
                    let (u, j) = behind[c];
                    let need = peak(u, j) + diameter;
                    let asked = if rigid[r] {
                        &mut level[r]
                    } else {
                        &mut lift[r][i]
                    };
                    if *asked < need - 1e-9 {
                        *asked = need;
                        raised = true;
                    }
                }
            }
            round += 1;
            if !raised || round == ROUNDS {
                break z;
            }
        };
        let paths: Vec<RopePath> = (0..ropes)
            .map(|rope| RopePath {
                closed: self.closed[rope],
                points: self.loops[rope][..self.drawn[rope]]
                    .iter()
                    .zip(&z[rope])
                    .map(|(p, &h)| [p[0], p[1], h])
                    .collect(),
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

/// The height at each arclength `s` of a rope of `length` and `radius` with a
/// plateau at each `(position, height, hold)`: flat for `hold` either side, then
/// down to the surface over 4 radii for a diameter of height (a cosine, so
/// the bend at its top and foot is the same whatever the height), the highest
/// of these wherever several reach, with every dip they leave filled.
fn heights(
    s: &[f64],
    plateaus: &[(f64, f64, f64)],
    length: f64,
    closed: bool,
    radius: f64,
) -> Vec<f64> {
    let ramp = RAMP * radius;
    let bump = |d: f64, h: f64, hold: f64| {
        let reach = ramp * (h / (2.0 * radius)).sqrt();
        let u = ((d - hold) / reach).clamp(0.0, 1.0);
        h * (1.0 + (PI * u).cos()) / 2.0
    };
    let raw: Vec<f64> = s
        .iter()
        .map(|&x| {
            plateaus
                .iter()
                .map(|&(at, h, hold)| bump(along(x, at, length, closed), h, hold))
                .fold(0.0, f64::max)
        })
        .collect();
    fill(s, &raw, length, closed, BRIDGE * radius)
}

/// `z` along a rope with every dip narrower than a disc of radius `rho`
/// filled: a closing in the (arclength, height) plane, so the curve turns
/// upward nowhere tighter than `rho`. Never lowers a point.
fn fill(s: &[f64], z: &[f64], length: f64, closed: bool, rho: f64) -> Vec<f64> {
    let n = s.len();
    // Each sample within `rho` of sample `i` along the rope, with the height
    // of the disc's rim above it.
    let rim = |i: usize| {
        let mut near = vec![(i, rho)];
        for forward in [true, false] {
            let mut j = i;
            for _ in 1..n {
                j = match (forward, closed) {
                    (true, true) => (j + 1) % n,
                    (false, true) => (j + n - 1) % n,
                    (true, false) if j + 1 < n => j + 1,
                    (false, false) if j > 0 => j - 1,
                    _ => break,
                };
                let d = along(s[i], s[j], length, closed);
                if d > rho {
                    break;
                }
                near.push((j, (rho * rho - d * d).sqrt()));
            }
        }
        near
    };
    let rims: Vec<Vec<(usize, f64)>> = (0..n).map(rim).collect();
    let up: Vec<f64> = rims
        .iter()
        .map(|near| near.iter().map(|&(j, r)| z[j] + r).fold(f64::MIN, f64::max))
        .collect();
    rims.iter()
        .map(|near| {
            near.iter()
                .map(|&(j, r)| up[j] - r)
                .fold(f64::MAX, f64::min)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::catalog;
    use crate::diagram::Rope;
    use crate::testing::{bights, clove_hitch};

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

    /// The tightest bend of a height profile: the least radius of the circle
    /// through three samples in a row, in the (arclength, height) plane.
    fn tightest(s: &[f64], z: &[f64]) -> f64 {
        (2..s.len())
            .map(|k| {
                let (a, b, c) = ([s[k - 2], z[k - 2]], [s[k - 1], z[k - 1]], [s[k], z[k]]);
                let side = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
                let twice_area =
                    ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs();
                side(a, b) * side(b, c) * side(c, a) / (2.0 * twice_area)
            })
            .fold(f64::MAX, f64::min)
    }

    #[test]
    fn a_rope_rests_on_the_surface_and_keeps_its_crossings() {
        let mut drawings: Vec<(String, RopeDiagram, f64, Vec<bool>)> = catalog()
            .iter()
            .map(|e| {
                let d = e.diagram().unwrap();
                let rigid = vec![false; d.components()];
                (e.id.to_string(), d, e.radius, rigid)
            })
            .collect();
        let reef = RopeDiagram::new(&bights(), "OUOOUO UOUUOU").unwrap();
        drawings.push(("bights".into(), reef, 0.16, vec![false; 2]));
        let clove = RopeDiagram::new(&clove_hitch(), "height").unwrap();
        drawings.push(("clove hitch".into(), clove, 0.16, vec![false, true]));
        for (id, d, radius, rigid) in drawings {
            let g = d.resting(radius, &rigid).unwrap();
            assert!(lowest(&g) >= 0.0, "{id}: below the surface");
            // Where the old heights hang a rope under the deck.
            assert!(lowest(&d.geometry(radius).unwrap()) < -radius, "{id}");
            let gaps = separations(&d, &g);
            assert_eq!(gaps.len(), d.drawn_crossings(), "{id}");
            let diameter = 2.0 * radius;
            assert!(gaps.iter().all(|&h| h > diameter - 1e-9), "{id}: {gaps:?}");
            let clear = g.min_clearance.unwrap();
            assert!(
                clear >= 1.0 - 1e-9,
                "{id}: the rope passes through itself ({clear:.3})"
            );
            for (rope, path) in g.ropes.iter().enumerate().filter(|&(r, _)| !rigid[r]) {
                let pts = &d.loops[rope][..d.drawn[rope]];
                let z: Vec<f64> = path.points.iter().map(|p| p[2]).collect();
                let bend = tightest(&arclength(pts), &z) / radius;
                assert!(bend >= 1.0, "{id} rope {rope}: bends to {bend:.2} radii");
            }
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
        let level: Vec<f64> = g.ropes[1].points.iter().map(|p| p[2]).collect();
        assert!(
            level
                .iter()
                .all(|&z| z == level[0] && (z - 2.0 * radius).abs() < 1e-9),
            "{level:?}"
        );
        let top = g.ropes[0]
            .points
            .iter()
            .map(|p| p[2])
            .fold(f64::MIN, f64::max);
        assert!((top - 4.0 * radius).abs() < 1e-9, "{top}");
        assert!(lowest(&g) >= 0.0);
        assert!(separations(&d, &g).iter().all(|&h| h > 2.0 * radius - 1e-9));
        assert!(g.min_clearance.unwrap() >= 1.0 - 1e-9);
        assert_eq!(
            d.resting(radius, &[true]),
            Err(DiagramError::Rigid { ropes: 2, got: 1 })
        );
    }

    #[test]
    fn the_rope_holds_on_top_comes_down_and_bridges_a_dip() {
        let s: Vec<f64> = (0..=600).map(|k| k as f64 * 0.05).collect();
        let at = |z: &[f64], x: f64| z[(x / 0.05).round() as usize];
        let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
        // Radius 1: a plateau 2 high holds for 1.5, comes down over 4.
        let one = heights(&s, &[(10.0, 2.0, 1.5)], 30.0, false, 1.0);
        assert!(near(at(&one, 10.0), 2.0));
        assert!(near(at(&one, 11.5), 2.0));
        assert!(near(at(&one, 13.5), 1.0));
        assert!(near(at(&one, 15.5), 0.0));
        assert!(near(at(&one, 0.0), 0.0));
        // Twice as high, it comes down over 4√2.
        let high = heights(&s, &[(10.0, 4.0, 1.5)], 30.0, false, 1.0);
        let half_way = (11.5 + 2.0 * 2f64.sqrt()) / 0.05;
        let (k, t) = (half_way.floor() as usize, half_way.fract());
        assert!((high[k] + t * (high[k + 1] - high[k]) - 2.0).abs() < 1e-3);
        // Closed, the plateau near the end reaches round to the start.
        let round = heights(&s, &[(29.5, 2.0, 1.5)], 30.0, true, 1.0);
        assert!(near(at(&round, 0.5), 2.0));
        // Two plateaus 6 apart leave a dip at 13; it is filled, never dug.
        let two = [(10.0, 2.0, 1.5), (16.0, 2.0, 1.5)];
        let raw: Vec<f64> = s
            .iter()
            .map(|&x| heights(&[x], &two, 30.0, false, 1.0)[0])
            .collect();
        let bridged = heights(&s, &two, 30.0, false, 1.0);
        assert!(bridged.iter().zip(&raw).all(|(b, r)| *b >= r - 1e-12));
        assert!(at(&bridged, 13.0) > at(&raw, 13.0) + 0.1);
        assert!(tightest(&s, &raw) < 0.5);
        assert!(tightest(&s, &bridged) >= 1.4, "{}", tightest(&s, &bridged));
    }
}
