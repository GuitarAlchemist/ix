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
//! A spar may be thicker than the rope. One of radius S resting on the surface
//! has its centre S − R above a rope's, and wherever two cross, the centre in
//! front is their two radii summed above the one behind, held for that sum
//! over the sine of their angle (clamped as above). Each path carries its own
//! radius, and the clearance is taken pair by pair over the two radii summed.
//!
//! Where a rope crosses a rigid one, the rigid one is a straight tube, so the
//! two clear each other by their true distance rather than stacked over the
//! whole overlap: a spar over a rope rests on the highest the rope gets less
//! what their axes are apart in plan, and a rope over a spar holds on top only
//! until it can come down along the spar's side. Stacked, a rope passing over
//! and then under the same spar a few diameters on (a turn round it) lifts the
//! spar onto its own ramp, and the spar lifts the rope, without end. A rope
//! within reach of a spar in plan but crossing it nowhere near, lower down,
//! lies under the spar's side: the spar rests on it too.
//!
//! Which passage is in front is never changed, so the path read by height
//! gives back the drawing's crossings. Heights that do not settle are refused
//! rather than returned.

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
    /// saying whether rope `r` is a spar that never bends, and each spar a
    /// tube of radius `spar`, at least the rope's.
    // @ai:invariant resting() keeps every tube on or above the surface, rigid ropes level, the passage in front the two radii summed above the one behind, and no bend in height tighter than one radius [T:test conf:0.85 src:resting::tests::a_rope_rests_on_the_surface_and_keeps_its_crossings]
    pub fn resting(
        &self,
        radius: f64,
        rigid: &[bool],
        spar: f64,
    ) -> Result<Geometry, DiagramError> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(DiagramError::Radius(radius));
        }
        if !(spar.is_finite() && spar >= radius) {
            return Err(DiagramError::Spar { spar, radius });
        }
        let ropes = self.components();
        if rigid.len() != ropes {
            return Err(DiagramError::Rigid {
                ropes,
                got: rigid.len(),
            });
        }
        let diameter = 2.0 * radius;
        let radii: Vec<f64> = rigid
            .iter()
            .map(|&r| if r { spar } else { radius })
            .collect();
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
        // The passage behind at each drawn crossing, as (rope, index in its
        // stops), and the two ropes' radii summed: how far apart their centres
        // must be.
        let mut behind = vec![(0, 0); self.crossings().len()];
        let mut front = vec![0; self.crossings().len()];
        let mut apart = vec![0.0; self.crossings().len()];
        for (rope, list) in stops.iter().enumerate() {
            for (i, &(.., c, over)) in list.iter().enumerate() {
                if over {
                    front[c] = rope;
                } else {
                    behind[c] = (rope, i);
                }
                apart[c] += radii[rope];
            }
        }
        // How long the rope in front holds over each crossing: either side, the
        // stretch where the two ropes overlap in plan, their radii summed over
        // the sine of the angle they cross at.
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
            .zip(&apart)
            .map(|(t, &sum)| match t[..] {
                [u, v] => {
                    let sin =
                        (u[0] * v[1] - u[1] * v[0]).abs() / (u[0].hypot(u[1]) * v[0].hypot(v[1]));
                    (sum / radius / sin).clamp(HOLD, MAX_HOLD) * radius
                }
                _ => HOLD * radius,
            })
            .collect();
        // Where a rope crosses a rigid one: the run of the rope's samples about
        // the crossing that lie within reach of the rigid rope in plan, each as
        // (sample, how far along the rope from the crossing, how far its centre
        // must be above or below the rigid rope's axis to clear it). A rope
        // within reach of a rigid one nowhere near a crossing lies beside it:
        // those samples, as (rope, sample, rise), by rigid rope.
        let mut cylinder: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); self.crossings().len()];
        let mut beside: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); ropes];
        for rope in (0..ropes).filter(|&r| !rigid[r]) {
            let (n, closed) = (arcs[rope].len(), self.closed[rope]);
            for other in (0..ropes).filter(|&r| rigid[r]) {
                let reach = radius + radii[other];
                let axis = &self.loops[other][..self.drawn[other]];
                let gap: Vec<f64> = self.loops[rope][..n]
                    .iter()
                    .map(|&p| plan_distance(p, axis, self.closed[other]))
                    .collect();
                let rise = |k: usize| (reach * reach - gap[k] * gap[k]).sqrt();
                let crossing: Vec<Stop> = stops[rope]
                    .iter()
                    .filter(|&&(.., c, over)| other == if over { behind[c].0 } else { front[c] })
                    .copied()
                    .collect();
                let mut taken = vec![false; n];
                for &(at, seg, _, c, _) in &crossing {
                    // No further than halfway to the next crossing of the two.
                    let half = crossing
                        .iter()
                        .map(|&(b, ..)| along(at, b, lengths[rope], closed))
                        .filter(|&d| d > 0.0)
                        .fold(f64::MAX, f64::min)
                        / 2.0;
                    for forward in [false, true] {
                        let mut k = if forward { (seg + 1) % n } else { seg };
                        loop {
                            let d = along(arcs[rope][k], at, lengths[rope], closed);
                            if taken[k] || gap[k] >= reach || d > half {
                                break;
                            }
                            taken[k] = true;
                            cylinder[c].push((k, d, rise(k)));
                            k = match (forward, closed) {
                                (true, true) => (k + 1) % n,
                                (false, true) => (k + n - 1) % n,
                                (true, false) if k + 1 < n => k + 1,
                                (false, false) if k > 0 => k - 1,
                                _ => break,
                            };
                        }
                    }
                }
                beside[other].extend(
                    (0..n)
                        .filter(|&k| gap[k] < reach && !taken[k])
                        .map(|k| (rope, k, rise(k))),
                );
            }
        }
        // The height asked of each passage in front, and of each rigid rope: a
        // spar on the surface has its centre its radius less the rope's up.
        let mut lift: Vec<Vec<f64>> = stops.iter().map(|l| vec![diameter; l.len()]).collect();
        let mut level: Vec<f64> = radii.iter().map(|r| r - radius).collect();
        // How long each passage in front holds on top.
        let mut held: Vec<Vec<f64>> = stops
            .iter()
            .map(|l| l.iter().map(|&(.., c, _)| hold[c]).collect())
            .collect();
        let profiles = |lift: &[Vec<f64>], level: &[f64], held: &[Vec<f64>]| -> Vec<Vec<f64>> {
            (0..ropes)
                .map(|r| {
                    if rigid[r] {
                        return vec![level[r]; arcs[r].len()];
                    }
                    let plateaus: Vec<(f64, f64, f64)> = stops[r]
                        .iter()
                        .zip(lift[r].iter().zip(&held[r]))
                        .filter(|((.., over), _)| *over)
                        .map(|(&(at, ..), (&h, &hold))| (at, h, hold))
                        .collect();
                    heights(&arcs[r], &plateaus, lengths[r], self.closed[r], radius)
                })
                .collect()
        };
        let mut round = 0;
        let z = loop {
            // A rope over a rigid one holds only until it can come down along
            // the rigid one's side.
            for (r, list) in stops.iter().enumerate().filter(|&(r, _)| !rigid[r]) {
                for (i, &(.., c, over)) in list.iter().enumerate() {
                    let u = behind[c].0;
                    if over && rigid[u] {
                        let floors: Vec<(f64, f64)> = cylinder[c]
                            .iter()
                            .map(|&(_, d, rise)| (d, level[u] + rise))
                            .collect();
                        let most = floors.iter().fold(hold[c], |m, &(d, _)| m.max(d));
                        held[r][i] = shortest_hold(lift[r][i], &floors, most, radius);
                    }
                }
            }
            let z = profiles(&lift, &level, &held);
            // Where the rope behind is at the crossing, and the highest it gets
            // while the one in front holds over it.
            let here = |u: usize, j: usize| {
                let (_, seg, t, ..) = stops[u][j];
                let n = z[u].len();
                z[u][seg] + t * (z[u][(seg + 1) % n] - z[u][seg])
            };
            let peak = |u: usize, j: usize| {
                let (at, .., c, _) = stops[u][j];
                arcs[u]
                    .iter()
                    .zip(&z[u])
                    .filter(|&(&s, _)| along(s, at, lengths[u], self.closed[u]) <= hold[c])
                    .fold(here(u, j), |m, (_, &h)| m.max(h))
            };
            let mut raised = false;
            for (r, list) in stops.iter().enumerate() {
                for (i, &(.., c, over)) in list.iter().enumerate() {
                    if !over {
                        continue;
                    }
                    let (u, j) = behind[c];
                    let need = if rigid[r] && !rigid[u] {
                        // A spar over a rope rests on it as a tube would.
                        cylinder[c]
                            .iter()
                            .fold(here(u, j) + apart[c], |m, &(k, _, rise)| {
                                m.max(z[u][k] + rise)
                            })
                    } else {
                        peak(u, j) + apart[c]
                    };
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
            // A rigid rope rests as well on a rope beside it lower down, under
            // its side.
            for (r, list) in beside.iter().enumerate() {
                let need = list
                    .iter()
                    .filter(|&&(u, k, _)| z[u][k] < level[r])
                    .map(|&(u, k, rise)| z[u][k] + rise)
                    .fold(f64::MIN, f64::max);
                if level[r] < need - 1e-9 {
                    level[r] = need;
                    raised = true;
                }
            }
            round += 1;
            if !raised {
                break z;
            }
            if round == ROUNDS {
                return Err(DiagramError::Unsettled(ROUNDS));
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
                radius: radii[rope],
            })
            .collect();
        let min_clearance = clearance(&paths, &arcs, &lengths, 3.0);
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
    let raw: Vec<f64> = s
        .iter()
        .map(|&x| {
            plateaus
                .iter()
                .map(|&(at, h, hold)| bump(along(x, at, length, closed), h, hold, radius))
                .fold(0.0, f64::max)
        })
        .collect();
    fill(s, &raw, length, closed, BRIDGE * radius)
}

/// A plateau of height `h` held for `hold`, `d` along the rope from its middle.
fn bump(d: f64, h: f64, hold: f64, radius: f64) -> f64 {
    let reach = RAMP * radius * (h / (2.0 * radius)).sqrt();
    let u = ((d - hold) / reach).clamp(0.0, 1.0);
    h * (1.0 + (PI * u).cos()) / 2.0
}

/// The shortest hold, from the least to `most`, for which a plateau of height
/// `h` stays on or above every `(distance along the rope, floor)`: `most` when
/// even that does not.
fn shortest_hold(h: f64, floors: &[(f64, f64)], most: f64, radius: f64) -> f64 {
    let clear = |hold: f64| {
        floors
            .iter()
            .all(|&(d, floor)| bump(d, h, hold, radius) >= floor - 1e-9)
    };
    let (mut low, mut high) = (HOLD * radius, most);
    if clear(low) || !clear(high) {
        return if clear(low) { low } else { high };
    }
    for _ in 0..40 {
        let mid = (low + high) / 2.0;
        if clear(mid) {
            high = mid;
        } else {
            low = mid;
        }
    }
    high
}

/// How far `p` is in plan from the nearest point of a drawn rope.
fn plan_distance(p: [f64; 3], line: &[[f64; 3]], closed: bool) -> f64 {
    let n = line.len();
    let segments = if closed { n } else { n.saturating_sub(1) };
    (0..segments)
        .map(|i| {
            let (a, b) = (line[i], line[(i + 1) % n]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy);
            let t = if t.is_finite() {
                t.clamp(0.0, 1.0)
            } else {
                0.0
            };
            (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
        })
        .fold(f64::MAX, f64::min)
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
    use crate::testing::{bights, clove_hitch, constrictor};

    /// For each drawn crossing, how far the passage in front is above the one
    /// behind on the resting path (what reading it by height would decide),
    /// over the two tubes' radii summed: below 1 they pass through each other.
    fn separations(d: &RopeDiagram, g: &Geometry) -> Vec<f64> {
        let mut at = vec![(f64::NAN, f64::NAN, 0.0); d.crossings().len()];
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
                at[p.crossing].2 += path.radius;
            }
        }
        at.into_iter()
            .filter(|(o, u, _)| !o.is_nan() && !u.is_nan())
            .map(|(o, u, apart)| (o - u) / apart)
            .collect()
    }

    /// The lowest point of any tube, above the surface (a rope's centre at 0
    /// lies on it, one radius up).
    fn lowest(g: &Geometry) -> f64 {
        g.ropes
            .iter()
            .flat_map(|r| r.points.iter().map(move |p| p[2] - (r.radius - g.radius)))
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

    /// A knot from the catalogue or a fixture: (name, drawing, rope radius,
    /// rigid ropes, spar radius).
    type Laid = (String, RopeDiagram, f64, Vec<bool>, f64);

    #[test]
    fn a_rope_rests_on_the_surface_and_keeps_its_crossings() {
        let mut drawings: Vec<Laid> = catalog()
            .iter()
            .map(|e| {
                let d = e.diagram().unwrap();
                let rigid = vec![false; d.components()];
                (e.id.to_string(), d, e.radius, rigid, e.radius)
            })
            .collect();
        let reef = RopeDiagram::new(&bights(), "OUOOUO UOUUOU").unwrap();
        drawings.push(("bights".into(), reef, 0.16, vec![false; 2], 0.16));
        for spar in [1.0, 2.0, 3.0] {
            let clove = RopeDiagram::new(&clove_hitch(), "height").unwrap();
            let name = format!("clove hitch, spar {spar} radii");
            drawings.push((name, clove, 0.16, vec![false, true], spar * 0.16));
        }
        let turned = RopeDiagram::new(&constrictor(), "height").unwrap();
        let name = "constrictor, spar 2.5 radii".to_string();
        drawings.push((name, turned, 0.16, vec![false, true], 2.5 * 0.16));
        for (id, d, radius, rigid, spar) in drawings {
            let g = d.resting(radius, &rigid, spar).unwrap();
            assert!(lowest(&g) >= 0.0, "{id}: below the surface");
            // Where the old heights hang a rope under the deck.
            assert!(lowest(&d.geometry(radius).unwrap()) < -radius, "{id}");
            let gaps = separations(&d, &g);
            assert_eq!(gaps.len(), d.drawn_crossings(), "{id}");
            assert!(gaps.iter().all(|&h| h > 1.0 - 1e-9), "{id}: {gaps:?}");
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

    /// A rope wound over and under a spar, four crossings.
    fn wound() -> RopeDiagram {
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
        d
    }

    /// The spar stays level, one diameter up, and the rope lies on top of it
    /// where it passes over.
    #[test]
    fn a_rigid_spar_stays_level() {
        let d = wound();
        let radius = 0.16;
        let g = d.resting(radius, &[false, true], radius).unwrap();
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
        assert!(separations(&d, &g).iter().all(|&h| h > 1.0 - 1e-9));
        assert!(g.min_clearance.unwrap() >= 1.0 - 1e-9);
        assert_eq!(
            d.resting(radius, &[true], radius),
            Err(DiagramError::Rigid { ropes: 2, got: 1 })
        );
    }

    /// A spar three rope radii thick sits on the rope where it passes over it,
    /// its centre the two radii summed up; the rope passing over the spar lies
    /// that much higher again, and the spar's path says how thick it is.
    #[test]
    fn a_thick_spar_rests_on_the_rope_and_the_rope_on_it() {
        let d = wound();
        let radius = 0.16;
        let spar = 3.0 * radius;
        let g = d.resting(radius, &[false, true], spar).unwrap();
        assert_eq!((g.ropes[0].radius, g.ropes[1].radius), (radius, spar));
        let level: Vec<f64> = g.ropes[1].points.iter().map(|p| p[2]).collect();
        assert!(
            level
                .iter()
                .all(|&z| z == level[0] && (z - 4.0 * radius).abs() < 1e-9),
            "{level:?}"
        );
        let top = g.ropes[0]
            .points
            .iter()
            .map(|p| p[2])
            .fold(f64::MIN, f64::max);
        assert!((top - 8.0 * radius).abs() < 1e-9, "{top}");
        assert!(lowest(&g) >= 0.0);
        assert!(separations(&d, &g).iter().all(|&h| h > 1.0 - 1e-9));
        assert!(g.min_clearance.unwrap() >= 1.0 - 1e-9);
        // With nothing under it, a thick spar lies on the surface: its centre
        // its radius less the rope's up.
        let pole = Rope {
            points: vec![[-3.6, 0.1, 0.0], [3.6, 0.1, 0.0]],
            closed: false,
        };
        let alone = RopeDiagram::new(&[pole], "").unwrap();
        let g = alone.resting(radius, &[true], spar).unwrap();
        assert!(g.ropes[0].points.iter().all(|p| p[2] == spar - radius));
        for bad in [0.5 * radius, f64::NAN] {
            assert!(matches!(
                d.resting(radius, &[false, true], bad),
                Err(DiagramError::Spar { .. })
            ));
        }
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
