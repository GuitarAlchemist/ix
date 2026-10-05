//! Knots tied in rope: ropes drawn in the plane, and at each crossing which
//! passage is in front.
//!
//! A [`RopeDiagram`] is drawn the way a knot book draws a knot. Each rope is a
//! smooth curve through control points, open (a standing part and a working
//! end) or closed (a grommet). The diagram finds where the curves cross, then
//! answers what a braid answers: the components, the writhe and the Jones
//! polynomial of the closure, and a 3D path for each rope that a renderer can
//! draw.
//!
//! Which passage is in front is given one of three ways:
//! - one letter per passage, `O` (over) or `U` (under), in the order the ropes
//!   reach them: rope 0 from its first control point to its last, then rope 1;
//! - `alternating`, for a single rope: over, under, over, … from its start;
//! - `height`: the control points carry a `z`, and the higher passage is in
//!   front.
//!
//! **Closure.** An open rope is closed by an arc from its end out past the
//! drawing, around, and back in to its start, lying above everything else.
//! Arcs that lie above the whole diagram and share their endpoints are
//! isotopic, so the closure is well defined: it is the knot that tables give
//! for a stopper knot. When several ropes are open, rope 0's arc is the
//! highest, then rope 1's.
//!
//! **Bracket.** The Kauffman bracket is contracted one crossing at a time, in
//! the order the ropes reach them, keeping for each partial state only how its
//! loose ends are joined. The cost grows with how wide the diagram is rather
//! than with 2^crossings.

use crate::gauss::{GaussCode, GaussRope};
use crate::jones::{normalize, Jones};
use crate::poly::Laurent;
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

/// The most ropes in one diagram.
pub const MAX_ROPES: usize = 8;
/// The most control points over all ropes.
pub const MAX_CONTROL_POINTS: usize = 256;
/// The most crossings, the closure's included.
pub const MAX_DIAGRAM_CROSSINGS: usize = 64;
/// Curve points per span between two control points.
pub const SAMPLES_PER_SPAN: usize = 16;
/// The most partial states the bracket keeps between two crossings.
const MAX_STATES: usize = 1 << 18;
/// Below this, a crossing parameter counts as sitting on a curve point.
const EPS: f64 = 1e-9;

/// One rope as drawn: control points `[x, y, z]`, where `z` is read only when
/// the diagram says `height`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rope {
    pub points: Vec<[f64; 3]>,
    pub closed: bool,
}

/// Why a drawing was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DiagramError {
    #[error("a diagram has 1 to {MAX_ROPES} ropes, got {0}")]
    Ropes(usize),
    #[error("at most {MAX_CONTROL_POINTS} control points over all ropes, got {0}")]
    ControlPoints(usize),
    #[error("rope {rope}: {reason}")]
    Rope { rope: usize, reason: String },
    #[error(
        "two passages meet at ({x:.3}, {y:.3}) without crossing cleanly; move a control point"
    )]
    Degenerate { x: f64, y: f64 },
    #[error("the drawing and its closure cross {0} times; the limit is {MAX_DIAGRAM_CROSSINGS}")]
    Crossings(usize),
    #[error("{0}")]
    Over(String),
    #[error("the diagram is too wide to evaluate: more than {MAX_STATES} partial states")]
    Width,
    #[error("a bracket coefficient overflowed i128")]
    Overflow,
    #[error("rope radius must be a positive number, got {0}")]
    Radius(f64),
    #[error("give one pull per rope: {ropes} ropes, {got} pulls")]
    Pulls { ropes: usize, got: usize },
    #[error("say for each rope whether it is rigid: {ropes} ropes, {got} given")]
    Rigid { ropes: usize, got: usize },
    #[error("a spar's radius must be a number at least the rope's ({radius}), got {spar}")]
    Spar { spar: f64, radius: f64 },
}

/// A crossing of the diagram or of its closure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crossing {
    /// Where, in the plane.
    pub at: [f64; 2],
    /// +1 when, looking along the passage in front, the one behind runs from
    /// right to left; the writhe is their sum.
    pub sign: i8,
    /// On a closing arc rather than in the drawing.
    pub closure: bool,
}

/// One rope's path in 3D: the drawn curve, lifted where it passes in front and
/// lowered where it passes behind.
#[derive(Debug, Clone, PartialEq)]
pub struct RopePath {
    pub closed: bool,
    pub points: Vec<[f64; 3]>,
    /// The tube's radius: the rope's, or a resting spar's own.
    pub radius: f64,
}

/// The ropes in 3D, for a renderer.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub radius: f64,
    pub ropes: Vec<RopePath>,
    /// The smallest distance between two rope centrelines that are not
    /// neighbours along a rope, over the sum of the two tubes' radii (in rope
    /// diameters when every tube is the rope's): below 1 the tubes pass
    /// through each other. `None` when no two such points exist.
    pub min_clearance: Option<f64>,
}

/// A passage of a rope (or of its closing arc) through a crossing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Passage {
    pub(crate) seg: usize,
    pub(crate) t: f64,
    pub(crate) crossing: usize,
    pub(crate) over: bool,
}

/// A knot or link as drawn: see the module documentation.
#[derive(Debug, Clone)]
pub struct RopeDiagram {
    /// The ropes as given, to redraw them with other crossings.
    ropes: Vec<Rope>,
    /// Each rope's smoothed curve, then (open ropes) its closing arc.
    pub(crate) loops: Vec<Vec<[f64; 3]>>,
    /// Points of each loop that belong to the drawn rope; the rest close it.
    pub(crate) drawn: Vec<usize>,
    pub(crate) closed: Vec<bool>,
    /// Each loop's passages through the crossings, in order along it.
    pub(crate) passages: Vec<Vec<Passage>>,
    crossings: Vec<Crossing>,
    writhe: i64,
    jones: Jones,
}

impl RopeDiagram {
    /// Read a drawing. `over` is `alternating`, `height`, or one letter `O` or
    /// `U` per passage of the drawn ropes (spaces and commas are ignored).
    pub fn new(ropes: &[Rope], over: &str) -> Result<Self, DiagramError> {
        if ropes.is_empty() || ropes.len() > MAX_ROPES {
            return Err(DiagramError::Ropes(ropes.len()));
        }
        let total: usize = ropes.iter().map(|r| r.points.len()).sum();
        if total > MAX_CONTROL_POINTS {
            return Err(DiagramError::ControlPoints(total));
        }
        for (i, rope) in ropes.iter().enumerate() {
            check_rope(i, rope)?;
        }
        let curves: Vec<Vec<[f64; 3]>> =
            ropes.iter().map(|r| smooth(&r.points, r.closed)).collect();
        let drawn: Vec<usize> = curves.iter().map(Vec::len).collect();
        let closed: Vec<bool> = ropes.iter().map(|r| r.closed).collect();
        let loops = close_open_ropes(&curves, &closed);

        let hits = intersections(&loops)?;
        if hits.len() > MAX_DIAGRAM_CROSSINGS {
            return Err(DiagramError::Crossings(hits.len()));
        }
        let mut passages: Vec<Vec<Passage>> = vec![Vec::new(); loops.len()];
        for (k, hit) in hits.iter().enumerate() {
            for &(rope, seg, t) in &[hit.a, hit.b] {
                passages[rope].push(Passage {
                    seg,
                    t,
                    crossing: k,
                    over: false,
                });
            }
        }
        for list in &mut passages {
            list.sort_by(|p, q| (p.seg, p.t).partial_cmp(&(q.seg, q.t)).unwrap());
        }
        let is_closure = |rope: usize, seg: usize| !closed[rope] && seg + 1 >= drawn[rope];
        decide_over(&mut passages, &hits, &loops, &is_closure, over)?;

        // Number the passages along the ropes; an edge runs from one passage to
        // the next, so a passage's outgoing edge has its own number.
        let mut base = Vec::with_capacity(passages.len());
        let mut edges = 0u32;
        for list in &passages {
            base.push(edges);
            edges += list.len() as u32;
        }
        let mut ends: Vec<[(usize, usize); 2]> = vec![[(0, 0); 2]; hits.len()];
        let mut seen = vec![0usize; hits.len()];
        for (rope, list) in passages.iter().enumerate() {
            for (j, p) in list.iter().enumerate() {
                // Slot 0 is the passage behind, slot 1 the one in front.
                ends[p.crossing][usize::from(p.over)] = (rope, j);
                seen[p.crossing] += 1;
            }
        }
        debug_assert!(seen.iter().all(|&n| n == 2));

        let incoming = |(rope, j): (usize, usize)| {
            let n = passages[rope].len();
            base[rope] + ((j + n - 1) % n) as u32
        };
        let outgoing = |(rope, j): (usize, usize)| base[rope] + j as u32;
        let direction = |(rope, j): (usize, usize)| {
            let pts = &loops[rope];
            let s = passages[rope][j].seg;
            let (a, b) = (pts[s], pts[(s + 1) % pts.len()]);
            [b[0] - a[0], b[1] - a[1]]
        };

        let mut crossings = Vec::with_capacity(hits.len());
        let mut slots = Vec::with_capacity(hits.len());
        let mut order = Vec::with_capacity(hits.len());
        for (k, hit) in hits.iter().enumerate() {
            let [under, front] = ends[k];
            let (u, v) = (direction(under), direction(front));
            let sign: i8 = if cross(v, u) > 0.0 { 1 } else { -1 };
            // Counterclockwise from the edge coming in behind: when the front
            // passage runs counterclockwise of the one behind, its incoming edge
            // comes next.
            let (a, c) = (incoming(under), outgoing(under));
            let (fin, fout) = (incoming(front), outgoing(front));
            slots.push(if cross(u, v) > 0.0 {
                [a, fin, c, fout]
            } else {
                [a, fout, c, fin]
            });
            crossings.push(Crossing {
                at: hit.at,
                sign,
                closure: is_closure(hit.a.0, hit.a.1) || is_closure(hit.b.0, hit.b.1),
            });
            let first = |(rope, j): (usize, usize)| base[rope] + j as u32;
            order.push((first(under).min(first(front)), k));
        }
        order.sort_unstable();
        let mut ordered: Vec<[u32; 4]> = order.iter().map(|&(_, k)| slots[k]).collect();
        cut(&mut ordered, edges);

        let free_loops = passages.iter().filter(|l| l.is_empty()).count();
        let mut bracket = contract(&ordered)?;
        let extra = if ordered.is_empty() {
            free_loops - 1
        } else {
            free_loops
        };
        for _ in 0..extra {
            bracket = bracket.checked_times_loop().ok_or(DiagramError::Overflow)?;
        }
        let writhe = crossings.iter().map(|c| i64::from(c.sign)).sum();
        let jones = normalize(&bracket, writhe);
        Ok(Self {
            ropes: ropes.to_vec(),
            loops,
            drawn,
            closed,
            passages,
            crossings,
            writhe,
            jones,
        })
    }

    /// Ropes, which are also the components of the closure.
    pub fn components(&self) -> usize {
        self.loops.len()
    }

    /// The ropes as given.
    pub fn ropes(&self) -> &[Rope] {
        &self.ropes
    }

    /// Each rope's passages through the crossings of the drawing (not those its
    /// closure adds), in the order `over` letters list them: the crossing's
    /// index in [`Self::crossings`], and whether this passage is in front.
    pub fn passes(&self) -> Vec<Vec<(usize, bool)>> {
        self.passages
            .iter()
            .map(|list| {
                list.iter()
                    .filter(|p| !self.crossings[p.crossing].closure)
                    .map(|p| (p.crossing, p.over))
                    .collect()
            })
            .collect()
    }

    /// The crossings of the drawing, then those its closure adds.
    pub fn crossings(&self) -> &[Crossing] {
        &self.crossings
    }

    /// The crossings of the drawing alone.
    pub fn drawn_crossings(&self) -> usize {
        self.crossings.iter().filter(|c| !c.closure).count()
    }

    /// Crossing signs summed, the closure's included.
    pub fn writhe(&self) -> i64 {
        self.writhe
    }

    /// The linking number of each two ropes, closed as for the Jones
    /// polynomial: half the sum of the signs of the crossings between them, as
    /// `((a, b), number)` with `a < b`. Not zero proves the two closed ropes
    /// cannot be pulled apart; zero proves nothing.
    // @ai:invariant crossings between two closed curves in the plane come in an even signed sum, so each linking number is a whole number [T:test conf:0.85 src:diagram::tests::linking_numbers_count_how_often_two_rings_wind_round]
    pub fn linking_numbers(&self) -> Vec<((usize, usize), i64)> {
        let n = self.loops.len();
        let mut ropes_at: Vec<Vec<usize>> = vec![Vec::new(); self.crossings.len()];
        for (rope, list) in self.passages.iter().enumerate() {
            for p in list {
                ropes_at[p.crossing].push(rope);
            }
        }
        let mut sums = vec![0i64; n * n];
        for (c, at) in self.crossings.iter().zip(&ropes_at) {
            if let [a, b] = at[..] {
                if a != b {
                    sums[a.min(b) * n + a.max(b)] += i64::from(c.sign);
                }
            }
        }
        (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .map(|(a, b)| ((a, b), sums[a * n + b] / 2))
            .collect()
    }

    /// The Jones polynomial of the closure.
    pub fn jones(&self) -> &Jones {
        &self.jones
    }

    /// The Gauss code of the drawing: each rope's passages through the
    /// crossings it draws (not its closure's), from its first control point,
    /// the crossings numbered in the order they are first passed.
    pub fn gauss_code(&self) -> GaussCode {
        let ropes = self
            .passages
            .iter()
            .zip(&self.closed)
            .map(|(list, &closed)| GaussRope {
                closed,
                passages: list
                    .iter()
                    .filter(|p| !self.crossings[p.crossing].closure)
                    .map(|p| (p.crossing as u32 + 1, p.over))
                    .collect(),
            })
            .collect();
        GaussCode { ropes }.canonical()
    }

    /// Each rope in 3D, as a tube of `radius`: lifted by 1.1 radii where it
    /// passes in front and lowered as much where it passes behind, held there
    /// for 1.5 radii of its length either side, then easing back to the plane
    /// over the next 3. The hold keeps two tubes apart while a shallow crossing
    /// is still close in the plane.
    pub fn geometry(&self, radius: f64) -> Result<Geometry, DiagramError> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(DiagramError::Radius(radius));
        }
        let (lift, hold, reach) = (1.1 * radius, 1.5 * radius, 3.0 * radius);
        let mut ropes = Vec::with_capacity(self.loops.len());
        let mut lengths = Vec::with_capacity(self.loops.len());
        let mut arcs = Vec::with_capacity(self.loops.len());
        for (rope, pts) in self.loops.iter().enumerate() {
            let pts = &pts[..self.drawn[rope]];
            let s = arclength(pts);
            let length = *s.last().unwrap();
            let length = if self.closed[rope] {
                length + dist2(pts[pts.len() - 1], pts[0]).sqrt()
            } else {
                length
            };
            let bumps: Vec<(f64, f64)> = self.passages[rope]
                .iter()
                .filter(|p| !self.crossings[p.crossing].closure)
                .map(|p| {
                    let next = if p.seg + 1 < pts.len() {
                        s[p.seg + 1]
                    } else {
                        length
                    };
                    let at = s[p.seg] + p.t * (next - s[p.seg]);
                    (at, if p.over { lift } else { -lift })
                })
                .collect();
            let points = pts
                .iter()
                .zip(&s)
                .map(|(p, &here)| {
                    let z: f64 = bumps
                        .iter()
                        .map(|&(at, h)| {
                            let d = along(here, at, length, self.closed[rope]) - hold;
                            if d <= 0.0 {
                                h
                            } else if d < reach {
                                h * (1.0 + (PI * d / reach).cos()) / 2.0
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    [p[0], p[1], z]
                })
                .collect();
            ropes.push(RopePath {
                closed: self.closed[rope],
                points,
                radius,
            });
            lengths.push(length);
            arcs.push(s);
        }
        let min_clearance = clearance(&ropes, &arcs, &lengths, reach / radius);
        Ok(Geometry {
            radius,
            ropes,
            min_clearance,
        })
    }
}

fn check_rope(i: usize, rope: &Rope) -> Result<(), DiagramError> {
    let bad = |reason: String| Err(DiagramError::Rope { rope: i, reason });
    let least = if rope.closed { 3 } else { 2 };
    if rope.points.len() < least {
        let kind = if rope.closed { "closed" } else { "open" };
        return bad(format!(
            "an {kind} rope needs at least {least} control points"
        ));
    }
    if let Some(p) = rope
        .points
        .iter()
        .find(|p| !p.iter().all(|v| v.is_finite()))
    {
        return bad(format!("control point {p:?} is not finite"));
    }
    let n = rope.points.len();
    let pairs = if rope.closed { n } else { n - 1 };
    for k in 0..pairs {
        let (a, b) = (rope.points[k], rope.points[(k + 1) % n]);
        if dist2(a, b) < EPS * EPS {
            return bad(format!("control points {k} and {} coincide", (k + 1) % n));
        }
    }
    Ok(())
}

/// A centripetal Catmull–Rom curve through the control points: no cusps or
/// loops inside a span. An open rope's ends are extended straight; a closed
/// rope's curve does not repeat its first point.
fn smooth(p: &[[f64; 3]], closed: bool) -> Vec<[f64; 3]> {
    let n = p.len() as isize;
    let at = |i: isize| -> [f64; 3] {
        if closed {
            p[i.rem_euclid(n) as usize]
        } else if i < 0 {
            lerp(p[0], p[1], -1.0)
        } else if i >= n {
            lerp(p[(n - 1) as usize], p[(n - 2) as usize], -1.0)
        } else {
            p[i as usize]
        }
    };
    let spans = if closed { n } else { n - 1 };
    let mut out = Vec::with_capacity(spans as usize * SAMPLES_PER_SPAN + 1);
    for i in 0..spans {
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        let t1 = dist2(p0, p1).sqrt().sqrt();
        let t2 = t1 + dist2(p1, p2).sqrt().sqrt();
        let t3 = t2 + dist2(p2, p3).sqrt().sqrt();
        for k in 0..SAMPLES_PER_SPAN {
            let t = t1 + (t2 - t1) * k as f64 / SAMPLES_PER_SPAN as f64;
            let a1 = mix(p0, p1, 0.0, t1, t);
            let a2 = mix(p1, p2, t1, t2, t);
            let a3 = mix(p2, p3, t2, t3, t);
            let b1 = mix(a1, a2, 0.0, t2, t);
            let b2 = mix(a2, a3, t1, t3, t);
            out.push(mix(b1, b2, t1, t2, t));
        }
    }
    if !closed {
        out.push(p[(n - 1) as usize]);
    }
    out
}

/// The point at parameter `t` on the line through `a` (at `ta`) and `b` (at `tb`).
fn mix(a: [f64; 3], b: [f64; 3], ta: f64, tb: f64, t: f64) -> [f64; 3] {
    lerp(a, b, (t - ta) / (tb - ta))
}

fn lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Squared distance in the plane.
pub(crate) fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)
}

fn cross(u: [f64; 2], v: [f64; 2]) -> f64 {
    u[0] * v[1] - u[1] * v[0]
}

/// Each rope as a closed loop: an open rope continues from its end straight out
/// past the drawing, around a circle, and straight back in to its start. Rope
/// `r`'s circle has radius `(2 + r)` times the drawing's, so no two arcs meet.
fn close_open_ropes(curves: &[Vec<[f64; 3]>], closed: &[bool]) -> Vec<Vec<[f64; 3]>> {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in curves.iter().flatten() {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, 0.0];
    let size = ((hi[0] - lo[0]).hypot(hi[1] - lo[1]) / 2.0).max(1.0);
    curves
        .iter()
        .enumerate()
        .map(|(rope, curve)| {
            let mut out = curve.clone();
            if closed[rope] {
                return out;
            }
            let (start, end) = (curve[0], curve[curve.len() - 1]);
            let radius = (2.0 + rope as f64) * size;
            let heading = |p: [f64; 3], other: [f64; 3]| {
                let (dx, dy) = (p[0] - centre[0], p[1] - centre[1]);
                let (ox, oy) = (p[0] - other[0], p[1] - other[1]);
                if dx.hypot(dy) > EPS {
                    dy.atan2(dx)
                } else if ox.hypot(oy) > EPS {
                    oy.atan2(ox)
                } else {
                    0.0
                }
            };
            let (from, to) = (heading(end, start), heading(start, end));
            let mut sweep = (to - from).rem_euclid(2.0 * PI);
            if sweep > PI {
                sweep -= 2.0 * PI;
            }
            let steps = (sweep.abs() / (PI / 18.0)).ceil().max(1.0) as usize;
            for k in 0..=steps {
                let a = from + sweep * k as f64 / steps as f64;
                let p = [
                    centre[0] + radius * a.cos(),
                    centre[1] + radius * a.sin(),
                    0.0,
                ];
                if dist2(p, out[out.len() - 1]) > EPS * EPS {
                    out.push(p);
                }
            }
            out
        })
        .collect()
}

/// A crossing of two segments: (rope, segment, parameter) for each, and where.
struct Hit {
    a: (usize, usize, f64),
    b: (usize, usize, f64),
    at: [f64; 2],
}

struct Segment {
    rope: usize,
    index: usize,
    p: [f64; 2],
    q: [f64; 2],
    lo: [f64; 2],
    hi: [f64; 2],
}

/// Every proper crossing between two segments of the loops that are not
/// neighbours, found by a sweep along x.
fn intersections(loops: &[Vec<[f64; 3]>]) -> Result<Vec<Hit>, DiagramError> {
    let mut segments = Vec::new();
    for (rope, pts) in loops.iter().enumerate() {
        let m = pts.len();
        for index in 0..m {
            let (a, b) = (pts[index], pts[(index + 1) % m]);
            segments.push(Segment {
                rope,
                index,
                p: [a[0], a[1]],
                q: [b[0], b[1]],
                lo: [a[0].min(b[0]), a[1].min(b[1])],
                hi: [a[0].max(b[0]), a[1].max(b[1])],
            });
        }
    }
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by(|&i, &j| {
        segments[i].lo[0]
            .total_cmp(&segments[j].lo[0])
            .then(i.cmp(&j))
    });
    let mut hits = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    for &i in &order {
        let s = &segments[i];
        active.retain(|&j| segments[j].hi[0] >= s.lo[0]);
        for &j in &active {
            let o = &segments[j];
            if o.hi[1] < s.lo[1] || o.lo[1] > s.hi[1] || neighbours(s, o, loops) {
                continue;
            }
            if let Some((t, u, at)) = meet(o, s)? {
                hits.push(Hit {
                    a: (o.rope, o.index, t),
                    b: (s.rope, s.index, u),
                    at,
                });
            }
        }
        active.push(i);
    }
    Ok(hits)
}

fn neighbours(a: &Segment, b: &Segment, loops: &[Vec<[f64; 3]>]) -> bool {
    if a.rope != b.rope {
        return false;
    }
    let m = loops[a.rope].len();
    a.index == b.index || (a.index + 1) % m == b.index || (b.index + 1) % m == a.index
}

/// Where segments `a` and `b` cross, as their parameters and the point; `None`
/// when they do not. A crossing at or near an end of either segment, or two
/// segments overlapping along a line, is refused rather than guessed at.
fn meet(a: &Segment, b: &Segment) -> Result<Option<(f64, f64, [f64; 2])>, DiagramError> {
    let d1 = [a.q[0] - a.p[0], a.q[1] - a.p[1]];
    let d2 = [b.q[0] - b.p[0], b.q[1] - b.p[1]];
    let w = [b.p[0] - a.p[0], b.p[1] - a.p[1]];
    let denom = cross(d1, d2);
    let (n1, n2) = (d1[0].hypot(d1[1]), d2[0].hypot(d2[1]));
    if denom.abs() <= 1e-12 * n1 * n2 {
        // Parallel: refuse only an overlap of positive length on one line.
        if cross(w, d1).abs() > 1e-12 * n1 * w[0].hypot(w[1]).max(n1) {
            return Ok(None);
        }
        let along = |p: [f64; 2]| ((p[0] - a.p[0]) * d1[0] + (p[1] - a.p[1]) * d1[1]) / (n1 * n1);
        let (s0, s1) = (along(b.p), along(b.q));
        let (lo, hi) = (s0.min(s1), s0.max(s1));
        if hi > EPS && lo < 1.0 - EPS {
            return Err(DiagramError::Degenerate {
                x: b.p[0],
                y: b.p[1],
            });
        }
        return Ok(None);
    }
    let t = cross(w, d2) / denom;
    let u = cross(w, d1) / denom;
    let inside = |x: f64| x > EPS && x < 1.0 - EPS;
    let near = |x: f64| x > -EPS && x < 1.0 + EPS;
    let at = [a.p[0] + t * d1[0], a.p[1] + t * d1[1]];
    if inside(t) && inside(u) {
        Ok(Some((t, u, at)))
    } else if near(t) && near(u) {
        Err(DiagramError::Degenerate { x: at[0], y: at[1] })
    } else {
        Ok(None)
    }
}

/// Settle which passage of each crossing is in front. A closing arc is above
/// the drawing, and rope 0's above rope 1's; the drawing follows `over`.
fn decide_over(
    passages: &mut [Vec<Passage>],
    hits: &[Hit],
    loops: &[Vec<[f64; 3]>],
    is_closure: &dyn Fn(usize, usize) -> bool,
    over: &str,
) -> Result<(), DiagramError> {
    // For each crossing, its two passages as (rope, index in that rope's list).
    let mut at: Vec<Vec<(usize, usize)>> = vec![Vec::new(); hits.len()];
    for (rope, list) in passages.iter().enumerate() {
        for (j, p) in list.iter().enumerate() {
            at[p.crossing].push((rope, j));
        }
    }
    // A crossing with a closing arc is settled by the closure, not by `over`.
    let on_arc: Vec<bool> = at
        .iter()
        .map(|pair| {
            pair.iter()
                .any(|&(rope, j)| is_closure(rope, passages[rope][j].seg))
        })
        .collect();
    let drawn: Vec<(usize, usize)> = passages
        .iter()
        .enumerate()
        .flat_map(|(rope, list)| {
            list.iter()
                .enumerate()
                .map(move |(j, p)| (rope, j, p.crossing))
        })
        .filter(|&(_, _, k)| !on_arc[k])
        .map(|(rope, j, _)| (rope, j))
        .collect();

    let word = over.trim();
    let letters: Option<Vec<bool>> = if word.eq_ignore_ascii_case("alternating") {
        if passages.len() > 1 {
            return Err(DiagramError::Over(
                "`alternating` is for a single rope; give one letter O or U per passage".into(),
            ));
        }
        Some((0..drawn.len()).map(|k| k % 2 == 0).collect())
    } else if word.eq_ignore_ascii_case("height") {
        None
    } else {
        let mut letters = Vec::new();
        for c in word.chars().filter(|c| !c.is_whitespace() && *c != ',') {
            match c.to_ascii_uppercase() {
                'O' => letters.push(true),
                'U' => letters.push(false),
                _ => {
                    return Err(DiagramError::Over(format!(
                        "`over` is `alternating`, `height`, or the letters O and U; found {c:?}"
                    )))
                }
            }
        }
        if letters.len() != drawn.len() {
            return Err(DiagramError::Over(format!(
                "`over` has {} letters, but the drawing has {} crossings, so {} passages",
                letters.len(),
                drawn.len() / 2,
                drawn.len()
            )));
        }
        Some(letters)
    };
    let mut front: HashMap<(usize, usize), bool> = HashMap::new();
    if let Some(letters) = &letters {
        for (k, &key) in drawn.iter().enumerate() {
            front.insert(key, letters[k]);
        }
    }
    for (k, pair) in at.iter().enumerate() {
        let [(ra, ja), (rb, jb)] = [pair[0], pair[1]];
        let (pa, pb) = (passages[ra][ja], passages[rb][jb]);
        let (ca, cb) = (is_closure(ra, pa.seg), is_closure(rb, pb.seg));
        let a_front = match (ca, cb) {
            (true, true) => ra < rb,
            (true, false) => true,
            (false, true) => false,
            (false, false) => match &letters {
                Some(_) => {
                    let (fa, fb) = (front[&(ra, ja)], front[&(rb, jb)]);
                    if fa == fb {
                        let word = if fa { "over" } else { "under" };
                        let place = hits[k].at;
                        return Err(DiagramError::Over(format!(
                            "the crossing at ({:.2}, {:.2}) is passed {word} both times",
                            place[0], place[1]
                        )));
                    }
                    fa
                }
                None => {
                    let (za, zb) = (height(loops, ra, pa), height(loops, rb, pb));
                    if (za - zb).abs() < EPS {
                        let place = hits[k].at;
                        return Err(DiagramError::Over(format!(
                            "both passages of the crossing at ({:.2}, {:.2}) are at height {za:.3}",
                            place[0], place[1]
                        )));
                    }
                    za > zb
                }
            },
        };
        passages[ra][ja].over = a_front;
        passages[rb][jb].over = !a_front;
    }
    Ok(())
}

fn height(loops: &[Vec<[f64; 3]>], rope: usize, p: Passage) -> f64 {
    let pts = &loops[rope];
    let (a, b) = (pts[p.seg], pts[(p.seg + 1) % pts.len()]);
    a[2] + (b[2] - a[2]) * p.t
}

/// Give the second end of the first edge a label of its own, so the loop
/// through it stays open and the bracket comes out normalized: the unknot's
/// is 1, not the loop factor.
fn cut(slots: &mut [[u32; 4]], fresh: u32) {
    let Some(first) = slots.first().map(|s| s[0]) else {
        return;
    };
    let mut seen = false;
    for label in slots.iter_mut().flatten() {
        if *label == first {
            if seen {
                *label = fresh;
                return;
            }
            seen = true;
        }
    }
}

/// The Kauffman bracket of the crossings `slots`, each its four edges
/// counterclockwise from the one coming in behind: A joins the first edge to
/// the second and the third to the fourth, A⁻¹ the first to the fourth and the
/// second to the third. A state is how the loose ends are joined so far.
fn contract(slots: &[[u32; 4]]) -> Result<Laurent, DiagramError> {
    let mut states: BTreeMap<Vec<(u32, u32)>, Laurent> =
        BTreeMap::from([(Vec::new(), Laurent::one())]);
    for &[a, b, c, d] in slots {
        let mut next: BTreeMap<Vec<(u32, u32)>, Laurent> = BTreeMap::new();
        for (state, coeff) in &states {
            for (pairs, shift) in [([(a, b), (c, d)], 1), ([(a, d), (b, c)], -1)] {
                let mut open: HashMap<u32, u32> =
                    state.iter().flat_map(|&(x, y)| [(x, y), (y, x)]).collect();
                let mut term = coeff.clone();
                for (x, y) in pairs {
                    if join(&mut open, x, y) {
                        term = term.checked_times_loop().ok_or(DiagramError::Overflow)?;
                    }
                }
                let mut key: Vec<(u32, u32)> = open
                    .iter()
                    .filter(|(x, y)| x < y)
                    .map(|(x, y)| (*x, *y))
                    .collect();
                key.sort_unstable();
                next.entry(key)
                    .or_default()
                    .checked_add_scaled(&term, 1, shift)
                    .ok_or(DiagramError::Overflow)?;
            }
        }
        next.retain(|_, c| *c != Laurent::default());
        if next.len() > MAX_STATES {
            return Err(DiagramError::Width);
        }
        states = next;
    }
    let mut total = Laurent::default();
    for coeff in states.values() {
        total
            .checked_add_scaled(coeff, 1, 0)
            .ok_or(DiagramError::Overflow)?;
    }
    Ok(total)
}

/// Join loose ends `x` and `y` by an arc of a smoothing; true when that closes
/// a loop. A label met for the first time becomes a loose end; met the second
/// time, the arc continues to wherever its other end is.
fn join(open: &mut HashMap<u32, u32>, x: u32, y: u32) -> bool {
    if x == y || open.get(&x) == Some(&y) {
        open.remove(&x);
        open.remove(&y);
        return true;
    }
    let mut end = |label: u32| match open.remove(&label) {
        Some(other) => {
            open.remove(&other);
            other
        }
        None => label,
    };
    let (ex, ey) = (end(x), end(y));
    open.insert(ex, ey);
    open.insert(ey, ex);
    false
}

/// Arclength at each point, from the first.
pub(crate) fn arclength(pts: &[[f64; 3]]) -> Vec<f64> {
    let mut s = Vec::with_capacity(pts.len());
    let mut total = 0.0;
    for (k, p) in pts.iter().enumerate() {
        if k > 0 {
            total += dist2(pts[k - 1], *p).sqrt();
        }
        s.push(total);
    }
    s
}

/// Distance along a rope of `length` between arclengths `a` and `b`.
pub(crate) fn along(a: f64, b: f64, length: f64, closed: bool) -> f64 {
    let d = (a - b).abs();
    if closed {
        d.min(length - d)
    } else {
        d
    }
}

/// The least 3D distance between two centreline points that are not within
/// `reach` of their rope's radii of each other along the same rope, over the
/// sum of their ropes' radii.
pub(crate) fn clearance(
    ropes: &[RopePath],
    arcs: &[Vec<f64>],
    lengths: &[f64],
    reach: f64,
) -> Option<f64> {
    let mut best: Option<f64> = None;
    for (r1, a) in ropes.iter().enumerate() {
        for (r2, b) in ropes.iter().enumerate().skip(r1) {
            for (i, p) in a.points.iter().enumerate() {
                let from = if r1 == r2 { i + 1 } else { 0 };
                for (j, q) in b.points.iter().enumerate().skip(from) {
                    if r1 == r2
                        && along(arcs[r1][i], arcs[r1][j], lengths[r1], a.closed) < reach * a.radius
                    {
                        continue;
                    }
                    let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                        .sqrt()
                        / (a.radius + b.radius);
                    if best.map_or(true, |b| d < b) {
                        best = Some(d);
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jones;
    use crate::testing::{ring, words};

    fn open(points: &[[f64; 2]]) -> Rope {
        Rope {
            points: points.iter().map(|p| [p[0], p[1], 0.0]).collect(),
            closed: false,
        }
    }

    fn circle(cx: f64, cy: f64, r: f64, n: usize) -> Rope {
        Rope {
            points: (0..n)
                .map(|k| {
                    let a = 2.0 * PI * k as f64 / n as f64;
                    [cx + r * a.cos(), cy + r * a.sin(), 0.0]
                })
                .collect(),
            closed: true,
        }
    }

    #[test]
    fn a_braid_closure_drawn_as_a_ring_agrees_with_the_braid() {
        for b in words(80) {
            let d = RopeDiagram::new(&ring(&b), "height").unwrap();
            assert_eq!(d.drawn_crossings(), b.crossings(), "crossings: {b}");
            assert_eq!(d.writhe(), b.writhe(), "writhe: {b}");
            assert_eq!(d.components(), b.components(), "components: {b}");
            assert_eq!(*d.jones(), jones(&b), "jones: {b}");
        }
    }

    #[test]
    fn unknots_unlinks_and_kinks() {
        let arc = RopeDiagram::new(&[open(&[[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])], "").unwrap();
        assert_eq!(arc.jones().to_string(), "1");
        assert_eq!(arc.crossings().len(), 0);
        let straight = RopeDiagram::new(&[open(&[[0.0, 0.0], [5.0, 0.0]])], "").unwrap();
        assert_eq!(straight.jones().to_string(), "1");

        let two =
            RopeDiagram::new(&[circle(0.0, 0.0, 1.0, 6), circle(5.0, 0.0, 1.0, 6)], "").unwrap();
        assert_eq!(two.jones().to_string(), "-t^(-1/2) - t^(1/2)");

        // A curl: one crossing, writhe ±1, still the unknot.
        let curl = [
            [0.0, 0.0],
            [4.0, 0.0],
            [5.0, 1.5],
            [4.0, 3.0],
            [3.0, 1.5],
            [4.0, -1.5],
            [8.0, -1.0],
        ];
        for (letters, writhe) in [("OU", -1), ("UO", 1)] {
            let d = RopeDiagram::new(&[open(&curl)], letters).unwrap();
            assert_eq!(d.drawn_crossings(), 1);
            assert_eq!(d.writhe(), writhe, "{letters}");
            assert_eq!(d.jones().to_string(), "1", "{letters}");
        }
    }

    #[test]
    fn the_closing_arc_passes_over_and_never_changes_the_knot() {
        // A spiral from its centre outward: its end is outside, its start in
        // the middle, so the arc closing it crosses every turn, always on top.
        let spiral: Vec<[f64; 2]> = (0..40)
            .map(|k| {
                let a = 0.45 * k as f64;
                let r = 0.4 + 0.12 * k as f64;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let d = RopeDiagram::new(&[open(&spiral)], "").unwrap();
        assert!(d.crossings().iter().any(|c| c.closure));
        assert_eq!(d.drawn_crossings(), 0);
        assert_eq!(d.jones().to_string(), "1");
    }

    #[test]
    fn the_hopf_link_and_the_mirror() {
        // Both circles reach the upper crossing first: rope 0 over it, rope 1
        // under it, and the other way at the lower one.
        let ropes = [circle(0.0, 0.0, 2.0, 8), circle(2.5, 0.0, 2.0, 8)];
        let d = RopeDiagram::new(&ropes, "OU UO").unwrap();
        assert_eq!(d.components(), 2);
        assert_eq!(d.jones().at_one(), -2);
        let hopf = ["-t^(1/2) - t^(5/2)", "-t^(-5/2) - t^(-1/2)"];
        assert!(
            hopf.contains(&d.jones().to_string().as_str()),
            "{}",
            d.jones()
        );
        let m = RopeDiagram::new(&ropes, "UO OU").unwrap();
        assert_eq!(*m.jones(), d.jones().mirror());
        // Splitting them is the unlink, whichever crossing is changed.
        let unlink = RopeDiagram::new(&ropes, "OOUU").unwrap();
        assert_eq!(unlink.jones().to_string(), "-t^(-1/2) - t^(1/2)");
        assert_eq!(d.linking_numbers()[0].1.abs(), 1);
        assert_eq!(m.linking_numbers()[0].1, -d.linking_numbers()[0].1);
        assert_eq!(unlink.linking_numbers(), vec![((0, 1), 0)]);
    }

    #[test]
    fn linking_numbers_count_how_often_two_rings_wind_round() {
        use crate::Braid;
        let lk = |text: &str| {
            let b = Braid::parse(None, text).unwrap();
            RopeDiagram::new(&ring(&b), "height")
                .unwrap()
                .linking_numbers()
        };
        assert_eq!(lk("s1^2"), vec![((0, 1), 1)]);
        assert_eq!(lk("s1^-6"), vec![((0, 1), -3)]);
        assert_eq!(lk("s1^2 s2^4"), vec![((0, 1), 1), ((0, 2), 0), ((1, 2), 2)]);
        // One rope: no pairs. The figure-eight's crossings are all its own.
        assert!(lk("s1 s2^-1 s1 s2^-1").is_empty());

        // Against the braid word: follow each strand through the word, and sum
        // the signs of the generators that cross two different components.
        for b in words(80) {
            let n = b.strands();
            let mut at: Vec<usize> = (0..n).collect();
            let mut crossed = Vec::new();
            for &g in b.word() {
                let k = g.unsigned_abs() as usize - 1;
                crossed.push((at[k], at[k + 1], g.signum() as i64));
                at.swap(k, k + 1);
            }
            // The closure joins the strand ending at position j to strand j;
            // ring() numbers the ropes by their least strand.
            let mut root: Vec<usize> = (0..n).collect();
            for (j, &s) in at.iter().enumerate() {
                let (a, b) = (root[j], root[s]);
                for r in root.iter_mut() {
                    if *r == a.max(b) {
                        *r = a.min(b);
                    }
                }
            }
            let mut firsts: Vec<usize> = root.clone();
            firsts.sort_unstable();
            firsts.dedup();
            let rope = |s: usize| firsts.iter().position(|&f| f == root[s]).unwrap();
            let mut want: Vec<((usize, usize), i64)> = (0..firsts.len())
                .flat_map(|a| (a + 1..firsts.len()).map(move |b| ((a, b), 0)))
                .collect();
            for (s, t, sign) in crossed {
                let (a, b) = (rope(s), rope(t));
                if a != b {
                    let key = (a.min(b), a.max(b));
                    want.iter_mut().find(|(k, _)| *k == key).unwrap().1 += sign;
                }
            }
            for w in &mut want {
                assert_eq!(w.1 % 2, 0, "{b}");
                w.1 /= 2;
            }
            let d = RopeDiagram::new(&ring(&b), "height").unwrap();
            assert_eq!(d.linking_numbers(), want, "{b}");
        }
    }

    #[test]
    fn refusals_name_the_problem() {
        let err = |ropes: &[Rope], over: &str| RopeDiagram::new(ropes, over).unwrap_err();
        assert_eq!(err(&[], ""), DiagramError::Ropes(0));
        assert!(matches!(
            err(&[open(&[[0.0, 0.0]])], ""),
            DiagramError::Rope { rope: 0, .. }
        ));
        assert!(matches!(
            err(&[open(&[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])], ""),
            DiagramError::Rope { rope: 0, .. }
        ));
        let ropes = [circle(0.0, 0.0, 2.0, 8), circle(2.5, 0.0, 2.0, 8)];
        assert!(matches!(err(&ropes, "OU"), DiagramError::Over(m) if m.contains("2 letters")));
        assert!(matches!(err(&ropes, "OUXO"), DiagramError::Over(m) if m.contains("'X'")));
        assert!(matches!(err(&ropes, "alternating"), DiagramError::Over(_)));
        assert!(matches!(err(&ropes, "OOOO"), DiagramError::Over(m) if m.contains("both times")));
        assert!(matches!(err(&ropes, "height"), DiagramError::Over(m) if m.contains("height")));
        // Two ropes laid along the same line.
        let a = open(&[[0.0, 0.0], [4.0, 0.0]]);
        let b = open(&[[1.0, 0.0], [3.0, 0.0]]);
        assert!(matches!(err(&[a, b], ""), DiagramError::Degenerate { .. }));
    }

    #[test]
    fn geometry_lifts_the_front_passage_and_measures_clearance() {
        let ropes = [circle(0.0, 0.0, 2.0, 8), circle(2.5, 0.0, 2.0, 8)];
        let d = RopeDiagram::new(&ropes, "OUUO").unwrap();
        let g = d.geometry(0.2).unwrap();
        assert_eq!(g.ropes.len(), 2);
        assert_eq!(g.ropes[0].points.len(), 8 * SAMPLES_PER_SPAN);
        let top = g.ropes[0]
            .points
            .iter()
            .map(|p| p[2])
            .fold(f64::MIN, f64::max);
        assert!((top - 0.22).abs() < 0.01, "{top}");
        let c = g.min_clearance.unwrap();
        assert!(c > 1.0, "{c}");
        assert_eq!(d.geometry(0.0), Err(DiagramError::Radius(0.0)));
        // Thick enough rope and the two circles collide.
        assert!(d.geometry(1.5).unwrap().min_clearance.unwrap() < 1.0);
    }
}
