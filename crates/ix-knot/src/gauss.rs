//! Knots spelled by their crossings: the Gauss code of a rope diagram, and a
//! drawing found from the code alone.
//!
//! Walk along each rope and write down every crossing it passes, `O` where it
//! passes over and `U` where it passes under, numbered so that the two passages
//! of one crossing carry the same number: `U1 O2 U3 O1 U2 O3` is the overhand
//! knot. An open rope is written as it is, a closed one in parentheses, and
//! ropes are separated by `|`: `(O1 U2) | (U1 O2)` is a Hopf link.
//!
//! **Drawing.** A code says in which order the crossings come along each rope,
//! not where they are. [`draw`] tries every way of laying them in the plane so
//! that the ropes meet nowhere else (a planar embedding: at each crossing, the
//! side the second passage comes in from), with every loose end on the outside.
//! Each embedding is drawn with Tutte's barycentric method, every free point at
//! the average of its neighbours, and handed to [`RopeDiagram`], which finds
//! the crossings in the drawing afresh: a drawing that does not give back the
//! code it came from is refused.
//!
//! **Which knot.** The letters fix which passage is in front, but not the
//! handedness: a drawing and its mirror image have the same code. Nor, when the
//! code is a sum of two knots, the handedness of each part: the code of two
//! overhand knots in a row draws the granny, its mirror image, and the reef.
//! Given the knot the closure must be, [`draw`] keeps the drawing that closes
//! into it. Without, it picks one handedness, and refuses a code that draws
//! knots which are not all one knot and its mirror image.

use crate::diagram::{DiagramError, Rope, RopeDiagram, MAX_ROPES};
use crate::jones::Jones;
use std::f64::consts::PI;
use std::fmt;
use std::str::FromStr;

/// The most crossings a code may have; the embeddings tried grow as
/// 2^crossings.
pub const MAX_GAUSS_CROSSINGS: usize = 20;
/// The most sign and end-order combinations tried.
const MAX_TRIALS: u64 = 1 << 22;
/// The most distinct embeddings drawn.
const MAX_EMBEDDINGS: usize = 16;
/// The radius of the circle the loose ends are drawn out to.
const CIRCLE: f64 = 10.0;
/// Points on the circle between two loose ends. Each pulls the face that
/// reaches the circle there outward; with one, the knot is squashed flat.
const GAP: usize = 8;

/// One rope of a code: the crossings it passes, in order, each with whether it
/// passes over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussRope {
    pub closed: bool,
    pub passages: Vec<(u32, bool)>,
}

/// A Gauss code: see the module documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussCode {
    pub ropes: Vec<GaussRope>,
}

/// Why a code was refused or could not be drawn.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GaussError {
    #[error("{0}")]
    Syntax(String),
    #[error("a code has 1 to {MAX_ROPES} ropes, got {0}")]
    Ropes(usize),
    #[error("crossing {label} is passed {count} times; each crossing is passed twice, once O and once U")]
    Count { label: u32, count: usize },
    #[error("crossing {label} is passed {} both times", if *.over { "over" } else { "under" })]
    Sides { label: u32, over: bool },
    #[error("{0} crossings; a code may have at most {MAX_GAUSS_CROSSINGS}")]
    Crossings(usize),
    #[error(
        "the ropes fall into pieces that do not cross each other; give each piece its own code"
    )]
    Split,
    #[error("no planar drawing has these crossings in this order with the ends outside: the code is not a knot drawn on paper")]
    NotPlanar,
    #[error("the code has too many possible drawings to try")]
    TooMany,
    #[error("the code draws {} different knots ({}); name the one it closes into", .0.len(), list(.0))]
    Ambiguous(Vec<Jones>),
    #[error("the code draws {}, not the knot it should close into", list(.0))]
    NoMatch(Vec<Jones>),
    #[error("no drawing of the code could be read back: {0}")]
    Drawing(String),
}

/// Jones polynomials named by their text, for an error message.
fn list(knots: &[Jones]) -> String {
    let texts: Vec<String> = knots.iter().map(|j| format!("V = {j}")).collect();
    texts.join("; ")
}

impl GaussCode {
    /// The number of crossings.
    pub fn crossings(&self) -> usize {
        self.ropes.iter().map(|r| r.passages.len()).sum::<usize>() / 2
    }

    /// The same code with its crossings numbered 1, 2, … in the order they are
    /// first passed.
    pub fn canonical(&self) -> GaussCode {
        let mut names: Vec<(u32, u32)> = Vec::new();
        let ropes = self
            .ropes
            .iter()
            .map(|r| GaussRope {
                closed: r.closed,
                passages: r
                    .passages
                    .iter()
                    .map(|&(label, over)| {
                        let name = match names.iter().find(|(l, _)| *l == label) {
                            Some(&(_, n)) => n,
                            None => {
                                let n = names.len() as u32 + 1;
                                names.push((label, n));
                                n
                            }
                        };
                        (name, over)
                    })
                    .collect(),
            })
            .collect();
        GaussCode { ropes }
    }

    fn check(&self) -> Result<(), GaussError> {
        if self.ropes.is_empty() || self.ropes.len() > MAX_ROPES {
            return Err(GaussError::Ropes(self.ropes.len()));
        }
        let mut seen: Vec<(u32, usize, bool)> = Vec::new();
        for &(label, over) in self.ropes.iter().flat_map(|r| &r.passages) {
            match seen.iter_mut().find(|(l, _, _)| *l == label) {
                Some(entry) => {
                    entry.1 += 1;
                    if entry.1 == 2 && entry.2 == over {
                        return Err(GaussError::Sides { label, over });
                    }
                }
                None => seen.push((label, 1, over)),
            }
        }
        if let Some(&(label, count, _)) = seen.iter().find(|(_, c, _)| *c != 2) {
            return Err(GaussError::Count { label, count });
        }
        if seen.len() > MAX_GAUSS_CROSSINGS {
            return Err(GaussError::Crossings(seen.len()));
        }
        Ok(())
    }
}

impl FromStr for GaussCode {
    type Err = GaussError;

    fn from_str(text: &str) -> Result<Self, GaussError> {
        let mut ropes = Vec::new();
        for part in text.split('|') {
            let part = part.trim();
            let (closed, body) = match part.strip_prefix('(') {
                Some(rest) => match rest.strip_suffix(')') {
                    Some(body) => (true, body),
                    None => {
                        return Err(GaussError::Syntax(format!(
                            "`{part}` opens a parenthesis and does not close it"
                        )))
                    }
                },
                None => (false, part),
            };
            let mut passages = Vec::new();
            for token in body
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|t| !t.is_empty())
            {
                let bad = || {
                    GaussError::Syntax(format!(
                        "`{token}` is not a passage: write O or U and the crossing's number, as O3"
                    ))
                };
                let mut chars = token.chars();
                let over = match chars.next() {
                    Some('O' | 'o') => true,
                    Some('U' | 'u') => false,
                    _ => return Err(bad()),
                };
                let label: u32 = chars.as_str().parse().map_err(|_| bad())?;
                passages.push((label, over));
            }
            ropes.push(GaussRope { closed, passages });
        }
        let code = GaussCode { ropes };
        code.check()?;
        Ok(code)
    }
}

impl fmt::Display for GaussCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, rope) in self.ropes.iter().enumerate() {
            if i > 0 {
                f.write_str(" | ")?;
            }
            let body: Vec<String> = rope
                .passages
                .iter()
                .map(|&(label, over)| format!("{}{label}", if over { 'O' } else { 'U' }))
                .collect();
            if rope.closed {
                write!(f, "({})", body.join(" "))?;
            } else {
                f.write_str(&body.join(" "))?;
            }
        }
        Ok(())
    }
}

/// A code drawn: the ropes to give [`RopeDiagram::new`], the letters that go
/// with them, the diagram they make, and a rope radius at which the rope
/// clears itself.
#[derive(Debug, Clone)]
pub struct Drawn {
    pub ropes: Vec<Rope>,
    pub over: String,
    pub diagram: RopeDiagram,
    pub radius: f64,
    /// At `radius`, in rope diameters.
    pub min_clearance: Option<f64>,
}

/// Draw `code`. With `closure`, the Jones polynomial of the knot it must close
/// into, keep the drawing that does. Without, pick one handedness, and refuse
/// a code whose drawings are not all one knot and its mirror image.
pub fn draw(code: &GaussCode, closure: Option<&Jones>) -> Result<Drawn, GaussError> {
    code.check()?;
    let code = code.canonical();
    let mut drawings = Vec::new();
    let mut failure = None;
    for signs in embeddings(&code)? {
        for mirror in [false, true] {
            match drawing(&code, &signs, mirror) {
                Ok(d) => drawings.push(d),
                Err(e) => failure = Some(e),
            }
        }
    }
    if drawings.is_empty() {
        return Err(GaussError::Drawing(
            failure.unwrap_or_else(|| "nothing was drawn".into()),
        ));
    }
    let mut knots: Vec<Jones> = Vec::new();
    for d in &drawings {
        if !knots.contains(d.diagram.jones()) {
            knots.push(d.diagram.jones().clone());
        }
    }
    let chosen = match closure {
        Some(want) => drawings
            .into_iter()
            .find(|d| d.diagram.jones() == want)
            .ok_or(GaussError::NoMatch(knots))?,
        None => {
            let (first, mirror) = (knots[0].clone(), knots[0].mirror());
            if knots.iter().any(|k| *k != first && *k != mirror) {
                return Err(GaussError::Ambiguous(knots));
            }
            drawings.swap_remove(0)
        }
    };
    fit(chosen)
}

/// Where each crossing's second passage comes in from, one choice per
/// crossing, plus the order of the loose ends around the outside.
#[derive(Debug, Clone)]
struct Signs {
    signs: Vec<bool>,
    hub: Vec<usize>,
}

/// The ropes as a graph: crossings are vertices `0..n`; the hub, vertex `n`,
/// stands for everything outside the drawing and every open rope starts and
/// ends there. Each edge runs along a rope from one vertex to the next.
struct Graph {
    n: usize,
    hub: bool,
    tail: Vec<usize>,
    head: Vec<usize>,
    /// Each rope's edges in order; a closed rope's first edge is the one that
    /// leads into its first passage.
    rope_edges: Vec<Vec<usize>>,
    /// Each crossing's two passages in reading order, as (edge in, edge out).
    passes: Vec<Vec<(usize, usize)>>,
    /// The darts leaving the hub, in a fixed order.
    hub_darts: Vec<usize>,
}

impl Graph {
    /// `cut`: a closed rope and the index of one of its edges, which then runs
    /// through the hub, so a drawing of only closed ropes has an outside too.
    fn new(code: &GaussCode, cut: Option<(usize, usize)>) -> Result<Graph, GaussError> {
        let n = code.crossings();
        let hub = cut.is_some() || code.ropes.iter().any(|r| !r.closed);
        let h = n;
        let mut g = Graph {
            n,
            hub,
            tail: Vec::new(),
            head: Vec::new(),
            rope_edges: Vec::new(),
            passes: vec![Vec::new(); n],
            hub_darts: Vec::new(),
        };
        for (r, rope) in code.ropes.iter().enumerate() {
            let at: Vec<usize> = rope.passages.iter().map(|p| p.0 as usize - 1).collect();
            if rope.closed && at.is_empty() {
                if code.ropes.len() > 1 {
                    return Err(GaussError::Split);
                }
                g.rope_edges.push(Vec::new());
                continue;
            }
            // The vertices along the rope, and where each passage sits among them.
            let mut seq: Vec<usize> = Vec::new();
            let mut slot = Vec::new();
            if !rope.closed {
                seq.push(h);
            }
            for (k, &c) in at.iter().enumerate() {
                slot.push(seq.len());
                seq.push(c);
                if cut == Some((r, k)) {
                    seq.push(h);
                }
            }
            if !rope.closed {
                seq.push(h);
            }
            let count = if rope.closed {
                seq.len()
            } else {
                seq.len() - 1
            };
            let first = g.tail.len();
            for k in 0..count {
                g.tail.push(seq[k]);
                g.head.push(seq[(k + 1) % seq.len()]);
            }
            let edge = |k: usize| first + (k + count) % count;
            for (i, &s) in slot.iter().enumerate() {
                g.passes[at[i]].push((edge(s.wrapping_add(count) - 1), edge(s)));
            }
            let mut edges: Vec<usize> = (0..count).map(edge).collect();
            if rope.closed {
                // Start from the edge leaving the last passage.
                edges.rotate_left(slot[slot.len() - 1]);
            }
            if !rope.closed {
                g.hub_darts.push(2 * edges[0]);
                g.hub_darts.push(2 * edges[edges.len() - 1] + 1);
            } else if let Some((cr, _)) = cut {
                if cr == r {
                    for &e in &edges {
                        if g.tail[e] == h {
                            g.hub_darts.push(2 * e);
                        }
                        if g.head[e] == h {
                            g.hub_darts.push(2 * e + 1);
                        }
                    }
                }
            }
            g.rope_edges.push(edges);
        }
        Ok(g)
    }

    fn vertices(&self) -> usize {
        self.n + usize::from(self.hub)
    }

    /// The vertex a dart leaves.
    fn from(&self, d: usize) -> usize {
        if d % 2 == 0 {
            self.tail[d / 2]
        } else {
            self.head[d / 2]
        }
    }

    /// The darts leaving each vertex counterclockwise, and each dart's place
    /// there.
    fn rotation(&self, s: &Signs) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut rot = vec![Vec::new(); self.vertices()];
        for (c, p) in self.passes.iter().enumerate() {
            let [(a_in, a_out), (b_in, b_out)] = [p[0], p[1]];
            let (ao, ai, bo, bi) = (2 * a_out, 2 * a_in + 1, 2 * b_out, 2 * b_in + 1);
            rot[c] = if s.signs[c] {
                vec![ao, bo, ai, bi]
            } else {
                vec![ao, bi, ai, bo]
            };
        }
        if self.hub {
            rot[self.n] = s.hub.iter().map(|&k| self.hub_darts[k]).collect();
        }
        let mut pos = vec![0; 2 * self.tail.len()];
        for list in &rot {
            for (i, &d) in list.iter().enumerate() {
                pos[d] = i;
            }
        }
        (rot, pos)
    }

    /// The faces of the embedding, each the darts around it in order.
    fn faces(&self, rot: &[Vec<usize>], pos: &[usize]) -> Vec<Vec<usize>> {
        let darts = 2 * self.tail.len();
        let mut seen = vec![false; darts];
        let mut faces = Vec::new();
        for start in 0..darts {
            if seen[start] {
                continue;
            }
            let mut face = Vec::new();
            let mut d = start;
            while !seen[d] {
                seen[d] = true;
                face.push(d);
                let back = d ^ 1;
                let at = &rot[self.from(back)];
                d = at[(pos[back] + 1) % at.len()];
            }
            faces.push(face);
        }
        faces
    }

    fn connected(&self) -> bool {
        let mut parent: Vec<usize> = (0..self.vertices()).collect();
        fn root(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for (&a, &b) in self.tail.iter().zip(&self.head) {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            parent[ra] = rb;
        }
        let r0 = root(&mut parent, 0);
        (0..self.vertices()).all(|v| root(&mut parent, v) == r0)
    }
}

/// Every planar embedding of the code with the first crossing's sign fixed;
/// its mirror image is drawn from each as well.
fn embeddings(code: &GaussCode) -> Result<Vec<Signs>, GaussError> {
    let g = Graph::new(code, None)?;
    if g.vertices() == 0 {
        // One closed rope with no crossings: a circle.
        return Ok(vec![Signs {
            signs: Vec::new(),
            hub: Vec::new(),
        }]);
    }
    if !g.connected() {
        return Err(GaussError::Split);
    }
    let orders = hub_orders(g.hub_darts.len());
    let free = g.n.saturating_sub(1);
    let trials = (1u64 << free).saturating_mul(orders.len() as u64);
    if trials > MAX_TRIALS {
        return Err(GaussError::TooMany);
    }
    let edges = g.tail.len();
    let mut found = Vec::new();
    for hub in &orders {
        for mask in 0..(1u64 << free) {
            let signs: Vec<bool> = (0..g.n)
                .map(|c| c == 0 || mask >> (c - 1) & 1 == 1)
                .collect();
            let s = Signs {
                signs,
                hub: hub.clone(),
            };
            let (rot, pos) = g.rotation(&s);
            if g.vertices() + g.faces(&rot, &pos).len() == edges + 2 {
                found.push(s);
                if found.len() > MAX_EMBEDDINGS {
                    return Err(GaussError::TooMany);
                }
            }
        }
    }
    if found.is_empty() {
        return Err(GaussError::NotPlanar);
    }
    Ok(found)
}

/// The cyclic orders of `m` darts around the hub: the first fixed, the rest in
/// every order.
fn hub_orders(m: usize) -> Vec<Vec<usize>> {
    if m == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    let mut rest: Vec<usize> = (1..m).collect();
    permute(&mut rest, 0, &mut out);
    out.into_iter()
        .map(|p| std::iter::once(0).chain(p).collect())
        .collect()
}

fn permute(items: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if k >= items.len() {
        out.push(items.clone());
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, out);
        items.swap(k, i);
    }
}

/// Draw one embedding (or its mirror image) and read it back.
fn drawing(code: &GaussCode, s: &Signs, mirror: bool) -> Result<Drawn, String> {
    let mut g = Graph::new(code, None).map_err(|e| e.to_string())?;
    let mut s = s.clone();
    if g.vertices() == 0 {
        return finish(code, circle(), mirror);
    }
    if !g.hub {
        // Only closed ropes: open the largest face to the outside by running
        // one of its edges through the hub.
        let (rot, pos) = g.rotation(&s);
        let faces = g.faces(&rot, &pos);
        let outer = faces.iter().max_by_key(|f| f.len()).unwrap();
        let e = outer[0] / 2;
        let (rope, k) = g
            .rope_edges
            .iter()
            .enumerate()
            .find_map(|(r, list)| list.iter().position(|&x| x == e).map(|_| (r, e)))
            .unwrap();
        let first = g.rope_edges[..rope].iter().map(Vec::len).sum::<usize>();
        g = Graph::new(code, Some((rope, k - first))).map_err(|e| e.to_string())?;
        s.hub = vec![0, 1];
    }
    let (rot, pos) = g.rotation(&s);
    let faces = g.faces(&rot, &pos);
    if g.vertices() + faces.len() != g.tail.len() + 2 {
        return Err("opening a face to the outside broke the embedding".into());
    }
    let ropes = tutte(&g, &rot, &pos, &faces, code)?;
    finish(code, ropes, mirror)
}

fn circle() -> Vec<Rope> {
    vec![Rope {
        points: (0..8)
            .map(|k| {
                let a = 2.0 * PI * k as f64 / 8.0;
                [CIRCLE * a.cos(), CIRCLE * a.sin(), 0.0]
            })
            .collect(),
        closed: true,
    }]
}

/// Build the diagram, mirrored if asked, and check it gives back the code.
fn finish(code: &GaussCode, mut ropes: Vec<Rope>, mirror: bool) -> Result<Drawn, String> {
    if mirror {
        for p in ropes.iter_mut().flat_map(|r| r.points.iter_mut()) {
            p[0] = -p[0];
        }
    }
    let over: Vec<&str> = code
        .ropes
        .iter()
        .flat_map(|r| &r.passages)
        .map(|&(_, o)| if o { "O" } else { "U" })
        .collect();
    let over = over.join(" ");
    let diagram = RopeDiagram::new(&ropes, &over).map_err(|e: DiagramError| e.to_string())?;
    let back = diagram.gauss_code();
    if back != *code {
        return Err(format!("the drawing reads back as {back}, not {code}"));
    }
    Ok(Drawn {
        ropes,
        over,
        diagram,
        radius: 0.0,
        min_clearance: None,
    })
}

/// A point of the drawing: solved for, or fixed on the outer circle.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Node {
    Free(usize),
    Fixed(usize),
}

/// Tutte's drawing. The hub is blown up into a circle: each dart leaving it
/// gets a point on the circle, in the hub's order, with `GAP` points between
/// each two. Every edge is split by two points, and every face gets a point at
/// its middle joined to everything on its boundary, so the free points are
/// pinned from all sides; each sits at the average of its neighbours. The
/// control points are the edge points and the circle points: the curve through
/// them passes each crossing without a control point on it.
///
/// A loop (an edge from a crossing back to itself) comes out collapsed, its
/// two points on one spot, since nothing in the equations tells its ends
/// apart; [`open_loops`] spreads them afterwards.
fn tutte(
    g: &Graph,
    rot: &[Vec<usize>],
    pos: &[usize],
    faces: &[Vec<usize>],
    code: &GaussCode,
) -> Result<Vec<Rope>, String> {
    let hub = &rot[g.n];
    let m = hub.len();
    let per = GAP + 1;
    let mut fixed: Vec<[f64; 2]> = (0..per * m)
        .map(|k| {
            // Each point turned a little, irregularly: no symmetry of the circle
            // maps the points onto themselves, so a symmetric code does not get
            // a symmetric drawing, where two passages can meet on a curve point.
            let wobble = 0.2 * ((k as f64 * 0.618_033_988_7).fract() - 0.5);
            let a = 2.0 * PI * (k as f64 + wobble) / (per * m) as f64;
            [CIRCLE * a.cos(), CIRCLE * a.sin()]
        })
        .collect();
    // On the circle: dart j's point is per * j, the gap after it the next GAP.
    let spoke = |d: usize| Node::Fixed(per * pos[d]);
    let edges = g.tail.len();
    let sub = |e: usize, k: usize| Node::Free(g.n + 2 * e + k);
    let centre = |f: usize| Node::Free(g.n + 2 * edges + f);
    let count = g.n + 2 * edges + faces.len();
    let mut near: Vec<Vec<Node>> = vec![Vec::new(); count];
    let link = |a: Node, b: Node, near: &mut Vec<Vec<Node>>| {
        if let Node::Free(i) = a {
            near[i].push(b);
        }
        if let Node::Free(j) = b {
            near[j].push(a);
        }
    };
    let end = |e: usize, start: bool| {
        let (v, d) = if start {
            (g.tail[e], 2 * e)
        } else {
            (g.head[e], 2 * e + 1)
        };
        if g.hub && v == g.n {
            spoke(d)
        } else {
            Node::Free(v)
        }
    };
    for e in 0..edges {
        link(end(e, true), sub(e, 0), &mut near);
        link(sub(e, 0), sub(e, 1), &mut near);
        link(sub(e, 1), end(e, false), &mut near);
    }
    for (f, face) in faces.iter().enumerate() {
        let mut around: Vec<Node> = Vec::new();
        for &d in face {
            let e = d / 2;
            let v = g.from(d);
            if g.hub && v == g.n {
                let j = pos[d];
                let before = (j + m - 1) % m;
                around.extend((0..per).map(|i| Node::Fixed(per * before + i)));
                around.push(Node::Fixed(per * j));
            } else {
                around.push(Node::Free(v));
            }
            around.extend([sub(e, 0), sub(e, 1)]);
        }
        let mut unique: Vec<Node> = Vec::new();
        for a in around {
            if !unique.contains(&a) {
                unique.push(a);
            }
        }
        for a in unique {
            link(centre(f), a, &mut near);
        }
    }
    let free = |node: Node| match node {
        Node::Free(i) => i,
        Node::Fixed(_) => unreachable!("edge points are free"),
    };
    // A face that reaches the circle once is pulled out along its arc: the
    // points of its boundary, in order from the dart leaving the hub to the one
    // coming back, are each tied to a point of the arc between the two spokes,
    // in the same order. Without, the knot is pulled flat between the faces'
    // middles.
    let turn = |p: [f64; 2]| p[1].atan2(p[0]);
    for face in faces {
        let leaving: Vec<usize> = (0..face.len())
            .filter(|&i| g.hub && g.from(face[i]) == g.n)
            .collect();
        let [start] = leaving[..] else { continue };
        let mut chain: Vec<usize> = Vec::new();
        for k in 0..face.len() {
            let d = face[(start + k) % face.len()];
            if k > 0 {
                chain.push(g.from(d));
            }
            let (e, forward) = (d / 2, d % 2 == 0);
            let (a, b) = if forward { (0, 1) } else { (1, 0) };
            chain.extend([free(sub(e, a)), free(sub(e, b))]);
        }
        let j = pos[face[start]];
        let before = (j + m - 1) % m;
        let to = turn(fixed[per * j]);
        let span = (to - turn(fixed[per * before])).rem_euclid(2.0 * PI);
        let steps = chain.len() as f64 + 1.0;
        for (k, &node) in chain.iter().enumerate() {
            let a = to - span * (k as f64 + 1.0) / steps;
            fixed.push([CIRCLE * a.cos(), CIRCLE * a.sin()]);
            link(Node::Free(node), Node::Fixed(fixed.len() - 1), &mut near);
        }
    }
    let mut at = solve(&near, &fixed)?;
    open_loops(g, rot, pos, &mut at, |e, k| free(sub(e, k)));
    let point = |node: Node| -> [f64; 3] {
        let p = match node {
            Node::Free(i) => at[i],
            Node::Fixed(k) => fixed[k],
        };
        [p[0], p[1], 0.0]
    };

    let mut ropes = Vec::with_capacity(code.ropes.len());
    for (r, rope) in code.ropes.iter().enumerate() {
        let list = &g.rope_edges[r];
        let mut points = Vec::new();
        // An open rope starts and ends on the circle. A closed rope never starts
        // or ends at the hub (its edges begin after its last passage), so where
        // it reaches the hub it runs along the circle to where it leaves.
        for (i, &e) in list.iter().enumerate() {
            if !rope.closed && g.tail[e] == g.n {
                points.push(point(spoke(2 * e)));
            }
            points.push(point(sub(e, 0)));
            points.push(point(sub(e, 1)));
            if g.hub && g.head[e] == g.n {
                if rope.closed {
                    let out = list[i + 1];
                    points.extend(through_hub(e, out, pos, m).map(|k| point(Node::Fixed(k))));
                } else {
                    points.push(point(spoke(2 * e + 1)));
                }
            }
        }
        ropes.push(Rope {
            points,
            closed: rope.closed,
        });
    }
    Ok(ropes)
}

/// The circle points a closed rope passes from edge `inn` (ending at the hub)
/// to edge `out` (leaving it): its two spokes and the gap between them.
fn through_hub(inn: usize, out: usize, pos: &[usize], m: usize) -> impl Iterator<Item = usize> {
    let per = GAP + 1;
    let (a, b) = (pos[2 * inn + 1], pos[2 * out]);
    let gap: Vec<usize> = if (a + 1) % m == b {
        (1..per).map(|i| per * a + i).collect()
    } else {
        (1..per).rev().map(|i| per * b + i).collect()
    };
    std::iter::once(per * a)
        .chain(gap)
        .chain(std::iter::once(per * b))
}

/// Spread each collapsed loop. A loop's two darts sit next to each other in
/// its crossing's rotation; they are laid at a third and two thirds of the
/// angle between the crossing's other two darts, on the side where Tutte's
/// drawing put the collapsed loop, at the distance it gave the loop.
fn open_loops(
    g: &Graph,
    rot: &[Vec<usize>],
    pos: &[usize],
    at: &mut [[f64; 2]],
    sub: impl Fn(usize, usize) -> usize,
) {
    // The point next to a crossing along a dart leaving it.
    let beside = |d: usize| {
        if d % 2 == 0 {
            sub(d / 2, 0)
        } else {
            sub(d / 2, 1)
        }
    };
    let angle = |at: &[[f64; 2]], c: usize, d: usize| {
        let (p, q) = (at[c], at[beside(d)]);
        (q[1] - p[1]).atan2(q[0] - p[0])
    };
    for e in 0..g.tail.len() {
        let c = g.tail[e];
        if c != g.head[e] || (g.hub && c == g.n) {
            continue;
        }
        let list = &rot[c];
        // The loop's darts at k and k + 1 (cyclically), the others after them.
        let (o, i) = (pos[2 * e], pos[2 * e + 1]);
        let k = if (o + 1) % 4 == i { o } else { i };
        let (next, last) = (list[(k + 2) % 4], list[(k + 3) % 4]);
        if [next, last].iter().any(|&d| g.tail[d / 2] == g.head[d / 2]) {
            // Two loops on one crossing: nothing fixed to measure from.
            continue;
        }
        let (from, to) = (angle(at, c, last), angle(at, c, next));
        let p = at[c];
        let q = at[sub(e, 0)];
        let reach = (q[0] - p[0]).hypot(q[1] - p[1]).max(1e-3);
        // Counterclockwise from `last` to `next`, unless the loop lies the
        // other way round; going from `last`, the dart at k comes first.
        let ccw = (to - from).rem_euclid(2.0 * PI);
        let toward = ((q[1] - p[1]).atan2(q[0] - p[0]) - from).rem_euclid(2.0 * PI);
        let sweep = if toward < ccw { ccw } else { ccw - 2.0 * PI };
        for (step, d) in [(1.0, list[k]), (2.0, list[(k + 1) % 4])] {
            let a = from + sweep * step / 3.0;
            at[beside(d)] = [p[0] + reach * a.cos(), p[1] + reach * a.sin()];
        }
    }
}

/// Each free point at the average of its neighbours: a linear system, solved
/// by Gaussian elimination with partial pivoting.
fn solve(near: &[Vec<Node>], fixed: &[[f64; 2]]) -> Result<Vec<[f64; 2]>, String> {
    let n = near.len();
    let mut a = vec![vec![0.0; n + 2]; n];
    for (i, list) in near.iter().enumerate() {
        if list.is_empty() {
            return Err(format!("point {i} has no neighbours"));
        }
        a[i][i] = list.len() as f64;
        for node in list {
            match *node {
                Node::Free(j) => a[i][j] -= 1.0,
                Node::Fixed(k) => {
                    a[i][n] += fixed[k][0];
                    a[i][n + 1] += fixed[k][1];
                }
            }
        }
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
            .unwrap();
        if a[pivot][col].abs() < 1e-12 {
            return Err("the drawing's equations are singular".into());
        }
        a.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            if f != 0.0 {
                let (above, below) = a.split_at_mut(row);
                for (x, p) in below[0][col..].iter_mut().zip(&above[col][col..]) {
                    *x -= f * p;
                }
            }
        }
    }
    let mut x = vec![[0.0; 2]; n];
    for row in (0..n).rev() {
        for (c, out) in [n, n + 1].into_iter().enumerate() {
            let mut v = a[row][out];
            for k in row + 1..n {
                v -= a[row][k] * x[k][c];
            }
            x[row][c] = v / a[row][row];
        }
    }
    Ok(x)
}

/// The rope radius: grown or shrunk until the rope clears itself by about a
/// tenth of its diameter, and never so thick it does not.
fn fit(mut d: Drawn) -> Result<Drawn, GaussError> {
    let err = |e: DiagramError| GaussError::Drawing(e.to_string());
    let target = 1.15;
    let mut radius = 0.5;
    let mut clearance = d.diagram.geometry(radius).map_err(err)?.min_clearance;
    for _ in 0..8 {
        let Some(c) = clearance else { break };
        if (c - target).abs() < 0.03 {
            break;
        }
        radius = (radius * c / target).min(CIRCLE / 8.0);
        clearance = d.diagram.geometry(radius).map_err(err)?.min_clearance;
    }
    for _ in 0..40 {
        match clearance {
            Some(c) if c < 1.0 => {
                radius *= 0.9;
                clearance = d.diagram.geometry(radius).map_err(err)?.min_clearance;
            }
            _ => break,
        }
    }
    d.radius = radius;
    d.min_clearance = clearance;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{catalog, find};
    use crate::testing::{ring, words};
    use crate::{jones, Braid};

    fn code(text: &str) -> GaussCode {
        text.parse().unwrap()
    }

    #[test]
    fn reads_prints_and_numbers_in_order() {
        let c = code("U7 O2 u9, O7 U2 O9");
        assert_eq!(c.crossings(), 3);
        assert_eq!(c.canonical().to_string(), "U1 O2 U3 O1 U2 O3");
        let hopf = code("(O1 U2) | (U1 O2)");
        assert_eq!(hopf.to_string(), "(O1 U2) | (U1 O2)");
        assert!(hopf.ropes.iter().all(|r| r.closed));
    }

    #[test]
    fn refusals_name_the_problem() {
        let err = |t: &str| t.parse::<GaussCode>().unwrap_err();
        assert!(matches!(err("O1 X2"), GaussError::Syntax(m) if m.contains("X2")));
        assert!(matches!(err("(O1 U1"), GaussError::Syntax(_)));
        assert_eq!(err("O1 U1 O2"), GaussError::Count { label: 2, count: 1 });
        assert_eq!(
            err("O1 O1"),
            GaussError::Sides {
                label: 1,
                over: true
            }
        );
        let many: String = (1..=21)
            .map(|k| format!("O{k} "))
            .chain((1..=21).map(|k| format!("U{k} ")))
            .collect();
        assert_eq!(err(&many), GaussError::Crossings(21));
        // Between the two passages of a crossing a plane curve passes an even
        // number of others; in 1 2 1 2 each has one.
        assert_eq!(
            draw(&code("O1 O2 U1 U2"), None).unwrap_err(),
            GaussError::NotPlanar
        );
        assert_eq!(
            draw(&code("(O1 U2 O3 U1 O2 U3) | ()"), None).unwrap_err(),
            GaussError::Split
        );
    }

    #[test]
    fn unknots_draw_too() {
        for text in ["", "()", "O1 U1", "(U1 O1)"] {
            let d = draw(&code(text), None).unwrap();
            assert_eq!(d.diagram.jones().to_string(), "1", "{text:?}");
            assert_eq!(d.diagram.gauss_code(), code(text).canonical(), "{text:?}");
        }
    }

    /// Every braid closure drawn as a ring: its code, drawn from the code
    /// alone, reads back as the same code and closes into the same knot.
    #[test]
    fn a_drawing_read_as_a_code_draws_the_same_knot_again() {
        let (mut drawn, mut split) = (0, 0);
        for b in words(80) {
            let d = RopeDiagram::new(&ring(&b), "height").unwrap();
            let c = d.gauss_code();
            // A braid closure falls apart exactly when some generator is missing.
            let whole = (1..b.strands() as i32).all(|k| b.word().iter().any(|g| g.abs() == k));
            match draw(&c, Some(d.jones())) {
                Ok(again) => {
                    assert!(whole, "{b}: {c}: drawn, though it falls apart");
                    assert_eq!(again.diagram.gauss_code(), c, "{b}");
                    assert_eq!(again.diagram.components(), d.components(), "{b}");
                    let mirror = d.jones().mirror();
                    match draw(&c, None) {
                        Ok(x) => assert!(
                            *x.diagram.jones() == *d.jones() || *x.diagram.jones() == mirror,
                            "{b}"
                        ),
                        Err(GaussError::Ambiguous(knots)) => assert!(knots.contains(d.jones())),
                        Err(e) => panic!("{b}: {c}: {e}"),
                    }
                    drawn += 1;
                }
                Err(GaussError::Split) => {
                    assert!(!whole, "{b}: {c}: refused as split");
                    split += 1;
                }
                Err(e) => panic!("{b}: {c}: {e}"),
            }
        }
        assert_eq!(drawn + split, 80);
        assert!(drawn >= 40, "only {drawn} of 80 are whole");
    }

    #[test]
    fn the_catalogue_knots_draw_from_their_codes_alone() {
        for e in catalog() {
            let d = e.diagram().unwrap();
            let c = d.gauss_code();
            let again = draw(&c, Some(d.jones())).unwrap();
            assert_eq!(again.diagram.gauss_code(), c, "{}", e.id);
            let clear = again.min_clearance.unwrap();
            assert!(clear >= 1.0, "{}: {clear:.3}", e.id);
        }
    }

    #[test]
    fn two_overhands_in_a_row_are_a_granny_or_a_reef() {
        let one = find("overhand").unwrap().diagram().unwrap().gauss_code();
        let n = one.crossings() as u32;
        let mut passages = one.ropes[0].passages.clone();
        passages.extend(one.ropes[0].passages.iter().map(|&(l, o)| (l + n, o)));
        let two = GaussCode {
            ropes: vec![GaussRope {
                closed: false,
                passages,
            }],
        };
        let granny = jones(&Braid::parse(None, "s1^3 s2^3").unwrap());
        let reef = jones(&Braid::parse(None, "s1^3 s2^-3").unwrap());
        match draw(&two, None).unwrap_err() {
            GaussError::Ambiguous(knots) => {
                assert!(knots.contains(&reef), "{knots:?}");
                assert!(knots.contains(&granny) || knots.contains(&granny.mirror()));
            }
            e => panic!("{e}"),
        }
        for want in [&granny, &reef] {
            assert_eq!(draw(&two, Some(want)).unwrap().diagram.jones(), want);
        }
    }

    #[test]
    fn a_wrong_closure_is_refused() {
        let trefoil = jones(&Braid::parse(None, "s1^3").unwrap());
        // The overhand with its middle crossing changed comes undone.
        let undone = code("U1 U2 U3 O1 O2 O3");
        match draw(&undone, Some(&trefoil)).unwrap_err() {
            GaussError::NoMatch(knots) => assert_eq!(knots[0].to_string(), "1"),
            e => panic!("{e}"),
        }
        assert_eq!(
            draw(&undone, None).unwrap().diagram.jones().to_string(),
            "1"
        );
    }
}
