//! The `.origami` language: a folding sequence written down so that IX can check it.
//!
//! A `.origami` file names a model, folds a square sheet step by step, as a diagram does, and
//! states what the result must be. IX replays the steps into a stated folded state (a FOLD
//! crease pattern, one folded frame and its `faceOrders`), runs every check of this crate on
//! it, and answers each `expect` line with what it found. Whoever writes the file, a person or a
//! model drafting under [`GRAMMAR`], proposes; IX decides.
//!
//! Every fold is a simple fold: it carries every layer on its moving side over the fold line, in
//! front for a valley and behind for a mountain. Each step therefore has exactly one result, and
//! the replay derives the layer order the steps dictate instead of searching for one. The replay
//! writes FOLD JSON and reads it back with [`Fold::from_value`]; it shares no code with the
//! checks, and its result is checked like any other fold.
//!
//! ```
//! use ix_origami::sequence::Sequence;
//!
//! let text = "
//! origami triangle
//! valley A to C      # the corner A onto the corner C, along the diagonal BD
//! expect ok
//! expect faces 2
//! expect ply 2
//! ";
//! let sequence: Sequence = text.parse().unwrap();
//! let checked = sequence.check().unwrap();
//! assert!(checked.holds());
//! ```

use std::collections::BTreeMap;
use std::str::FromStr;

use serde_json::{json, Value};

use crate::fold::{Assignment, Fold, Point};
use crate::layers::{analyse_within, Report, Unchecked};
use crate::limits::Limits;

/// The most bytes a `.origami` file may have.
pub const MAX_ORIGAMI_FILE: usize = 64 * 1024;
/// The most faces a replay may make. A simple fold can double the faces, so the steps alone
/// do not bound them.
pub const MAX_FACES: usize = 1_000;
/// The largest coordinate a step may write.
pub const MAX_COORDINATE: f64 = 1e6;
/// Two positions closer than this are one place; a point closer than this to a line is on it.
const EPS: f64 = 1e-9;

/// The grammar of a `.origami` file, in EBNF. One statement per line; blank lines and comments
/// anywhere. A comment starts at a `#` that begins the line or follows a space.
pub const GRAMMAR: &str = r##"file        = { line } ;
line        = [ statement ] , [ comment ] , newline ;
comment     = "#" , { any character } ;              (* at the line's start or after a space *)
statement   = "origami" , name                         (* first, once *)
            | "sheet" , "square"                       (* optional, before the first fold: the unit square *)
            | fold
            | "expect" , expectation ;
fold        = ( "valley" | "mountain" ) ,              (* the moving side goes in front, or behind *)
              ( point , "to" , point                   (* Huzita-Justin O2: the first point onto the second; its side moves *)
              | line , "to" , line                     (* O3: the first line onto the second, direction onto direction; its side moves *)
              | "through" , point , point , "moving" , point ) ;   (* O1: the side of the last point moves *)
point       = corner                                   (* A (0,0), B (1,0), C (1,1), D (0,1): a point of the paper *)
            | "m" , corner , corner                    (* the midpoint of a side: mAB, mBC, mCD, mDA, either order *)
            | "O"                                      (* the centre *)
            | "(" , number , "," , number , ")" ;      (* a position in the folded state at that step *)
line        = corner , corner                          (* two different corners: AB, AC *)
            | point , "-" , point ;                    (* no space around "-" *)
corner      = "A" | "B" | "C" | "D" ;
number      = [ "-" ] , digits , [ "." , digits ]
            | [ "-" ] , digits , "/" , digits ;        (* a fraction: 1/3 *)
expectation = "ok"                                     (* every check of ix-origami passes *)
            | "faces" , integer
            | "ply" , integer ;                        (* the most faces stacked at one point *)
name        = letter , { letter | digit | "-" } ;
"##;

/// Why a file was refused, and on which line (1-based; 0 for the file as a whole).
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceError {
    pub line: usize,
    pub reason: String,
}

impl std::fmt::Display for SequenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            0 => write!(f, "{}", self.reason),
            line => write!(f, "line {line}: {}", self.reason),
        }
    }
}

impl std::error::Error for SequenceError {}

fn fail<T>(line: usize, reason: impl Into<String>) -> Result<T, SequenceError> {
    Err(SequenceError {
        line,
        reason: reason.into(),
    })
}

/// What a point names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Landmark {
    /// A point of the paper, at these crease-pattern coordinates: a corner, a side's midpoint or
    /// the centre. It is used where it lies at the step that names it.
    Paper(Point),
    /// A position in the folded state at that step: the coordinates fallback.
    At(Point),
}

/// A point as written, and what it names.
#[derive(Debug, Clone, PartialEq)]
pub struct Named {
    pub text: String,
    pub landmark: Landmark,
}

/// A line as written, and its two points in order: its direction runs from the first to the
/// second.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub text: String,
    pub points: [Named; 2],
}

impl Line {
    /// The same line written the other way round: `DC` for `CD`, `Q-P` for `P-Q`.
    pub fn reversed(&self) -> String {
        let [p, q] = &self.points;
        if self.text.contains('-') {
            format!("{}-{}", q.text, p.text)
        } else {
            format!("{}{}", q.text, p.text)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Valley,
    Mountain,
}

/// How a step builds its fold line, and which side moves.
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// O2: the first point onto the second; the first point's side moves.
    PointOnto(Named, Named),
    /// O3: the first line onto the second, direction onto direction; the first line's side
    /// moves.
    LineOnto(Line, Line),
    /// O1: the line through two points; the third point's side moves.
    Through(Named, Named, Named),
}

/// One fold.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub line: usize,
    /// The statement as written, comment removed.
    pub text: String,
    pub kind: Kind,
    pub target: Target,
}

/// What the folded state must be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    /// Every check passes.
    Ok,
    Faces(usize),
    /// The most faces stacked at one point.
    Ply(usize),
}

/// One `expect` line.
#[derive(Debug, Clone, PartialEq)]
pub struct Expectation {
    pub line: usize,
    pub text: String,
    pub expect: Expect,
}

/// A parsed `.origami` file.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub name: String,
    pub steps: Vec<Step>,
    pub expectations: Vec<Expectation>,
}

/// An expectation checked: what IX found, and whether it is what was expected.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub line: usize,
    pub text: String,
    pub got: String,
    pub holds: bool,
}

/// A sequence replayed and checked.
#[derive(Debug, Clone)]
pub struct Checked {
    /// The stated folded state the steps give.
    pub fold: Fold,
    /// The checks' report, or the structure problems that stopped them.
    pub report: Result<Report, Vec<String>>,
    pub verdicts: Vec<Verdict>,
}

impl Checked {
    /// Every expectation holds.
    pub fn holds(&self) -> bool {
        self.verdicts.iter().all(|v| v.holds)
    }
}

// ---- reading ----------------------------------------------------------------------------

/// The line up to a `#` that begins it or follows a space.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'#' && (i == 0 || bytes[i - 1].is_ascii_whitespace()) {
            return &line[..i];
        }
    }
    line
}

/// The statement's words. A point in parentheses is one word, whatever spaces it holds.
fn words(line: usize, statement: &str) -> Result<Vec<String>, SequenceError> {
    let (mut out, mut cur, mut open) = (Vec::new(), String::new(), false);
    for c in statement.chars() {
        match c {
            '(' if open => return fail(line, "a `(` inside a point"),
            '(' => {
                open = true;
                cur.push(c);
            }
            ')' if !open => return fail(line, "a `)` with no `(`"),
            ')' => {
                open = false;
                cur.push(c);
            }
            c if c.is_whitespace() => {
                if !open && !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if open {
        return fail(line, "a `(` with no `)`");
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Ok(out)
}

/// A decimal or a fraction; no exponent, `inf` or `nan`.
fn number(s: &str) -> Option<f64> {
    let v = match s.split_once('/') {
        Some((a, b)) => {
            if b.is_empty() || !b.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let (a, b) = (a.parse::<i64>().ok()?, b.parse::<i64>().ok()?);
            if b == 0 {
                return None;
            }
            a as f64 / b as f64
        }
        None => {
            if !s
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
            {
                return None;
            }
            s.parse::<f64>().ok()?
        }
    };
    (v.is_finite() && v.abs() <= MAX_COORDINATE).then_some(v)
}

fn corner(c: char) -> Option<Point> {
    match c {
        'A' => Some([0.0, 0.0]),
        'B' => Some([1.0, 0.0]),
        'C' => Some([1.0, 1.0]),
        'D' => Some([0.0, 1.0]),
        _ => None,
    }
}

/// The point a word names, if it names one.
fn point(word: &str) -> Option<Named> {
    let named = |landmark| {
        Some(Named {
            text: word.to_string(),
            landmark,
        })
    };
    let chars: Vec<char> = word.chars().collect();
    match chars.as_slice() {
        ['O'] => named(Landmark::Paper([0.5, 0.5])),
        [c] => named(Landmark::Paper(corner(*c)?)),
        ['m', a, b] => {
            let (p, q) = (corner(*a)?, corner(*b)?);
            // A side joins two corners next to each other: AB, BC, CD, DA.
            let next = |x: char| match x {
                'A' => 'B',
                'B' => 'C',
                'C' => 'D',
                _ => 'A',
            };
            if next(*a) != *b && next(*b) != *a {
                return None;
            }
            named(Landmark::Paper([(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0]))
        }
        ['(', .., ')'] => {
            let (x, y) = word[1..word.len() - 1].split_once(',')?;
            named(Landmark::At([number(x)?, number(y)?]))
        }
        _ => None,
    }
}

/// The line a word names, if it names one: two corners (`AC`) or two points joined by `-`.
fn line_of(word: &str) -> Option<Line> {
    let chars: Vec<char> = word.chars().collect();
    let points = if let [a, b] = chars.as_slice() {
        if a == b || corner(*a).is_none() || corner(*b).is_none() {
            return None;
        }
        [point(&a.to_string())?, point(&b.to_string())?]
    } else {
        // The first `-` outside parentheses that follows a point.
        let mut depth = 0;
        let mut split = None;
        for (i, c) in word.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                '-' if depth == 0 && i > 0 => {
                    split = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let i = split?;
        [point(&word[..i])?, point(&word[i + 1..])?]
    };
    Some(Line {
        text: word.to_string(),
        points,
    })
}

fn fold_step(line: usize, text: &str, w: &[String]) -> Result<Step, SequenceError> {
    let kind = if w[0] == "valley" {
        Kind::Valley
    } else {
        Kind::Mountain
    };
    let pt = |s: &str| {
        point(s).ok_or_else(|| SequenceError {
            line,
            reason: format!(
                "`{s}` is not a point: a corner A to D, a midpoint such as mAB, the centre O, or (x, y)"
            ),
        })
    };
    let w: Vec<&str> = w.iter().map(String::as_str).collect();
    let target = match w.as_slice() {
        [_, "through", p, q, "moving", r] => Target::Through(pt(p)?, pt(q)?, pt(r)?),
        [_, "through", ..] => {
            return fail(
                line,
                "a fold through two points says which side moves: `through P Q moving R`",
            )
        }
        [_, x, "to", y] => match (point(x), point(y)) {
            (Some(p), Some(q)) => Target::PointOnto(p, q),
            _ => match (line_of(x), line_of(y)) {
                (Some(l1), Some(l2)) => Target::LineOnto(l1, l2),
                _ => {
                    return fail(
                        line,
                        format!(
                            "`to` puts a point onto a point or a line onto a line, and `{x}` and `{y}` are not both"
                        ),
                    )
                }
            },
        },
        [_, _, "to", _, "moving", ..] => {
            return fail(
                line,
                "`moving` goes with `through` only: in `X to Y`, the side of X moves",
            )
        }
        _ => {
            return fail(
                line,
                "a fold is `valley X to Y`, `valley through P Q moving R`, or the same with `mountain`",
            )
        }
    };
    Ok(Step {
        line,
        text: text.to_string(),
        kind,
        target,
    })
}

fn expectation(line: usize, text: &str, w: &[String]) -> Result<Expectation, SequenceError> {
    let count = |s: &str, what: &str| {
        s.parse::<usize>().map_err(|_| SequenceError {
            line,
            reason: format!("`expect {what}` takes a whole number, not `{s}`"),
        })
    };
    let w: Vec<&str> = w[1..].iter().map(String::as_str).collect();
    let expect = match w.as_slice() {
        ["ok"] => Expect::Ok,
        ["faces", n] => Expect::Faces(count(n, "faces")?),
        ["ply", n] => Expect::Ply(count(n, "ply")?),
        _ => return fail(line, "an expectation is `ok`, `faces <n>` or `ply <n>`"),
    };
    Ok(Expectation {
        line,
        text: text.to_string(),
        expect,
    })
}

fn read_name(line: usize, w: &[String]) -> Result<String, SequenceError> {
    let ok = |s: &str| {
        s.len() <= 64
            && s.starts_with(|c: char| c.is_ascii_alphabetic())
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
    };
    match w {
        [name] if ok(name) => Ok(name.clone()),
        _ => fail(
            line,
            "a name is one word of at most 64 letters, digits and `-`, starting with a letter",
        ),
    }
}

impl FromStr for Sequence {
    type Err = SequenceError;

    fn from_str(text: &str) -> Result<Self, SequenceError> {
        if text.len() > MAX_ORIGAMI_FILE {
            return fail(
                0,
                format!("a .origami file has at most {MAX_ORIGAMI_FILE} bytes"),
            );
        }
        let (mut name, mut sheet) = (None, false);
        let (mut steps, mut expectations) = (Vec::new(), Vec::new());
        for (i, raw) in text.lines().enumerate() {
            let line = i + 1;
            let statement = strip_comment(raw).trim();
            if statement.is_empty() {
                continue;
            }
            let w = words(line, statement)?;
            let head = w[0].as_str();
            if name.is_none() && head != "origami" {
                return fail(line, "a .origami file starts with `origami <name>`");
            }
            match head {
                "origami" => {
                    if name.is_some() {
                        return fail(line, "`origami` is given twice");
                    }
                    name = Some(read_name(line, &w[1..])?);
                }
                "sheet" => {
                    if sheet {
                        return fail(line, "`sheet` is given twice");
                    }
                    if !steps.is_empty() {
                        return fail(line, "`sheet` comes before the first fold");
                    }
                    if w.len() != 2 || w[1] != "square" {
                        return fail(line, "the only sheet is `sheet square`, the unit square");
                    }
                    sheet = true;
                }
                "valley" | "mountain" => steps.push(fold_step(line, statement, &w)?),
                "expect" => expectations.push(expectation(line, statement, &w)?),
                other => {
                    let known = "origami, sheet, valley, mountain or expect";
                    return fail(line, format!("`{other}` is not a statement: {known}"));
                }
            }
        }
        let Some(name) = name else {
            return fail(0, "a .origami file starts with `origami <name>`");
        };
        if steps.is_empty() {
            return fail(0, "a .origami file folds at least once");
        }
        Ok(Self {
            name,
            steps,
            expectations,
        })
    }
}

// ---- replaying ----------------------------------------------------------------------------

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1]]
}
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}
fn scale(a: Point, k: f64) -> Point {
    [a[0] * k, a[1] * k]
}
fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: Point, b: Point) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn dist(a: Point, b: Point) -> f64 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}
fn unit(a: Point) -> Point {
    scale(a, 1.0 / dot(a, a).sqrt())
}

/// Positive left of the line through `a` with direction `d`, negative right of it.
fn side(a: Point, d: Point, p: Point) -> f64 {
    cross(d, sub(p, a))
}

/// The line through `a` with direction `d`, turned so that `m` is on its left.
fn toward(a: Point, d: Point, m: Point) -> (Point, Point) {
    if side(a, d, m) > 0.0 {
        (a, d)
    } else {
        (a, scale(d, -1.0))
    }
}

/// A face of the sheet: its crease-pattern polygon (counterclockwise), the assignment of each
/// side (`labels[i]` for `cp[i]` to `cp[i + 1]`), the isometry `[a, b, c, d, tx, ty]` that places
/// it in the folded state, and whether that isometry turns it over.
#[derive(Debug, Clone)]
struct Face {
    cp: Vec<Point>,
    labels: Vec<Assignment>,
    m: [f64; 6],
    flip: bool,
}

impl Face {
    fn folded(&self, p: Point) -> Point {
        let [a, b, c, d, tx, ty] = self.m;
        [a * p[0] + b * p[1] + tx, c * p[0] + d * p[1] + ty]
    }

    /// `p` is in the face's crease-pattern polygon, its boundary included.
    fn holds(&self, p: Point) -> bool {
        let n = self.cp.len();
        (0..n).all(|i| cross(sub(self.cp[(i + 1) % n], self.cp[i]), sub(p, self.cp[i])) >= -EPS)
    }
}

/// The isometry `m` followed by the reflection across the line through `a` with unit
/// direction `d`.
fn reflect_then(m: [f64; 6], a: Point, d: Point) -> [f64; 6] {
    let r = [
        2.0 * d[0] * d[0] - 1.0,
        2.0 * d[0] * d[1],
        2.0 * d[0] * d[1],
        2.0 * d[1] * d[1] - 1.0,
    ];
    let [ma, mb, mc, md, tx, ty] = m;
    let (px, py) = (tx - a[0], ty - a[1]);
    [
        r[0] * ma + r[1] * mc,
        r[0] * mb + r[1] * md,
        r[2] * ma + r[3] * mc,
        r[2] * mb + r[3] * md,
        r[0] * px + r[1] * py + a[0],
        r[2] * px + r[3] * py + a[1],
    ]
}

/// The parts of `f` left and right of the line through `a` with direction `d`, in the folded
/// state; `None` for an empty part. The cut gets `crease`.
fn split(f: &Face, a: Point, d: Point, crease: Assignment) -> (Option<Face>, Option<Face>) {
    let s: Vec<f64> =
        f.cp.iter()
            .map(|&q| {
                let v = side(a, d, f.folded(q));
                if v.abs() < EPS {
                    0.0
                } else {
                    v
                }
            })
            .collect();
    if s.iter().all(|&v| v >= 0.0) {
        return (Some(f.clone()), None);
    }
    if s.iter().all(|&v| v <= 0.0) {
        return (None, Some(f.clone()));
    }
    let n = f.cp.len();
    let mut halves = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
    for i in 0..n {
        let (p, q, sp, sq) = (f.cp[i], f.cp[(i + 1) % n], s[i], s[(i + 1) % n]);
        for (h, sign) in [(0usize, 1.0), (1, -1.0)] {
            let (pts, labs) = &mut halves[h];
            if sp * sign >= 0.0 {
                pts.push(p);
                // The side from p runs on to q, or to the cut point; or, p on the line and q
                // beyond it, it is the cut.
                labs.push(if sq * sign >= 0.0 || sp != 0.0 {
                    f.labels[i]
                } else {
                    crease
                });
            }
            if sp * sq < 0.0 {
                let t = sp / (sp - sq);
                pts.push(add(p, scale(sub(q, p), t)));
                // Leaving this half, the cut starts here; entering it, the rest of p to q.
                labs.push(if sp * sign > 0.0 { crease } else { f.labels[i] });
            }
        }
    }
    let [(p0, l0), (p1, l1)] = halves;
    let part = |cp, labels| Face {
        cp,
        labels,
        m: f.m,
        flip: f.flip,
    };
    (Some(part(p0, l0)), Some(part(p1, l1)))
}

/// Area of the intersection of two convex polygons, either winding.
fn overlap_area(p: &[Point], q: &[Point]) -> f64 {
    let ccw = |poly: &[Point]| {
        let n = poly.len();
        let a: f64 = (0..n).map(|i| cross(poly[i], poly[(i + 1) % n])).sum();
        if a > 0.0 {
            poly.to_vec()
        } else {
            poly.iter().rev().copied().collect()
        }
    };
    let mut out = ccw(p);
    let q = ccw(q);
    for i in 0..q.len() {
        let (a, b) = (q[i], q[(i + 1) % q.len()]);
        let inside = |x: Point| cross(sub(b, a), sub(x, a));
        let mut next = Vec::with_capacity(out.len() + 1);
        for j in 0..out.len() {
            let (c, e) = (out[j], out[(j + 1) % out.len()]);
            let (sc, se) = (inside(c), inside(e));
            if sc >= 0.0 {
                next.push(c);
            }
            if sc * se < 0.0 {
                next.push(add(c, scale(sub(e, c), sc / (sc - se))));
            }
        }
        out = next;
        if out.len() < 3 {
            return 0.0;
        }
    }
    let n = out.len();
    (0..n)
        .map(|i| cross(out[i], out[(i + 1) % n]))
        .sum::<f64>()
        .abs()
        / 2.0
}

/// The sheet as the steps leave it: its faces, and their order from the bottom of the stack to
/// the top, as a viewer above the sheet sees it. Simple folds keep one total order.
struct Sheet {
    faces: Vec<Face>,
    order: Vec<usize>,
}

impl Sheet {
    fn square() -> Self {
        Self {
            faces: vec![Face {
                cp: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                labels: vec![Assignment::B; 4],
                m: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                flip: false,
            }],
            order: vec![0],
        }
    }

    /// Where a point lies now.
    fn place(&self, n: &Named) -> Point {
        match n.landmark {
            Landmark::At(p) => p,
            Landmark::Paper(p) => self
                .faces
                .iter()
                .find(|f| f.holds(p))
                .map(|f| f.folded(p))
                .expect("every point of the square lies on a face"),
        }
    }

    /// The fold line, as a point on it and its unit direction turned so that the moving side is
    /// on its left; or why the step has none.
    fn fold_line(&self, target: &Target) -> Result<(Point, Point), String> {
        match target {
            Target::PointOnto(p, q) => {
                let (a, b) = (self.place(p), self.place(q));
                if dist(a, b) < EPS {
                    return Err(format!(
                        "{} and {} lie at the same place, so no line folds one onto the other",
                        p.text, q.text
                    ));
                }
                let d = unit([a[1] - b[1], b[0] - a[0]]);
                Ok(toward(scale(add(a, b), 0.5), d, a))
            }
            Target::Through(p, q, r) => {
                let (a, b, m) = (self.place(p), self.place(q), self.place(r));
                if dist(a, b) < EPS {
                    return Err(format!(
                        "{} and {} lie at the same place, so they make no line",
                        p.text, q.text
                    ));
                }
                let d = unit(sub(b, a));
                if side(a, d, m).abs() < EPS {
                    return Err(format!(
                        "{} lies on the fold line, so it names no side",
                        r.text
                    ));
                }
                Ok(toward(a, d, m))
            }
            Target::LineOnto(l1, l2) => {
                let [p1, p2] = l1.points.each_ref().map(|n| self.place(n));
                let [q1, q2] = l2.points.each_ref().map(|n| self.place(n));
                for (l, [x, y]) in [(l1, [p1, p2]), (l2, [q1, q2])] {
                    if dist(x, y) < EPS {
                        return Err(format!(
                            "the two points of {} lie at the same place, so they make no line",
                            l.text
                        ));
                    }
                }
                let (u, v) = (unit(sub(p2, p1)), unit(sub(q2, q1)));
                let (a, d) = if cross(u, v).abs() < EPS {
                    if dot(u, v) < 0.0 {
                        return Err(format!(
                            "{} and {} run in opposite directions, so no fold lands one on the other: write {} to {}",
                            l1.text,
                            l2.text,
                            l1.text,
                            l2.reversed()
                        ));
                    }
                    let foot = add(q1, scale(v, dot(sub(p1, q1), v)));
                    if dist(p1, foot) < EPS {
                        return Err(format!("{} and {} are the same line", l1.text, l2.text));
                    }
                    (scale(add(p1, foot), 0.5), u)
                } else {
                    // The fold line crosses where the two lines do, and halves their angle so
                    // that the first direction lands on the second.
                    let t = cross(sub(q1, p1), v) / cross(u, v);
                    (add(p1, scale(u, t)), unit(add(u, v)))
                };
                let sides: Vec<f64> = [p1, p2]
                    .into_iter()
                    .map(|p| side(a, d, p))
                    .filter(|s| s.abs() >= EPS)
                    .collect();
                if sides.is_empty() {
                    return Err(format!(
                        "{} lies on the fold line, so it names no side",
                        l1.text
                    ));
                }
                if sides.iter().any(|s| s.signum() != sides[0].signum()) {
                    return Err(format!(
                        "{} crosses the fold line, so its side is ambiguous: write this fold with `through ... moving`",
                        l1.text
                    ));
                }
                Ok(if sides[0] > 0.0 {
                    (a, d)
                } else {
                    (a, scale(d, -1.0))
                })
            }
        }
    }

    /// A simple fold along the line through `a` with direction `d`: every layer left of it moves.
    fn fold(&mut self, kind: Kind, a: Point, d: Point) -> Result<(), String> {
        let mut faces = Vec::new();
        let (mut fixed, mut moved) = (Vec::new(), Vec::new());
        for &i in &self.order {
            let f = &self.faces[i];
            // Seen from the paper's own side, the viewer's valley is a mountain on a face that
            // is turned over.
            let crease = if (kind == Kind::Valley) != f.flip {
                Assignment::V
            } else {
                Assignment::M
            };
            let (left, right) = split(f, a, d, crease);
            if let Some(r) = right {
                faces.push(r);
                fixed.push(faces.len() - 1);
            }
            if let Some(mut l) = left {
                l.m = reflect_then(l.m, a, d);
                l.flip = !l.flip;
                faces.push(l);
                moved.push(faces.len() - 1);
            }
        }
        if moved.is_empty() || fixed.is_empty() {
            return Err("the fold line does not cross the paper".into());
        }
        if faces.len() > MAX_FACES {
            return Err(format!("this fold makes more than {MAX_FACES} faces"));
        }
        // The moving stack turns over as it goes: its top layer lands at the far end.
        moved.reverse();
        self.order = match kind {
            Kind::Valley => [fixed, moved].concat(),
            Kind::Mountain => [moved, fixed].concat(),
        };
        self.faces = faces;
        Ok(())
    }

    /// The stated folded state, as FOLD 1.2.
    fn to_fold(&self, name: &str) -> Result<Fold, String> {
        let (mut cp, mut folded): (Vec<Point>, Vec<Point>) = (Vec::new(), Vec::new());
        let mut faces_vertices = Vec::with_capacity(self.faces.len());
        for f in &self.faces {
            let mut fv = Vec::with_capacity(f.cp.len());
            for &p in &f.cp {
                let at = f.folded(p);
                let found = cp
                    .iter()
                    .position(|&q| (q[0] - p[0]).abs() < EPS && (q[1] - p[1]).abs() < EPS);
                let v = match found {
                    Some(v) => {
                        // Every face that holds a point of the sheet places it at one spot.
                        if dist(folded[v], at) > 1e-7 {
                            return Err(format!("the folded sheet is torn at {p:?}"));
                        }
                        v
                    }
                    None => {
                        cp.push(p);
                        folded.push(at);
                        cp.len() - 1
                    }
                };
                fv.push(v);
            }
            faces_vertices.push(fv);
        }
        let mut edges: BTreeMap<(usize, usize), Assignment> = BTreeMap::new();
        for (f, fv) in self.faces.iter().zip(&faces_vertices) {
            for (i, &label) in f.labels.iter().enumerate() {
                let (a, b) = (fv[i], fv[(i + 1) % fv.len()]);
                let e = (a.min(b), a.max(b));
                if *edges.entry(e).or_insert(label) != label {
                    return Err(format!("edge {e:?} has two assignments"));
                }
            }
        }
        let mut rank = vec![0; self.faces.len()];
        for (r, &f) in self.order.iter().enumerate() {
            rank[f] = r;
        }
        let polys: Vec<Vec<Point>> = self
            .faces
            .iter()
            .map(|f| f.cp.iter().map(|&p| f.folded(p)).collect())
            .collect();
        let mut orders = Vec::new();
        for f in 0..self.faces.len() {
            for g in f + 1..self.faces.len() {
                if overlap_area(&polys[f], &polys[g]) > EPS {
                    // +1 when f lies on the side g's normal points to; g's normal faces the
                    // viewer unless g is turned over.
                    let s = if (rank[f] > rank[g]) != self.faces[g].flip {
                        1
                    } else {
                        -1
                    };
                    orders.push(json!([f, g, s]));
                }
            }
        }
        let points = |ps: &[Point]| Value::Array(ps.iter().map(|p| json!([p[0], p[1]])).collect());
        let value = json!({
            "file_spec": 1.2,
            "file_creator": "ix-origami sequence replay",
            "file_title": name,
            "frame_classes": ["creasePattern"],
            "vertices_coords": points(&cp),
            "edges_vertices": edges.keys().map(|&(a, b)| json!([a, b])).collect::<Vec<_>>(),
            "edges_assignment": edges.values().map(|a| a.letter()).collect::<Vec<_>>(),
            "faces_vertices": faces_vertices,
            "file_frames": [{
                "frame_classes": ["foldedForm"],
                "frame_parent": 0,
                "frame_inherit": true,
                "vertices_coords": points(&folded),
                "faceOrders": orders,
            }],
        });
        Fold::from_value(&value).map_err(|e| e.to_string())
    }
}

fn rule_names(report: &Report) -> String {
    report
        .rejected_by
        .iter()
        .map(|r| {
            serde_json::to_value(r)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("{r:?}"))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn verdict(e: &Expectation, fold: &Fold, report: &Result<Report, Vec<String>>) -> Verdict {
    let (got, holds) = match (e.expect, report) {
        (Expect::Ok, Ok(r)) if r.ok => ("ok".to_string(), true),
        (Expect::Ok, Ok(r)) if !r.rejected_by.is_empty() => {
            (format!("not ok: rejected by {}", rule_names(r)), false)
        }
        (Expect::Ok, Ok(_)) => ("not ok".to_string(), false),
        (Expect::Ok, Err(p)) => (format!("unchecked: {}", p.join("; ")), false),
        (Expect::Faces(n), _) => {
            let got = fold.faces_vertices.len();
            (got.to_string(), got == n)
        }
        (Expect::Ply(n), Ok(r)) => {
            let got = r.layers.cells.max_layers;
            (got.to_string(), got == n)
        }
        (Expect::Ply(_), Err(_)) => ("unchecked".to_string(), false),
    };
    Verdict {
        line: e.line,
        text: e.text.clone(),
        got,
        holds,
    }
}

impl Sequence {
    /// The stated folded state the steps give. `Err` with the line of the first step that has
    /// no fold: its points coincide, its side is ambiguous, its line misses the paper, or it
    /// makes more than [`MAX_FACES`] faces.
    pub fn replay(&self) -> Result<Fold, SequenceError> {
        let mut sheet = Sheet::square();
        for step in &self.steps {
            let at = |reason| SequenceError {
                line: step.line,
                reason,
            };
            let (a, d) = sheet.fold_line(&step.target).map_err(at)?;
            sheet.fold(step.kind, a, d).map_err(at)?;
        }
        sheet.to_fold(&self.name).map_err(|r| SequenceError {
            line: 0,
            reason: format!("the replay is inconsistent, which is a bug in ix-origami: {r}"),
        })
    }

    /// [`Sequence::check_within`] with no limit.
    pub fn check(&self) -> Result<Checked, SequenceError> {
        self.check_within(&Limits::NONE)
    }

    /// Replays the steps, runs every check on the result and answers each expectation. A step
    /// with no fold, or a count over `limits`, is refused; failed expectations are answered.
    pub fn check_within(&self, limits: &Limits) -> Result<Checked, SequenceError> {
        let fold = self.replay()?;
        let report = match analyse_within(&fold, limits) {
            Ok(r) => Ok(r),
            Err(Unchecked::Structure(problems)) => Err(problems),
            Err(Unchecked::OverLimit(e)) => return fail(0, format!("refused, not checked: {e}")),
        };
        Ok(self.answer(fold, report))
    }

    /// Answers each expectation from the replayed `fold` and its checks: the report, or the
    /// structure problems that stopped them. For a caller that runs the checks itself.
    pub fn answer(&self, fold: Fold, report: Result<Report, Vec<String>>) -> Checked {
        let verdicts = self
            .expectations
            .iter()
            .map(|e| verdict(e, &fold, &report))
            .collect();
        Checked {
            fold,
            report,
            verdicts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every word the parser matches is in the grammar.
    #[test]
    fn the_grammar_names_every_word_the_parser_reads() {
        for word in [
            "origami", "sheet", "square", "valley", "mountain", "to", "through", "moving",
            "expect", "ok", "faces", "ply", "m", "O", "A", "B", "C", "D", "-", "/", "#",
        ] {
            assert!(GRAMMAR.contains(&format!("\"{word}\"")), "{word}");
        }
    }

    #[test]
    fn points_and_lines_are_read_as_written() {
        assert_eq!(point("mBA").unwrap().landmark, Landmark::Paper([0.5, 0.0]));
        assert_eq!(point("mDA").unwrap().landmark, Landmark::Paper([0.0, 0.5]));
        assert!(point("mAC").is_none(), "AC is a diagonal, not a side");
        assert_eq!(
            point("(1/3,-0.5)").unwrap().landmark,
            Landmark::At([1.0 / 3.0, -0.5])
        );
        for bad in ["(1e3,0)", "(inf,0)", "(1/0,0)", "(1,2,3)", "E", "AB"] {
            assert!(point(bad).is_none(), "{bad}");
        }
        let l = line_of("(-1,0)-mCD").unwrap();
        assert_eq!(l.points[0].landmark, Landmark::At([-1.0, 0.0]));
        assert_eq!(l.points[1].landmark, Landmark::Paper([0.5, 1.0]));
        assert_eq!(line_of("AC").unwrap().points[1].text, "C");
        assert!(line_of("AA").is_none());
        assert!(line_of("A").is_none());
    }

    #[test]
    fn a_point_in_parentheses_is_one_word() {
        assert_eq!(
            words(1, "valley through (1/3, 0) ( 1/3 , 1 ) moving (0, 1/2)").unwrap(),
            ["valley", "through", "(1/3,0)", "(1/3,1)", "moving", "(0,1/2)"]
        );
        assert_eq!(words(4, "valley (0, 0").unwrap_err().line, 4);
    }
}
