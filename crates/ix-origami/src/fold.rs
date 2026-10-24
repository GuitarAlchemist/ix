//! FOLD files (spec v1.2, Demaine, Ku, Lang; <https://github.com/edemaine/fold/blob/main/doc/spec.md>):
//! one crease-pattern frame and one folded frame that inherits from it.
//!
//! Field names and assignment letters are the spec's. Reading keeps every key it does not
//! interpret, so [`Fold::to_value`] writes back what was read, in the 1.2 layout.

use std::collections::HashMap;

use serde_json::{Map, Value};
use thiserror::Error;

use crate::geometry::area;

/// A point in the plane.
pub type Point = [f64; 2];

/// An edge's assignment, with the spec's letters: border, mountain, valley, flat, unassigned,
/// cut, join.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Assignment {
    B,
    M,
    V,
    F,
    U,
    C,
    J,
}

impl Assignment {
    pub fn from_letter(s: &str) -> Option<Self> {
        Some(match s {
            "B" => Self::B,
            "M" => Self::M,
            "V" => Self::V,
            "F" => Self::F,
            "U" => Self::U,
            "C" => Self::C,
            "J" => Self::J,
            _ => return None,
        })
    }

    pub fn letter(self) -> &'static str {
        match self {
            Self::B => "B",
            Self::M => "M",
            Self::V => "V",
            Self::F => "F",
            Self::U => "U",
            Self::C => "C",
            Self::J => "J",
        }
    }

    /// A mountain or a valley: a crease folded flat.
    pub fn is_fold(self) -> bool {
        matches!(self, Self::M | Self::V)
    }
}

/// Why a file could not be read as a FOLD crease pattern with folded frames.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FoldError {
    #[error("not JSON: {0}")]
    Json(String),
    #[error("{field}: {problem}")]
    Field { field: String, problem: String },
    #[error("FOLD 1.1 layout: frame 1 must be a crease pattern without faceOrders")]
    Layout,
}

fn bad(field: &str, problem: &str) -> FoldError {
    FoldError::Field {
        field: field.to_string(),
        problem: problem.to_string(),
    }
}

/// The folded frame: its coordinates and its stated layer order.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldedFrame {
    pub frame_classes: Vec<String>,
    pub frame_parent: Option<i64>,
    pub frame_inherit: Option<bool>,
    pub vertices_coords: Vec<Point>,
    pub edges_fold_angle: Option<Vec<f64>>,
    /// `[f, g, s]` triples as written; [`Fold::structure`] checks their range.
    pub face_orders: Vec<[i64; 3]>,
    /// Every other key of the frame, as read.
    pub other: Map<String, Value>,
}

/// A crease pattern (the top-level frame) and its folded frames, in the FOLD 1.2 layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Fold {
    pub file_spec: Option<f64>,
    pub frame_classes: Vec<String>,
    pub vertices_coords: Vec<Point>,
    pub edges_vertices: Vec<[usize; 2]>,
    pub edges_assignment: Vec<Assignment>,
    pub faces_vertices: Vec<Vec<usize>>,
    pub frames: Vec<FoldedFrame>,
    /// Every other top-level key (`file_title`, `file_author`, ...), as read.
    pub other: Map<String, Value>,
}

/// Faces on each edge, found by matching each face side to an edge; and the face sides that are
/// no edge, as `(face, [lower vertex, higher vertex])`.
pub type EdgeFaces = (Vec<Vec<usize>>, Vec<(usize, [usize; 2])>);

impl Fold {
    pub fn from_json_str(text: &str) -> Result<Self, FoldError> {
        let v: Value = serde_json::from_str(text).map_err(|e| FoldError::Json(e.to_string()))?;
        Self::from_value(&v)
    }

    /// Reads a FOLD object. A file in the FOLD 1.1 layout (folded state at the top level,
    /// crease pattern in frame 1, `faceOrders` at the top) is rearranged into the 1.2 layout:
    /// places change, no value does.
    pub fn from_value(v: &Value) -> Result<Self, FoldError> {
        let mut top = v
            .as_object()
            .cloned()
            .ok_or_else(|| bad("(root)", "not a JSON object"))?;
        let mut frames = match top.remove("file_frames") {
            None => Vec::new(),
            Some(Value::Array(a)) => a
                .into_iter()
                .enumerate()
                .map(|(i, f)| match f {
                    Value::Object(m) => Ok(m),
                    _ => Err(bad(&format!("file_frames[{i}]"), "not an object")),
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(bad("file_frames", "not an array")),
        };
        if has_class(&top, "foldedForm") {
            to_spec_1_2(&mut top, &mut frames)?;
        }
        let frames = frames
            .into_iter()
            .enumerate()
            .map(|(i, m)| FoldedFrame::from_map(m, &format!("file_frames[{i}].")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            file_spec: top.remove("file_spec").and_then(|x| x.as_f64()),
            frame_classes: take_strings(&mut top, "frame_classes", "")?,
            vertices_coords: take_points(&mut top, "vertices_coords", "")?,
            edges_vertices: take_pairs(&mut top, "edges_vertices")?,
            edges_assignment: take_assignments(&mut top)?,
            faces_vertices: take_lists(&mut top, "faces_vertices")?,
            frames,
            other: top,
        })
    }

    /// The FOLD 1.2 object: everything read, with the crease pattern at the top level.
    pub fn to_value(&self) -> Value {
        let mut top = self.other.clone();
        if let Some(s) = self.file_spec {
            top.insert("file_spec".into(), s.into());
        }
        top.insert("frame_classes".into(), self.frame_classes.clone().into());
        top.insert(
            "vertices_coords".into(),
            points_value(&self.vertices_coords),
        );
        top.insert(
            "edges_vertices".into(),
            Value::Array(
                self.edges_vertices
                    .iter()
                    .map(|e| e.to_vec().into())
                    .collect(),
            ),
        );
        top.insert(
            "edges_assignment".into(),
            Value::Array(
                self.edges_assignment
                    .iter()
                    .map(|a| a.letter().into())
                    .collect(),
            ),
        );
        top.insert(
            "faces_vertices".into(),
            Value::Array(
                self.faces_vertices
                    .iter()
                    .map(|f| f.clone().into())
                    .collect(),
            ),
        );
        top.insert(
            "file_frames".into(),
            Value::Array(self.frames.iter().map(FoldedFrame::to_value).collect()),
        );
        Value::Object(top)
    }

    /// The folded frame. Call it once [`Fold::structure`] is empty, which guarantees there is
    /// exactly one.
    pub fn folded(&self) -> &FoldedFrame {
        &self.frames[0]
    }

    pub fn folded_mut(&mut self) -> &mut FoldedFrame {
        &mut self.frames[0]
    }

    /// The spec's field meanings: lengths, index ranges, fold angles. Empty = ok.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let nv = self.vertices_coords.len();
        if self.file_spec != Some(1.2) {
            problems.push("file_spec is not 1.2".to_string());
        }
        if self.edges_assignment.len() != self.edges_vertices.len() {
            problems.push("edges_assignment and edges_vertices differ in length".to_string());
        }
        for (k, &[u, w]) in self.edges_vertices.iter().enumerate() {
            if u >= nv || w >= nv || u == w {
                problems.push(format!("edge {k} has bad vertices {u},{w}"));
            }
        }
        for (k, f) in self.faces_vertices.iter().enumerate() {
            if f.len() < 3 || f.iter().any(|&v| v >= nv) {
                problems.push(format!("face {k} is malformed"));
            }
        }
        for fr in &self.frames {
            if fr.vertices_coords.len() != nv {
                problems.push("folded frame has a different vertex count".to_string());
            }
            if let Some(fa) = &fr.edges_fold_angle {
                if fa.len() != self.edges_vertices.len() {
                    problems.push("edges_foldAngle length differs from edges".to_string());
                }
                for (k, (&a, &s)) in fa.iter().zip(&self.edges_assignment).enumerate() {
                    if !(-180.0..=180.0).contains(&a) {
                        problems.push(format!("edge {k} fold angle out of range"));
                    }
                    if (s == Assignment::V && a <= 0.0) || (s == Assignment::M && a >= 0.0) {
                        problems.push(format!(
                            "edge {k}: fold angle sign {a:+.3} disagrees with {}",
                            s.letter()
                        ));
                    }
                    // The spec: "zero for flat, unassigned, and border folds".
                    if matches!(s, Assignment::F | Assignment::U | Assignment::B) && a != 0.0 {
                        problems.push(format!(
                            "edge {k} ({}): fold angle {a:+.3}, where the spec sets 0 on flat, \
                             unassigned and border edges",
                            s.letter()
                        ));
                    }
                }
            }
        }
        problems
    }

    /// What the layer checks need on top of [`Fold::validate`]: one folded frame inheriting from
    /// the crease pattern without overriding its edges or faces, `faceOrders` in range and on
    /// the folded frame only, faces counterclockwise in the crease pattern, every face side an
    /// edge, one face on a border edge and two on any other. Empty = ok.
    pub fn structure(&self) -> Vec<String> {
        let mut problems = self.validate();
        if !problems.is_empty() {
            return problems;
        }
        if self.frames.len() != 1 {
            problems.push(format!(
                "expected one folded frame, found {}",
                self.frames.len()
            ));
            return problems;
        }
        let fr = self.folded();
        if !fr.frame_classes.iter().any(|c| c == "foldedForm") {
            problems.push("frame 1 is not a foldedForm".to_string());
        }
        if fr.frame_parent != Some(0) || fr.frame_inherit != Some(true) {
            problems.push("frame 1 does not inherit from frame 0".to_string());
        }
        // The checks read these from the crease pattern and the orders from the folded frame;
        // anything else would be silently ignored.
        for key in ["edges_vertices", "edges_assignment", "faces_vertices"] {
            if fr.other.contains_key(key) {
                problems.push(format!(
                    "the folded frame overrides {key}, which the checks read from the crease pattern"
                ));
            }
        }
        if self.other.contains_key("faceOrders") {
            problems.push(
                "faceOrders on the crease pattern; the checks read them from the folded frame"
                    .to_string(),
            );
        }
        let nf = self.faces_vertices.len() as i64;
        for (k, t) in fr.face_orders.iter().enumerate() {
            let in_range = (0..nf).contains(&t[0]) && (0..nf).contains(&t[1]);
            if !in_range || t[0] == t[1] || !(-1..=1).contains(&t[2]) {
                problems.push(format!("faceOrders[{k}] = {t:?} is out of range"));
            }
        }
        for (fi, f) in self.faces_vertices.iter().enumerate() {
            let poly: Vec<Point> = f.iter().map(|&v| self.vertices_coords[v]).collect();
            if area(&poly) <= 0.0 {
                problems.push(format!(
                    "face {fi} is not counterclockwise in the crease pattern"
                ));
            }
        }
        let (ef, missing) = self.edge_faces();
        for (fi, [a, b]) in missing {
            problems.push(format!("face {fi} side ({a}, {b}) is not an edge"));
        }
        for (k, a) in self.edges_assignment.iter().enumerate() {
            let want = if *a == Assignment::B { 1 } else { 2 };
            if ef[k].len() != want {
                problems.push(format!(
                    "edge {k} ({}) has {} faces",
                    a.letter(),
                    ef[k].len()
                ));
            }
        }
        problems
    }

    /// Faces on each edge, in face order; and the face sides that are no edge.
    pub fn edge_faces(&self) -> EdgeFaces {
        let index: HashMap<[usize; 2], usize> = self
            .edges_vertices
            .iter()
            .enumerate()
            .map(|(k, &[u, w])| ([u.min(w), u.max(w)], k))
            .collect();
        let mut out = vec![Vec::new(); self.edges_vertices.len()];
        let mut missing = Vec::new();
        for (fi, f) in self.faces_vertices.iter().enumerate() {
            for i in 0..f.len() {
                let (a, b) = (f[i], f[(i + 1) % f.len()]);
                let key = [a.min(b), a.max(b)];
                match index.get(&key) {
                    Some(&k) => out[k].push(fi),
                    None => missing.push((fi, key)),
                }
            }
        }
        (out, missing)
    }
}

impl FoldedFrame {
    fn from_map(mut m: Map<String, Value>, at: &str) -> Result<Self, FoldError> {
        let edges_fold_angle = match m.remove("edges_foldAngle") {
            None => None,
            Some(Value::Array(a)) => Some(
                a.iter()
                    .map(Value::as_f64)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| bad(&format!("{at}edges_foldAngle"), "not numbers"))?,
            ),
            Some(_) => return Err(bad(&format!("{at}edges_foldAngle"), "not an array")),
        };
        let face_orders = match m.remove("faceOrders") {
            None => Vec::new(),
            Some(Value::Array(a)) => a
                .iter()
                .enumerate()
                .map(|(k, t)| {
                    let ints = t
                        .as_array()
                        .filter(|t| t.len() == 3)
                        .and_then(|t| t.iter().map(Value::as_i64).collect::<Option<Vec<_>>>());
                    match ints {
                        Some(v) => Ok([v[0], v[1], v[2]]),
                        None => Err(bad(&format!("{at}faceOrders[{k}]"), "not three integers")),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(bad(&format!("{at}faceOrders"), "not an array")),
        };
        Ok(Self {
            frame_classes: take_strings(&mut m, "frame_classes", at)?,
            frame_parent: m.remove("frame_parent").and_then(|x| x.as_i64()),
            frame_inherit: m.remove("frame_inherit").and_then(|x| x.as_bool()),
            vertices_coords: take_points(&mut m, "vertices_coords", at)?,
            edges_fold_angle,
            face_orders,
            other: m,
        })
    }

    fn to_value(&self) -> Value {
        let mut m = self.other.clone();
        m.insert("frame_classes".into(), self.frame_classes.clone().into());
        if let Some(p) = self.frame_parent {
            m.insert("frame_parent".into(), p.into());
        }
        if let Some(i) = self.frame_inherit {
            m.insert("frame_inherit".into(), i.into());
        }
        m.insert(
            "vertices_coords".into(),
            points_value(&self.vertices_coords),
        );
        if let Some(fa) = &self.edges_fold_angle {
            m.insert("edges_foldAngle".into(), fa.clone().into());
        }
        m.insert(
            "faceOrders".into(),
            Value::Array(self.face_orders.iter().map(|t| t.to_vec().into()).collect()),
        );
        Value::Object(m)
    }
}

fn has_class(m: &Map<String, Value>, class: &str) -> bool {
    m.get("frame_classes")
        .and_then(Value::as_array)
        .is_some_and(|a| a.iter().any(|c| c.as_str() == Some(class)))
}

/// Rabbit Ear's FOLD 1.1 layout to the 1.2 layout: the folded state at the top level and the
/// crease pattern in frame 1 swap places, and `faceOrders` moves into the folded frame.
fn to_spec_1_2(
    top: &mut Map<String, Value>,
    frames: &mut [Map<String, Value>],
) -> Result<(), FoldError> {
    let fr = frames.first_mut().ok_or(FoldError::Layout)?;
    if !has_class(fr, "creasePattern") || fr.contains_key("faceOrders") {
        return Err(FoldError::Layout);
    }
    top.insert("file_spec".into(), 1.2.into());
    for key in ["frame_classes", "vertices_coords"] {
        let (a, b) = (top.remove(key), fr.remove(key));
        if let Some(b) = b {
            top.insert(key.into(), b);
        }
        if let Some(a) = a {
            fr.insert(key.into(), a);
        }
    }
    if let Some(orders) = top.remove("faceOrders") {
        fr.insert("faceOrders".into(), orders);
    }
    Ok(())
}

fn take_array(m: &mut Map<String, Value>, key: &str, at: &str) -> Result<Vec<Value>, FoldError> {
    match m.remove(key) {
        Some(Value::Array(a)) => Ok(a),
        Some(_) => Err(bad(&format!("{at}{key}"), "not an array")),
        None => Err(bad(&format!("{at}{key}"), "missing")),
    }
}

fn take_strings(m: &mut Map<String, Value>, key: &str, at: &str) -> Result<Vec<String>, FoldError> {
    if !m.contains_key(key) {
        return Ok(Vec::new());
    }
    take_array(m, key, at)?
        .iter()
        .map(|s| s.as_str().map(str::to_string))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| bad(&format!("{at}{key}"), "not strings"))
}

fn take_points(m: &mut Map<String, Value>, key: &str, at: &str) -> Result<Vec<Point>, FoldError> {
    take_array(m, key, at)?
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let c = p.as_array().filter(|c| c.len() == 2);
            match c.map(|c| (c[0].as_f64(), c[1].as_f64())) {
                Some((Some(x), Some(y))) => Ok([x, y]),
                _ => Err(bad(&format!("{at}{key}[{i}]"), "not a 2D point")),
            }
        })
        .collect()
}

fn index(v: &Value) -> Option<usize> {
    v.as_u64().and_then(|x| usize::try_from(x).ok())
}

fn take_pairs(m: &mut Map<String, Value>, key: &str) -> Result<Vec<[usize; 2]>, FoldError> {
    take_array(m, key, "")?
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let e = e.as_array().filter(|e| e.len() == 2);
            match e.map(|e| (index(&e[0]), index(&e[1]))) {
                Some((Some(u), Some(w))) => Ok([u, w]),
                _ => Err(bad(&format!("{key}[{i}]"), "not two vertex indices")),
            }
        })
        .collect()
}

fn take_lists(m: &mut Map<String, Value>, key: &str) -> Result<Vec<Vec<usize>>, FoldError> {
    take_array(m, key, "")?
        .iter()
        .enumerate()
        .map(|(i, f)| {
            f.as_array()
                .and_then(|f| f.iter().map(index).collect::<Option<Vec<_>>>())
                .ok_or_else(|| bad(&format!("{key}[{i}]"), "not vertex indices"))
        })
        .collect()
}

fn take_assignments(m: &mut Map<String, Value>) -> Result<Vec<Assignment>, FoldError> {
    take_array(m, "edges_assignment", "")?
        .iter()
        .enumerate()
        .map(|(k, a)| {
            a.as_str()
                .and_then(Assignment::from_letter)
                .ok_or_else(|| bad(&format!("edges_assignment[{k}]"), "unknown assignment"))
        })
        .collect()
}

fn points_value(points: &[Point]) -> Value {
    Value::Array(points.iter().map(|p| p.to_vec().into()).collect())
}
