//! `ix_braid` and `ix_knot` — the agent-facing surface for `ix-knot`.
//!
//! `ix_braid`: a braid word in, the knot or link its closure makes out: the
//! strand permutation, components, writhe and Jones polynomial, plus on request
//! a 3D layout of the strands a renderer can draw (a ComfyUI control image, for
//! one).
//!
//! `ix_knot`: a knot tied in rope, named from the catalogue or drawn by the
//! caller, with the same invariants and on request each rope's 3D path.
//!
//! Pure computations over the caller's input and the built-in catalogue: no
//! filesystem, no network, no state.

use ix_knot::catalog::{catalog, closure_braid, find};
use ix_knot::gauss::draw;
use ix_knot::{
    jones, layout, Braid, GaussCode, GaussError, Jones, Rope, RopeDiagram, MAX_CONTROL_POINTS,
    MAX_CROSSINGS, MAX_GAUSS_CROSSINGS, MAX_POINTS, MAX_ROPES, MAX_STRANDS,
};
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

fn braid_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "word": {
                "type": "string",
                "description": "Braid word: generators s1, s2^-1, s1^3 (σ may stand for s) or signed integers, separated by spaces or commas. sk: the strand in position k crosses in front of the one in position k+1; sk^-1 the other way. A sailor's plait is \"s1 s2^-1\", laid rope \"s1 s2\", the reef knot \"s1^3 s2^-3\", the granny \"s1^3 s2^3\"."
            },
            "strands": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_STRANDS,
                "description": "Number of strands; defaults to one more than the largest generator"
            },
            "repeat": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
                "description": format!("Write the word this many times; at most {MAX_CROSSINGS} crossings in all")
            },
            "geometry": {
                "type": "boolean",
                "default": false,
                "description": "Also return each strand's 3D path, for drawing the braid"
            },
            "samples_per_crossing": {
                "type": "integer",
                "minimum": 2,
                "default": 16,
                "description": format!("Points per crossing in each path; at most {MAX_POINTS} points over all strands")
            }
        },
        "required": ["word"]
    })
}

fn braid_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "word": { "type": "string", "description": "The word as read, runs written as powers" },
            "generators": { "type": "array", "items": { "type": "integer" }, "description": "±k for each crossing, in order" },
            "strands": { "type": "integer" },
            "crossings": { "type": "integer" },
            "writhe": { "type": "integer", "description": "Crossing signs summed: +1 per sk, -1 per sk^-1" },
            "permutation": { "type": "array", "items": { "type": "integer" }, "description": "permutation[i]: the 0-based position where the strand starting at position i ends" },
            "components": { "type": "integer", "description": "Components of the closure: 1 for a knot" },
            "jones": {
                "type": "object",
                "description": "Jones polynomial of the closure. `text` in t; `terms` as [power, coefficient] with power a multiple of 1/2. The closure of s1^3 is the right-handed trefoil, t + t^3 - t^4."
            },
            "jones_symmetric": { "type": "boolean", "description": "V(t) = V(1/t). Every amphichiral link has it; having it does not prove a link amphichiral" },
            "geometry": {
                "type": "object",
                "description": "With `geometry`: `strands`, each {start, points: [[x, y, z], ...]} in a right-handed frame (x across the strands, y along the braid, one crossing per unit, z toward the viewer). Drawn with rows growing downward it shows the mirror braid."
            }
        }
    })
}

/// The knot or link a braid word closes into: permutation, components, writhe
/// and Jones polynomial, and on request the strands' 3D layout for drawing it.
///
/// The Jones polynomial is the Kauffman bracket evaluated in the
/// Temperley–Lieb algebra, so the cost grows with the crossings rather than
/// 2^crossings. It tells a reef knot (s1^3 s2^-3, symmetric) from a granny
/// (s1^3 s2^3, not), though the two share an Alexander polynomial.
#[ix_skill(
    domain = "knot",
    name = "braid",
    governance = "deterministic",
    schema_fn = "crate::skills::knot::braid_schema",
    output_schema_fn = "crate::skills::knot::braid_output_schema"
)]
pub fn braid(params: Value) -> Result<Value, String> {
    let word = params
        .get("word")
        .and_then(Value::as_str)
        .ok_or("`word` is required: a braid word such as \"s1 s2^-1\"")?;
    let strands = optional_count(&params, "strands")?;
    let repeat = optional_count(&params, "repeat")?.unwrap_or(1);
    if repeat == 0 {
        return Err("`repeat` must be at least 1".to_string());
    }
    let braid = Braid::parse(strands, word)
        .and_then(|b| b.repeat(repeat))
        .map_err(|e| e.to_string())?;

    let v = jones(&braid);
    let mut out = json!({
        "word": braid.to_string(),
        "generators": braid.word(),
        "strands": braid.strands(),
        "crossings": braid.crossings(),
        "writhe": braid.writhe(),
        "permutation": braid.permutation(),
        "components": braid.components(),
        "jones": jones_json(&v)?,
        "jones_symmetric": v.is_symmetric(),
    });
    if params
        .get("geometry")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let samples = optional_count(&params, "samples_per_crossing")?.unwrap_or(16);
        let paths = layout(&braid, samples).map_err(|e| e.to_string())?;
        let strands: Vec<Value> = paths
            .iter()
            .map(|p| json!({ "start": p.start, "points": p.points }))
            .collect();
        out["geometry"] = json!({ "strands": strands });
    }
    Ok(out)
}

/// `{text, terms}`: the polynomial in t, and its terms as [power, coefficient].
fn jones_json(v: &Jones) -> Result<Value, String> {
    let terms = v
        .terms()
        .iter()
        .map(|&(half, coeff)| {
            i64::try_from(coeff)
                .map(|c| json!([f64::from(half) / 2.0, c]))
                .map_err(|_| format!("a Jones coefficient ({coeff}) does not fit a JSON integer"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({ "text": v.to_string(), "terms": terms }))
}

fn knot_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "A knot from the catalogue, e.g. \"overhand\" or \"figure-eight\"; `list` names them all. Omit it to read a drawing of your own from `ropes`."
            },
            "list": {
                "type": "boolean",
                "default": false,
                "description": "Return the catalogue (id, English and French names, family, Ashley number, closure) instead of a knot"
            },
            "ropes": {
                "type": "array",
                "maxItems": MAX_ROPES,
                "description": format!("A drawing of your own: each rope {{points: [[x, y], ...], closed}}, its control points in order (a smooth curve passes through them), y up; [x, y, z] when `over` is \"height\". At most {MAX_CONTROL_POINTS} points over all ropes."),
                "items": {
                    "type": "object",
                    "properties": {
                        "points": { "type": "array", "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 3 } },
                        "closed": { "type": "boolean", "default": false }
                    },
                    "required": ["points"]
                }
            },
            "over": {
                "type": "string",
                "description": "With `ropes`: which passage of each crossing is in front. One letter O (over) or U (under) per passage, in the order the ropes reach them (rope 0 from its first point, then rope 1); \"alternating\" for one rope, starting over; or \"height\", the higher z in front. An open rope is closed by an arc over everything from its end back to its start."
            },
            "gauss": {
                "type": "string",
                "description": format!("A knot spelled by its crossings, drawn from the code alone: along each rope, O (over) or U (under) and a number the two passages of one crossing share. \"U1 O2 U3 O1 U2 O3\" is the overhand knot; a closed rope goes in parentheses, \"(O1 U2 O3 U1 O2 U3)\" the trefoil; ropes are separated by |. At most {MAX_GAUSS_CROSSINGS} crossings. The letters do not fix handedness: see `closure`.")
            },
            "closure": {
                "type": "string",
                "description": "With `gauss`: the knot the closure must be, in Rolfsen's notation from \"0_1\" to \"6_3\", `m` in front for the mirror image; the drawing that closes into it is kept. Without it, a code whose drawings close into different knots (two overhands in a row: the granny or the reef) is refused with the candidates."
            },
            "radius": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "With `ropes` and `geometry`: the rope radius, in the drawing's units. A catalogue knot has its own, and a drawing from `gauss` one that clears itself."
            },
            "geometry": {
                "type": "boolean",
                "default": false,
                "description": "Also return each rope's 3D path, for drawing the knot"
            }
        }
    })
}

fn knot_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "entries": { "type": "array", "description": "With `list`: the catalogue" },
            "id": { "type": "string", "description": "Catalogue knots: the id, with `en`, `fr`, `family`, `abok` and `closure` (the closure's knot in Rolfsen's table, `m` in front for the mirror)" },
            "crossings": { "type": "integer", "description": "Crossings of the drawing" },
            "closure_crossings": { "type": "integer", "description": "Crossings the arcs closing open ropes add" },
            "writhe": { "type": "integer", "description": "Crossing signs summed, the closure's included" },
            "components": { "type": "integer", "description": "Components of the closure: one per rope" },
            "jones": { "type": "object", "description": "Jones polynomial of the closure: `text` in t, `terms` as [power, coefficient]" },
            "jones_symmetric": { "type": "boolean", "description": "V(t) = V(1/t)" },
            "gauss": { "type": "string", "description": "The drawing's Gauss code, crossings numbered in the order they are first passed" },
            "drawing": {
                "type": "object",
                "description": "With `gauss`: the drawing found, as `ropes` and `over` that `ropes`/`over` take back"
            },
            "geometry": {
                "type": "object",
                "description": "With `geometry`: `radius`, `ropes` (each {closed, points: [[x, y, z], ...]}, z toward the viewer: up where the rope passes in front) and `min_clearance`, the least distance between two parts of the ropes in rope diameters (below 1 the tubes pass through each other)"
            }
        }
    })
}

/// A knot tied in rope, from the catalogue or drawn by the caller: its
/// closure's components, writhe and Jones polynomial, and on request each
/// rope's 3D path.
///
/// A drawing is ropes through control points and, at each crossing, which
/// passage is in front; the crossings are found from the curves. Each
/// catalogue entry is tested against the knot its closure must be.
#[ix_skill(
    domain = "knot",
    name = "knot",
    governance = "deterministic",
    schema_fn = "crate::skills::knot::knot_schema",
    output_schema_fn = "crate::skills::knot::knot_output_schema"
)]
pub fn knot(params: Value) -> Result<Value, String> {
    if params.get("list").and_then(Value::as_bool).unwrap_or(false) {
        let entries: Vec<Value> = catalog()
            .iter()
            .map(|e| {
                json!({
                    "id": e.id, "en": e.en, "fr": e.fr, "family": e.family,
                    "abok": e.abok, "closure": e.closure,
                })
            })
            .collect();
        return Ok(json!({ "entries": entries }));
    }
    let geometry = params
        .get("geometry")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let given: Vec<&str> = ["name", "ropes", "gauss"]
        .into_iter()
        .filter(|k| !matches!(params.get(*k), None | Some(Value::Null)))
        .collect();
    if let [a, b, ..] = given[..] {
        return Err(format!(
            "give one of `name`, `ropes` and `gauss`, not both `{a}` and `{b}`"
        ));
    }
    let closure = match params.get("closure") {
        None | Some(Value::Null) => None,
        Some(_) if given != ["gauss"] => return Err("`closure` goes with `gauss`".into()),
        Some(Value::String(k)) => Some(k.as_str()),
        Some(other) => return Err(format!("`closure` must be a string, got {other}")),
    };
    let radius = match params.get("radius") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_f64()
                .ok_or_else(|| format!("`radius` must be a number, got {v}"))?,
        ),
    };
    let mut drawing = None;
    let (entry, diagram, radius) = match given.first().copied() {
        Some("name") => {
            let id = params["name"]
                .as_str()
                .ok_or_else(|| format!("`name` must be a string, got {}", params["name"]))?;
            let e = find(id).ok_or_else(|| {
                format!("no knot {id:?} in the catalogue; `list: true` names them")
            })?;
            (Some(e), e.diagram().map_err(|e| e.to_string())?, e.radius)
        }
        Some("gauss") => {
            let text = params["gauss"]
                .as_str()
                .ok_or_else(|| format!("`gauss` must be a string, got {}", params["gauss"]))?;
            let code: GaussCode = text.parse().map_err(|e: GaussError| e.to_string())?;
            let want = match closure {
                None => None,
                Some(k) => Some(jones(&closure_braid(k).ok_or_else(|| {
                    format!("no knot {k:?} to close into: `closure` is one of \"0_1\" to \"6_3\", `m` in front for the mirror image")
                })?)),
            };
            let drawn = draw(&code, want.as_ref()).map_err(|e| e.to_string())?;
            let ropes: Vec<Value> = drawn
                .ropes
                .iter()
                .map(|r| {
                    let points: Vec<[f64; 2]> = r.points.iter().map(|p| [p[0], p[1]]).collect();
                    json!({ "closed": r.closed, "points": points })
                })
                .collect();
            drawing = Some(json!({ "ropes": ropes, "over": drawn.over }));
            (None, drawn.diagram, radius.unwrap_or(drawn.radius))
        }
        _ => {
            let ropes = read_ropes(&params)?;
            let over = match params.get("over") {
                None | Some(Value::Null) => "",
                Some(Value::String(s)) => s.as_str(),
                Some(other) => return Err(format!("`over` must be a string, got {other}")),
            };
            let diagram = RopeDiagram::new(&ropes, over).map_err(|e| e.to_string())?;
            let radius = match radius {
                None if geometry => {
                    return Err(
                        "`radius` is required with `geometry` for a drawing of your own".into(),
                    )
                }
                r => r.unwrap_or(1.0),
            };
            (None, diagram, radius)
        }
    };

    let drawn = diagram.drawn_crossings();
    let v = diagram.jones();
    let mut out = json!({
        "crossings": drawn,
        "closure_crossings": diagram.crossings().len() - drawn,
        "writhe": diagram.writhe(),
        "components": diagram.components(),
        "jones": jones_json(v)?,
        "jones_symmetric": v.is_symmetric(),
        "gauss": diagram.gauss_code().to_string(),
    });
    if let Some(d) = drawing {
        out["drawing"] = d;
    }
    if let Some(k) = closure {
        out["closure"] = json!(k);
    }
    if let Some(e) = entry {
        for (key, value) in [
            ("id", json!(e.id)),
            ("en", json!(e.en)),
            ("fr", json!(e.fr)),
            ("family", json!(e.family)),
            ("abok", json!(e.abok)),
            ("closure", json!(e.closure)),
        ] {
            out[key] = value;
        }
    }
    if geometry {
        let g = diagram.geometry(radius).map_err(|e| e.to_string())?;
        let ropes: Vec<Value> = g
            .ropes
            .iter()
            .map(|r| json!({ "closed": r.closed, "points": r.points }))
            .collect();
        out["geometry"] = json!({
            "radius": g.radius,
            "min_clearance": g.min_clearance,
            "ropes": ropes,
        });
    }
    Ok(out)
}

/// `ropes` as the caller drew them; the diagram checks the rest.
fn read_ropes(params: &Value) -> Result<Vec<Rope>, String> {
    let ropes = params.get("ropes").and_then(Value::as_array).ok_or(
        "give `name` (a catalogue knot), `ropes` (a drawing of your own) or `gauss` (a code)",
    )?;
    ropes
        .iter()
        .enumerate()
        .map(|(i, rope)| {
            let points = rope
                .get("points")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("rope {i}: `points` must be an array of [x, y] points"))?;
            let points = points
                .iter()
                .map(|p| {
                    let xyz: Option<Vec<f64>> = p
                        .as_array()
                        .filter(|a| a.len() == 2 || a.len() == 3)
                        .map(|a| a.iter().filter_map(Value::as_f64).collect());
                    match xyz {
                        Some(v) if v.len() == 2 => Ok([v[0], v[1], 0.0]),
                        Some(v) if v.len() == 3 => Ok([v[0], v[1], v[2]]),
                        _ => Err(format!("rope {i}: a point is [x, y] or [x, y, z], got {p}")),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let closed = match rope.get("closed") {
                None | Some(Value::Null) => false,
                Some(v) => v
                    .as_bool()
                    .ok_or_else(|| format!("rope {i}: `closed` must be a boolean"))?,
            };
            Ok(Rope { points, closed })
        })
        .collect()
}

/// A present count must be a non-negative integer; absent or `null` is `None`.
fn optional_count(params: &Value, key: &str) -> Result<Option<usize>, String> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| format!("`{key}` must be a non-negative integer, got {v}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plait_closes_into_the_figure_eight_then_the_borromean_rings() {
        let out = braid(json!({ "word": "s1 s2^-1", "repeat": 2 })).unwrap();
        assert_eq!(out["word"], "s1 s2^-1 s1 s2^-1");
        assert_eq!(out["components"], 1);
        assert_eq!(out["writhe"], 0);
        assert_eq!(out["jones"]["text"], "t^-2 - t^-1 + 1 - t + t^2");
        assert_eq!(out["jones"]["terms"][0], json!([-2.0, 1]));
        assert_eq!(out["jones_symmetric"], true);
        assert!(out.get("geometry").is_none());

        let out = braid(json!({ "word": "s1 s2^-1", "repeat": 3 })).unwrap();
        assert_eq!(out["components"], 3);
        assert_eq!(out["permutation"], json!([0, 1, 2]));
    }

    #[test]
    fn tells_the_reef_knot_from_the_granny() {
        let reef = braid(json!({ "word": "s1^3 s2^-3" })).unwrap();
        let granny = braid(json!({ "word": "s1^3 s2^3" })).unwrap();
        assert_eq!(reef["jones_symmetric"], true);
        assert_eq!(granny["jones_symmetric"], false);
        assert_eq!(
            braid(json!({ "word": "s1^2" })).unwrap()["jones"]["terms"],
            json!([[0.5, -1], [2.5, -1]])
        );
    }

    #[test]
    fn returns_the_layout_on_request() {
        let out = braid(json!({
            "word": "1 -2",
            "geometry": true,
            "samples_per_crossing": 4
        }))
        .unwrap();
        let strands = out["geometry"]["strands"].as_array().unwrap();
        assert_eq!(strands.len(), 3);
        assert_eq!(strands[0]["points"].as_array().unwrap().len(), 2 * 4 + 1);
        assert_eq!(strands[0]["points"][0], json!([0.0, 0.0, 0.0]));
        // σ₁ first: strand 0 moves from position 0 to 1, in front halfway.
        let mid: Vec<f64> = strands[0]["points"][2]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert!(
            (mid[0] - 0.5).abs() < 1e-12 && mid[1] == 0.5 && mid[2] == 1.0,
            "{mid:?}"
        );
    }

    #[test]
    fn names_the_catalogue_and_draws_its_knots() {
        let list = knot(json!({ "list": true })).unwrap();
        let ids: Vec<&str> = list["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap())
            .collect();
        assert!(
            ids.contains(&"overhand") && ids.contains(&"figure-eight"),
            "{ids:?}"
        );

        let out = knot(json!({ "name": "figure-eight", "geometry": true })).unwrap();
        assert_eq!(out["fr"], "Nœud en huit");
        assert_eq!(out["abok"], 570);
        assert_eq!(out["closure"], "4_1");
        assert_eq!(out["crossings"], 4);
        assert_eq!(out["closure_crossings"], 0);
        assert_eq!(out["jones"]["text"], "t^-2 - t^-1 + 1 - t + t^2");
        assert_eq!(out["geometry"]["ropes"].as_array().unwrap().len(), 1);
        assert!(out["geometry"]["min_clearance"].as_f64().unwrap() >= 1.0);
        let overhand = knot(json!({ "name": "overhand" })).unwrap();
        assert_eq!(overhand["jones"]["text"], "t + t^3 - t^4");
        assert!(overhand.get("geometry").is_none());
    }

    fn circle(cx: f64) -> Value {
        let points: Vec<[f64; 2]> = (0..8)
            .map(|k| {
                let a = std::f64::consts::PI * k as f64 / 4.0;
                [cx + 2.0 * a.cos(), 2.0 * a.sin()]
            })
            .collect();
        json!({ "closed": true, "points": points })
    }

    #[test]
    fn reads_a_drawing_of_your_own() {
        // Two overlapping circles, each reaching the upper crossing first:
        // rope 0 over there and under at the lower one is the Hopf link.
        let ropes = json!([circle(0.0), circle(2.5)]);
        let hopf = knot(json!({ "ropes": ropes, "over": "OU UO" })).unwrap();
        assert_eq!(hopf["components"], 2);
        assert_eq!(hopf["crossings"], 2);
        assert!(hopf.get("id").is_none());
        let text = hopf["jones"]["text"].as_str().unwrap();
        assert!(
            ["-t^(1/2) - t^(5/2)", "-t^(-5/2) - t^(-1/2)"].contains(&text),
            "{text}"
        );
        let unlink =
            knot(json!({ "ropes": ropes, "over": "OOUU", "geometry": true, "radius": 0.2 }))
                .unwrap();
        assert_eq!(unlink["jones"]["text"], "-t^(-1/2) - t^(1/2)");
        assert_eq!(unlink["geometry"]["radius"], 0.2);
    }

    #[test]
    fn draws_a_knot_from_its_gauss_code_alone() {
        let overhand = "U1 O2 U3 O1 U2 O3";
        let right = knot(json!({ "gauss": overhand, "closure": "3_1", "geometry": true })).unwrap();
        assert_eq!(right["jones"]["text"], "t + t^3 - t^4");
        assert_eq!(right["gauss"], overhand);
        assert_eq!(right["closure"], "3_1");
        assert!(right["geometry"]["min_clearance"].as_f64().unwrap() >= 1.0);
        let left = knot(json!({ "gauss": overhand, "closure": "m3_1" })).unwrap();
        assert_eq!(left["jones"]["text"], "-t^-4 + t^-3 + t^-1");

        // The drawing found goes back in as a drawing of your own.
        let again = knot(json!({
            "ropes": right["drawing"]["ropes"],
            "over": right["drawing"]["over"],
        }))
        .unwrap();
        assert_eq!(again["gauss"], overhand);
        assert_eq!(again["jones"], right["jones"]);

        // Two overhands in a row draw the granny or the reef: the code alone
        // is refused, and so is a closure neither is, each time with the
        // polynomials the code does draw.
        let two = "U1 O2 U3 O1 U2 O3 U4 O5 U6 O4 U5 O6";
        let err = knot(json!({ "gauss": two })).unwrap_err();
        assert!(err.contains("V ="), "{err}");
        let err = knot(json!({ "gauss": two, "closure": "3_1" })).unwrap_err();
        assert!(err.contains("V ="), "{err}");
    }

    #[test]
    fn knot_refuses_with_a_reason() {
        let err = |p: Value| knot(p).unwrap_err();
        assert!(err(json!({})).contains("give `name`"));
        assert!(err(json!({ "name": "granny-bend" })).contains("no knot"));
        assert!(err(json!({ "name": 3 })).contains("must be a string"));
        assert!(err(json!({ "name": "overhand", "ropes": [] })).contains("not both"));
        assert!(err(json!({ "name": "overhand", "gauss": "O1 U1" })).contains("not both"));
        assert!(err(json!({ "name": "overhand", "closure": "3_1" })).contains("goes with"));
        assert!(err(json!({ "gauss": "O1 U1", "closure": "7_1" })).contains("no knot \"7_1\""));
        assert!(err(json!({ "gauss": "O1 O2 U1 U2" })).contains("not a knot drawn on paper"));
        assert!(err(json!({ "gauss": "O1 X2" })).contains("X2"));
        assert!(err(json!({ "ropes": [{ "points": [[0, 0], [1]] }] })).contains("[x, y]"));
        assert!(
            err(json!({ "ropes": [{ "points": [[0, 0], [1, 1]], "closed": 1 }] }))
                .contains("`closed` must be a boolean")
        );
        assert!(
            err(json!({ "ropes": [{ "points": [[0, 0], [1, 1]] }], "geometry": true }))
                .contains("`radius` is required")
        );
        let ropes = json!([circle(0.0), circle(2.5)]);
        assert!(err(json!({ "ropes": ropes, "over": "OU" })).contains("2 letters"));
        assert!(
            err(json!({ "ropes": ropes, "over": "OUUO", "geometry": true, "radius": -1 }))
                .contains("positive")
        );
    }

    #[test]
    fn refuses_with_a_reason() {
        let err = |p: Value| braid(p).unwrap_err();
        assert!(err(json!({})).contains("`word` is required"));
        assert!(err(json!({ "word": "s0" })).contains("cannot read"));
        assert!(err(json!({ "word": "s1", "strands": 9 })).contains("1 to 8 strands"));
        assert!(err(json!({ "word": "s1", "repeat": 65 })).contains("at most 64 crossings"));
        assert!(err(json!({ "word": "s1", "repeat": -1 })).contains("non-negative integer"));
        assert!(err(json!({ "word": "s1", "repeat": 0 })).contains("at least 1"));
        assert!(
            err(json!({ "word": "s1", "geometry": true, "samples_per_crossing": 1 }))
                .contains("at least 2")
        );
    }
}
