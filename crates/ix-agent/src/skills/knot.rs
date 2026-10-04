//! `ix_braid` — the agent-facing surface for `ix-knot`.
//!
//! A braid word in, the knot or link its closure makes out: the strand
//! permutation, components, writhe and Jones polynomial, plus on request a 3D
//! layout of the strands a renderer can draw (a ComfyUI control image, for one).
//! A pure computation over the caller's word: no filesystem, no network, no
//! state.

use ix_knot::{jones, layout, Braid, MAX_CROSSINGS, MAX_POINTS, MAX_STRANDS};
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
    let terms = v
        .terms()
        .iter()
        .map(|&(half, coeff)| {
            i64::try_from(coeff)
                .map(|c| json!([f64::from(half) / 2.0, c]))
                .map_err(|_| format!("a Jones coefficient ({coeff}) does not fit a JSON integer"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut out = json!({
        "word": braid.to_string(),
        "generators": braid.word(),
        "strands": braid.strands(),
        "crossings": braid.crossings(),
        "writhe": braid.writhe(),
        "permutation": braid.permutation(),
        "components": braid.components(),
        "jones": { "text": v.to_string(), "terms": terms },
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
