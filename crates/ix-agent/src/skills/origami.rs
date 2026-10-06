//! `ix_origami_check` — the agent-facing surface for `ix-origami`.
//!
//! The tool takes a FOLD object (a crease pattern with one folded frame holding `faceOrders`)
//! and returns every check `ix_origami::analyse` runs on that stated flat folded state:
//! structure, face isometry, crease orientation, Kawasaki and Maekawa, and the four layer
//! rules. With `census`, it also flips each stated pair alone and tallies which rules reject
//! the flip. It is a pure computation: no filesystem, no network, no state. The caller passes
//! the FOLD object itself; reading a `.fold` file from disk is deliberately not offered.
//!
//! The checks test the layer order the file states. They do not compute one, and they do not
//! prove the sheet folds flat: global flat-foldability is NP-complete (Bern and Hayes 1996).

use std::collections::BTreeMap;

use ix_origami::{analyse, swap_census, Context, Fold, Rule};
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

/// The largest fold checked, in faces. The overlay and the layer rules grow with the number of
/// faces and of overlay cells, and a request crossing a process boundary must not be able to
/// tie up the server.
pub const MAX_FACES: usize = 1000;
/// The largest census, in stated pairs: it re-checks the rules once per pair.
pub const MAX_CENSUS_PAIRS: usize = 5000;

fn origami_check_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "fold": {
                "type": "object",
                "description": "A FOLD object (spec 1.2): a crease pattern at the top level and one folded frame in file_frames (frame_parent 0, frame_inherit true) whose faceOrders state the layer order. Rabbit Ear's FOLD 1.1 layout, with the folded state at the top level, is accepted and rearranged; no value changes."
            },
            "census": {
                "type": "boolean",
                "default": false,
                "description": "Also flip each stated faceOrders pair alone and tally which layer rules reject the flip. A rule that rejects no flip cannot fail, and proves nothing."
            }
        },
        "required": ["fold"]
    })
}

fn origami_check_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "ok": { "type": "boolean", "description": "Every check passes. Unstated or undetermined items are reported, not failed." },
            "structure": { "type": "array", "items": { "type": "string" }, "description": "Problems that stop the checks from running (indices, frames, face order, non-convex faces). When not empty, only `ok` comes with it." },
            "counts": { "type": "object", "description": "vertices, edges, faces, assignment letters, faceOrders" },
            "orientation": { "type": "object", "description": "Faces that keep (up) or reverse (down) their crease-pattern orientation in the folded frame" },
            "converse_conflicts": { "type": "array", "description": "Ordered face pairs where a stated converse disagrees with the spec's rule" },
            "isometry": { "type": "object", "description": "Face shapes, crease pattern against folded frame, up to one global scale" },
            "crease_orientation": { "type": "object", "description": "Mountain and valley creases must turn one face over relative to the other" },
            "local_theorems": { "type": "object", "description": "Kawasaki and Maekawa at interior vertices whose creases are all mountains or valleys" },
            "overlap": { "type": "object", "description": "Overlapping face pairs from the overlay, against the stated pairs" },
            "layers": { "type": "object", "description": "adjacency, cells (cycles), taco_tortilla, taco_taco: each with what was checked and the violations" },
            "tacos_on_opposite_sides": { "type": "integer" },
            "rejected_by": { "type": "array", "items": { "type": "string" }, "description": "Layer rules with at least one violation" },
            "census": { "type": "object", "description": "With census: pairs, rejected, the accepted pairs, and a tally keyed '<adjacent|non-adjacent>: <rules>'" }
        }
    })
}

/// Check a stated flat folded state read from a FOLD object.
///
/// Returns the report of `ix_origami::analyse`. A fold whose structure stops the checks is a
/// verdict (`ok: false` with the `structure` problems), not an error; input that is not a FOLD
/// object is an error. Folds above [`MAX_FACES`] faces, and censuses above
/// [`MAX_CENSUS_PAIRS`] pairs, are refused.
#[ix_skill(
    domain = "origami",
    name = "origami.check",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::origami::origami_check_schema",
    output_schema_fn = "crate::skills::origami::origami_check_output_schema"
)]
pub fn origami_check(params: Value) -> Result<Value, String> {
    let value = params
        .get("fold")
        .ok_or("supply `fold`: a FOLD object with one folded frame")?;
    let census = match params.get("census") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or("`census` must be a boolean")?,
    };
    let fold = Fold::from_value(value).map_err(|e| e.to_string())?;
    if fold.faces_vertices.len() > MAX_FACES {
        return Err(format!(
            "the fold has {} faces; this tool checks at most {MAX_FACES}",
            fold.faces_vertices.len()
        ));
    }
    let report = match analyse(&fold) {
        Ok(r) => r,
        Err(problems) => return Ok(json!({ "ok": false, "structure": problems })),
    };
    let mut out = serde_json::to_value(&report).map_err(|e| e.to_string())?;
    out["structure"] = json!([]);
    if census {
        out["census"] = census_tally(&fold)?;
    }
    Ok(out)
}

fn census_tally(fold: &Fold) -> Result<Value, String> {
    let pairs = fold
        .folded()
        .face_orders
        .iter()
        .filter(|t| t[2] != 0)
        .count();
    if pairs > MAX_CENSUS_PAIRS {
        return Err(format!(
            "the census would flip {pairs} pairs; this tool flips at most {MAX_CENSUS_PAIRS}"
        ));
    }
    let ctx = Context::new(fold).map_err(|p| p.join("; "))?;
    let rows = swap_census(fold, &ctx);
    let mut by_rules: BTreeMap<String, usize> = BTreeMap::new();
    for row in &rows {
        let rules: Vec<String> = row.rejected_by.iter().map(|&r| rule_name(r)).collect();
        let kind = if row.adjacent {
            "adjacent"
        } else {
            "non-adjacent"
        };
        let rules = if rules.is_empty() {
            "none".to_string()
        } else {
            rules.join(" + ")
        };
        *by_rules.entry(format!("{kind}: {rules}")).or_insert(0) += 1;
    }
    let accepted: Vec<[usize; 2]> = rows
        .iter()
        .filter(|r| r.rejected_by.is_empty())
        .map(|r| r.pair)
        .collect();
    Ok(json!({
        "pairs": rows.len(),
        "rejected": rows.len() - accepted.len(),
        "accepted": accepted,
        "by_rules": by_rules,
    }))
}

fn rule_name(rule: Rule) -> String {
    serde_json::to_value(rule)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{rule:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit square folded in half along x = 0.5 by one valley crease: the right half
    /// (face 1) lands on top of the left half (face 0), turned over. `s` is the stated order
    /// of face 1 relative to face 0.
    fn square_folded_in_half(s: i64) -> Value {
        json!({
            "file_spec": 1.2,
            "frame_classes": ["creasePattern"],
            "vertices_coords": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [0.5, 1]],
            "edges_vertices": [[0, 4], [4, 1], [1, 2], [2, 5], [5, 3], [3, 0], [4, 5]],
            "edges_assignment": ["B", "B", "B", "B", "B", "B", "V"],
            "faces_vertices": [[0, 4, 5, 3], [4, 1, 2, 5]],
            "file_frames": [{
                "frame_classes": ["foldedForm"],
                "frame_parent": 0,
                "frame_inherit": true,
                "vertices_coords": [[0, 0], [0, 0], [0, 1], [0, 1], [0.5, 0], [0.5, 1]],
                "faceOrders": [[1, 0, s]]
            }]
        })
    }

    #[test]
    fn a_square_folded_in_half_passes_every_check() {
        let out = origami_check(json!({ "fold": square_folded_in_half(1) })).unwrap();
        assert_eq!(out["ok"], true, "{out}");
        assert_eq!(out["structure"], json!([]));
        assert_eq!(out["orientation"], json!({ "up": 1, "down": 1 }));
        assert_eq!(out["overlap"]["overlapping_pairs"], 1);
        assert_eq!(out["overlap"]["stated_and_overlapping"], 1);
        assert_eq!(out["layers"]["adjacency"]["checked"], 1);
        assert_eq!(out["isometry"]["ok"], true);
        assert!(out.get("census").is_none());
    }

    #[test]
    fn the_wrong_layer_order_breaks_the_adjacency_rule() {
        let out = origami_check(json!({ "fold": square_folded_in_half(-1) })).unwrap();
        assert_eq!(out["ok"], false);
        assert_eq!(out["rejected_by"], json!(["adjacency"]));
        assert_eq!(out["layers"]["adjacency"]["violations"], json!([6]));
    }

    #[test]
    fn the_census_shows_the_one_flip_is_rejected() {
        let out =
            origami_check(json!({ "fold": square_folded_in_half(1), "census": true })).unwrap();
        assert_eq!(
            out["census"],
            json!({
                "pairs": 1,
                "rejected": 1,
                "accepted": [],
                "by_rules": { "adjacent: adjacency": 1 }
            })
        );
    }

    #[test]
    fn a_structure_problem_is_a_verdict_not_an_error() {
        let mut fold = square_folded_in_half(1);
        fold["file_frames"][0]["faceOrders"] = json!([[1, 7, 1]]);
        let out = origami_check(json!({ "fold": fold })).unwrap();
        assert_eq!(out["ok"], false);
        let problems = out["structure"].as_array().unwrap();
        assert!(
            problems
                .iter()
                .any(|p| p.as_str().unwrap().contains("out of range")),
            "{out}"
        );
        assert!(out.get("layers").is_none());
    }

    #[test]
    fn input_that_is_not_a_fold_is_an_error() {
        assert!(origami_check(json!({})).is_err());
        assert!(origami_check(json!({ "fold": [1, 2, 3] })).is_err());
        assert!(origami_check(json!({ "fold": { "vertices_coords": [] } })).is_err());
        let err = origami_check(json!({ "fold": square_folded_in_half(1), "census": "yes" }))
            .unwrap_err();
        assert!(err.contains("census"), "{err}");
    }

    #[test]
    fn a_fold_above_the_face_limit_is_refused_before_any_check() {
        let faces = vec![json!([0, 1, 2]); MAX_FACES + 1];
        let fold = json!({
            "file_spec": 1.2,
            "vertices_coords": [[0, 0], [1, 0], [0, 1]],
            "edges_vertices": [[0, 1], [1, 2], [2, 0]],
            "edges_assignment": ["B", "B", "B"],
            "faces_vertices": faces
        });
        let err = origami_check(json!({ "fold": fold })).unwrap_err();
        assert!(err.contains(&format!("at most {MAX_FACES}")), "{err}");
    }
}
