//! `ix_origami_check` — the agent-facing surface for `ix-origami`.
//!
//! The tool takes a FOLD object (a crease pattern with one folded frame holding `faceOrders`)
//! and returns every check `ix_origami::analyse` runs on that stated flat folded state:
//! structure, face isometry, crease orientation, Kawasaki and Maekawa, and the five layer
//! rules. With `census`, it also flips each stated pair alone and tallies which rules reject
//! the flip. With `thickness`, it also reports the stack that thickness implies (see
//! `ix_origami::thickness`). It is a pure computation: no filesystem, no network, no state. The
//! caller passes the FOLD object itself; reading a `.fold` file from disk is deliberately not
//! offered.
//!
//! The checks test the layer order the file states. They do not compute one, and they do not
//! prove the sheet folds flat: global flat-foldability is NP-complete (Bern and Hayes 1996).

use std::collections::BTreeMap;

use ix_origami::{
    analyse_within, stack, swap_census_within, Context, Fold, Limits, Rule, Unchecked,
};
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

/// The most work one request may ask for. The overlay, the tortillas and tacos, and the census
/// grow faster than the request, and a request crossing a process boundary must not be able to
/// tie up the server or run it out of memory. The crane (59 faces, 216 face-vertex incidences,
/// 83 cells, 16580 cell face pairs, 1245 tortillas and tacos, about 15 million census steps)
/// is checked with room to spare: 17 times its faces, 18 times its incidences.
pub const LIMITS: Limits = Limits {
    faces: 1_000,
    face_vertices: 4_000,
    cells: 20_000,
    cell_pairs: 1_000_000,
    tacos: 100_000,
    census_steps: 200_000_000,
};

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
            },
            "thickness": {
                "type": "object",
                "properties": {
                    "t": { "type": "number", "minimum": 0, "description": "The sheet's full thickness, in the crease pattern's units." }
                },
                "required": ["t"],
                "description": "Also report the stack this thickness implies: the most faces at one point, and whether parallel rigid panels, one height each, can realise the stated order. Never changes the top-level verdict."
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
            "layers": { "type": "object", "description": "adjacency, cells (cycles), taco_tortilla, taco_taco, tortilla_tortilla: each with what was checked and the violations" },
            "tacos_on_opposite_sides": { "type": "integer" },
            "rejected_by": { "type": "array", "items": { "type": "string" }, "description": "Layer rules with at least one violation" },
            "census": { "type": "object", "description": "With census: pairs, rejected, the accepted pairs, and a tally keyed '<adjacent|non-adjacent>: <rules>'" },
            "thickness": { "type": "object", "description": "With thickness: t, ply (most faces at one point) and ply_height (the least height of any stack), and rigid: parallel rigid panels (faces joined by flat joints), with acyclic, cycle (stated [below, above] pairs), levels, levels_exact, height, face_levels, undetermined_pairs, refused (t > 0 and a cycle) and ok (the top-level ok and not refused). A refusal means no stack of parallel panels, not that the fold cannot be made." }
        }
    })
}

/// Check a stated flat folded state read from a FOLD object.
///
/// Returns the report of `ix_origami::analyse`. A fold whose structure stops the checks is a
/// verdict (`ok: false` with the `structure` problems), not an error; input that is not a FOLD
/// object is an error, and so is a fold or census over [`LIMITS`]: refused, never checked in
/// part.
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
    let thickness = match params.get("thickness") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.get("t")
                .and_then(Value::as_f64)
                .filter(|t| *t >= 0.0)
                .ok_or("`thickness.t` must be a number, 0 or more: the sheet's thickness")?,
        ),
    };
    let fold = Fold::from_value(value).map_err(|e| e.to_string())?;
    let report = match analyse_within(&fold, &LIMITS) {
        Ok(r) => r,
        Err(Unchecked::Structure(problems)) => {
            return Ok(json!({ "ok": false, "structure": problems }))
        }
        Err(Unchecked::OverLimit(e)) => return Err(refused(&e)),
    };
    let mut out = serde_json::to_value(&report).map_err(|e| e.to_string())?;
    out["structure"] = json!([]);
    if census {
        out["census"] = census_tally(&fold)?;
    }
    if let Some(t) = thickness {
        let ctx = Context::within(&fold, &LIMITS).map_err(|e| refused(&e))?;
        let stack = stack(&ctx, t).map_err(|e| e.to_string())?;
        let mut v = serde_json::to_value(&stack).map_err(|e| e.to_string())?;
        v["rigid"]["ok"] = json!(report.ok && !stack.rigid.refused);
        out["thickness"] = v;
    }
    Ok(out)
}

fn refused(e: &dyn std::fmt::Display) -> String {
    format!("refused, not checked: {e} (the limits of this tool)")
}

fn census_tally(fold: &Fold) -> Result<Value, String> {
    let ctx = Context::within(fold, &LIMITS).map_err(|e| refused(&e))?;
    let rows = swap_census_within(fold, &ctx, &LIMITS).map_err(|e| refused(&e))?;
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
        assert!(out.get("thickness").is_none());
    }

    #[test]
    fn a_thickness_reports_the_stack_it_implies() {
        let out = origami_check(json!({
            "fold": square_folded_in_half(1),
            "thickness": { "t": 0.5 }
        }))
        .unwrap();
        assert_eq!(out["ok"], true);
        assert_eq!(
            out["thickness"],
            json!({
                "t": 0.5,
                "ply": 2,
                "ply_height": 1.0,
                "rigid": {
                    "acyclic": true,
                    "cycle": null,
                    "levels": 2,
                    "levels_exact": true,
                    "height": 1.0,
                    "face_levels": [0, 1],
                    "undetermined_pairs": 0,
                    "refused": false,
                    "ok": true
                }
            })
        );
    }

    #[test]
    fn a_fold_the_checks_reject_is_never_reported_as_stackable() {
        let out = origami_check(json!({
            "fold": square_folded_in_half(-1),
            "thickness": { "t": 0.5 }
        }))
        .unwrap();
        assert_eq!(out["ok"], false);
        assert_eq!(out["thickness"]["rigid"]["refused"], false);
        assert_eq!(out["thickness"]["rigid"]["ok"], false);
    }

    #[test]
    fn a_thickness_that_is_not_a_number_0_or_more_is_an_error() {
        for thickness in [
            json!({ "t": -1 }),
            json!({ "t": "thin" }),
            json!({}),
            json!(0.5),
            // Two layers: 2 · t overflows.
            json!({ "t": f64::MAX }),
        ] {
            let err = origami_check(json!({
                "fold": square_folded_in_half(1),
                "thickness": thickness
            }))
            .unwrap_err();
            assert!(err.contains("thickness.t"), "{thickness}: {err}");
        }
        // Checked before the fold: bad input is an error even when the fold is no verdict.
        let mut fold = square_folded_in_half(1);
        fold["file_frames"][0]["faceOrders"] = json!([[1, 7, 1]]);
        let err = origami_check(json!({ "fold": fold, "thickness": { "t": -1 } })).unwrap_err();
        assert!(err.contains("thickness.t"), "{err}");
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
        let faces = vec![json!([0, 1, 2]); LIMITS.faces + 1];
        let fold = json!({
            "file_spec": 1.2,
            "vertices_coords": [[0, 0], [1, 0], [0, 1]],
            "edges_vertices": [[0, 1], [1, 2], [2, 0]],
            "edges_assignment": ["B", "B", "B"],
            "faces_vertices": faces
        });
        let err = origami_check(json!({ "fold": fold })).unwrap_err();
        assert!(
            err.contains(&format!("faces: more than {}", LIMITS.faces)),
            "{err}"
        );
    }

    /// One convex face with as many vertices as the limit allows plus one, on a circle: under
    /// the face limit, but its vertex pairs grow as the square.
    #[test]
    fn one_face_with_too_many_vertices_is_refused_before_any_check() {
        let n = LIMITS.face_vertices + 1;
        let circle: Vec<Value> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                json!([a.cos(), a.sin()])
            })
            .collect();
        let fold = json!({
            "file_spec": 1.2,
            "vertices_coords": circle,
            "edges_vertices": (0..n).map(|i| json!([i, (i + 1) % n])).collect::<Vec<_>>(),
            "edges_assignment": vec!["B"; n],
            "faces_vertices": [(0..n).collect::<Vec<_>>()],
            "file_frames": [{
                "frame_classes": ["foldedForm"],
                "frame_parent": 0,
                "frame_inherit": true,
                "vertices_coords": circle
            }]
        });
        let err = origami_check(json!({ "fold": fold })).unwrap_err();
        assert!(err.contains("face-vertex incidences"), "{err}");
    }

    /// The crane, with its census, fits well inside the limits.
    #[test]
    fn the_crane_and_its_census_fit_inside_the_limits() {
        let text = include_str!("../../../ix-origami/tests/fixtures/crane-f325f3fd8a.fold");
        let fold: Value = serde_json::from_str(text).unwrap();
        let out = origami_check(json!({ "fold": fold, "census": true })).unwrap();
        assert_eq!(out["ok"], true, "{}", out["structure"]);
        assert_eq!(out["census"]["pairs"], 838);
        assert_eq!(out["census"]["rejected"], 838);
    }
}
