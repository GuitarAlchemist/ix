//! The L1 loop over tracer A (examples/l1): what it may change, what it may not, and how it
//! chooses. Nothing here calls the chatbot.

#[path = "../examples/l1/plan.rs"]
mod plan;

use ix_pipeline::spec::PipelineSpec;
use serde_json::{json, Value};

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/stream/music-theory-oracle"
);

fn graph() -> Value {
    let spec = PipelineSpec::from_file(format!("{DIR}/ix.yaml")).unwrap();
    spec.stages["tracer"].args["graph"].clone()
}

/// The questions of both splits: the spec's 10 and the held-out 7.
fn questions() -> Vec<Value> {
    let mut all = plan::node(&graph(), "questions").unwrap()["values"]
        .as_array()
        .unwrap()
        .clone();
    let held = std::fs::read_to_string(format!("{DIR}/held-out.json")).unwrap();
    all.extend(serde_json::from_str::<Vec<Value>>(&held).unwrap());
    all
}

#[test]
fn a_run_whose_judge_or_extract_changed_is_refused() {
    let g = graph();
    let want = plan::frozen(&g).unwrap();
    let values = vec![json!({ "id": "x", "prompt": "chords in D?" })];
    let ok = plan::candidate(&g, &want, plan::TEMPLATES[1].1, &values).unwrap();
    // What a run may change: the chatbot's message and the questions, nothing frozen.
    assert_eq!(plan::frozen(&ok).unwrap(), want);
    assert_eq!(
        plan::node(&ok, "chatbot").unwrap()["http"]["body"]["message"],
        plan::TEMPLATES[1].1
    );
    assert_eq!(
        plan::node(&ok, "questions").unwrap()["values"],
        json!(values)
    );

    for name in ["judge", "extract", "report", "ga"] {
        let mut bad = g.clone();
        for n in bad["nodes"].as_array_mut().unwrap() {
            if n["name"] == name {
                match n.get_mut("sql") {
                    Some(sql) => *sql = json!(format!("{} -- edited", sql.as_str().unwrap())),
                    None => n["http"]["timeout_ms"] = json!(1),
                }
            }
        }
        let err = plan::candidate(&bad, &want, plan::TEMPLATES[0].1, &values).unwrap_err();
        assert!(
            err.starts_with("refused: the judge changed"),
            "{name}: {err}"
        );
    }
}

#[test]
fn no_template_holds_a_note_a_chord_an_interval_or_an_expected_value() {
    let mut expected: Vec<String> = Vec::new();
    for q in questions() {
        let e = &q["expect"];
        for c in e["chords"].as_array().into_iter().flatten() {
            expected.push(c.as_str().unwrap().to_string());
        }
        if let Some(n) = e["note"].as_str() {
            expected.push(n.to_string());
        }
    }
    assert!(expected.len() > 20, "{expected:?}");
    for (name, t) in plan::TEMPLATES {
        assert_eq!(plan::refuse(t), None, "{name}");
        let own = t.replace("{prompt}", " ");
        for w in own.split(|c: char| !c.is_alphanumeric() && !"#♯♭".contains(c)) {
            assert!(
                !expected.iter().any(|e| e == w),
                "{name} holds `{w}`, an expected value"
            );
        }
    }
    // The guard itself: each of these is refused.
    for bad in [
        "{prompt} e.g. C major (C-E-G)",
        "{prompt} Answer with a perfect fifth.",
        "{prompt} {expect}",
        "{prompt} {prompt}",
        "no prompt",
        "{prompt} like F#m7b5 or Bb",
    ] {
        assert!(plan::refuse(bad).is_some(), "{bad}");
    }
}

#[test]
fn a_template_that_checks_fewer_claims_or_more_u_cannot_win() {
    let run = |template: &str, run: u32, t: i64, f: i64, u: i64, claims: i64| plan::Run {
        template: template.into(),
        run,
        questions: 10,
        t,
        f,
        u,
        claims,
        verified: claims - f,
    };
    let base = plan::TEMPLATES[0].0;
    let (first, notes, key) = (
        plan::TEMPLATES[2].0,
        plan::TEMPLATES[1].0,
        plan::TEMPLATES[3].0,
    );
    // `first` scores best but checks fewer claims; `notes` raises U; `key` wins.
    let runs = [
        run(base, 1, 5, 4, 1, 30),
        run(base, 2, 5, 4, 1, 30),
        run(first, 1, 9, 1, 0, 20),
        run(first, 2, 9, 1, 0, 20),
        run(notes, 1, 8, 0, 2, 40),
        run(notes, 2, 8, 0, 2, 40),
        run(key, 1, 6, 3, 1, 30),
        run(key, 2, 7, 2, 1, 31),
    ];
    let (scores, chosen) = plan::choose(&runs);
    assert_eq!(chosen, key);
    let s = |name: &str| scores.iter().find(|s| s.template == name).unwrap().clone();
    assert!(!s(first).eligible && s(first).score > s(key).score);
    assert!(!s(notes).eligible);
    assert_eq!((s(key).score, s(key).runs), (8, 2));
    // Precision is reported, not scored: `notes` has every claim verified and still cannot win.
    assert_eq!(s(key).precision, 56.0 / 61.0);
    assert_eq!(s(notes).precision, 1.0);

    // Without a template above the baseline, the baseline stays.
    let (_, chosen) = plan::choose(
        &runs[..2]
            .iter()
            .cloned()
            .chain([run(key, 1, 5, 4, 1, 30)])
            .collect::<Vec<_>>(),
    );
    assert_eq!(chosen, base);
}
