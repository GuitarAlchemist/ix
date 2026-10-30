//! `stream.run` as a stage of a real `ix.yaml`: lowered by ix-pipeline, run by its executor.

use ix_pipeline::executor::{execute, NoCache};
use ix_pipeline::lower::lower;
use ix_pipeline::spec::PipelineSpec;
use ix_stream::petri::Shape;
use ix_stream::skill::stream_run;
use ix_stream::{Bus, Graph};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

/// A source, a stage of two workers, and a sink.
fn chain() -> Value {
    json!({
        "nodes": [
            { "name": "src", "source": true },
            { "name": "work", "cap": 2, "workers": 2 },
            { "name": "sink" }
        ],
        "edges": [{ "from": 0, "to": 1 }, { "from": 1, "to": 2 }]
    })
}

/// A source split into two lanes that join, a batching node, and a sink.
fn diamond() -> Value {
    json!({
        "nodes": [
            { "name": "src", "source": true },
            { "name": "a", "cap": 1 },
            { "name": "b", "cap": 3, "workers": 3 },
            { "name": "join", "cap": 2 },
            { "name": "count", "batching": true },
            { "name": "sink" }
        ],
        "edges": [
            { "from": 0, "to": 1 }, { "from": 0, "to": 2 },
            { "from": 1, "to": 3 }, { "from": 2, "to": 3 },
            { "from": 3, "to": 4 }, { "from": 4, "to": 5 }
        ]
    })
}

fn counts(out: &Value) -> Vec<(String, u64, u64)> {
    out["receipts"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            (
                n["name"].as_str().unwrap().to_string(),
                n["received"].as_u64().unwrap(),
                n["emitted"].as_u64().unwrap(),
            )
        })
        .collect()
}

fn row(name: &str, received: u64, emitted: u64) -> (String, u64, u64) {
    (name.to_string(), received, emitted)
}

#[test]
fn a_pipeline_stage_runs_a_graph_and_hands_its_count_to_the_next() {
    let spec: PipelineSpec = serde_json::from_value(json!({
        "version": "1",
        "stages": {
            "first": { "skill": "stream.run", "args": { "graph": chain(), "max_events": 40 } },
            "second": { "skill": "stream.run", "args": {
                "graph": diamond(), "max_events": { "from": "first.events" } } }
        }
    }))
    .unwrap();
    let result = execute(&lower(&spec).unwrap(), &HashMap::new(), &NoCache).unwrap();

    let first = result.output("first").unwrap();
    assert_eq!(first["stopped_by"], "max_events");
    assert_eq!(first["events"], 40);
    assert_eq!(first["steps"], 80);
    assert_eq!(
        counts(first),
        [row("src", 0, 40), row("work", 40, 40), row("sink", 40, 0)]
    );
    assert_eq!(first["receipts"]["petri"]["method"], "acyclic");
    assert_eq!(first["receipts"]["petri"]["deadlock_free"], true);
    assert_eq!(
        first["receipts"]["budget"],
        json!({ "cap": 1_000_000, "spent": 80 })
    );

    // 40 events through the diamond: each lane takes all 40, the join both lanes' 80, and the
    // batching node sends its one count to the sink.
    let second = result.output("second").unwrap();
    assert_eq!(second["events"], 40);
    assert_eq!(
        counts(second),
        [
            row("src", 0, 80),
            row("a", 40, 40),
            row("b", 40, 40),
            row("join", 80, 80),
            row("count", 80, 1),
            row("sink", 1, 0)
        ]
    );
    assert_eq!(second["steps"], 40 + 40 + 80 + 80 + 1);
}

#[test]
fn the_stream_stops_at_max_events() {
    let out = stream_run(json!({ "graph": chain(), "max_events": 7 })).unwrap();
    assert_eq!(out["stopped_by"], "max_events");
    assert_eq!(out["events"], 7);
    assert_eq!(
        counts(&out),
        [row("src", 0, 7), row("work", 7, 7), row("sink", 7, 0)]
    );
}

#[test]
fn the_budget_stops_the_stream_before_it_is_spent() {
    // Each event costs two steps here, so a budget of 9 pays for 4 of the 100 asked.
    let out = stream_run(json!({ "graph": chain(), "max_events": 100, "budget": 9 })).unwrap();
    assert_eq!(out["stopped_by"], "budget");
    assert_eq!(out["events"], 4);
    assert_eq!(out["steps"], 8);
    assert_eq!(out["receipts"]["budget"], json!({ "cap": 9, "spent": 8 }));
}

#[test]
fn the_answer_does_not_depend_on_thread_timing() {
    let args = json!({ "graph": diamond(), "max_events": 500, "budget": 2_000 });
    let first = stream_run(args.clone()).unwrap();
    for _ in 0..5 {
        assert_eq!(stream_run(args.clone()).unwrap(), first);
    }
    // 6 steps an event and 1 for the count: 333 events fit in 2 000.
    assert_eq!(first["events"], 333);
    assert_eq!(first["steps"], 6 * 333 + 1);
}

#[test]
fn graphs_it_cannot_run_deterministically_are_refused() {
    let refused = |graph: Value| stream_run(json!({ "graph": graph })).unwrap_err();
    let cycle = json!({
        "nodes": [{ "name": "src", "source": true }, { "name": "a" }, { "name": "b" }],
        "edges": [{ "from": 0, "to": 1 }, { "from": 1, "to": 2 }, { "from": 2, "to": 1 }]
    });
    assert!(refused(cycle).contains("cycle"));
    let lossy = json!({
        "nodes": [{ "name": "src", "source": true }, { "name": "a" }],
        "edges": [{ "from": 0, "to": 1, "lossy": true }]
    });
    assert!(refused(lossy).contains("lossy"));
    let sourceless = json!({ "nodes": [{ "name": "a" }], "edges": [] });
    assert!(refused(sourceless).contains("no source"));
    let into_source = json!({
        "nodes": [{ "name": "src", "source": true }, { "name": "a" }],
        "edges": [{ "from": 1, "to": 0 }]
    });
    assert!(refused(into_source).contains("a link into a source"));
    let e = stream_run(json!({ "graph": chain(), "max_events": 0 })).unwrap_err();
    assert_eq!(
        e,
        "max_events must be a whole number from 1 to 1000000, got 0"
    );
}

#[test]
fn a_shape_reads_back_what_it_writes() {
    let mut g = Graph::new(&Bus::new());
    let src = g.source("src", |_| {});
    let a = g.stage("a", 3, 2, |_, _| {});
    let b = g.stage("b", 1, 1, |_, _| {});
    g.link(src, a);
    g.link_lossy(src, b);
    g.link(a, b);
    let shape = g.shape();
    assert_eq!(Shape::from_json(&shape.json()).unwrap(), shape);
}

#[test]
fn the_skill_is_registered_with_its_schemas() {
    let d = ix_registry::by_name("stream.run").expect("stream.run is registered");
    assert_eq!(d.domain, "stream");
    assert!((d.json_schema)()["properties"]["graph"].is_object());
    assert!((d.output_schema)()["properties"]["receipts"].is_object());
}

#[test]
fn a_source_with_values_sends_them_and_stops_when_it_has_none_left() {
    let graph = json!({
        "nodes": [
            { "name": "src", "source": true, "values": [{ "q": 1 }, { "q": 2 }, { "q": 3 }] },
            { "name": "work" },
            { "name": "sink" }
        ],
        "edges": [{ "from": 0, "to": 1 }, { "from": 1, "to": 2 }]
    });
    let out = stream_run(json!({ "graph": graph, "max_events": 10 })).unwrap();
    assert_eq!(out["stopped_by"], "sources");
    assert_eq!(out["events"], 3);
    assert_eq!(
        counts(&out),
        [row("src", 0, 3), row("work", 3, 3), row("sink", 3, 0)]
    );
    let out = stream_run(json!({ "graph": graph, "max_events": 2 })).unwrap();
    assert_eq!(out["stopped_by"], "max_events");
    assert_eq!(out["events"], 2);
}

/// A source of one value, an http node calling `url`, and a sink.
fn calling(url: &str) -> Value {
    json!({
        "nodes": [
            { "name": "src", "source": true, "values": [{ "q": 1 }] },
            { "name": "call", "http": { "url": url } },
            { "name": "sink" }
        ],
        "edges": [{ "from": 0, "to": 1 }, { "from": 1, "to": 2 }]
    })
}

#[cfg(not(feature = "world"))]
#[test]
fn world_nodes_need_the_world_feature() {
    let e = stream_run(json!({ "graph": calling("http://127.0.0.1:5232/api") })).unwrap_err();
    assert!(e.contains("`world` feature"), "{e}");
}

#[cfg(feature = "world")]
#[test]
fn an_http_node_off_the_allowlist_is_refused_before_the_run() {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.set_nonblocking(true).unwrap();
    let url = format!("http://{}/api", l.local_addr().unwrap());
    let e = stream_run(json!({ "graph": calling(&url) })).unwrap_err();
    assert!(e.contains("not an allowed origin"), "{e}");
    let e = stream_run(json!({ "graph": calling("http://example.com/api") })).unwrap_err();
    assert!(e.contains("not an allowed origin"), "{e}");
    assert!(l.accept().is_err(), "nothing connected");
}

#[cfg(feature = "world")]
#[test]
#[ignore = "needs the duckdb CLI"]
fn a_sql_node_answers_rows_a_file_sink_writes_and_the_query_reaches_no_file() {
    let dir = std::env::temp_dir().join(format!("ix-stream-world-{}", std::process::id()));
    let graph = |sql: &str| {
        json!({
            "nodes": [
                { "name": "src", "source": true, "values": [{ "k": "a", "n": 1 }, { "k": "b", "n": 2 }, { "k": "a", "n": 3 }] },
                { "name": "sum", "batching": true, "sql": sql },
                { "name": "out", "file": "sums.jsonl" }
            ],
            "edges": [{ "from": 0, "to": 1 }, { "from": 1, "to": 2 }]
        })
    };
    let out = stream_run(json!({
        "graph": graph("SELECT k, sum(n)::BIGINT AS n FROM batch GROUP BY k ORDER BY k"),
        "out_dir": dir.display().to_string()
    }))
    .unwrap();
    assert_eq!(out["receipts"]["faults"], json!([]));
    assert_eq!(out["receipts"]["files"][0]["rows"], 2);
    let text = std::fs::read_to_string(dir.join("sums.jsonl")).unwrap();
    let rows: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        rows,
        [json!({ "k": "a", "n": 4 }), json!({ "k": "b", "n": 2 })]
    );

    let out = stream_run(json!({
        "graph": graph("SELECT * FROM read_text('Cargo.toml')"),
        "out_dir": dir.display().to_string()
    }))
    .unwrap();
    let faults = out["receipts"]["faults"].to_string();
    assert!(faults.contains("disabled by configuration"), "{faults}");
    assert_eq!(out["receipts"]["files"][0]["rows"], 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The rows tracer A wrote (examples/stream/music-theory-oracle/ix.yaml): each question's
/// verdict is an ix-types hexavalent symbol, and the one its claims call for.
#[test]
fn the_tracer_verdicts_follow_from_their_claims() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../state/stream-tracer/2026-10-17-music-theory-oracle/judge.jsonl"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let rows: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let questions: Vec<&Value> = rows.iter().filter(|r| r["row"] == "question").collect();
    assert_eq!(questions.len(), 10);
    for q in questions {
        let symbol = q["verdict"].as_str().unwrap().chars().next().unwrap();
        let verdict = ix_types::Hexavalent::from_char(symbol).unwrap();
        let (claims, verified) = (
            q["claims"].as_u64().unwrap(),
            q["verified"].as_u64().unwrap(),
        );
        let mine = rows
            .iter()
            .filter(|r| r["row"] == "claim" && r["id"] == q["id"]);
        assert_eq!(mine.clone().count() as u64, claims, "{}", q["id"]);
        assert_eq!(
            mine.filter(|r| r["match"] == true).count() as u64,
            verified,
            "{}",
            q["id"]
        );
        let failed = !q["chatbot_error"].is_null() || q["answer"].is_null();
        let expected = if failed || claims == 0 {
            ix_types::Hexavalent::Unknown
        } else if verified == claims {
            ix_types::Hexavalent::True
        } else {
            ix_types::Hexavalent::False
        };
        assert_eq!(verdict, expected, "{}", q["id"]);
    }
}

const TRACER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/stream/music-theory-oracle/ix.yaml"
);
const WITNESS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../state/stream-tracer/2026-10-17-music-theory-oracle/witness"
);

fn lines(path: &str) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// The SQL of a node of tracer A's graph.
fn tracer_sql(node: &str) -> String {
    let spec = PipelineSpec::from_file(TRACER).unwrap();
    let nodes = spec.stages["tracer"].args["graph"]["nodes"].clone();
    let node = nodes.as_array().unwrap().iter().find(|n| n["name"] == node);
    node.unwrap()["sql"].as_str().unwrap().to_string()
}

/// A copy of a stored answer: its edits, applied in order ("" replaces the whole answer), and
/// the verdict the judge must give it.
struct Copy {
    of: &'static str,
    kind: &'static str,
    edits: &'static [(&'static str, &'static str)],
    want: char,
}

/// The judge's verdicts on the stored answers, in their order.
const STORED: &[(&str, char)] = &[
    ("m-94", 'T'),
    ("m-37", 'F'),
    ("m-81", 'T'),
    ("f-1", 'T'),
    ("f-20", 'T'),
    ("f-28", 'T'),
    ("f-12", 'T'),
    ("f-63", 'T'),
    ("f-73", 'T'),
    ("f-75", 'F'),
];

const M37_FIX: (&str, &str) = (
    "D\u{202f}augmented (D+)",
    "B♭ major in first inversion (B♭/D)",
);
const F75_FIX: (&str, &str) = (
    "",
    "From C up to A♭ is a **minor sixth** (m6): 8 semitones. A♭ is the sixth degree of C minor; \
     an augmented fifth would be spelled G♯.",
);

#[rustfmt::skip]
const COPIES: &[Copy] = &[
    // m-94: voicings, checked by their frets.
    Copy { of: "m-94", kind: "wrong note", edits: &[("x-9-x-x-10-10", "x-9-x-x-11-10")], want: 'F' },
    Copy { of: "m-94", kind: "wrong note", edits: &[("**D/A** `5-x-4-x-3-x`", "**D/F#** `5-x-4-x-3-x`")], want: 'F' },
    Copy { of: "m-94", kind: "wrong chord name", edits: &[("**D** `10-12-x-11-10-10`", "**E** `10-12-x-11-10-10`")], want: 'F' },
    Copy { of: "m-94", kind: "neutral", edits: &[("**D/Gb**", "**D/F♯**")], want: 'T' },
    // m-37: D-F-A# named D augmented; corrected, it names the triad GA lists on those notes.
    Copy { of: "m-37", kind: "neutral", edits: &[("D\u{202f}augmented (D+)", "D\u{202f}aug (D+)")], want: 'F' },
    Copy { of: "m-37", kind: "corrected", edits: &[M37_FIX], want: 'T' },
    Copy { of: "m-37", kind: "wrong chord name", edits: &[M37_FIX, ("B♭ major in", "D minor in")], want: 'F' },
    Copy { of: "m-37", kind: "neutral", edits: &[M37_FIX, ("B♭ major in first inversion (B♭/D)", "A♯ major in first inversion (A♯/D)")], want: 'T' },
    // m-81: the triads of D harmonic minor, with their notes.
    Copy { of: "m-81", kind: "wrong note", edits: &[("B♭ major (B♭\u{2011}D\u{2011}F)", "B♭ major (B♭\u{2011}D♯\u{2011}F)")], want: 'F' },
    Copy { of: "m-81", kind: "wrong chord name", edits: &[("G minor (G", "G major (G")], want: 'F' },
    Copy { of: "m-81", kind: "neutral", edits: &[("B♭ major (B♭\u{2011}D\u{2011}F)", "A♯ major (A♯\u{2011}D\u{2011}F)")], want: 'T' },
    Copy { of: "m-81", kind: "neutral", edits: &[("D minor (D", "Dm (D")], want: 'T' },
    // f-1: E A B in D, and its open shapes.
    Copy { of: "f-1", kind: "wrong chord name", edits: &[("**D \u{2013} G \u{2013} A**", "**D \u{2013} G \u{2013} B**")], want: 'F' },
    Copy { of: "f-1", kind: "wrong note", edits: &[("x\u{2011}0\u{2011}2\u{2011}2\u{2011}2\u{2011}0", "x\u{2011}0\u{2011}2\u{2011}2\u{2011}3\u{2011}0")], want: 'F' },
    Copy { of: "f-1", kind: "neutral", edits: &[("**D \u{2013} G \u{2013} A**", "**D, G, A**")], want: 'T' },
    Copy { of: "f-1", kind: "neutral", edits: &[("**D \u{2013} G \u{2013} A**", "**D\u{202f}G\u{202f}A**")], want: 'T' },
    Copy { of: "f-1", kind: "neutral", edits: &[("**D \u{2013} G \u{2013} A**", "**D \u{2192} G \u{2192} A**")], want: 'T' },
    Copy { of: "f-1", kind: "neutral", edits: &[("x\u{2011}0\u{2011}2\u{2011}2\u{2011}2\u{2011}0", "x 0 2 2 2 0")], want: 'T' },
    Copy { of: "f-1", kind: "wrong note", edits: &[("x\u{2011}0\u{2011}2\u{2011}2\u{2011}2\u{2011}0", "x 0 2 2 3 0")], want: 'F' },
    // f-20: G up to E.
    Copy { of: "f-20", kind: "wrong interval", edits: &[("**major sixth (M6)**", "**minor sixth (m6)**")], want: 'F' },
    Copy { of: "f-20", kind: "wrong interval", edits: &[("9 semitones", "8 semitones")], want: 'F' },
    Copy { of: "f-20", kind: "neutral", edits: &[("major sixth (M6)", "major 6th (M6)")], want: 'T' },
    Copy { of: "f-20", kind: "neutral", edits: &[("9 semitones", "9 half steps")], want: 'T' },
    // f-28: the diatonic chords of D major.
    Copy { of: "f-28", kind: "wrong chord name", edits: &[("F♯\u{202f}m", "G\u{202f}m")], want: 'F' },
    Copy { of: "f-28", kind: "wrong chord name", edits: &[("C♯\u{202f}dim", "C♯\u{202f}m")], want: 'F' },
    Copy { of: "f-28", kind: "neutral", edits: &[("D\u{202f}maj", "D major")], want: 'T' },
    Copy { of: "f-28", kind: "neutral", edits: &[("F♯\u{202f}m", "G♭\u{202f}m")], want: 'T' },
    // f-12: the notes E and C share.
    Copy { of: "f-12", kind: "wrong note", edits: &[("The note **E** is common", "The note **G** is common")], want: 'F' },
    Copy { of: "f-12", kind: "wrong note", edits: &[("E \u{2013} G♯ \u{2013} B", "E \u{2013} G \u{2013} B")], want: 'F' },
    Copy { of: "f-12", kind: "wrong chord name", edits: &[("**C\u{202f}maj**", "**C\u{202f}min**")], want: 'F' },
    Copy { of: "f-12", kind: "neutral", edits: &[("E \u{2013} G♯ \u{2013} B", "E, G♯, B")], want: 'T' },
    // f-63: the notes of Bb.
    Copy { of: "f-63", kind: "wrong note", edits: &[("Bb, D, and F", "Bb, D#, and F")], want: 'F' },
    Copy { of: "f-63", kind: "wrong chord name", edits: &[("Bb major chord", "B major chord")], want: 'F' },
    Copy { of: "f-63", kind: "neutral", edits: &[("Bb major chord contains Bb, D, and F.", "The B♭ major triad is made of B♭, D and F.")], want: 'T' },
    // f-73: Am F C G up a whole step.
    Copy { of: "f-73", kind: "wrong chord name", edits: &[("**Bm \u{2013} G \u{2013} D \u{2013} A**", "**Bm \u{2013} G \u{2013} D \u{2013} E**")], want: 'F' },
    Copy { of: "f-73", kind: "wrong chord name", edits: &[("**Bm \u{2013} G", "**B \u{2013} G")], want: 'F' },
    Copy { of: "f-73", kind: "neutral", edits: &[("**Bm \u{2013} G \u{2013} D \u{2013} A**", "**Bm, G, D, A**")], want: 'T' },
    // f-75: C up to Ab, answered with the A minor scale; corrected, it names the interval.
    Copy { of: "f-75", kind: "neutral", edits: &[("The A minor scale has 7 notes", "The A natural minor scale has seven notes")], want: 'F' },
    Copy { of: "f-75", kind: "corrected", edits: &[F75_FIX], want: 'T' },
    Copy { of: "f-75", kind: "wrong interval", edits: &[F75_FIX, ("**minor sixth**", "**augmented fifth**")], want: 'F' },
    Copy { of: "f-75", kind: "wrong interval", edits: &[F75_FIX, ("8 semitones", "9 semitones")], want: 'F' },
    Copy { of: "f-75", kind: "neutral", edits: &[F75_FIX, ("**minor sixth**", "**minor 6th**")], want: 'T' },
];

/// The witness of tracer A's judge, offline: the 10 answers it judged on 2026-10-17
/// (witness/answers.jsonl), copies with one wrong note, interval or chord name, which it must
/// call F, and neutral rewordings, which must keep their verdict. A mutant is made from a T
/// answer, so the two F ones first get a corrected copy, which must be T. The extract and judge
/// SQL are the spec's; GaApi's answers are recorded (witness/ga.jsonl, and those of the answers'
/// own requests), and a request without one fails the test.
#[test]
#[ignore = "needs the duckdb CLI"]
fn the_tracer_judge_calls_every_mutant_false_and_keeps_its_verdict_on_rewordings() {
    let answers = lines(&format!("{WITNESS}/answers.jsonl"));
    let mut recorded: HashMap<String, Value> = HashMap::new();
    for a in &answers {
        let (paths, res) = (a["ga"].as_array().unwrap(), a["ga_res"].as_array().unwrap());
        for (p, r) in paths.iter().zip(res) {
            recorded.insert(p.as_str().unwrap().to_string(), r.clone());
        }
    }
    for r in lines(&format!("{WITNESS}/ga.jsonl")) {
        recorded.insert(r["path"].as_str().unwrap().to_string(), r["res"].clone());
    }
    let stored: HashMap<&str, char> = STORED.iter().copied().collect();
    // The verdict a copy's edits start from: the stored answer's, or the corrected copy's.
    let base = |c: &Copy| match &c.edits[..c.edits.len() - 1] {
        [] => stored[c.of],
        prefix => {
            COPIES
                .iter()
                .find(|b| b.of == c.of && b.edits == prefix)
                .unwrap()
                .want
        }
    };

    // What the chatbot node hands on: the rows without GaApi's answers.
    let mut rows: Vec<Value> = answers.clone();
    for r in &mut rows {
        r.as_object_mut().unwrap().remove("ga_res");
    }
    for (n, c) in COPIES.iter().enumerate() {
        let mut row = rows.iter().find(|r| r["id"] == c.of).unwrap().clone();
        let mut text = row["chatbot"]["body"]["answer"]
            .as_str()
            .unwrap()
            .to_string();
        for (find, put) in c.edits {
            if find.is_empty() {
                text = put.to_string();
                continue;
            }
            assert!(text.contains(find), "{}#{n}: {find:?} is not in it", c.of);
            text = text.replacen(find, put, 1);
        }
        row["chatbot"]["body"]["answer"] = json!(text);
        row["id"] = json!(format!("{}#{n}", c.of));
        rows.push(row);
    }

    let dir = std::env::temp_dir().join(format!("ix-stream-witness-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let extract = tracer_sql("extract");
    let mut rows =
        ix_stream::sql::query_sealed(&rows, &extract, &dir.join("extract.jsonl")).unwrap();
    for r in &mut rows {
        let res: Vec<Value> = r["ga"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let got = recorded.get(p.as_str().unwrap());
                got.unwrap_or_else(|| panic!("{}: no recorded GaApi answer for {p}", r["id"]))
                    .clone()
            })
            .collect();
        r["ga_res"] = json!(res);
    }
    let out = ix_stream::sql::query_sealed(&rows, &tracer_sql("judge"), &dir.join("judge.jsonl"));
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.unwrap();
    let verdict: HashMap<&str, char> = out
        .iter()
        .filter(|r| r["row"] == "question")
        .map(|r| {
            let v = r["verdict"].as_str().unwrap().chars().next().unwrap();
            (r["id"].as_str().unwrap(), v)
        })
        .collect();

    let mine: Vec<(&str, char)> = answers
        .iter()
        .map(|a| a["id"].as_str().unwrap())
        .map(|id| (id, verdict[id]))
        .collect();
    assert_eq!(mine, STORED);
    let claims = out
        .iter()
        .filter(|r| r["row"] == "claim" && !r["id"].as_str().unwrap().contains('#'));
    eprintln!(
        "stored answers: {} claims, {} verified",
        claims.clone().count(),
        claims.filter(|r| r["match"] == true).count()
    );

    let mut tally: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut missed = Vec::new();
    for (n, c) in COPIES.iter().enumerate() {
        match c.kind {
            "neutral" => assert_eq!(c.want, base(c), "{}#{n}", c.of),
            "corrected" => assert_eq!(c.want, 'T', "{}#{n}", c.of),
            _ => assert!(c.want == 'F' && base(c) == 'T', "{}#{n}", c.of),
        }
        let got = verdict[format!("{}#{n}", c.of).as_str()];
        let t = tally.entry(c.kind).or_default();
        t.1 += 1;
        if got == c.want {
            t.0 += 1;
        } else {
            missed.push(format!("{}#{n} ({}): {got}, not {}", c.of, c.kind, c.want));
        }
    }
    eprintln!("as wanted, of: {tally:?}");
    assert!(missed.is_empty(), "{missed:#?}");
}
