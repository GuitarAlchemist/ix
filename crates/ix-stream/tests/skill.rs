//! `stream.run` as a stage of a real `ix.yaml`: lowered by ix-pipeline, run by its executor.

use ix_pipeline::executor::{execute, NoCache};
use ix_pipeline::lower::lower;
use ix_pipeline::spec::PipelineSpec;
use ix_stream::petri::Shape;
use ix_stream::skill::stream_run;
use ix_stream::{Bus, Graph};
use serde_json::{json, Value};
use std::collections::HashMap;

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
