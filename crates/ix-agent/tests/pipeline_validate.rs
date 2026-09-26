//! `ix_node_catalog` + `ix_pipeline_validate` — the node catalog and the
//! offline validator for `ix_pipeline_run` specs, exercised through the same
//! `call_with_ctx` entry point `main.rs` uses for `tools/call`.

use ix_agent::registry_bridge::shared_loop_detector;
use ix_agent::server_context::ServerContext;
use ix_agent::tools::{
    ToolRegistry, CONTEXT_ROUTED_TOOLS, MAX_PIPELINE_STEPS, MAX_STEP_REFERENCES,
};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

/// Since ix#350 these manual tools go through the approval gate, loop
/// detector included, and this binary calls `ix_pipeline_validate` far more
/// than 10 times. Each call starts from a clear slot for its tool; the lock
/// keeps parallel tests from refilling the window between the clear and the
/// call.
fn call(name: &str, args: Value) -> Result<Value, String> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    reset_loop_budget(&[name]);
    let (ctx, _rx) = ServerContext::new();
    ToolRegistry::new().call_with_ctx(name, args, &ctx)
}

fn validate(spec: Value) -> Value {
    call("ix_pipeline_validate", spec).expect("ix_pipeline_validate never fails the call")
}

fn catalog_nodes() -> Vec<Value> {
    let catalog = call("ix_node_catalog", json!({})).expect("ix_node_catalog");
    catalog["nodes"].as_array().expect("nodes array").clone()
}

fn node<'a>(nodes: &'a [Value], name: &str) -> &'a Value {
    nodes
        .iter()
        .find(|n| n["name"] == name)
        .unwrap_or_else(|| panic!("{name} in catalog"))
}

fn error_codes(report: &Value) -> Vec<(String, Value)> {
    report["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .map(|e| (e["code"].as_str().unwrap().to_string(), e["step"].clone()))
        .collect()
}

/// Clear the tools' slots in the process-wide loop-detector window before a
/// test actually runs a pipeline. The window is shared by every test in this
/// binary (10 calls / 5 min per tool), so without this the last tests to run
/// would start with someone else's budget already spent.
fn reset_loop_budget(tools: &[&str]) {
    for tool in tools {
        shared_loop_detector().clear_key(tool);
    }
}

fn stats_step(id: &str) -> Value {
    json!({ "id": id, "tool": "ix_stats", "arguments": { "data": [1.0] } })
}

#[test]
fn catalog_lists_every_tool_with_schema_tier_and_gating() {
    let catalog = call("ix_node_catalog", json!({})).unwrap();
    let nodes = catalog["nodes"].as_array().expect("nodes array");
    assert_eq!(catalog["count"], nodes.len());
    assert_eq!(nodes.len(), ToolRegistry::new().tool_names().count());

    let stats = node(nodes, "ix_stats");
    assert_eq!(stats["dispatch"], "registry");
    assert_eq!(stats["gated"], true);
    assert_eq!(stats["input_schema"]["type"], "object");
    assert_eq!(stats["required_inputs"], json!(["data"]));
    assert_eq!(stats["approval"]["action_kind"], "read");
    assert_eq!(stats["approval"]["tier"], "tier_one");
    assert_eq!(stats["approval"]["effect"], "auto_approved");

    let cache = node(nodes, "ix_cache");
    assert_eq!(cache["approval"]["tier"], "tier_two");
    assert_eq!(cache["approval"]["effect"], "auto_approved");

    let own = node(nodes, "ix_node_catalog");
    assert_eq!(own["dispatch"], "manual");
    assert_eq!(own["output_schema"], Value::Null);
    assert_eq!(own["approval"]["tier"], "tier_one");

    // Every node's effect follows from gated + tier: a tier_three tool is
    // never reported as merely "requiring approval".
    for n in nodes {
        let expected = match (
            n["gated"].as_bool().unwrap(),
            n["approval"]["tier"].as_str().unwrap(),
        ) {
            (false, _) => "not_gated",
            (true, "tier_three") => "blocked",
            (true, _) => "auto_approved",
        };
        assert_eq!(n["approval"]["effect"], expected, "{}", n["name"]);
    }
}

/// `gated` must describe the real dispatch path. A call that passes through the
/// middleware chain is counted by the shared loop detector; one that does not is
/// not. When manual tools start going through the gate (ix#352), the manual
/// probe starts being counted and this test fails until `gated` is updated.
#[test]
fn catalog_gated_flag_matches_real_dispatch() {
    let nodes = catalog_nodes();
    let detector = shared_loop_detector();
    let reg = ToolRegistry::new();

    // Neither tool is called anywhere else in this test binary, so no
    // concurrent test can reset the counter between the two reads.
    let probes = [
        (
            "ix_distance",
            json!({ "a": [0.0], "b": [1.0], "metric": "euclidean" }),
        ),
        (
            "ix_pipeline_list",
            json!({ "root": "no/such/dir/for/gating/probe" }),
        ),
    ];
    for (tool, args) in probes {
        let before = detector.count(tool);
        let _ = reg.call(tool, args);
        let counted = detector.count(tool) > before;
        assert_eq!(
            node(&nodes, tool)["gated"],
            counted,
            "{tool}: catalog 'gated' disagrees with the dispatch path"
        );
    }
}

#[test]
fn valid_chained_pipeline_passes_with_order_and_tier() {
    let report = validate(json!({
        "steps": [
            { "id": "a", "tool": "ix_stats", "arguments": { "data": [1.0, 2.0, 3.0] } },
            {
                "id": "b",
                "tool": "ix_cache",
                "depends_on": ["a"],
                "arguments": { "operation": "set", "key": "k", "value": "$a.mean" }
            }
        ]
    }));
    assert_eq!(report["valid"], true, "{report:#}");
    assert!(
        report["warnings"].as_array().unwrap().is_empty(),
        "{report:#}"
    );
    assert_eq!(report["execution_order"], json!(["a", "b"]));
    assert_eq!(report["approval"]["max_gated_tier"], "tier_two");
    assert_eq!(report["approval"]["verdict"], "auto_approved");
    assert_eq!(report["approval"]["gated_steps"], 2);
    assert_eq!(report["approval"]["ungated_steps"], json!([]));
    assert_eq!(report["steps"][0]["tier"], "tier_one");
    assert_eq!(report["steps"][0]["gated"], true);
    assert_eq!(report["steps"][1]["effect"], "auto_approved");
}

/// Manual tools went through no gate before ix#352 and were reported as
/// `ungated_steps`; they are now classified like every other step.
#[test]
fn manual_tool_steps_are_gated() {
    let report = validate(json!({
        "steps": [{ "id": "cat", "tool": "ix_node_catalog", "arguments": {} }]
    }));
    assert_eq!(report["valid"], true, "{report:#}");
    assert_eq!(report["approval"]["ungated_steps"], json!([]));
    assert_eq!(report["approval"]["gated_steps"], 1);
    assert_eq!(report["approval"]["max_gated_tier"], "tier_one");
    assert_eq!(report["approval"]["verdict"], "auto_approved");
    assert_eq!(report["steps"][0]["effect"], "auto_approved");
    assert!(
        report["warnings"].as_array().unwrap().is_empty(),
        "{report:#}"
    );
}

/// Vacuous while no gated tool is Tier 3 (parity.rs forbids unclassified
/// registry tools); becomes live once manual Tier-3 tools are gated (ix#352).
#[test]
fn gated_tier_three_step_is_an_error() {
    let nodes = catalog_nodes();
    for n in nodes
        .iter()
        .filter(|n| n["approval"]["effect"] == "blocked")
    {
        let report = validate(json!({
            "steps": [{ "id": "s", "tool": n["name"], "arguments": {} }]
        }));
        assert!(
            error_codes(&report)
                .iter()
                .any(|(c, _)| c == "blocked_by_approval_gate"),
            "{report:#}"
        );
    }
}

#[test]
fn execution_order_is_deterministic_and_follows_step_order() {
    let spec = json!({
        "steps": [stats_step("d"), stats_step("b"), stats_step("c"), stats_step("a")]
    });
    for _ in 0..20 {
        assert_eq!(
            validate(spec.clone())["execution_order"],
            json!(["d", "b", "c", "a"])
        );
    }
}

#[test]
fn execution_order_matches_what_pipeline_run_executes() {
    let spec = json!({
        "steps": [
            stats_step("z"),
            { "id": "y", "tool": "ix_stats", "arguments": { "data": [2.0] }, "depends_on": ["x"] },
            stats_step("x"),
        ]
    });
    let order = validate(spec.clone())["execution_order"].clone();
    reset_loop_budget(&["ix_stats"]);
    let run = call("ix_pipeline_run", spec).expect("pipeline runs");
    assert_eq!(order, run["execution_order"]);
    assert_eq!(order, json!(["z", "x", "y"]));
}

#[test]
fn cycle_is_reported_on_the_cycle_steps_only() {
    let report = validate(json!({
        "steps": [
            { "id": "a", "tool": "ix_stats", "arguments": { "data": [1.0] }, "depends_on": ["b"] },
            { "id": "b", "tool": "ix_stats", "arguments": { "data": [1.0] }, "depends_on": ["a"] },
            { "id": "c", "tool": "ix_stats", "arguments": { "data": [1.0] }, "depends_on": ["b"] }
        ]
    }));
    assert_eq!(report["valid"], false);
    assert_eq!(
        error_codes(&report),
        vec![
            ("cycle".to_string(), json!("a")),
            ("cycle".to_string(), json!("b"))
        ],
        "{report:#}"
    );
    assert_eq!(report["execution_order"], Value::Null);
}

#[test]
fn unknown_tool_is_reported_against_its_step() {
    let report = validate(json!({
        "steps": [{ "id": "x", "tool": "ix_does_not_exist", "arguments": {} }]
    }));
    assert_eq!(report["valid"], false);
    assert_eq!(
        error_codes(&report),
        vec![("unknown_tool".to_string(), json!("x"))]
    );
    assert_eq!(report["errors"][0]["index"], 0);
    // An unknown tool has no tier to contribute, and no step was classified.
    assert_eq!(report["approval"]["max_gated_tier"], Value::Null);
    assert_eq!(report["approval"]["verdict"], "unknown");
}

#[test]
fn missing_required_input_is_reported() {
    let report = validate(json!({
        "steps": [{ "id": "s", "tool": "ix_stats", "arguments": {} }]
    }));
    assert_eq!(report["valid"], false);
    let errors = report["errors"].as_array().unwrap();
    assert_eq!(errors.len(), 1, "{report:#}");
    assert_eq!(errors[0]["code"], "missing_required_input");
    assert_eq!(errors[0]["step"], "s");
    assert!(errors[0]["message"].as_str().unwrap().contains("'data'"));
}

#[test]
fn references_to_undefined_steps_are_reported() {
    let report = validate(json!({
        "steps": [{
            "id": "s",
            "tool": "ix_stats",
            "depends_on": ["ghost"],
            "arguments": { "data": "$phantom.values" }
        }]
    }));
    assert_eq!(
        error_codes(&report),
        vec![
            ("unknown_step_reference".to_string(), json!("s")),
            ("unknown_step_reference".to_string(), json!("s")),
        ],
        "{report:#}"
    );
}

/// `ix_pipeline_run` substitutes a step's arguments before the step runs, so
/// reading its own output fails on every run.
#[test]
fn self_reference_is_an_error_and_really_fails_at_run_time() {
    let spec = json!({
        "steps": [{ "id": "a", "tool": "ix_stats", "arguments": { "data": "$a.values" } }]
    });
    let report = validate(spec.clone());
    assert_eq!(
        error_codes(&report),
        vec![("self_reference".to_string(), json!("a"))]
    );
    reset_loop_budget(&["ix_stats"]);
    let err = call("ix_pipeline_run", spec).expect_err("run must fail");
    assert!(err.contains("has no result yet"), "{err}");
}

/// A `$ref` whose target runs *after* the referring step can never resolve.
/// The rule is the runner's: the target must precede in the execution order.
#[test]
fn reference_to_a_step_that_runs_later_is_an_error() {
    let spec = json!({
        "steps": [
            { "id": "b", "tool": "ix_stats", "arguments": { "data": "$a.values" } },
            stats_step("a"),
        ]
    });
    let report = validate(spec.clone());
    assert_eq!(report["valid"], false);
    assert_eq!(
        error_codes(&report),
        vec![("undeclared_dependency".to_string(), json!("b"))],
        "{report:#}"
    );
    reset_loop_budget(&["ix_stats"]);
    let err = call("ix_pipeline_run", spec).expect_err("run must fail");
    assert!(err.contains("has no result yet"), "{err}");
}

/// The mirror case: the target runs first, so `ix_pipeline_run` resolves the
/// reference and the spec is valid. Nothing declares the order though, so it
/// gets an `order_dependent_reference` warning rather than an error.
#[test]
fn reference_to_a_step_that_runs_earlier_is_valid_and_warns() {
    let spec = json!({
        "steps": [
            stats_step("a"),
            { "id": "b", "tool": "ix_cache",
              "arguments": { "operation": "set", "key": "order_dep_probe", "value": "$a.count" } },
        ]
    });
    let report = validate(spec.clone());
    assert_eq!(report["valid"], true, "{report:#}");
    assert_eq!(report["warnings"][0]["code"], "order_dependent_reference");
    assert_eq!(report["warnings"][0]["step"], "b");
    reset_loop_budget(&["ix_stats", "ix_cache"]);
    let run = call("ix_pipeline_run", spec).expect("the runner resolves it");
    assert_eq!(run["execution_order"], json!(["a", "b"]));
    assert_eq!(run["results"]["b"]["ok"], true);
}

/// A declared dependency resolves the reference with no warning at all.
#[test]
fn declared_dependency_reference_is_clean() {
    let report = validate(json!({
        "steps": [
            stats_step("a"),
            { "id": "b", "tool": "ix_stats", "depends_on": ["a"], "arguments": { "data": "$a.values" } },
        ]
    }));
    assert_eq!(report["valid"], true, "{report:#}");
    let unrelated: Vec<&Value> = report["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|w| w["code"] == "order_dependent_reference")
        .collect();
    assert!(unrelated.is_empty(), "{report:#}");
}

#[test]
fn transitive_upstream_reference_is_accepted() {
    let report = validate(json!({
        "steps": [
            { "id": "c", "tool": "ix_stats", "depends_on": ["b"], "arguments": { "data": "$a.values" } },
            { "id": "b", "tool": "ix_stats", "depends_on": ["a"], "arguments": { "data": [1.0] } },
            stats_step("a"),
        ]
    }));
    assert_eq!(report["valid"], true, "{report:#}");
    assert_eq!(report["execution_order"], json!(["a", "b", "c"]));
}

#[test]
fn context_routed_tools_are_rejected_as_steps_and_really_fail_there() {
    for tool in CONTEXT_ROUTED_TOOLS {
        let spec = json!({ "steps": [{ "id": "nested", "tool": tool, "arguments": {} }] });
        let report = validate(spec.clone());
        assert!(
            error_codes(&report)
                .iter()
                .any(|(c, _)| c == "unsupported_in_pipeline"),
            "{tool}: {report:#}"
        );
        reset_loop_budget(&["ix_pipeline_run", tool]);
        let err = call("ix_pipeline_run", spec).expect_err("nested step must fail");
        assert!(err.contains("'nested'"), "{tool}: {err}");
    }
}

#[test]
fn more_same_tool_steps_than_the_loop_threshold_warns() {
    let threshold = shared_loop_detector().config().threshold;
    let steps: Vec<Value> = (0..=threshold)
        .map(|i| stats_step(&format!("s{i}")))
        .collect();
    let report = validate(json!({ "steps": steps }));
    assert_eq!(
        report["warnings"][0]["code"], "loop_detect_threshold",
        "{report:#}"
    );

    let steps: Vec<Value> = (0..threshold)
        .map(|i| stats_step(&format!("s{i}")))
        .collect();
    let report = validate(json!({ "steps": steps }));
    assert!(
        report["warnings"].as_array().unwrap().is_empty(),
        "{report:#}"
    );
}

#[test]
fn duplicate_id_does_not_leak_errors_onto_the_first_step() {
    let report = validate(json!({
        "steps": [
            stats_step("a"),
            { "id": "b", "tool": "ix_stats", "arguments": { "data": [1.0] }, "depends_on": ["a"] },
            { "id": "a", "tool": "nope", "depends_on": ["b"], "arguments": { "data": "$zzz.x" } }
        ]
    }));
    let errors = report["errors"].as_array().unwrap();
    assert_eq!(errors.len(), 1, "{report:#}");
    assert_eq!(errors[0]["code"], "duplicate_id");
    assert_eq!(errors[0]["index"], 2);
}

#[test]
fn oversized_pipeline_is_rejected_quickly() {
    let steps: Vec<Value> = (0..8000)
        .map(|i| json!({ "id": format!("s{i}"), "tool": "ix_stats", "arguments": { "data": "$s0.values" } }))
        .collect();
    let started = Instant::now();
    let report = validate(json!({ "steps": steps }));
    let elapsed = started.elapsed();
    assert_eq!(
        error_codes(&report),
        vec![("too_many_steps".to_string(), Value::Null)]
    );
    assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
}

/// Worst case the review measured (reverse-ordered chain, every step reading
/// `$s0`) at the step cap: reachability is computed once, not per reference.
#[test]
fn validation_at_the_step_cap_stays_fast() {
    let n = MAX_PIPELINE_STEPS;
    let steps: Vec<Value> = (0..n)
        .rev()
        .map(|i| {
            let mut s = json!({ "id": format!("s{i}"), "tool": "ix_stats", "arguments": { "data": "$s0.values" } });
            if i > 0 {
                s["depends_on"] = json!([format!("s{}", i - 1)]);
                s["arguments"]["data"] = json!("$s0.values");
            } else {
                s["arguments"]["data"] = json!([1.0]);
            }
            s
        })
        .collect();
    let started = Instant::now();
    let report = validate(json!({ "steps": steps }));
    let elapsed = started.elapsed();
    let codes = error_codes(&report);
    assert!(codes.is_empty(), "{:?}", &codes[..codes.len().min(3)]);
    assert!(elapsed < Duration::from_secs(10), "took {elapsed:?}");
}

#[test]
fn too_many_references_in_one_step_is_an_error() {
    let deps: Vec<String> = (0..=MAX_STEP_REFERENCES).map(|_| "a".to_string()).collect();
    let report = validate(json!({
        "steps": [stats_step("a"), { "id": "b", "tool": "ix_stats", "arguments": { "data": [1.0] }, "depends_on": deps }]
    }));
    assert_eq!(
        error_codes(&report),
        vec![("too_many_references".to_string(), json!("b"))]
    );
}

/// Stricter than `ix_pipeline_run`, which ignores a non-array `depends_on`.
#[test]
fn malformed_depends_on_is_an_error() {
    let report = validate(json!({
        "steps": [{ "id": "a", "tool": "ix_stats", "arguments": { "data": [1.0] }, "depends_on": "nope" }]
    }));
    assert_eq!(
        error_codes(&report),
        vec![("invalid_depends_on".to_string(), json!("a"))]
    );
}

#[test]
fn missing_or_empty_steps_is_a_structured_error_not_a_call_failure() {
    assert_eq!(validate(json!({}))["errors"][0]["code"], "missing_steps");
    assert_eq!(
        validate(json!({ "steps": [] }))["errors"][0]["code"],
        "empty_steps"
    );
}

/// `missing_id` has no step name to hang off, so it must at least carry the
/// position an editor can point at.
#[test]
fn missing_id_error_carries_the_step_index() {
    let report = validate(json!({
        "steps": [stats_step("a"), { "tool": "ix_stats", "arguments": { "data": [1.0] } }]
    }));
    assert_eq!(report["errors"][0]["code"], "missing_id");
    assert_eq!(report["errors"][0]["step"], Value::Null);
    assert_eq!(report["errors"][0]["index"], 1);
}

/// The `unsupported_in_pipeline` guard is only as good as its list. Read the
/// arms of `call_with_ctx` out of the source and require the two to match, so
/// a tool added to the `match` without the const cannot slip through.
#[test]
fn context_routed_tools_const_matches_call_with_ctx() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tools.rs"),
    )
    .expect("read tools.rs");
    let body = {
        let start = src
            .find("pub fn call_with_ctx(")
            .expect("call_with_ctx in tools.rs");
        let rest = &src[start..];
        let end = rest.find("\n    }").expect("end of call_with_ctx");
        &rest[..end]
    };
    let mut intercepted: Vec<String> = body
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let name = line.strip_prefix('"')?;
            let (name, tail) = name.split_once('"')?;
            tail.trim_start()
                .starts_with("=>")
                .then(|| name.to_string())
        })
        .collect();
    intercepted.sort();
    let mut expected: Vec<String> = CONTEXT_ROUTED_TOOLS.iter().map(|s| s.to_string()).collect();
    expected.sort();
    assert!(
        !intercepted.is_empty(),
        "no match arms found — parser drifted"
    );
    assert_eq!(
        intercepted, expected,
        "CONTEXT_ROUTED_TOOLS must list exactly the tools call_with_ctx intercepts"
    );
}
