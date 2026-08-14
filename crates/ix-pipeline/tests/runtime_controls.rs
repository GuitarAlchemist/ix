use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ix_pipeline::dag::Dag;
use ix_pipeline::executor::{
    execute, execute_with_options, CancellationToken, ExecutionOptions, PipelineCache,
    PipelineError, PipelineNode,
};
use serde_json::Value;

struct MemoryCache {
    values: Mutex<HashMap<String, Value>>,
}

impl PipelineCache for MemoryCache {
    fn get(&self, key: &str) -> Option<Value> {
        self.values.lock().unwrap().get(key).cloned()
    }

    fn set(&self, key: &str, value: &Value) {
        self.values
            .lock()
            .unwrap()
            .insert(key.to_string(), value.clone());
    }
}

fn pipeline(call_count: Arc<AtomicUsize>, multiplier: i64) -> Dag<PipelineNode> {
    let mut dag = Dag::new();
    dag.add_node(
        "compute",
        PipelineNode {
            name: "compute".into(),
            compute: Box::new(move |_| {
                call_count.fetch_add(1, Ordering::SeqCst);
                Ok(Value::from(multiplier))
            }),
            input_map: HashMap::new(),
            cost: 1.0,
            cacheable: true,
        },
    )
    .unwrap();
    dag
}

#[test]
fn cache_identity_includes_the_nodes_logic_digest() {
    let calls = Arc::new(AtomicUsize::new(0));
    let cache = MemoryCache {
        values: Mutex::new(HashMap::new()),
    };
    let v1 = ExecutionOptions::new().with_logic_digest(
        "compute",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    );
    let v2 = ExecutionOptions::new().with_logic_digest(
        "compute",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    );

    let first =
        execute_with_options(&pipeline(calls.clone(), 1), &HashMap::new(), &cache, &v1).unwrap();
    let replay =
        execute_with_options(&pipeline(calls.clone(), 999), &HashMap::new(), &cache, &v1).unwrap();
    let changed =
        execute_with_options(&pipeline(calls.clone(), 2), &HashMap::new(), &cache, &v2).unwrap();

    assert_eq!(Value::from(1), first.output("compute").unwrap().clone());
    assert_eq!(Value::from(1), replay.output("compute").unwrap().clone());
    assert_eq!(Value::from(2), changed.output("compute").unwrap().clone());
    assert_eq!(2, calls.load(Ordering::SeqCst));
}

#[test]
fn legacy_execute_preserves_the_existing_cache_key_format() {
    let cache = MemoryCache {
        values: Mutex::new(HashMap::new()),
    };
    execute(
        &pipeline(Arc::new(AtomicUsize::new(0)), 1),
        &HashMap::new(),
        &cache,
    )
    .expect("legacy execution succeeds");

    let keys = cache
        .values
        .lock()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(vec!["pipeline:compute:675868731199239589"], keys);
}

#[test]
fn strict_runtime_refuses_an_unversioned_cacheable_node() {
    let cache = MemoryCache {
        values: Mutex::new(HashMap::new()),
    };
    let error = execute_with_options(
        &pipeline(Arc::new(AtomicUsize::new(0)), 1),
        &HashMap::new(),
        &cache,
        &ExecutionOptions::new(),
    )
    .expect_err("strict cache identity must fail closed");

    assert!(matches!(error, PipelineError::MissingLogicDigest(node) if node == "compute"));
}

#[test]
fn cancellation_stops_the_pipeline_before_the_next_node_runs() {
    let calls = Arc::new(AtomicUsize::new(0));
    let token = CancellationToken::new();
    token.cancel();
    let options = ExecutionOptions::new().with_cancellation(token);

    let error = execute_with_options(
        &pipeline(calls.clone(), 1),
        &HashMap::new(),
        &ix_pipeline::executor::NoCache,
        &options,
    )
    .expect_err("cancelled execution must stop");

    assert!(matches!(error, PipelineError::Cancelled));
    assert_eq!(0, calls.load(Ordering::SeqCst));
}

#[test]
fn independent_nodes_execute_concurrently() {
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let mut dag = Dag::new();
    for id in ["first", "second"] {
        let active = active.clone();
        let maximum = maximum.clone();
        dag.add_node(
            id,
            PipelineNode {
                name: id.into(),
                compute: Box::new(move |_| {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(100));
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(Value::Null)
                }),
                input_map: HashMap::new(),
                cost: 1.0,
                cacheable: false,
            },
        )
        .unwrap();
    }

    let result = execute_with_options(
        &dag,
        &HashMap::new(),
        &ix_pipeline::executor::NoCache,
        &ExecutionOptions::new(),
    )
    .expect("same-level nodes must overlap");

    assert_eq!(2, result.node_results.len());
    assert_eq!(2, maximum.load(Ordering::SeqCst));
}
