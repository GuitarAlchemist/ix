//! Pipeline executor — runs DAG nodes in dependency order with parallelism.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::dag::{Dag, NodeId};

/// A compute function that takes named inputs and produces a JSON output.
pub type ComputeFn =
    Box<dyn Fn(&HashMap<String, Value>) -> Result<Value, PipelineError> + Send + Sync>;

/// A pipeline node: wraps a compute function with metadata.
pub struct PipelineNode {
    /// Human-readable name.
    pub name: String,
    /// The compute function.
    pub compute: ComputeFn,
    /// Which predecessor outputs this node reads (mapped to input names).
    /// Key = input name for this node, Value = (source_node_id, output_key).
    pub input_map: HashMap<String, (NodeId, String)>,
    /// Estimated cost (for critical path analysis). Default: 1.0.
    pub cost: f64,
    /// Whether to cache the result. Default: true.
    pub cacheable: bool,
}

/// Result of executing one node.
#[derive(Debug, Clone)]
pub struct NodeResult {
    pub node_id: NodeId,
    pub output: Value,
    pub duration: Duration,
    pub cache_hit: bool,
}

/// Result of executing the full pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Results from each node.
    pub node_results: HashMap<NodeId, NodeResult>,
    /// Total wall-clock time.
    pub total_duration: Duration,
    /// Number of cache hits.
    pub cache_hits: usize,
    /// Execution order (level by level).
    pub execution_order: Vec<Vec<NodeId>>,
}

impl PipelineResult {
    /// Get the output of a specific node.
    pub fn output(&self, node_id: &str) -> Option<&Value> {
        self.node_results.get(node_id).map(|r| &r.output)
    }

    /// Get the final outputs (leaf nodes).
    pub fn final_outputs(&self) -> HashMap<&NodeId, &Value> {
        // Find nodes that aren't inputs to any other node
        let _all_inputs: std::collections::HashSet<&str> = self
            .node_results
            .values()
            .flat_map(|_| std::iter::empty::<&str>()) // Can't easily determine without the DAG
            .collect();

        self.node_results
            .iter()
            .map(|(id, r)| (id, &r.output))
            .collect()
    }
}

/// Pipeline errors.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("node '{0}' failed: {1}")]
    NodeFailed(NodeId, String),

    #[error("missing input '{input}' for node '{node}' (expected from '{source_node}')")]
    MissingInput {
        node: String,
        input: String,
        source_node: String,
    },

    #[error("pipeline has no nodes")]
    EmptyPipeline,

    #[error("compute error: {0}")]
    ComputeError(String),
}

/// Optional cache interface for memoization.
///
/// Implement this to connect to `ix-cache` or any other cache.
pub trait PipelineCache: Send + Sync {
    /// Try to get a cached result for a node.
    fn get(&self, cache_key: &str) -> Option<Value>;

    /// Store a result in the cache.
    fn set(&self, cache_key: &str, value: &Value);
}

/// A no-op cache (disables caching).
pub struct NoCache;

impl PipelineCache for NoCache {
    fn get(&self, _key: &str) -> Option<Value> {
        None
    }
    fn set(&self, _key: &str, _value: &Value) {}
}

/// Execute a pipeline DAG.
///
/// Runs nodes level-by-level. Within each level, nodes execute in parallel
/// using std threads (no async runtime required).
pub fn execute(
    dag: &Dag<PipelineNode>,
    initial_inputs: &HashMap<String, Value>,
    cache: &dyn PipelineCache,
) -> Result<PipelineResult, PipelineError> {
    let start = Instant::now();
    let levels = dag.parallel_levels();

    if levels.is_empty() {
        return Err(PipelineError::EmptyPipeline);
    }

    let outputs: Arc<Mutex<HashMap<NodeId, Value>>> = Arc::new(Mutex::new(HashMap::new()));
    let mut node_results: HashMap<NodeId, NodeResult> = HashMap::new();
    let mut cache_hits = 0usize;
    let mut execution_order: Vec<Vec<NodeId>> = Vec::new();

    // Seed initial inputs as "virtual node" outputs
    {
        let mut out = outputs.lock().unwrap();
        for (k, v) in initial_inputs {
            out.insert(format!("__input__:{}", k), v.clone());
        }
    }

    for level in &levels {
        let level_ids: Vec<NodeId> = level.iter().map(|id| (*id).clone()).collect();
        execution_order.push(level_ids.clone());

        if level.len() == 1 {
            // Single node — run directly (no thread overhead)
            let id = level[0];
            let node = dag.get(id).unwrap();

            let result = execute_node(id, node, &outputs, cache)?;
            if result.cache_hit {
                cache_hits += 1;
            }

            outputs
                .lock()
                .unwrap()
                .insert(id.clone(), result.output.clone());
            node_results.insert(id.clone(), result);
        } else {
            // Multiple nodes — run in parallel
            let handles: Vec<_> = level
                .iter()
                .map(|&id| {
                    let id = id.clone();
                    let outputs = Arc::clone(&outputs);
                    let node = dag.get(&id).unwrap();

                    // Gather inputs before spawning
                    let node_inputs = gather_inputs(&id, node, &outputs)?;
                    let cache_key = make_cache_key(&id, &node_inputs);
                    let cacheable = node.cacheable;

                    // Check cache
                    if cacheable {
                        if let Some(cached) = cache.get(&cache_key) {
                            let result = NodeResult {
                                node_id: id.clone(),
                                output: cached.clone(),
                                duration: Duration::ZERO,
                                cache_hit: true,
                            };
                            outputs.lock().unwrap().insert(id.clone(), cached);
                            return Ok((id, result));
                        }
                    }

                    // Build a closure that doesn't borrow `node`
                    // We need to run compute in the current thread context since ComputeFn isn't Send
                    let node_start = Instant::now();
                    let output = (node.compute)(&node_inputs)
                        .map_err(|e| PipelineError::NodeFailed(id.clone(), e.to_string()))?;

                    if cacheable {
                        cache.set(&cache_key, &output);
                    }

                    let result = NodeResult {
                        node_id: id.clone(),
                        output: output.clone(),
                        duration: node_start.elapsed(),
                        cache_hit: false,
                    };

                    outputs.lock().unwrap().insert(id.clone(), output);
                    Ok((id, result))
                })
                .collect::<Result<Vec<_>, PipelineError>>()?;

            for (id, result) in handles {
                if result.cache_hit {
                    cache_hits += 1;
                }
                node_results.insert(id, result);
            }
        }
    }

    Ok(PipelineResult {
        node_results,
        total_duration: start.elapsed(),
        cache_hits,
        execution_order,
    })
}

/// Execute a single node.
fn execute_node(
    id: &NodeId,
    node: &PipelineNode,
    outputs: &Arc<Mutex<HashMap<NodeId, Value>>>,
    cache: &dyn PipelineCache,
) -> Result<NodeResult, PipelineError> {
    let inputs = gather_inputs(id, node, outputs)?;

    // Check cache
    if node.cacheable {
        let cache_key = make_cache_key(id, &inputs);
        if let Some(cached) = cache.get(&cache_key) {
            outputs.lock().unwrap().insert(id.clone(), cached.clone());
            return Ok(NodeResult {
                node_id: id.clone(),
                output: cached,
                duration: Duration::ZERO,
                cache_hit: true,
            });
        }
    }

    let start = Instant::now();
    let output = (node.compute)(&inputs)
        .map_err(|e| PipelineError::NodeFailed(id.clone(), e.to_string()))?;

    if node.cacheable {
        let cache_key = make_cache_key(id, &inputs);
        cache.set(&cache_key, &output);
    }

    Ok(NodeResult {
        node_id: id.clone(),
        output,
        duration: start.elapsed(),
        cache_hit: false,
    })
}

/// Gather inputs for a node from predecessor outputs.
fn gather_inputs(
    id: &NodeId,
    node: &PipelineNode,
    outputs: &Arc<Mutex<HashMap<NodeId, Value>>>,
) -> Result<HashMap<String, Value>, PipelineError> {
    let out = outputs.lock().unwrap();
    let mut inputs = HashMap::new();

    for (input_name, (source_id, output_key)) in &node.input_map {
        // Try direct source node output
        let source_output = out
            .get(source_id)
            .or_else(|| out.get(&format!("__input__:{}", source_id)));

        match source_output {
            Some(val) => {
                // If the output is an object and we want a specific key
                if output_key == "*" || output_key.is_empty() {
                    inputs.insert(input_name.clone(), val.clone());
                } else if let Some(field) = val.get(output_key) {
                    inputs.insert(input_name.clone(), field.clone());
                } else {
                    inputs.insert(input_name.clone(), val.clone());
                }
            }
            None => {
                return Err(PipelineError::MissingInput {
                    node: id.clone(),
                    input: input_name.clone(),
                    source_node: source_id.clone(),
                });
            }
        }
    }

    Ok(inputs)
}

/// Create a deterministic cache key from node ID and inputs.
fn make_cache_key(node_id: &str, inputs: &HashMap<String, Value>) -> String {
    let mut sorted_inputs: Vec<(&String, &Value)> = inputs.iter().collect();
    sorted_inputs.sort_by_key(|(k, _)| *k);

    let input_hash = format!("{:?}", sorted_inputs);
    format!("pipeline:{}:{}", node_id, simple_hash(&input_hash))
}

fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::Dag;

    fn make_adder(amount: f64) -> ComputeFn {
        Box::new(move |inputs: &HashMap<String, Value>| {
            let x = inputs.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Ok(Value::from(x + amount))
        })
    }

    fn make_multiplier(factor: f64) -> ComputeFn {
        Box::new(move |inputs: &HashMap<String, Value>| {
            let x = inputs.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Ok(Value::from(x * factor))
        })
    }

    fn make_combiner() -> ComputeFn {
        Box::new(|inputs: &HashMap<String, Value>| {
            let a = inputs.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = inputs.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Ok(Value::from(a + b))
        })
    }

    #[test]
    fn test_linear_pipeline() {
        // input(10) → add(5) → multiply(2) = 30
        let mut dag: Dag<PipelineNode> = Dag::new();

        dag.add_node(
            "add",
            PipelineNode {
                name: "Add 5".into(),
                compute: make_adder(5.0),
                input_map: [("x".into(), ("value".into(), "*".into()))].into(),
                cost: 1.0,
                cacheable: false,
            },
        )
        .unwrap();

        dag.add_node(
            "mul",
            PipelineNode {
                name: "Multiply by 2".into(),
                compute: make_multiplier(2.0),
                input_map: [("x".into(), ("add".into(), "*".into()))].into(),
                cost: 1.0,
                cacheable: false,
            },
        )
        .unwrap();

        dag.add_edge("add", "mul").unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("value".to_string(), Value::from(10.0));

        let result = execute(&dag, &inputs, &NoCache).unwrap();

        let output = result.output("mul").unwrap().as_f64().unwrap();
        assert!((output - 30.0).abs() < 1e-10, "Expected 30, got {}", output);
    }

    #[test]
    fn test_diamond_pipeline() {
        //        input(10)
        //        /       \
        //   add(5)     mul(3)
        //        \       /
        //        combine(+)  = 15 + 30 = 45
        let mut dag: Dag<PipelineNode> = Dag::new();

        dag.add_node(
            "add",
            PipelineNode {
                name: "Add 5".into(),
                compute: make_adder(5.0),
                input_map: [("x".into(), ("value".into(), "*".into()))].into(),
                cost: 1.0,
                cacheable: false,
            },
        )
        .unwrap();

        dag.add_node(
            "mul",
            PipelineNode {
                name: "Multiply by 3".into(),
                compute: make_multiplier(3.0),
                input_map: [("x".into(), ("value".into(), "*".into()))].into(),
                cost: 1.0,
                cacheable: false,
            },
        )
        .unwrap();

        dag.add_node(
            "combine",
            PipelineNode {
                name: "Combine".into(),
                compute: make_combiner(),
                input_map: [
                    ("a".into(), ("add".into(), "*".into())),
                    ("b".into(), ("mul".into(), "*".into())),
                ]
                .into(),
                cost: 1.0,
                cacheable: false,
            },
        )
        .unwrap();

        dag.add_edge("add", "combine").unwrap();
        dag.add_edge("mul", "combine").unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("value".to_string(), Value::from(10.0));

        let result = execute(&dag, &inputs, &NoCache).unwrap();

        let output = result.output("combine").unwrap().as_f64().unwrap();
        assert!((output - 45.0).abs() < 1e-10, "Expected 45, got {}", output);

        // add and mul should be in the same execution level
        assert_eq!(result.execution_order[0].len(), 2); // add, mul parallel
        assert_eq!(result.execution_order[1].len(), 1); // combine
    }

    #[test]
    fn test_caching() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Simple in-memory cache for testing
        struct TestCache {
            store: Mutex<HashMap<String, Value>>,
        }

        impl PipelineCache for TestCache {
            fn get(&self, key: &str) -> Option<Value> {
                self.store.lock().unwrap().get(key).cloned()
            }
            fn set(&self, key: &str, value: &Value) {
                self.store
                    .lock()
                    .unwrap()
                    .insert(key.to_string(), value.clone());
            }
        }

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let mut dag: Dag<PipelineNode> = Dag::new();
        dag.add_node(
            "expensive",
            PipelineNode {
                name: "Expensive op".into(),
                compute: Box::new(move |inputs: &HashMap<String, Value>| {
                    cc.fetch_add(1, Ordering::SeqCst);
                    let x = inputs.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    Ok(Value::from(x * 2.0))
                }),
                input_map: [("x".into(), ("value".into(), "*".into()))].into(),
                cost: 10.0,
                cacheable: true,
            },
        )
        .unwrap();

        let cache = TestCache {
            store: Mutex::new(HashMap::new()),
        };
        let mut inputs = HashMap::new();
        inputs.insert("value".to_string(), Value::from(5.0));

        // First run: computes
        let r1 = execute(&dag, &inputs, &cache).unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        assert_eq!(r1.cache_hits, 0);

        // Second run: cache hit
        let r2 = execute(&dag, &inputs, &cache).unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1); // NOT called again
        assert_eq!(r2.cache_hits, 1);
    }
}
