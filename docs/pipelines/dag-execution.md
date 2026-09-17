# DAG Pipeline Execution

## The Problem

You are building a model training workflow with five steps: load data, compute statistics, normalize features, train model, evaluate. Some steps depend on others (you cannot normalize without statistics), but some are independent (statistics and a separate feature-engineering branch can run in parallel). You need a system that:

1. Understands dependencies and refuses to create circular ones.
2. Automatically determines which steps can run in parallel.
3. Executes everything in the correct order.
4. Passes outputs from one step to the inputs of the next.

This is the ETL pipeline problem, the CI/CD pipeline problem, the DAG-based workflow orchestration problem. ix's `ix-pipeline` crate solves it with a typed, cycle-checked DAG and a level-parallel executor.

---

## The Intuition

Think of a recipe with multiple cooks in a kitchen:

- **Cook A** chops vegetables (no dependencies).
- **Cook B** boils water (no dependencies).
- **Cook C** makes the sauce (needs chopped vegetables from Cook A).
- **Cook D** cooks the pasta (needs boiling water from Cook B).
- **Cook E** plates the dish (needs sauce from Cook C *and* pasta from Cook D).

Cooks A and B can work simultaneously (level 0). Cooks C and D can start as soon as their inputs are ready (level 1). Cook E waits for both C and D (level 2). This is a DAG --- a directed acyclic graph. The "acyclic" part means nobody waits for themselves, which would be a deadlock.

The pipeline builder lets you declare nodes (cooks) and edges (who-needs-what), then the executor figures out the parallelism and runs everything.

---

## How It Works

### The DAG data structure

A `Dag<N>` is a directed graph that **rejects cycles at insertion time**. Every `add_edge(from, to)` call runs a BFS from `to` to `from` --- if a path exists, the edge would create a cycle and is rejected.

| Operation | Complexity | What it does |
|-----------|:----------:|--------------|
| `add_node(id, data)` | O(1) | Register a node |
| `add_edge(from, to)` | O(V + E) | Add a dependency edge (with cycle check) |
| `topological_sort()` | O(V + E) | Kahn's algorithm: order nodes so dependencies come first |
| `parallel_levels()` | O(V + E) | Group nodes into parallelizable levels |
| `has_path(a, b)` | O(V + E) | BFS reachability check |
| `critical_path(cost_fn)` | O(V + E) | Longest weighted path through the DAG |

### Error types

The `DagError` enum catches structural problems early:

| Variant | When it fires |
|---------|--------------|
| `DuplicateNode(id)` | Calling `add_node` with an ID that already exists |
| `NodeNotFound(id)` | Referencing a node in `add_edge` that was never added |
| `CycleDetected(from, to)` | The proposed edge would create a cycle |
| `SelfLoop(id)` | An edge from a node to itself |

### The PipelineBuilder

The builder API lets you declare nodes with compute functions and wire them together:

```
PipelineBuilder::new()
    .node(id, |builder| builder
        .compute(|inputs| -> Result<Value>)
        .input(name, source_node)
        .cost(estimated_time)
        .no_cache()
    )
    .edge(from, to)          // explicit edge
    .build()                  // returns Dag<PipelineNode>
```

Edges declared via `.input(name, source_node)` are **auto-detected**: if a node's input references another node by ID, the builder creates the edge automatically. You do not need to call `.edge()` separately unless you want additional ordering constraints.

### The executor

`execute(dag, initial_inputs, cache)` runs the pipeline:

1. Compute `parallel_levels()` to group nodes by dependency depth.
2. For each level, run all nodes in that level. (Single-node levels run directly; multi-node levels run concurrently.)
3. Each node's compute function receives a `HashMap<String, Value>` of its declared inputs, resolved from predecessor outputs or `initial_inputs`.
4. Results are collected into a `PipelineResult` with per-node outputs, timings, cache hits, and execution order.

---

## In Rust

> Full runnable example: [`examples/pipeline/dag_pipeline.rs`](../../examples/pipeline/dag_pipeline.rs)

### ETL pipeline with the builder API

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, NoCache};
use serde_json::{json, Value};
use std::collections::HashMap;

let pipeline = PipelineBuilder::new()
    // Source node: produces raw data
    .source("raw_data", || {
        Ok(json!({"values": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    })
    // Compute statistics (depends on raw_data)
    .node("stats", |b| {
        b.input("data", "raw_data")
         .compute(|inputs| {
             let vals = inputs["data"]["values"].as_array().unwrap();
             let sum: f64 = vals.iter().map(|v| v.as_f64().unwrap()).sum();
             let mean = sum / vals.len() as f64;
             Ok(json!({"mean": mean, "count": vals.len()}))
         })
    })
    // Normalize (depends on BOTH raw_data and stats)
    .node("normalize", |b| {
        b.input("data", "raw_data")
         .input("stats", "stats")
         .compute(|inputs| {
             let vals = inputs["data"]["values"].as_array().unwrap();
             let mean = inputs["stats"]["mean"].as_f64().unwrap();
             let normalized: Vec<f64> = vals.iter()
                 .map(|v| v.as_f64().unwrap() - mean)
                 .collect();
             Ok(json!(normalized))
         })
    })
    .build()
    .unwrap();

let result = execute(&pipeline, &HashMap::new(), &NoCache).unwrap();
println!("Normalized: {}", result.output("normalize").unwrap());
println!("Executed in {} levels", result.execution_order.len());
```

### Diamond pattern (parallel branches)

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, NoCache};
use serde_json::{json, Value};
use std::collections::HashMap;

let pipeline = PipelineBuilder::new()
    .source("data", || Ok(json!(100.0)))
    .node("branch_a", |b| b
        .input("x", "data")
        .compute(|inputs| {
            let x = inputs["x"].as_f64().unwrap();
            Ok(json!(x + 50.0))  // 150
        })
    )
    .node("branch_b", |b| b
        .input("x", "data")
        .compute(|inputs| {
            let x = inputs["x"].as_f64().unwrap();
            Ok(json!(x * 0.5))  // 50
        })
    )
    .node("merge", |b| b
        .input("a", "branch_a")
        .input("b", "branch_b")
        .compute(|inputs| {
            let a = inputs["a"].as_f64().unwrap();
            let b = inputs["b"].as_f64().unwrap();
            Ok(json!(a - b))  // 100
        })
    )
    .build()
    .unwrap();

// Execution levels:
//   Level 0: [data]
//   Level 1: [branch_a, branch_b]  <-- parallel
//   Level 2: [merge]
let result = execute(&pipeline, &HashMap::new(), &NoCache).unwrap();
assert_eq!(result.execution_order[1].len(), 2);  // branch_a and branch_b in parallel
println!("Result: {}", result.output("merge").unwrap());  // 100
```

### Using the raw DAG API

```rust
use ix_pipeline::dag::{Dag, DagError};

let mut dag = Dag::new();
dag.add_node("load",    "Load CSV").unwrap();
dag.add_node("clean",   "Remove nulls").unwrap();
dag.add_node("feature", "Engineer features").unwrap();
dag.add_node("train",   "Train model").unwrap();
dag.add_node("eval",    "Evaluate").unwrap();

dag.add_edge("load", "clean").unwrap();
dag.add_edge("clean", "feature").unwrap();
dag.add_edge("clean", "train").unwrap();   // train can start after clean
dag.add_edge("feature", "train").unwrap(); // but also needs features
dag.add_edge("train", "eval").unwrap();

// Topological sort
let sorted = dag.topological_sort();
println!("Execution order: {:?}", sorted);

// Parallel levels
let levels = dag.parallel_levels();
for (i, level) in levels.iter().enumerate() {
    let ids: Vec<&str> = level.iter().map(|s| s.as_str()).collect();
    println!("Level {}: {:?}", i, ids);
}

// Cycle detection
let err = dag.add_edge("eval", "load");
assert!(matches!(err, Err(DagError::CycleDetected(_, _))));

// Path queries
assert!(dag.has_path("load", "eval"));
assert!(!dag.has_path("eval", "load"));

// Critical path
let (path, cost) = dag.critical_path(|_, _| 1.0);
println!("Critical path: {:?} (cost: {})", path, cost);
```

### External inputs

```rust
use ix_pipeline::builder::PipelineBuilder;
use ix_pipeline::executor::{execute, NoCache};
use serde_json::Value;
use std::collections::HashMap;

let pipeline = PipelineBuilder::new()
    .node("greet", |b| b
        .input("name", "name")  // "name" is an external input, not a node
        .compute(|inputs| {
            let name = inputs["name"].as_str().unwrap_or("world");
            Ok(Value::from(format!("Hello, {}!", name)))
        })
        .no_cache()  // non-deterministic or side-effecting nodes should skip cache
    )
    .build()
    .unwrap();

let mut inputs = HashMap::new();
inputs.insert("name".to_string(), Value::from("ix"));

let result = execute(&pipeline, &inputs, &NoCache).unwrap();
println!("{}", result.output("greet").unwrap());  // "Hello, ix!"
```

---

## When To Use This

| Situation | DAG Pipeline | Sequential loop | Thread pool |
|-----------|:-----------:|:---------------:|:-----------:|
| Tasks with complex dependencies | Best | Manual ordering | Manual sync |
| Automatic parallelism of independent tasks | Yes | No | Manual |
| Cycle detection at build time | Yes | N/A | N/A |
| Data flow between tasks | Built-in (input/output maps) | Manual variables | Manual channels |
| Caching / memoization | Built-in | Manual | Manual |
| Critical path analysis | Built-in | Manual | Manual |

**Rule of thumb:** use a DAG pipeline when you have more than 3 steps with non-trivial dependency relationships, especially if some branches can run in parallel.

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| Node `compute` | The function that runs for this step | Takes `&HashMap<String, Value>`, returns `Result<Value, PipelineError>` |
| Node `input(name, source)` | Where this node gets its data | `source` is another node's ID or an external input name |
| Node `input_field(name, source, field)` | Read a specific JSON field from a source | Use when a source outputs an object and you want one key |
| Node `cost` | Estimated execution time (for critical path) | Default: 1.0. Set higher for expensive nodes to get meaningful critical path analysis |
| Node `no_cache()` | Disable caching for this node | Use for non-deterministic or side-effecting nodes |
| `initial_inputs` | External values passed to the pipeline | Available to any node via `input(name, external_key)` |

---

## Pitfalls

1. **Cycle detection runs on every `add_edge`.** For DAGs with thousands of nodes and edges, this BFS cycle check can be expensive. Build large DAGs in dependency order (add edges from sources toward sinks) to keep the reachability check short.

2. **Auto-edges only work with the builder.** If you construct a `Dag` manually and set `input_map` entries without calling `add_edge`, the executor will fail with `MissingInput`. The builder's `build()` method calls `auto_edges()` for you; the raw DAG does not.

3. **JSON values everywhere.** The pipeline passes data as `serde_json::Value`. This is flexible but means you lose type safety inside compute functions. Parse and validate inputs early in each compute closure.

4. **Parallelism is level-based, not full.** The executor runs all nodes at a given dependency level before moving to the next. This is simpler than full async scheduling but means a slow node in level 1 blocks all of level 2, even if some level-2 nodes only depend on a fast level-1 node.

5. **`PipelineError::MissingInput` at runtime.** If a node declares `.input("x", "nonexistent_node")` and that node is neither in the pipeline nor in `initial_inputs`, you get a runtime error, not a build-time error. Validate your pipeline structure before execution.

---

## Validating MCP Pipeline Specs Offline

The MCP surface has two read-only (Tier 1) companions to `ix_pipeline_run`, modelled on ComfyUI's `object_info` endpoint and `comfy validate`:

- **`ix_node_catalog`** (no arguments) returns one entry per registered tool: `name`, `description`, `dispatch` (`registry` or `manual`), `input_schema`, `required_inputs`, `output_schema` (`null` when the skill declares none) and `approval` (`action_kind` + `tier` as computed by `ix-approval`).
- **`ix_pipeline_validate`** takes the exact `{"steps": [...]}` spec `ix_pipeline_run` consumes and checks it without running anything.

```json
{
  "steps": [
    { "id": "a", "tool": "ix_stats", "arguments": { "data": [1.0, 2.0, 3.0] } },
    { "id": "b", "tool": "ix_cache", "depends_on": ["a"],
      "arguments": { "operation": "set", "key": "k", "value": "$a.mean" } }
  ]
}
```

returns `valid: true`, `execution_order: ["a", "b"]`, per-step tiers, `max_tier: "tier_two"` and `requires_approval: false`. Problems come back as structured `{code, step, message}` entries so an editor can pin each one to a node:

| Code | Meaning |
|------|---------|
| `unknown_tool` | `tool` is not in the registry |
| `missing_required_input` | an input listed in the tool's schema `required` is absent from `arguments` |
| `unknown_step_reference` | a `depends_on` entry or a `"$step.field"` argument names an undefined step |
| `cycle` | a `depends_on` edge would close a cycle (rejected by `ix_pipeline::dag::Dag`) |
| `missing_steps`, `empty_steps`, `missing_id`, `duplicate_id`, `missing_tool`, `invalid_arguments`, `invalid_depends_on` | malformed spec |

The warning `undeclared_dependency` flags a `"$step.field"` reference to a step that is not upstream through `depends_on` — it may not have run when the reference is substituted. The checks are structural: argument *types* are not validated against the schema yet. The example above is pinned by `crates/ix-agent/tests/pipeline_validate.rs::valid_chained_pipeline_passes_with_order_and_tier`.

---

## Going Further

- **[Caching and Memoization](./caching-and-memoization.md)** covers the `PipelineCache` trait, per-node cacheability, and how to connect to `ix-cache` for incremental recomputation.
- **Critical path analysis** (`dag.critical_path(cost_fn)`) identifies the bottleneck chain in your pipeline. Use it to decide which nodes to optimize or move to GPU.
- **`parallel_levels()`** returns the execution schedule. You can use this for visualization, progress bars, or custom scheduling without running the full executor.
- The `Dag` is generic over node data (`Dag<N>`). The pipeline uses `Dag<PipelineNode>`, but you can use `Dag<String>`, `Dag<MyTask>`, or any other type for non-pipeline DAG workloads (dependency graphs, build systems, task schedulers).
