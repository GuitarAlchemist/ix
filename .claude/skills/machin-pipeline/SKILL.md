---
name: machin-pipeline
description: DAG pipeline orchestration with parallel execution and caching
---

# Pipeline

Design and execute multi-step data processing pipelines as DAGs.

## When to Use
When the user has a multi-step workflow where steps have dependencies, branches can run in parallel, and results should be cached.

## Builder API
```rust
use machin_pipeline::builder::PipelineBuilder;
use machin_pipeline::executor::{execute, NoCache};
use serde_json::{json, Value};
use std::collections::HashMap;

let pipeline = PipelineBuilder::new()
    .source("load", || Ok(json!({"data": [1, 2, 3]})))
    .node("process", |b| b
        .input("x", "load")
        .compute(|inputs| {
            let data = inputs["x"].as_array().unwrap();
            Ok(json!(data.len()))
        })
    )
    .node("transform", |b| b
        .input("x", "load")
        .compute(|inputs| Ok(json!("transformed")))
    )
    .node("merge", |b| b
        .input("count", "process")
        .input("result", "transform")
        .compute(|inputs| Ok(json!({"count": inputs["count"], "result": inputs["result"]})))
    )
    .build()
    .unwrap();

// Edges auto-detected from .input() calls
// "process" and "transform" run in parallel (same level)
// "merge" waits for both

let result = execute(&pipeline, &HashMap::new(), &NoCache).unwrap();
```

## Features
- Auto-edge detection from input mappings
- Parallel execution of independent branches
- Critical path analysis (`.critical_path()`)
- PipelineCache trait — plug into machin-cache for memoization
- Cycle detection on every edge insertion
