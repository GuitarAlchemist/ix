//! Fluent builder API for constructing pipelines.
//!
//! ```ignore
//! let pipeline = PipelineBuilder::new()
//!     .node("load", |b| b
//!         .compute(|_| Ok(json!({"data": [1, 2, 3]})))
//!     )
//!     .node("process", |b| b
//!         .input("data", "load")
//!         .compute(|inputs| {
//!             let data = inputs["data"].as_array().unwrap();
//!             Ok(json!(data.len()))
//!         })
//!     )
//!     .edge("load", "process")
//!     .build()
//!     .unwrap();
//! ```

use std::collections::HashMap;

use serde_json::Value;

use crate::dag::{Dag, DagError, NodeId};
use crate::executor::{ComputeFn, PipelineNode};

/// Builder for constructing a pipeline DAG.
pub struct PipelineBuilder {
    nodes: Vec<(NodeId, PipelineNode)>,
    edges: Vec<(NodeId, NodeId)>,
}

/// Builder for a single node.
pub struct NodeBuilder {
    name: String,
    compute: Option<ComputeFn>,
    input_map: HashMap<String, (NodeId, String)>,
    cost: f64,
    cacheable: bool,
}

impl NodeBuilder {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            compute: None,
            input_map: HashMap::new(),
            cost: 1.0,
            cacheable: true,
        }
    }

    /// Set the compute function.
    pub fn compute<F>(mut self, f: F) -> Self
    where
        F: Fn(&HashMap<String, Value>) -> Result<Value, crate::executor::PipelineError> + Send + Sync + 'static,
    {
        self.compute = Some(Box::new(f));
        self
    }

    /// Map an input name to a source node's output.
    /// `source` is the node ID, reads the full output.
    pub fn input(mut self, name: &str, source: &str) -> Self {
        self.input_map.insert(name.to_string(), (source.to_string(), "*".to_string()));
        self
    }

    /// Map an input name to a specific field of a source node's output.
    pub fn input_field(mut self, name: &str, source: &str, field: &str) -> Self {
        self.input_map.insert(name.to_string(), (source.to_string(), field.to_string()));
        self
    }

    /// Set the estimated cost (for critical path analysis).
    pub fn cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// Disable caching for this node (e.g., non-deterministic operations).
    pub fn no_cache(mut self) -> Self {
        self.cacheable = false;
        self
    }

    fn build(self) -> PipelineNode {
        PipelineNode {
            name: self.name,
            compute: self.compute.expect("Node must have a compute function"),
            input_map: self.input_map,
            cost: self.cost,
            cacheable: self.cacheable,
        }
    }
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node using a builder closure.
    pub fn node<F>(mut self, id: &str, builder_fn: F) -> Self
    where
        F: FnOnce(NodeBuilder) -> NodeBuilder,
    {
        let builder = NodeBuilder::new(id);
        let node = builder_fn(builder).build();
        self.nodes.push((id.to_string(), node));
        self
    }

    /// Add a source node (no inputs, just produces a value).
    pub fn source<F>(self, id: &str, f: F) -> Self
    where
        F: Fn() -> Result<Value, crate::executor::PipelineError> + Send + Sync + 'static,
    {
        self.node(id, |b| {
            b.compute(move |_| f())
        })
    }

    /// Add a directed edge (dependency).
    pub fn edge(mut self, from: &str, to: &str) -> Self {
        self.edges.push((from.to_string(), to.to_string()));
        self
    }

    /// Auto-detect edges from input mappings.
    /// If a node's input references another node, create the edge automatically.
    fn auto_edges(&mut self) {
        let node_ids: std::collections::HashSet<String> = self.nodes.iter()
            .map(|(id, _)| id.clone())
            .collect();

        let mut auto = Vec::new();
        for (id, node) in &self.nodes {
            for (source, _) in node.input_map.values() {
                if node_ids.contains(source) {
                    let edge = (source.clone(), id.clone());
                    if !self.edges.contains(&edge) {
                        auto.push(edge);
                    }
                }
            }
        }
        self.edges.extend(auto);
    }

    /// Build the pipeline DAG.
    pub fn build(mut self) -> Result<Dag<PipelineNode>, DagError> {
        self.auto_edges();

        let mut dag = Dag::new();

        for (id, node) in self.nodes {
            dag.add_node(id, node)?;
        }

        for (from, to) in &self.edges {
            dag.add_edge(from.as_str(), to.as_str())?;
        }

        Ok(dag)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: create a simple transform node.
pub fn transform<F>(input_node: &str, f: F) -> NodeBuilder
where
    F: Fn(Value) -> Result<Value, crate::executor::PipelineError> + Send + Sync + 'static,
{
    NodeBuilder::new("transform")
        .input("x", input_node)
        .compute(move |inputs| {
            let x = inputs.get("x").cloned().unwrap_or(Value::Null);
            f(x)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{execute, NoCache};

    #[test]
    fn test_builder_linear() {
        let pipeline = PipelineBuilder::new()
            .source("input", || Ok(Value::from(10.0)))
            .node("double", |b| b
                .input("x", "input")
                .compute(|inputs| {
                    let x = inputs["x"].as_f64().unwrap();
                    Ok(Value::from(x * 2.0))
                })
            )
            .node("add_one", |b| b
                .input("x", "double")
                .compute(|inputs| {
                    let x = inputs["x"].as_f64().unwrap();
                    Ok(Value::from(x + 1.0))
                })
            )
            .build()
            .unwrap();

        let result = execute(&pipeline, &HashMap::new(), &NoCache).unwrap();
        let output = result.output("add_one").unwrap().as_f64().unwrap();
        assert!((output - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_diamond() {
        let pipeline = PipelineBuilder::new()
            .source("data", || Ok(Value::from(100.0)))
            .node("branch_a", |b| b
                .input("x", "data")
                .compute(|inputs| {
                    let x = inputs["x"].as_f64().unwrap();
                    Ok(Value::from(x + 50.0))
                })
            )
            .node("branch_b", |b| b
                .input("x", "data")
                .compute(|inputs| {
                    let x = inputs["x"].as_f64().unwrap();
                    Ok(Value::from(x * 0.5))
                })
            )
            .node("merge", |b| b
                .input("a", "branch_a")
                .input("b", "branch_b")
                .compute(|inputs| {
                    let a = inputs["a"].as_f64().unwrap();
                    let b = inputs["b"].as_f64().unwrap();
                    Ok(Value::from(a - b))
                })
            )
            .build()
            .unwrap();

        let levels = pipeline.parallel_levels();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[1].len(), 2); // branch_a and branch_b in parallel

        let result = execute(&pipeline, &HashMap::new(), &NoCache).unwrap();
        let output = result.output("merge").unwrap().as_f64().unwrap();
        // (100 + 50) - (100 * 0.5) = 150 - 50 = 100
        assert!((output - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_auto_edges() {
        // No explicit .edge() calls — edges inferred from .input()
        let pipeline = PipelineBuilder::new()
            .source("a", || Ok(Value::from(1)))
            .node("b", |b| b.input("x", "a").compute(|i| Ok(i["x"].clone())))
            .build()
            .unwrap();

        assert_eq!(pipeline.edge_count(), 1);
        assert_eq!(pipeline.successors("a").len(), 1);
    }

    #[test]
    fn test_builder_cycle_rejected() {
        let result = PipelineBuilder::new()
            .source("a", || Ok(Value::Null))
            .source("b", || Ok(Value::Null))
            .edge("a", "b")
            .edge("b", "a")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_with_external_inputs() {
        let pipeline = PipelineBuilder::new()
            .node("greet", |b| b
                .input("name", "name")  // References external input "name"
                .compute(|inputs| {
                    let name = inputs["name"].as_str().unwrap_or("world");
                    Ok(Value::from(format!("Hello, {}!", name)))
                })
                .no_cache()
            )
            .build()
            .unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("name".to_string(), Value::from("MachinDeOuf"));

        let result = execute(&pipeline, &inputs, &NoCache).unwrap();
        let output = result.output("greet").unwrap().as_str().unwrap();
        assert_eq!(output, "Hello, MachinDeOuf!");
    }
}
