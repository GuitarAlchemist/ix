//! Demo mode — curated real-world scenarios that chain multiple ix tools.
//!
//! Each scenario generates deterministic synthetic data (seeded RNG),
//! executes a sequence of real tool calls through `ToolRegistry::call`,
//! and annotates each step with a narrative explaining what's happening.
//!
//! # Usage
//!
//! - MCP: `ix_demo { "action": "list" }` / `ix_demo { "action": "run", "scenario": "chaos-detective" }`
//! - CLI: `ix demo list` / `ix demo run chaos-detective`

pub mod scenarios;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ── Types ────────────────────────────────────────────────────────

/// Difficulty tier for scenario categorization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Difficulty {
    Beginner,
    Intermediate,
    Advanced,
}

/// Static metadata for a demo scenario.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioMeta {
    pub id: &'static str,
    pub title: &'static str,
    pub tagline: &'static str,
    pub description: &'static str,
    pub difficulty: Difficulty,
    pub tags: &'static [&'static str],
    pub tools_used: &'static [&'static str],
}

/// A demo scenario is a sequence of tool calls with narrative glue.
pub trait DemoScenario: Send + Sync {
    fn meta(&self) -> &ScenarioMeta;

    /// Build steps for this scenario.
    /// `seed` controls RNG for reproducible data. `verbosity`: 0=terse, 1=normal, 2=verbose.
    fn steps(&self, seed: u64, verbosity: u8) -> Vec<DemoStep>;
}

/// A glue function that transforms one step's output into the next step's input.
pub type GlueFn = Box<dyn Fn(&Value) -> Result<Value, String> + Send + Sync>;

/// How a step gets its input JSON.
pub enum StepInput {
    /// Static JSON blob (first step or no dependency).
    Static(Value),
    /// Transform previous step's output into this step's input.
    Glue(GlueFn),
}

/// One step in a demo scenario.
pub struct DemoStep {
    pub label: String,
    pub tool: String,
    pub input: StepInput,
    /// Narrative text shown before execution.
    pub narrative: String,
    /// Optional function to interpret the output in human terms.
    pub interpret: Option<fn(&Value) -> String>,
}

// ── Registry ─────────────────────────────────────────────────────

/// Find a scenario by id.
pub fn find_scenario(id: &str) -> Option<Box<dyn DemoScenario>> {
    scenarios::all().into_iter().find(|s| s.meta().id == id)
}

/// List all scenario metadata.
pub fn list_scenarios() -> Vec<&'static ScenarioMeta> {
    scenarios::all_meta()
}

// ── Executor ─────────────────────────────────────────────────────

/// Execute a demo scenario by calling real tool handlers.
pub fn run_scenario(id: &str, seed: u64, verbosity: u8) -> Result<Value, String> {
    let scenario = find_scenario(id)
        .ok_or_else(|| format!("unknown scenario: {id}"))?;

    let registry = crate::tools::ToolRegistry::new();
    let steps = scenario.steps(seed, verbosity);
    let mut step_results: Vec<Value> = Vec::with_capacity(steps.len());
    let mut prev_output: Option<Value> = None;
    let mut total_ms: u128 = 0;
    let mut succeeded = 0u32;
    let mut failed = 0u32;
    let mut tools_used: Vec<String> = Vec::new();

    for (i, step) in steps.iter().enumerate() {
        let input = match &step.input {
            StepInput::Static(v) => v.clone(),
            StepInput::Glue(f) => {
                let prev = prev_output.as_ref()
                    .ok_or_else(|| format!("step {i} has Glue input but no previous output"))?;
                f(prev)?
            }
        };

        let start = std::time::Instant::now();
        let result = registry.call(&step.tool, input.clone());
        let elapsed = start.elapsed().as_millis();
        total_ms += elapsed;

        let (output, success) = match result {
            Ok(v) => { succeeded += 1; (v, true) }
            Err(e) => { failed += 1; (json!({"error": e}), false) }
        };

        let interpretation = step.interpret
            .and_then(|f| if success { Some(f(&output)) } else { None });

        if !tools_used.contains(&step.tool) {
            tools_used.push(step.tool.clone());
        }

        step_results.push(json!({
            "index": i,
            "label": step.label,
            "tool": step.tool,
            "narrative": step.narrative,
            "input_summary": summarize_input(&input),
            "output": output,
            "interpretation": interpretation,
            "duration_ms": elapsed,
            "success": success,
        }));

        prev_output = Some(output);
    }

    let meta = scenario.meta();
    Ok(json!({
        "scenario": {
            "id": meta.id,
            "title": meta.title,
            "tagline": meta.tagline,
            "difficulty": meta.difficulty,
            "tags": meta.tags,
        },
        "steps": step_results,
        "summary": {
            "total_steps": steps.len(),
            "succeeded": succeeded,
            "failed": failed,
            "total_duration_ms": total_ms,
            "tools_exercised": tools_used,
            "seed": seed,
        }
    }))
}

/// Summarize large input arrays to keep output readable.
fn summarize_input(input: &Value) -> Value {
    match input {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                match v {
                    Value::Array(arr) if arr.len() > 10 => {
                        out.insert(k.clone(), json!(format!("[{} elements]", arr.len())));
                    }
                    _ => { out.insert(k.clone(), v.clone()); }
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

// ── MCP Handler ──────────────────────────────────────────────────

/// MCP tool handler for `ix_demo`.
pub fn ix_demo(params: Value) -> Result<Value, String> {
    let action = params.get("action")
        .and_then(|v| v.as_str())
        .ok_or("missing 'action' (list | run | describe)")?;

    match action {
        "list" => {
            let scenarios: Vec<Value> = list_scenarios().iter().map(|m| {
                json!({
                    "id": m.id,
                    "title": m.title,
                    "tagline": m.tagline,
                    "difficulty": m.difficulty,
                    "tags": m.tags,
                    "tools_used": m.tools_used,
                })
            }).collect();
            Ok(json!({ "scenarios": scenarios }))
        }
        "describe" => {
            let id = params.get("scenario")
                .and_then(|v| v.as_str())
                .ok_or("'scenario' required for describe")?;
            let scenario = find_scenario(id)
                .ok_or_else(|| format!("unknown scenario: {id}"))?;
            let meta = scenario.meta();
            let steps: Vec<Value> = scenario.steps(42, 1).iter().enumerate().map(|(i, s)| {
                json!({
                    "index": i,
                    "label": s.label,
                    "tool": s.tool,
                    "narrative": s.narrative,
                })
            }).collect();
            Ok(json!({
                "id": meta.id,
                "title": meta.title,
                "tagline": meta.tagline,
                "description": meta.description,
                "difficulty": meta.difficulty,
                "tags": meta.tags,
                "tools_used": meta.tools_used,
                "steps": steps,
            }))
        }
        "run" => {
            let id = params.get("scenario")
                .and_then(|v| v.as_str())
                .ok_or("'scenario' required for run")?;
            let seed = params.get("seed")
                .and_then(|v| v.as_u64())
                .unwrap_or(42);
            let verbosity = params.get("verbosity")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as u8;
            run_scenario(id, seed, verbosity)
        }
        other => Err(format!("unknown action: {other} (expected list | run | describe)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_scenarios() {
        let result = ix_demo(json!({"action": "list"})).unwrap();
        let scenarios = result["scenarios"].as_array().unwrap();
        assert!(!scenarios.is_empty(), "should have at least one scenario");
        for s in scenarios {
            assert!(s["id"].is_string());
            assert!(s["title"].is_string());
            assert!(s["difficulty"].is_string());
        }
    }

    #[test]
    fn test_describe_scenario() {
        let all = list_scenarios();
        let first_id = all[0].id;
        let result = ix_demo(json!({"action": "describe", "scenario": first_id})).unwrap();
        assert_eq!(result["id"].as_str().unwrap(), first_id);
        assert!(result["steps"].as_array().unwrap().len() >= 2);
    }

    #[test]
    fn test_run_scenario_deterministic() {
        let all = list_scenarios();
        let first_id = all[0].id;
        let r1 = ix_demo(json!({"action": "run", "scenario": first_id, "seed": 42})).unwrap();
        let r2 = ix_demo(json!({"action": "run", "scenario": first_id, "seed": 42})).unwrap();
        // Same seed → same outputs
        assert_eq!(r1["steps"], r2["steps"]);
    }

    #[test]
    fn test_unknown_scenario() {
        let result = ix_demo(json!({"action": "run", "scenario": "nonexistent"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_action() {
        let result = ix_demo(json!({"action": "foobar"}));
        assert!(result.is_err());
    }
}
