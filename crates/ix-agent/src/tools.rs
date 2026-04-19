//! Tool registry — defines MCP tools and dispatches calls.

use serde_json::{json, Value};
use std::collections::HashMap;

use crate::handlers;
use crate::registry_bridge;

/// Strip a ```json ... ``` (or plain ``` ... ```) markdown code fence
/// from an LLM response. The LLM is told not to emit fences, but in
/// practice it sometimes does anyway — this helper keeps the compiler
/// robust without punishing the first invalid response.
///
/// Only strips if the input starts with a triple backtick; otherwise
/// returns the original string unchanged. Trailing text after the
/// closing fence (e.g. "Here you go!") is preserved outside the fence
/// and not returned — the parser only sees the content between fences.
fn strip_markdown_fence(s: &str) -> &str {
    let trimmed = s.trim();
    if !trimmed.starts_with("```") {
        return trimmed;
    }
    // Drop the opening fence line (```json, ```, etc.)
    let after_open = match trimmed.find('\n') {
        Some(i) => &trimmed[i + 1..],
        None => return trimmed,
    };
    // Drop the closing fence and anything after it.
    match after_open.rfind("```") {
        Some(i) => after_open[..i].trim(),
        None => after_open.trim(),
    }
}

/// Walk a JSON value and replace any string of the form
/// `"$step_id.field.subfield"` with the corresponding value from
/// `upstream_results`. A bare `"$step_id"` replaces with the full
/// result. Useful for pipeline argument chaining without requiring
/// the LLM to marshal upstream outputs by hand.
fn substitute_refs(
    value: &Value,
    upstream: &HashMap<String, Value>,
) -> Result<Value, String> {
    match value {
        Value::String(s) if s.starts_with('$') => {
            let path = &s[1..];
            let mut parts = path.split('.');
            let step_id = parts
                .next()
                .ok_or_else(|| format!("empty reference '{s}'"))?;
            let mut current = upstream
                .get(step_id)
                .ok_or_else(|| format!("reference '{s}': step '{step_id}' has no result yet"))?
                .clone();
            for key in parts {
                // Numeric keys index into arrays when the current
                // value is an array. This is the minimal extension
                // needed to let pipeline specs say `$s.features.0`
                // for "the first row of the features matrix" (see
                // FINDINGS §5.C on substitution weakness). Non-numeric
                // keys always walk the object, as before.
                current = match (&current, key.parse::<usize>()) {
                    (Value::Array(arr), Ok(idx)) => arr.get(idx).cloned().ok_or_else(|| {
                        format!(
                            "reference '{s}': index {idx} out of bounds for array of length {}",
                            arr.len()
                        )
                    })?,
                    _ => current
                        .get(key)
                        .ok_or_else(|| format!("reference '{s}': missing field '{key}'"))?
                        .clone(),
                };
            }
            Ok(current)
        }
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                out.insert(k.clone(), substitute_refs(v, upstream)?);
            }
            Ok(Value::Object(out))
        }
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                out.push(substitute_refs(v, upstream)?);
            }
            Ok(Value::Array(out))
        }
        other => Ok(other.clone()),
    }
}

/// An MCP tool definition.
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
    pub handler: fn(Value) -> Result<Value, String>,
}

/// JSON schema for the `ix_supervised` tool. Extracted out of
/// `register_advanced_learning` because its 14-variant operation enum
/// plus 14 argument fields were the sole reason that method's
/// cyclomatic metric stayed above 20 after the P0.1 register_all
/// decomposition.
fn schema_ix_supervised() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["linear_regression", "logistic_regression", "svm", "knn", "naive_bayes", "decision_tree", "metrics", "cross_validate", "confusion_matrix", "roc_auc"],
                "description": "Algorithm, 'metrics' for evaluation, 'cross_validate' for k-fold CV, 'confusion_matrix' for confusion matrix, 'roc_auc' for ROC/AUC"
            },
            "x_train": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Training features matrix"
            },
            "y_train": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Training labels (class indices for classification, values for regression)"
            },
            "x_test": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Test features matrix"
            },
            "k": { "type": "integer", "description": "K for KNN (default 3), or number of CV folds (default 5)" },
            "c": { "type": "number", "description": "Regularization for SVM (default 1.0)" },
            "max_depth": { "type": "integer", "description": "Max depth for decision tree (default 5)" },
            "y_true": { "type": "array", "items": { "type": "number" }, "description": "True labels (for metrics/confusion_matrix)" },
            "y_pred": { "type": "array", "items": { "type": "number" }, "description": "Predicted labels (for metrics/confusion_matrix)" },
            "y_scores": { "type": "array", "items": { "type": "number" }, "description": "Predicted probabilities for positive class (for roc_auc)" },
            "metric_type": { "type": "string", "enum": ["mse", "accuracy"], "description": "Metric type: 'mse' for regression, 'accuracy' for classification" },
            "model": { "type": "string", "enum": ["knn", "decision_tree", "naive_bayes", "logistic_regression"], "description": "Model for cross_validate (default 'decision_tree')" },
            "n_classes": { "type": "integer", "description": "Number of classes (for confusion_matrix, auto-detected if omitted)" },
            "seed": { "type": "integer", "description": "Random seed for cross-validation (default 42)" }
        },
        "required": ["operation"]
    })
}

/// JSON schema for `ix_grammar_weights`. Extracted from
/// `register_core_symbolic` for the same reason as
/// `schema_ix_supervised`: the nested rule / observation objects
/// dominated that method's cyclomatic count.
fn schema_ix_grammar_weights() -> Value {
    json!({
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "description": "Grammar rules with weights",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":     { "type": "string" },
                        "alpha":  { "type": "number" },
                        "beta":   { "type": "number" },
                        "weight": { "type": "number" },
                        "level":  { "type": "integer" },
                        "source": { "type": "string" }
                    },
                    "required": ["id"]
                }
            },
            "observation": {
                "type": "object",
                "description": "Optional: apply a Bayesian update to one rule",
                "properties": {
                    "rule_id": { "type": "string" },
                    "success": { "type": "boolean" }
                },
                "required": ["rule_id", "success"]
            },
            "temperature": {
                "type": "number",
                "description": "Softmax temperature (default 1.0)",
                "minimum": 0
            }
        },
        "required": ["rules"]
    })
}

/// JSON schema for `ix_grammar_evolve`. Extracted from
/// `register_core_symbolic`.
fn schema_ix_grammar_evolve() -> Value {
    json!({
        "type": "object",
        "properties": {
            "species": {
                "type": "array",
                "description": "Initial grammar species",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":          { "type": "string" },
                        "proportion":  { "type": "number" },
                        "fitness":     { "type": "number" },
                        "is_stable":   { "type": "boolean" }
                    },
                    "required": ["id", "proportion", "fitness"]
                }
            },
            "steps": {
                "type": "integer",
                "description": "Number of simulation steps",
                "minimum": 1
            },
            "dt": {
                "type": "number",
                "description": "Time step (default 0.05)",
                "minimum": 0
            },
            "prune_threshold": {
                "type": "number",
                "description": "Proportion below which species are pruned (default 1e-6)",
                "minimum": 0
            }
        },
        "required": ["species", "steps"]
    })
}

/// JSON schema for `ix_grammar_search`. Extracted from
/// `register_core_symbolic`.
fn schema_ix_grammar_search() -> Value {
    json!({
        "type": "object",
        "properties": {
            "grammar_ebnf": {
                "type": "string",
                "description": "Grammar in EBNF notation (one rule per line: name ::= alt | alt)"
            },
            "max_iterations": {
                "type": "integer",
                "description": "MCTS iterations (default 500)",
                "minimum": 1
            },
            "exploration": {
                "type": "number",
                "description": "UCB1 exploration constant (default 1.41)",
                "minimum": 0
            },
            "max_depth": {
                "type": "integer",
                "description": "Max grammar expansion depth (default 20)",
                "minimum": 1
            },
            "seed": {
                "type": "integer",
                "description": "RNG seed for reproducibility (default 42)",
                "minimum": 0
            }
        },
        "required": ["grammar_ebnf"]
    })
}

/// Registry of all available tools.
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut reg = Self { tools: Vec::new() };
        reg.register_all();
        reg
    }

    /// List all tools as MCP tool definitions.
    pub fn list(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })
            })
            .collect();
        json!({ "tools": tools })
    }

    /// Call a tool by name with the given arguments.
    ///
    /// Registry-backed tools (handler == `registry_handler_marker`) are
    /// dispatched via `registry_bridge::dispatch`, which routes through
    /// `ix_registry::invoke`. Manual tools are called directly.
    pub fn call(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("Unknown tool: {}", name))?;
        if registry_bridge::is_registry_backed(tool.handler) {
            registry_bridge::dispatch(name, arguments)
        } else {
            (tool.handler)(arguments)
        }
    }

    /// Call a tool with a [`ServerContext`] available. Tools that need
    /// server-initiated JSON-RPC (e.g. MCP sampling) are intercepted by
    /// name and routed to their context-aware handler; everything else
    /// falls through to [`Self::call`]. Keeping this as a separate entry
    /// point avoids touching the `Tool` struct shape and the registry
    /// bridge's fn-pointer marker protocol.
    pub fn call_with_ctx(
        &self,
        name: &str,
        arguments: Value,
        ctx: &crate::server_context::ServerContext,
    ) -> Result<Value, String> {
        match name {
            "ix_explain_algorithm" => handlers::explain_algorithm_with_ctx(arguments, ctx),
            "ix_triage_session" => handlers::triage_session_with_ctx(arguments, ctx),
            "ix_pipeline_run" => self.run_pipeline(arguments),
            "ix_pipeline_compile" => self.compile_pipeline(arguments, ctx),
            _ => self.call(name, arguments),
        }
    }

    /// Execute a pipeline spec by topologically sorting steps and
    /// calling each step's tool via [`Self::call`].
    ///
    /// - **R1** (`ix-roadmap-plan-v1.md` §4.1): turns 13 hand-chained
    ///   MCP calls into a single submission with `$step_id.field`
    ///   cross-step argument references.
    /// - **R2 Phase 1** (§4.2): content-addressed cache hits when a
    ///   step declares an `asset_name`. The cache key is
    ///   `blake3(asset_name || tool || canonical_json(resolved_args))`,
    ///   so replaying the same pipeline skips every tool invocation
    ///   whose inputs are unchanged.
    ///
    /// Input format:
    /// ```json
    /// {
    ///   "steps": [
    ///     {
    ///       "id": "s1",
    ///       "tool": "ix_stats",
    ///       "arguments": {"data": [1, 2, 3]},
    ///       "asset_name": "stats_of_my_data"
    ///     },
    ///     {
    ///       "id": "s2",
    ///       "tool": "ix_fft",
    ///       "arguments": {"signal": "$s1.values"},
    ///       "depends_on": ["s1"]
    ///     }
    ///   ]
    /// }
    /// ```
    ///
    /// Output:
    /// ```json
    /// {
    ///   "results":          { "s1": {...}, "s2": {...} },
    ///   "execution_order":  ["s1", "s2"],
    ///   "durations_ms":     { "s1": 12, "s2": 3 },
    ///   "cache_hits":       ["s1"],
    ///   "cache_keys":       { "s1": "blake3:abc...", "s2": null }
    /// }
    /// ```
    fn run_pipeline(&self, args: Value) -> Result<Value, String> {
        use ix_pipeline::dag::Dag;
        use std::collections::HashMap;
        use std::time::Instant;

        let steps = args
            .get("steps")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "ix_pipeline_run: missing 'steps' array".to_string())?;

        // First pass: build a Dag<(tool, arguments, asset_name)>, remember
        // declared dependency edges, and validate. R2 Phase 2: also
        // capture per-step `depends_on` in a parallel map so we can emit
        // a provenance DAG in the response.
        let mut dag: Dag<(String, Value, Option<String>)> = Dag::new();
        let mut depends_on: HashMap<String, Vec<String>> = HashMap::new();
        for step in steps {
            let id = step
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or("ix_pipeline_run: each step needs 'id'")?;
            let tool = step
                .get("tool")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("ix_pipeline_run: step '{id}' missing 'tool'"))?;
            let arguments = step.get("arguments").cloned().unwrap_or_else(|| json!({}));
            let asset_name = step
                .get("asset_name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            dag.add_node(id, (tool.to_string(), arguments, asset_name))
                .map_err(|e| format!("ix_pipeline_run: step '{id}': {e}"))?;
            depends_on.insert(id.to_string(), Vec::new());
        }
        for step in steps {
            let id = step.get("id").and_then(|v| v.as_str()).unwrap();
            if let Some(deps) = step.get("depends_on").and_then(|v| v.as_array()) {
                for dep in deps {
                    if let Some(dep_id) = dep.as_str() {
                        dag.add_edge(dep_id, id).map_err(|e| {
                            format!("ix_pipeline_run: edge {dep_id} -> {id}: {e}")
                        })?;
                        depends_on
                            .get_mut(id)
                            .expect("id registered")
                            .push(dep_id.to_string());
                    }
                }
            }
        }

        // Second pass: topological sort, then execute each step.
        // topological_sort() borrows &dag, so clone the order to a
        // Vec<String> before we mutate results.
        let order: Vec<String> = dag.topological_sort().into_iter().cloned().collect();
        let mut results: HashMap<String, Value> = HashMap::new();
        let mut durations: HashMap<String, u128> = HashMap::new();
        let mut cache_hits: Vec<String> = Vec::new();
        let mut cache_keys: HashMap<String, Value> = HashMap::new();
        // R2 Phase 2: per-step provenance records.
        let mut lineage: HashMap<String, Value> = HashMap::new();
        let mut tools_by_step: HashMap<String, String> = HashMap::new();
        let mut assets_by_step: HashMap<String, Option<String>> = HashMap::new();

        let cache = crate::handlers::global_cache();

        for id in &order {
            let (tool, raw_args, asset_name) = dag
                .get(id)
                .ok_or_else(|| format!("ix_pipeline_run: missing node '{id}'"))?
                .clone();

            // Substitute any "$step_id.field" references with upstream results.
            let resolved = substitute_refs(&raw_args, &results).map_err(|e| {
                format!("ix_pipeline_run: step '{id}' arg substitution: {e}")
            })?;

            // Derive cache key if the step declared an asset_name.
            let cache_key: Option<String> = asset_name.as_ref().map(|name| {
                // Canonical JSON via serde_json::to_string: maps are
                // emitted in insertion order which is not deterministic
                // across runs. For R2 Phase 1 we accept this — the
                // LLM-submitted pipelines use fixed key order, and a
                // full canonicalization is a Phase 2 item.
                let canon = format!(
                    "asset={name}||tool={tool}||args={}",
                    serde_json::to_string(&resolved).unwrap_or_default()
                );
                let hash = blake3::hash(canon.as_bytes());
                format!("ix_pipeline_run:{}", hash.to_hex())
            });

            tools_by_step.insert(id.clone(), tool.clone());
            assets_by_step.insert(id.clone(), asset_name.clone());

            // Try cache lookup before dispatching the tool.
            if let Some(key) = &cache_key {
                if let Some(cached) = cache.get::<Value>(key) {
                    results.insert(id.clone(), cached);
                    durations.insert(id.clone(), 0);
                    cache_hits.push(id.clone());
                    cache_keys.insert(id.clone(), json!(key));
                    continue;
                }
            }

            let start = Instant::now();
            let result = self.call(&tool, resolved).map_err(|e| {
                format!("ix_pipeline_run: step '{id}' (tool '{tool}') failed: {e}")
            })?;
            let elapsed = start.elapsed().as_millis();

            // Store in cache on miss (only when asset_name was declared).
            if let Some(key) = &cache_key {
                cache.set(key, &result);
                cache_keys.insert(id.clone(), json!(key));
            } else {
                cache_keys.insert(id.clone(), Value::Null);
            }

            results.insert(id.clone(), result);
            durations.insert(id.clone(), elapsed);
        }

        // R2 Phase 2: build the provenance DAG. For each step, record:
        //   - tool + asset_name
        //   - its own content-addressed cache_key (Some on asset-backed
        //     steps, None otherwise)
        //   - depends_on: the list of upstream step ids
        //   - upstream_cache_keys: the cache keys of those upstream
        //     steps, which is what downstream audits walk to prove
        //     "where did this decision come from"
        for id in &order {
            let deps = depends_on
                .get(id)
                .cloned()
                .unwrap_or_default();
            let upstream_keys: Vec<Value> = deps
                .iter()
                .map(|dep| cache_keys.get(dep).cloned().unwrap_or(Value::Null))
                .collect();
            lineage.insert(
                id.clone(),
                json!({
                    "tool": tools_by_step.get(id).cloned().unwrap_or_default(),
                    "asset_name": assets_by_step.get(id).and_then(|o| o.clone()),
                    "cache_key": cache_keys.get(id).cloned().unwrap_or(Value::Null),
                    "depends_on": deps,
                    "upstream_cache_keys": upstream_keys,
                }),
            );
        }

        Ok(json!({
            "results": results,
            "execution_order": order,
            "durations_ms": durations,
            "cache_hits": cache_hits,
            "cache_keys": cache_keys,
            "lineage": lineage,
        }))
    }

    /// Compile a natural-language sentence into a `pipeline.json` DAG via
    /// MCP sampling. The handler builds a system + user prompt containing a
    /// trimmed registry summary (tool name + description) and two worked
    /// examples, asks the client's LLM to emit a JSON pipeline spec,
    /// strips any markdown fencing, validates the spec against the
    /// registry, and returns both the spec and the validation report.
    ///
    /// Input format:
    /// ```json
    /// {
    ///   "sentence": "analyse 5 ix crates for refactor candidates",
    ///   "max_steps": 12,
    ///   "context": { "note": "optional free-form hints" }
    /// }
    /// ```
    ///
    /// Output:
    /// ```json
    /// {
    ///   "status": "ok" | "invalid" | "parse_error",
    ///   "sentence": "...",
    ///   "spec": { "steps": [...] },
    ///   "validation": { "errors": [...], "warnings": [...] },
    ///   "raw_llm_response": "the unparsed LLM output (for debugging)"
    /// }
    /// ```
    ///
    /// A `status: "ok"` result is guaranteed to parse as JSON, have a
    /// `steps` array, reference only registered tools, and contain no
    /// dependency cycles. Callers can pass `spec` directly to
    /// `ix_pipeline_run`.
    fn compile_pipeline(
        &self,
        args: Value,
        ctx: &crate::server_context::ServerContext,
    ) -> Result<Value, String> {
        let sentence = args
            .get("sentence")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "ix_pipeline_compile: missing 'sentence' (string)".to_string())?;
        let max_steps = args
            .get("max_steps")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize;
        let context_hint = args
            .get("context")
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_default();

        // Build a compact registry summary: one line per tool.
        // Limit to 80 tools to keep the prompt within a few KB.
        let mut registry_summary = String::new();
        for tool in self.tools.iter().take(80) {
            registry_summary.push_str(&format!("- {}: {}\n", tool.name, tool.description));
        }

        let examples = r#"Example 1 — "baseline stats on AWS cost data":
{
  "steps": [
    {
      "id": "baseline",
      "tool": "ix_stats",
      "asset_name": "cost.baseline",
      "arguments": { "data": [994.24, 993.08, 995.55] }
    }
  ]
}

Example 2 — "cluster crates by complexity then classify":
{
  "steps": [
    {
      "id": "s01_baseline",
      "tool": "ix_stats",
      "asset_name": "demo.baseline",
      "arguments": { "data": [10.0, 12.0, 9.0, 50.0, 11.0] }
    },
    {
      "id": "s02_clusters",
      "tool": "ix_kmeans",
      "asset_name": "demo.clusters",
      "depends_on": ["s01_baseline"],
      "arguments": {
        "data": [[10.0], [12.0], [9.0], [50.0], [11.0]],
        "k": 2,
        "max_iter": 100
      }
    }
  ]
}
"#;

        let system_prompt = "You are the ix pipeline compiler. Given a natural-language \
            sentence and the ix tool registry, emit a JSON pipeline spec of the form \
            {\"steps\": [...]} that satisfies the request. \
            RULES: \
            (1) Use only tool names from the supplied registry. Do not invent names. \
            (2) Every step must have a unique 'id' (snake_case, prefixed 's01_', 's02_' ...) and a 'tool' name. \
            (3) Set 'asset_name' on every step using the pattern '<demo>.<slug>' so the runner can cache results. \
            (4) Use 'depends_on': [\"prev_id\"] for ordered steps. The DAG must be acyclic. \
            (5) Arguments must match the tool's expected schema; when in doubt, use conservative constants (e.g. small arrays, k=2, max_iter=100). \
            (6) Output ONLY the JSON pipeline spec. No prose, no markdown fences, no preamble. \
            (7) Keep it short and coherent — one sensible chain is better than 12 random tools.";

        let user_text = format!(
            "Request: {sentence}\n\n\
             Max steps: {max_steps}\n\n\
             Context hints: {context_hint}\n\n\
             {examples}\n\n\
             Available tools (name: description):\n{registry_summary}\n\n\
             Emit ONLY the pipeline JSON now, matching the {{\"steps\": [...]}} shape."
        );

        // Ask the client's LLM to compile the sentence.
        let raw = ctx.sample(&user_text, system_prompt, 4096)?;

        // Strip any ```json ... ``` markdown fence the LLM may have added.
        let cleaned = strip_markdown_fence(&raw);

        // Parse as JSON.
        let spec: Value = match serde_json::from_str(cleaned) {
            Ok(v) => v,
            Err(e) => {
                return Ok(json!({
                    "status": "parse_error",
                    "sentence": sentence,
                    "spec": Value::Null,
                    "validation": {
                        "errors": [format!("LLM response was not valid JSON: {e}")],
                        "warnings": [],
                    },
                    "raw_llm_response": raw,
                }));
            }
        };

        // Validate against the registry.
        let (errors, warnings) = self.validate_pipeline_spec(&spec);
        let status = if errors.is_empty() { "ok" } else { "invalid" };

        Ok(json!({
            "status": status,
            "sentence": sentence,
            "spec": spec,
            "validation": {
                "errors": errors,
                "warnings": warnings,
            },
            "raw_llm_response": raw,
        }))
    }

    /// Shape-check a pipeline spec against the tool registry. Returns
    /// `(errors, warnings)`. An empty errors list means the spec is safe
    /// to pass to [`Self::run_pipeline`].
    ///
    /// Checks performed:
    /// - `steps` is a non-empty JSON array
    /// - every step has string `id` and string `tool`
    /// - step ids are unique
    /// - every `tool` name exists in `self.tools`
    /// - `depends_on` references resolve to earlier step ids
    /// - the DAG has no cycles (uses `ix_pipeline::dag::Dag`)
    /// - missing `asset_name` is a warning, not an error
    pub fn validate_pipeline_spec(&self, spec: &Value) -> (Vec<String>, Vec<String>) {
        use ix_pipeline::dag::Dag;
        use std::collections::HashSet;

        let mut errors: Vec<String> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        let Some(steps) = spec.get("steps").and_then(|v| v.as_array()) else {
            errors.push("missing 'steps' array".to_string());
            return (errors, warnings);
        };
        if steps.is_empty() {
            errors.push("'steps' array is empty".to_string());
            return (errors, warnings);
        }

        let known_tools: HashSet<&str> = self.tools.iter().map(|t| t.name).collect();
        let mut seen_ids: HashSet<String> = HashSet::new();
        let mut dag: Dag<()> = Dag::new();

        // First pass: per-step structural checks and DAG node population.
        for (i, step) in steps.iter().enumerate() {
            let id = match step.get("id").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => {
                    errors.push(format!("step[{i}]: missing 'id' string"));
                    continue;
                }
            };
            if !seen_ids.insert(id.to_string()) {
                errors.push(format!("step[{i}]: duplicate id '{id}'"));
                continue;
            }
            match step.get("tool").and_then(|v| v.as_str()) {
                Some(name) => {
                    if !known_tools.contains(name) {
                        errors.push(format!("step '{id}': unknown tool '{name}'"));
                    }
                }
                None => {
                    errors.push(format!("step '{id}': missing 'tool' string"));
                }
            }
            if step.get("asset_name").is_none() {
                warnings.push(format!(
                    "step '{id}': no 'asset_name' — results will not be cached"
                ));
            }
            let _ = dag.add_node(id, ());
        }

        // Second pass: edges, cycle detection.
        for step in steps {
            let Some(id) = step.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(deps) = step.get("depends_on").and_then(|v| v.as_array()) else {
                continue;
            };
            for dep in deps {
                let Some(dep_id) = dep.as_str() else {
                    errors.push(format!("step '{id}': non-string entry in depends_on"));
                    continue;
                };
                if !seen_ids.contains(dep_id) {
                    errors.push(format!(
                        "step '{id}': depends_on references unknown step '{dep_id}'"
                    ));
                    continue;
                }
                if let Err(e) = dag.add_edge(dep_id, id) {
                    errors.push(format!("step '{id}': edge {dep_id} -> {id}: {e}"));
                }
            }
        }

        (errors, warnings)
    }

    /// Merge registry-sourced skills into the tool list, with registry
    /// taking precedence over any manual entry of the same name. Called at
    /// the end of [`Self::register_all`].
    fn merge_registry_tools(&mut self) {
        let registry_tools = registry_bridge::all_registry_tools();
        let registry_names: std::collections::HashSet<&str> =
            registry_tools.iter().map(|t| t.name).collect();
        // Drop manual tools that have been migrated to the registry.
        self.tools.retain(|t| !registry_names.contains(t.name));
        // Append the registry-sourced tools.
        self.tools.extend(registry_tools);
    }

    /// Top-level tool registration. Delegates to two sub-methods
    /// grouping the ~56 MCP tools by rough stability / layer — this
    /// exists purely to bound the single-function complexity. Prior
    /// to the split this function was 1,534 SLOC with cyclomatic 108
    /// (the worst function in the workspace per the adversarial
    /// refactor oracle's measurement of `ix-agent/src/tools.rs`).
    /// Each sub-method still has ~25 pushes but they're now bounded.
    /// Subsequent sessions may split further as the catalog / tool
    /// surface grows.
    fn register_all(&mut self) {
        self.register_all_core();
        self.register_all_advanced();
        self.register_all_governance_and_session();
        // Merge registry-sourced skills. Registry wins on name collision.
        self.merge_registry_tools();
    }

    /// First half of the tool registrations: core math, optimisation,
    /// classical ML, signal + chaos, search + game theory, probabilistic
    /// data structures, grammar, advanced math. Dispatches to four
    /// sub-methods to keep per-function complexity bounded
    /// (P0.1 structural refactor, round 2).
    fn register_all_core(&mut self) {
        self.register_core_numerics();
        self.register_core_search_sim();
        self.register_core_symbolic();
        self.register_core_structures();
    }

    /// Core sub-group 1: basic statistics, distance, optimisation,
    /// linear regression / k-means, FFT, and Markov chains.
    fn register_core_numerics(&mut self) {
        self.tools.push(Tool {
            name: "ix_stats",
            description: "Compute statistics (mean, std, min, max, median) on a list of numbers.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "List of numbers to compute statistics on"
                    }
                },
                "required": ["data"]
            }),
            handler: handlers::stats,
        });

        self.tools.push(Tool {
            name: "ix_distance",
            description: "Compute distance between two vectors (euclidean, cosine, or manhattan).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "a": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "First vector"
                    },
                    "b": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Second vector"
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["euclidean", "cosine", "manhattan"],
                        "description": "Distance metric to use"
                    }
                },
                "required": ["a", "b", "metric"]
            }),
            handler: handlers::distance,
        });

        self.tools.push(Tool {
            name: "ix_optimize",
            description: "Minimize a benchmark function (sphere, rosenbrock, rastrigin) using SGD, Adam, PSO, or simulated annealing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "enum": ["sphere", "rosenbrock", "rastrigin"],
                        "description": "Benchmark function to minimize"
                    },
                    "dimensions": {
                        "type": "integer",
                        "description": "Number of dimensions",
                        "minimum": 1
                    },
                    "method": {
                        "type": "string",
                        "enum": ["sgd", "adam", "pso", "annealing"],
                        "description": "Optimization method"
                    },
                    "max_iter": {
                        "type": "integer",
                        "description": "Maximum iterations",
                        "minimum": 1
                    }
                },
                "required": ["function", "dimensions", "method", "max_iter"]
            }),
            handler: handlers::optimize,
        });

        self.tools.push(Tool {
            name: "ix_linear_regression",
            description: "Fit ordinary least squares linear regression and return weights, bias, and predictions.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Feature matrix (rows=samples, cols=features)"
                    },
                    "y": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Target values"
                    }
                },
                "required": ["x", "y"]
            }),
            handler: handlers::linear_regression,
        });

        self.tools.push(Tool {
            name: "ix_kmeans",
            description: "K-Means clustering with K-Means++ initialization.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Data matrix (rows=samples, cols=features)"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of clusters",
                        "minimum": 1
                    },
                    "max_iter": {
                        "type": "integer",
                        "description": "Maximum iterations",
                        "minimum": 1
                    }
                },
                "required": ["data", "k", "max_iter"]
            }),
            handler: handlers::kmeans,
        });

        self.tools.push(Tool {
            name: "ix_fft",
            description: "Compute the Fast Fourier Transform of a real-valued signal. Returns frequency bins and magnitudes.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Real-valued signal samples"
                    }
                },
                "required": ["signal"]
            }),
            handler: handlers::fft,
        });

        self.tools.push(Tool {
            name: "ix_markov",
            description: "Analyze a Markov chain: compute stationary distribution after a number of steps.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "transition_matrix": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Row-stochastic transition matrix (rows sum to 1)"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of power-iteration steps for stationary distribution",
                        "minimum": 1
                    }
                },
                "required": ["transition_matrix", "steps"]
            }),
            handler: handlers::markov,
        });
    }

    /// Core sub-group 2: search / decision (Viterbi, A* / MCTS, Nash),
    /// chaos + adversarial, and Bloom filter.
    fn register_core_search_sim(&mut self) {
        self.tools.push(Tool {
            name: "ix_viterbi",
            description: "HMM Viterbi decoding: find the most likely hidden state sequence given observations.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "initial": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Initial state distribution (sums to 1)"
                    },
                    "transition": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "State transition matrix (row-stochastic)"
                    },
                    "emission": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Emission probability matrix (row-stochastic)"
                    },
                    "observations": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Observation sequence (indices into emission columns)"
                    }
                },
                "required": ["initial", "transition", "emission", "observations"]
            }),
            handler: handlers::viterbi,
        });

        self.tools.push(Tool {
            name: "ix_search",
            description: "Get information about search algorithms (A*, BFS, DFS) including descriptions and complexity.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["astar", "bfs", "dfs"],
                        "description": "Search algorithm to describe"
                    },
                    "description": {
                        "type": "boolean",
                        "description": "Whether to include a description"
                    }
                },
                "required": ["algorithm"]
            }),
            handler: handlers::search_info,
        });

        self.tools.push(Tool {
            name: "ix_game_nash",
            description: "Find Nash equilibria of a 2-player bimatrix game via support enumeration.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "payoff_a": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Player A payoff matrix"
                    },
                    "payoff_b": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Player B payoff matrix"
                    }
                },
                "required": ["payoff_a", "payoff_b"]
            }),
            handler: handlers::game_nash,
        });

        self.tools.push(Tool {
            name: "ix_chaos_lyapunov",
            description: "Compute the maximal Lyapunov exponent of the logistic map for a given parameter r.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "map": {
                        "type": "string",
                        "enum": ["logistic"],
                        "description": "Map type (currently only 'logistic')"
                    },
                    "parameter": {
                        "type": "number",
                        "description": "Map parameter (r for logistic map, 0 < r <= 4)"
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Number of iterations for Lyapunov computation",
                        "minimum": 1
                    }
                },
                "required": ["map", "parameter", "iterations"]
            }),
            handler: handlers::chaos_lyapunov,
        });

        self.tools.push(Tool {
            name: "ix_adversarial_fgsm",
            description: "Fast Gradient Sign Method: compute adversarial perturbation of an input given its loss gradient.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Original input vector"
                    },
                    "gradient": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Loss gradient w.r.t. input"
                    },
                    "epsilon": {
                        "type": "number",
                        "description": "Perturbation magnitude"
                    }
                },
                "required": ["input", "gradient", "epsilon"]
            }),
            handler: handlers::adversarial_fgsm,
        });

        self.tools.push(Tool {
            name: "ix_bloom_filter",
            description: "Create a Bloom filter from items and check membership. Returns whether query items are (probably) in the set.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "check"],
                        "description": "Operation: 'create' inserts items, 'check' tests membership"
                    },
                    "items": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Items to insert into the Bloom filter"
                    },
                    "query": {
                        "type": "string",
                        "description": "Item to check membership for (used with 'check')"
                    },
                    "false_positive_rate": {
                        "type": "number",
                        "description": "Desired false positive rate (e.g. 0.01)",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["items", "false_positive_rate"]
            }),
            handler: handlers::bloom_filter,
        });
    }

    /// Core sub-group 3: symbolic — grammar weighting / evolution /
    /// search, plus rotation / number theory / fractal / sedenion
    /// advanced-math primitives.
    fn register_core_symbolic(&mut self) {
        self.tools.push(Tool {
            name: "ix_grammar_weights",
            description: "Bayesian (Beta-Binomial) update of grammar rule weights and softmax probability query. \
                Returns updated rule weights and selection probabilities.",
            input_schema: schema_ix_grammar_weights(),
            handler: handlers::grammar_weights,
        });

        self.tools.push(Tool {
            name: "ix_grammar_evolve",
            description: "Simulate grammar rule competition via replicator dynamics. \
                Returns the final species proportions, full trajectory, and detected ESS.",
            input_schema: schema_ix_grammar_evolve(),
            handler: handlers::grammar_evolve,
        });

        self.tools.push(Tool {
            name: "ix_grammar_search",
            description: "Grammar-guided MCTS derivation search. \
                Finds the best sentence derivation from an EBNF grammar using Monte Carlo Tree Search.",
            input_schema: schema_ix_grammar_search(),
            handler: handlers::grammar_search,
        });

        // ── New crate tools (Phase 4) ──────────────────────────────

        self.tools.push(Tool {
            name: "ix_rotation",
            description: "3D rotation operations: quaternion from axis-angle, SLERP interpolation, Euler angle conversion, rotation matrix.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["quaternion", "slerp", "euler_to_quat", "quat_to_euler", "rotate_point", "rotation_matrix"],
                        "description": "Rotation operation to perform"
                    },
                    "axis": { "type": "array", "items": { "type": "number" }, "description": "Rotation axis [x,y,z]" },
                    "angle": { "type": "number", "description": "Rotation angle in radians" },
                    "axis2": { "type": "array", "items": { "type": "number" }, "description": "Second rotation axis (for SLERP)" },
                    "angle2": { "type": "number", "description": "Second angle (for SLERP)" },
                    "t": { "type": "number", "description": "Interpolation parameter 0..1 (for SLERP)" },
                    "roll": { "type": "number", "description": "Roll in radians (for Euler)" },
                    "pitch": { "type": "number", "description": "Pitch in radians (for Euler)" },
                    "yaw": { "type": "number", "description": "Yaw in radians (for Euler)" },
                    "point": { "type": "array", "items": { "type": "number" }, "description": "Point [x,y,z] to rotate" },
                    "quaternion": { "type": "array", "items": { "type": "number" }, "description": "Quaternion [w,x,y,z]" }
                },
                "required": ["operation"]
            }),
            handler: handlers::rotation,
        });

        self.tools.push(Tool {
            name: "ix_number_theory",
            description: "Number theory: prime sieve, primality testing, modular arithmetic (mod_pow, gcd, lcm, mod_inverse).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["sieve", "is_prime", "mod_pow", "gcd", "lcm", "mod_inverse", "prime_gaps"],
                        "description": "Number theory operation"
                    },
                    "limit": { "type": "integer", "description": "Upper limit for sieve", "minimum": 2 },
                    "n": { "type": "integer", "description": "Number to test for primality" },
                    "base": { "type": "integer", "description": "Base for mod_pow" },
                    "exp": { "type": "integer", "description": "Exponent for mod_pow" },
                    "modulus": { "type": "integer", "description": "Modulus for mod_pow/mod_inverse" },
                    "a": { "type": "integer", "description": "First number for gcd/lcm" },
                    "b": { "type": "integer", "description": "Second number for gcd/lcm" }
                },
                "required": ["operation"]
            }),
            handler: handlers::number_theory,
        });

        self.tools.push(Tool {
            name: "ix_fractal",
            description: "Generate fractal data: Takagi curve, Hilbert/Peano space-filling curves, Morton encoding.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["takagi", "hilbert", "peano", "morton_encode", "morton_decode"],
                        "description": "Fractal operation"
                    },
                    "n_points": { "type": "integer", "description": "Number of points for Takagi curve", "minimum": 2 },
                    "terms": { "type": "integer", "description": "Number of terms for Takagi", "minimum": 1 },
                    "order": { "type": "integer", "description": "Order for space-filling curves", "minimum": 1 },
                    "x": { "type": "integer", "description": "X coordinate for Morton encode" },
                    "y": { "type": "integer", "description": "Y coordinate for Morton encode" },
                    "z": { "type": "integer", "description": "Z-order value for Morton decode" }
                },
                "required": ["operation"]
            }),
            handler: handlers::fractal,
        });

        self.tools.push(Tool {
            name: "ix_sedenion",
            description: "Hypercomplex algebra: sedenion/octonion multiplication, conjugate, norm, Cayley-Dickson construction.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["multiply", "conjugate", "norm", "cayley_dickson_multiply"],
                        "description": "Sedenion operation"
                    },
                    "a": { "type": "array", "items": { "type": "number" }, "description": "First element components (16 for sedenion)" },
                    "b": { "type": "array", "items": { "type": "number" }, "description": "Second element components (for multiply)" }
                },
                "required": ["operation", "a"]
            }),
            handler: handlers::sedenion,
        });
    }

    /// Core sub-group 4: structural — topology, category theory,
    /// neural network forward pass, bandit / evolution / random
    /// forest / gradient boosting.
    fn register_core_structures(&mut self) {
        self.tools.push(Tool {
            name: "ix_topo",
            description: "Topological data analysis: persistent homology, Betti numbers, Betti curves from point clouds.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["persistence", "betti_at_radius", "betti_curve"],
                        "description": "TDA operation"
                    },
                    "points": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Point cloud (rows=points, cols=dimensions)"
                    },
                    "max_dim": { "type": "integer", "description": "Maximum homology dimension (default 1)", "minimum": 0 },
                    "max_radius": { "type": "number", "description": "Maximum filtration radius (default 2.0)" },
                    "radius": { "type": "number", "description": "Radius for betti_at_radius" },
                    "n_steps": { "type": "integer", "description": "Number of steps for betti_curve (default 50)" }
                },
                "required": ["operation", "points"]
            }),
            handler: handlers::topo,
        });

        self.tools.push(Tool {
            name: "ix_category",
            description: "Category theory: verify monad laws for Option/Result monads with sample functions.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["monad_laws", "free_forgetful"],
                        "description": "Category theory operation"
                    },
                    "monad": {
                        "type": "string",
                        "enum": ["option", "result"],
                        "description": "Which monad to verify laws for"
                    },
                    "value": { "type": "integer", "description": "Value to test monad laws with" },
                    "elements": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Elements for free-forgetful adjunction"
                    }
                },
                "required": ["operation"]
            }),
            handler: handlers::category,
        });

        self.tools.push(Tool {
            name: "ix_nn_forward",
            description: "Neural network forward pass: dense layer, MSE/BCE loss, attention, positional encodings.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["dense_forward", "mse_loss", "bce_loss", "sinusoidal_encoding", "attention"],
                        "description": "Neural network operation"
                    },
                    "input": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Input matrix (rows=batch, cols=features)"
                    },
                    "target": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Target matrix (for loss computation)"
                    },
                    "output_size": { "type": "integer", "description": "Output size for dense layer" },
                    "max_len": { "type": "integer", "description": "Max sequence length for positional encoding" },
                    "d_model": { "type": "integer", "description": "Model dimension for positional encoding" },
                    "seed": { "type": "integer", "description": "RNG seed (default 42)" }
                },
                "required": ["operation"]
            }),
            handler: handlers::nn_forward,
        });

        self.tools.push(Tool {
            name: "ix_bandit",
            description: "Multi-armed bandit simulation: run epsilon-greedy, UCB1, or Thompson sampling for N rounds.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["epsilon_greedy", "ucb1", "thompson"],
                        "description": "Bandit algorithm"
                    },
                    "n_arms": { "type": "integer", "description": "Number of arms", "minimum": 1 },
                    "true_means": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "True mean rewards for each arm (for simulation)"
                    },
                    "rounds": { "type": "integer", "description": "Number of rounds to simulate", "minimum": 1 },
                    "epsilon": { "type": "number", "description": "Epsilon for epsilon-greedy (default 0.1)" }
                },
                "required": ["algorithm", "true_means", "rounds"]
            }),
            handler: handlers::bandit,
        });

        self.tools.push(Tool {
            name: "ix_evolution",
            description: "Evolutionary optimization: genetic algorithm or differential evolution on benchmark functions.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["genetic", "differential"],
                        "description": "Evolution algorithm"
                    },
                    "function": {
                        "type": "string",
                        "enum": ["sphere", "rosenbrock", "rastrigin"],
                        "description": "Benchmark function to minimize"
                    },
                    "dimensions": { "type": "integer", "description": "Number of dimensions", "minimum": 1 },
                    "generations": { "type": "integer", "description": "Number of generations", "minimum": 1 },
                    "population_size": { "type": "integer", "description": "Population size (default 50)" },
                    "mutation_rate": { "type": "number", "description": "Mutation rate for GA (default 0.1)" }
                },
                "required": ["algorithm", "function", "dimensions", "generations"]
            }),
            handler: handlers::evolution,
        });

        self.tools.push(Tool {
            name: "ix_random_forest",
            description: "Random forest classifier: train on data and predict class labels with probability estimates.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x_train": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Training feature matrix"
                    },
                    "y_train": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Training labels (class indices)"
                    },
                    "x_test": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Test feature matrix to predict"
                    },
                    "n_trees": { "type": "integer", "description": "Number of trees (default 10)", "minimum": 1 },
                    "max_depth": { "type": "integer", "description": "Max tree depth (default 5)", "minimum": 1 }
                },
                "required": ["x_train", "y_train", "x_test"]
            }),
            handler: handlers::random_forest,
        });

        self.tools.push(Tool {
            name: "ix_gradient_boosting",
            description: "Gradient boosted trees classifier: train on data and predict class labels with probability estimates. Supports binary and multiclass classification.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x_train": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Training feature matrix"
                    },
                    "y_train": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Training labels (class indices)"
                    },
                    "x_test": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Test feature matrix to predict"
                    },
                    "n_estimators": { "type": "integer", "description": "Number of boosting rounds (default 50)", "minimum": 1 },
                    "learning_rate": { "type": "number", "description": "Step size shrinkage (default 0.1)", "minimum": 0.001 },
                    "max_depth": { "type": "integer", "description": "Max weak learner depth (default 3)", "minimum": 1 }
                },
                "required": ["x_train", "y_train", "x_test"]
            }),
            handler: handlers::gradient_boosting,
        });
    }

    /// Second half of the tool registrations: supervised learning,
    /// graph, probabilistic sketches, autograd, pipeline orchestration,
    /// catalogs, source adapters. Dispatches to four sub-methods to
    /// keep per-function complexity bounded (P0.1 structural refactor).
    fn register_all_advanced(&mut self) {
        self.register_advanced_learning();
        self.register_advanced_pipelines();
        self.register_advanced_catalogs();
        self.register_advanced_misc();
    }

    /// Advanced sub-group 1: supervised learning, graph algorithms,
    /// and probabilistic sketches.
    fn register_advanced_learning(&mut self) {
        self.tools.push(Tool {
            name: "ix_supervised",
            description: "Supervised learning: train and predict with linear/logistic regression, SVM, KNN, naive Bayes, decision tree. Compute metrics (accuracy, confusion matrix, ROC/AUC, log loss). Cross-validate models with k-fold.",
            input_schema: schema_ix_supervised(),
            handler: handlers::supervised,
        });

        self.tools.push(Tool {
            name: "ix_graph",
            description: "Graph algorithms: Dijkstra shortest path, BFS, DFS, PageRank, topological sort on weighted directed/undirected graphs.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["dijkstra", "shortest_path", "pagerank", "bfs", "dfs", "topological_sort"],
                        "description": "Graph algorithm"
                    },
                    "n_nodes": { "type": "integer", "description": "Number of nodes in the graph" },
                    "edges": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "Edges as [from, to, weight] triples"
                    },
                    "directed": { "type": "boolean", "description": "Whether graph is directed (default true)" },
                    "source": { "type": "integer", "description": "Source node for path/traversal algorithms" },
                    "target": { "type": "integer", "description": "Target node for shortest_path" },
                    "damping": { "type": "number", "description": "Damping factor for PageRank (default 0.85)" },
                    "iterations": { "type": "integer", "description": "Iterations for PageRank (default 100)" }
                },
                "required": ["operation", "n_nodes", "edges"]
            }),
            handler: handlers::graph_ops,
        });

        self.tools.push(Tool {
            name: "ix_hyperloglog",
            description: "HyperLogLog cardinality estimation: estimate unique item count with configurable precision, or merge multiple sketches.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["estimate", "merge"],
                        "description": "'estimate' for single set, 'merge' for union of multiple sets"
                    },
                    "items": {
                        "type": "array",
                        "description": "Items to count (strings or numbers) — for 'estimate'"
                    },
                    "sets": {
                        "type": "array",
                        "items": { "type": "array" },
                        "description": "Array of item arrays — for 'merge'"
                    },
                    "precision": { "type": "integer", "description": "HLL precision 4-18 (default 14, ~0.81% error)", "minimum": 4, "maximum": 18 }
                },
                "required": ["operation"]
            }),
            handler: handlers::hyperloglog,
        });
    }

    /// Advanced sub-group 2: autograd execution and pipeline compile /
    /// run primitives (R7 Week 2 + ix_pipeline_* family).
    fn register_advanced_pipelines(&mut self) {
        self.tools.push(Tool {
            name: "ix_autograd_run",
            description: "R7 Week 2: run a differentiable tool (LinearRegressionTool, StatsVarianceTool) end-to-end, returning forward outputs AND per-input gradients in a single MCP call. Inputs are nested f64 arrays; shape is inferred from nesting depth. Used for pipeline-level gradient descent where the caller maintains an Adam/SGD loop outside the MCP boundary.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": ["linear_regression", "stats_variance"],
                        "description": "Name of the differentiable tool to run"
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Map of input name → nested f64 array (1-D, 2-D, or scalar). For linear_regression: x, w, b, y. For stats_variance: x.",
                        "additionalProperties": true
                    }
                },
                "required": ["tool", "inputs"]
            }),
            handler: handlers::autograd_run,
        });

        self.tools.push(Tool {
            name: "ix_pipeline_run",
            description: "Execute a DAG pipeline end-to-end: topologically sorts steps, dispatches each step's tool with substituted upstream references, and returns per-step results + durations. Replaces hand-chaining of MCP calls. Reference upstream outputs in step arguments via the string `\"$step_id.field\"`. Handled by ToolRegistry::call_with_ctx; this entry exists only for tools/list discovery.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "Step identifier, used for cross-step references" },
                                "tool": { "type": "string", "description": "Name of the MCP tool to invoke, e.g. 'ix_stats'" },
                                "arguments": { "type": "object", "description": "Arguments to pass to the tool; may contain $step_id.field references" },
                                "depends_on": { "type": "array", "items": { "type": "string" }, "description": "IDs of prerequisite steps that must complete before this one" }
                            },
                            "required": ["id", "tool"]
                        },
                        "description": "Ordered or unordered set of pipeline steps; execution order is derived from depends_on"
                    }
                },
                "required": ["steps"]
            }),
            handler: handlers::pipeline_run_placeholder,
        });

        self.tools.push(Tool {
            name: "ix_pipeline_compile",
            description: "Compile a natural-language sentence into a pipeline.json DAG via MCP sampling. The handler asks the client's LLM to emit a JSON {steps: [...]} spec using only registered ix tools, then validates the result against the registry (unknown tools, duplicate ids, unresolved depends_on, cycles). Returns status 'ok' when the spec is safe to pass directly to ix_pipeline_run. Handled by ToolRegistry::call_with_ctx; this entry exists only for tools/list discovery.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "sentence": {
                        "type": "string",
                        "description": "Natural-language description of the analysis you want to run"
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Upper bound on the number of steps the compiler is allowed to emit. Default 12."
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional free-form context hints (data bindings, preferred tools, etc.) forwarded to the LLM prompt",
                        "additionalProperties": true
                    }
                },
                "required": ["sentence"]
            }),
            handler: handlers::pipeline_compile_placeholder,
        });
    }

    /// Advanced sub-group 3: catalog tools (code / grammar / RFC /
    /// meta) plus workspace introspection (cargo deps, git log).
    fn register_advanced_catalogs(&mut self) {
        self.tools.push(Tool {
            name: "ix_catalog_list",
            description: "Meta-tool: list every registered ix catalog (code_analysis, grammar, rfc, ...) with its name, scope, and entry count. Use this to discover what catalogs ix exposes before issuing a specific ix_*_catalog query.",
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
            handler: handlers::catalog_list,
        });

        self.tools.push(Tool {
            name: "ix_grammar_catalog",
            description: "Query a curated catalog of ~30 real-world grammar sources across EBNF, ABNF, PEG, ANTLR G4, W3C EBNF, and BNF notations. Covers programming languages (Python, Go, ECMAScript, Rust, C11, ...), data formats (JSON, TOML, YAML, SQL-2016, GraphQL), IETF protocols (HTTP, TLS, DNS, SMTP, IMAP, OAuth, WebSockets), and meta-grammars (ABNF, ISO 14977 EBNF). Filter by language, format, or topic.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Case-insensitive language filter (e.g. 'python', 'http'). Meta-entries with language='many' always pass."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["ebnf", "w3c_ebnf", "abnf", "peg", "antlr_g4", "bnf"],
                        "description": "Filter by grammar notation format."
                    },
                    "topic": {
                        "type": "string",
                        "description": "Case-insensitive substring match against topic tags (e.g. 'web', 'protocol', 'meta')."
                    }
                }
            }),
            handler: handlers::grammar_catalog,
        });

        self.tools.push(Tool {
            name: "ix_rfc_catalog",
            description: "Query a curated catalog of ~70 IETF RFCs covering the modern internet stack: IP, TCP/QUIC, HTTP/1.1/2/3, TLS 1.3, DNS+DNSSEC, SMTP/IMAP, OAuth/JOSE/JWT, SSH, SIP/RTP, JSON/CBOR/UUID, ABNF, and BCPs. Includes the obsolescence graph — passing current_standard=true filters out obsoleted entries, and obsolescence_chain=N walks both directions from the seed RFC (useful for 'what replaced RFC 2616' questions).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "Exact RFC number lookup (e.g. 9110)."
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic substring (e.g. 'http', 'dns', 'tls', 'auth'). Case-insensitive."
                    },
                    "status": {
                        "type": "string",
                        "enum": [
                            "internet_standard",
                            "proposed_standard",
                            "draft_standard",
                            "experimental",
                            "informational",
                            "obsoleted"
                        ],
                        "description": "Filter by publication status."
                    },
                    "current_standard": {
                        "type": "boolean",
                        "description": "When true, excludes obsoleted entries. Combine with topic to get 'the current spec for X'."
                    },
                    "obsolescence_chain": {
                        "type": "integer",
                        "description": "Return the complete obsolescence chain (walked in both directions) for this RFC number. Overrides other filters."
                    }
                }
            }),
            handler: handlers::rfc_catalog,
        });

        self.tools.push(Tool {
            name: "ix_code_catalog",
            description: "Query a curated catalog of external mathematical tools for analysing programming-language repositories (static analysers, formal verifiers, safety / memory checkers, statistical + behavioural analysis tools, documentation generators, and numeric libraries). Filter by language, category, or technique substring. Use this to route users to the right specialist rather than over-stretching ix_code_analyze.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Case-insensitive language filter (e.g. 'rust', 'python'). Language-agnostic tools are always included."
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "static_analysis",
                            "formal_verification",
                            "safety_memory",
                            "statistical_analysis",
                            "documentation",
                            "numeric_library",
                            "ml_framework",
                            "fuzzing",
                            "supply_chain"
                        ],
                        "description": "One of nine categories; omit to include all."
                    },
                    "technique": {
                        "type": "string",
                        "description": "Substring match against the 'technique' field (e.g. 'cyclomatic', 'abstract interpretation', 'model checking')."
                    }
                }
            }),
            handler: handlers::code_catalog,
        });

        self.tools.push(Tool {
            name: "ix_cargo_deps",
            description: "P1.2 — walk a Rust workspace, parse every crates/<name>/Cargo.toml for intra-workspace ix-* dependencies, and emit a {nodes, edges, n_nodes} structure that ix_graph can consume directly. Each node records {id, name, sloc, file_count, dep_count}. Edges are [from_id, to_id, 1.0] triples. Default workspace_root is the process CWD.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "workspace_root": {
                        "type": "string",
                        "description": "Absolute or relative path to the workspace root (the directory containing 'crates/'). Defaults to the process CWD."
                    }
                }
            }),
            handler: handlers::cargo_deps,
        });

        self.tools.push(Tool {
            name: "ix_git_log",
            description: "P1.1 — shell out to `git log` on a repo-internal path and return a normalized commit cadence time series. Buckets commits into per-day or per-week dense arrays so downstream tools (ix_fft, ix_stats, ix_chaos_lyapunov) can consume the output directly. Every argument is passed through Command::arg(), not shell concatenation, and 'path' is whitelist-validated to reject '..', absolute prefixes, and shell metacharacters.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative repo-internal path to scope the git log to (e.g. 'crates/ix-agent'). Must not contain '..', absolute prefixes, or shell metacharacters."
                    },
                    "since_days": {
                        "type": "integer",
                        "description": "Window size in days ending today. Default 90. Must be in 1..=3650."
                    },
                    "bucket": {
                        "type": "string",
                        "enum": ["day", "week"],
                        "description": "Bucket size for the output time series. Default 'day'."
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Optional absolute path to a git repository root. When provided, git runs as if invoked from that directory via `git -C <root>`. Use this when the MCP server's CWD is not the repo root."
                    }
                },
                "required": ["path"]
            }),
            handler: handlers::git_log,
        });
    }

    /// Advanced sub-group 4: pipeline discovery / execution helpers
    /// and the in-memory cache.
    fn register_advanced_misc(&mut self) {
        self.tools.push(Tool {
            name: "ix_pipeline_list",
            description: "Discover canonical-showcase pipeline.json specs under a directory (default 'examples/canonical-showcase'). Returns metadata for each spec — name, description, step count, and the list of tools it uses. Companion to ix_pipeline_run for pipeline browsing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Directory to scan for '<subdir>/pipeline.json' specs. Relative paths resolve against CWD. Default: 'examples/canonical-showcase'."
                    }
                }
            }),
            handler: handlers::pipeline_list,
        });

        self.tools.push(Tool {
            name: "ix_pipeline",
            description: "DAG pipeline analysis: define steps with dependencies, get topological order, parallel execution levels, and critical path info.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["info"],
                        "description": "Pipeline operation"
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "Step identifier" },
                                "description": { "type": "string", "description": "Step description" },
                                "depends_on": { "type": "array", "items": { "type": "string" }, "description": "IDs of prerequisite steps" }
                            },
                            "required": ["id"]
                        },
                        "description": "Pipeline step definitions"
                    }
                },
                "required": ["operation", "steps"]
            }),
            handler: handlers::pipeline_exec,
        });

        self.tools.push(Tool {
            name: "ix_cache",
            description: "In-memory cache operations: set, get, delete, or list keys.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["set", "get", "delete", "keys"],
                        "description": "Cache operation to perform"
                    },
                    "key": {
                        "type": "string",
                        "description": "Cache key (required for set/get/delete)"
                    },
                    "value": {
                        "description": "Value to store (required for set, any JSON value)"
                    }
                },
                "required": ["operation"]
            }),
            handler: handlers::cache_op,
        });
    }

    /// Third section: governance, federation bridges, and
    /// session / telemetry / explainer utilities. Dispatches to four
    /// sub-methods (P0.1 round 3) to keep per-function complexity
    /// bounded.
    fn register_all_governance_and_session(&mut self) {
        self.register_governance();
        self.register_federation_discovery();
        self.register_ml_and_code();
        self.register_bridges_and_session();
    }

    /// Governance sub-group 1: the four Demerzel governance primitives
    /// (compliance check, persona loader, belief query, policy).
    fn register_governance(&mut self) {
        self.tools.push(Tool {
            name: "ix_governance_check",
            description: "Check a proposed action against the Demerzel constitution for compliance. Optionally accepts a 'lineage' object emitted by ix_pipeline_run to record upstream provenance alongside the verdict (R2 Phase 2 audit trail).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The proposed action to check for compliance"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context for the action"
                    },
                    "lineage": {
                        "type": "object",
                        "description": "Optional lineage map emitted by ix_pipeline_run. When provided, the response includes a 'lineage_audit' summary with step-by-step provenance (tool, asset_name, cache_key, upstream_cache_keys) so auditors can trace which assets fed into the decision.",
                        "additionalProperties": true
                    }
                },
                "required": ["action"]
            }),
            handler: handlers::governance_check,
        });

        self.tools.push(Tool {
            name: "ix_governance_persona",
            description: "Load a Demerzel persona by name — returns capabilities, constraints, voice, interaction patterns",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "persona": {
                        "type": "string",
                        "enum": ["default", "kaizen-optimizer", "reflective-architect", "skeptical-auditor", "system-integrator"],
                        "description": "Persona name to load"
                    }
                },
                "required": ["persona"]
            }),
            handler: handlers::governance_persona,
        });

        self.tools.push(Tool {
            name: "ix_governance_belief",
            description: "Manage beliefs with tetravalent logic (True/False/Unknown/Contradictory)",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "update", "resolve"],
                        "description": "Belief operation"
                    },
                    "proposition": {
                        "type": "string",
                        "description": "The proposition to evaluate"
                    },
                    "truth_value": {
                        "type": "string",
                        "enum": ["T", "F", "U", "C"],
                        "description": "Initial truth value (True/False/Unknown/Contradictory)"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level 0.0–1.0"
                    },
                    "supporting": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Supporting evidence claims"
                    },
                    "contradicting": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Contradicting evidence claims"
                    }
                },
                "required": ["operation", "proposition"]
            }),
            handler: handlers::governance_belief,
        });

        self.tools.push(Tool {
            name: "ix_governance_policy",
            description: "Query Demerzel governance policies — alignment thresholds, rollback triggers, self-modification rules",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "policy": {
                        "type": "string",
                        "enum": ["alignment", "rollback", "self-modification"],
                        "description": "Policy to query"
                    },
                    "query": {
                        "type": "string",
                        "enum": ["thresholds", "triggers", "allowed"],
                        "description": "What aspect of the policy to query"
                    }
                },
                "required": ["policy"]
            }),
            handler: handlers::governance_policy,
        });
    }

    /// Governance sub-group 2: federation discovery and trace ingest.
    fn register_federation_discovery(&mut self) {
        self.tools.push(Tool {
            name: "ix_federation_discover",
            description: "Discover capabilities across the GuitarAlchemist ecosystem (ix, tars, ga)",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Filter by domain (e.g. 'math', 'grammar', 'music-theory')"
                    },
                    "query": {
                        "type": "string",
                        "description": "Free-text search across tool names and descriptions"
                    }
                }
            }),
            handler: handlers::federation_discover,
        });

        self.tools.push(Tool {
            name: "ix_trace_ingest",
            description: "Ingest GA trace files from a directory and compute statistics (counts, durations, percentiles, event breakdowns)",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "dir": {
                        "type": "string",
                        "description": "Path to trace directory (default: ~/.ga/traces/)"
                    }
                }
            }),
            handler: handlers::trace_ingest,
        });
    }

    /// Governance sub-group 3: higher-level ML (ml_pipeline, ml_predict)
    /// and source code analysis.
    fn register_ml_and_code(&mut self) {
        self.tools.push(Tool {
            name: "ix_ml_pipeline",
            description: "End-to-end ML pipeline: load data, preprocess, train a model, evaluate metrics, and optionally persist. Supports classification (KNN, decision tree, random forest), regression (linear), and clustering (K-Means). Set task/model to 'auto' for automatic selection.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source": {
                        "type": "object",
                        "description": "Data source configuration",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["csv", "json", "inline"],
                                "description": "Source type"
                            },
                            "path": {
                                "type": "string",
                                "description": "File path (for csv/json)"
                            },
                            "data": {
                                "type": "array",
                                "items": { "type": "array", "items": { "type": "number" } },
                                "description": "Inline data as array of rows (for type=inline)"
                            },
                            "has_header": {
                                "type": "boolean",
                                "description": "Whether CSV has a header row (default: true)"
                            },
                            "target_column": {
                                "description": "Target column: integer index or string name. Omit for unsupervised tasks."
                            }
                        },
                        "required": ["type"]
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classify", "regress", "cluster", "auto"],
                        "description": "ML task (default: auto)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name: knn, decision_tree, random_forest, linear_regression, kmeans, or 'auto'"
                    },
                    "model_params": {
                        "type": "object",
                        "description": "Model hyperparameters (e.g. {\"k\": 5} for KNN, {\"max_depth\": 10} for decision tree)"
                    },
                    "preprocess": {
                        "type": "object",
                        "properties": {
                            "normalize": { "type": "boolean", "description": "Z-score normalize features (default: false)" },
                            "drop_nan": { "type": "boolean", "description": "Drop rows with NaN (default: true)" },
                            "pca_components": { "type": "integer", "description": "Reduce to N principal components" }
                        }
                    },
                    "split": {
                        "type": "object",
                        "properties": {
                            "test_ratio": { "type": "number", "description": "Fraction for test set (default: 0.2)" },
                            "seed": { "type": "integer", "description": "Random seed (default: 42)" }
                        }
                    },
                    "persist": { "type": "boolean", "description": "Save trained model to cache (default: false)" },
                    "persist_key": { "type": "string", "description": "Cache key for persisted model" },
                    "return_predictions": { "type": "boolean", "description": "Include predictions in response (default: false)" },
                    "max_rows": { "type": "integer", "description": "Max rows allowed (default: 50000)" },
                    "max_features": { "type": "integer", "description": "Max feature columns allowed (default: 500)" }
                },
                "required": ["source"]
            }),
            handler: handlers::ml_pipeline,
        });

        self.tools.push(Tool {
            name: "ix_ml_predict",
            description: "Run predictions using a previously persisted ML model. Provide the persist_key from a prior ix_ml_pipeline call and new data rows.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "persist_key": {
                        "type": "string",
                        "description": "The persist_key used when the model was saved"
                    },
                    "data": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "New data rows (each row is a feature vector)"
                    }
                },
                "required": ["persist_key", "data"]
            }),
            handler: handlers::ml_predict,
        });

        // ── ix_code_analyze ─────────────────────────────────────

        self.tools.push(Tool {
            name: "ix_code_analyze",
            description: "Analyze source code for complexity metrics (cyclomatic, cognitive, Halstead, SLOC, maintainability index). Supports Rust, Python, JS, TS, C/C++, Java, Go, C#, F#. Returns file-level and per-function metrics with ML-ready feature vectors.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source code string to analyze"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["rust", "python", "javascript", "typescript", "cpp", "java", "go", "csharp", "fsharp"],
                        "description": "Programming language"
                    },
                    "path": {
                        "type": "string",
                        "description": "File path to analyze (alternative to source+language). Language auto-detected from extension."
                    }
                }
            }),
            handler: handlers::code_analyze,
        });
    }

    /// Governance sub-group 4: cross-repo bridges (TARS, GA) plus the
    /// session-facing utilities (demo, explain, triage).
    fn register_bridges_and_session(&mut self) {
        self.tools.push(Tool {
            name: "ix_optick_search",
            description: "Search the OPTIC-K voicing index by embedding similarity. Memory-mapped brute-force cosine search over 228-dim musical embeddings. Returns top-k most similar voicings with diagrams and metadata.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "228-dim query embedding vector (will be L2-normalized internally)"
                    },
                    "instrument": {
                        "type": "string",
                        "enum": ["guitar", "bass", "ukulele"],
                        "description": "Optional instrument filter"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)"
                    },
                    "index_path": {
                        "type": "string",
                        "description": "Path to optick.index file (default: state/voicings/optick.index)"
                    }
                },
                "required": ["query"]
            }),
            handler: handlers::optick_search,
        });

        self.tools.push(Tool {
            name: "ix_tars_bridge",
            description: "Cross-repo bridge to TARS. Prepares ix analysis results (trace stats, pattern data, grammar weights) in the format TARS expects for ingestion. Returns structured payload ready for TARS tools (ingest_ga_traces, run_promotion_pipeline).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["prepare_traces", "prepare_patterns", "export_grammar"],
                        "description": "Bridge action: prepare_traces (format trace stats for TARS), prepare_patterns (format discovered patterns for promotion), export_grammar (export current grammar weights)"
                    },
                    "trace_dir": {
                        "type": "string",
                        "description": "Trace directory (default: ~/.ga/traces/)"
                    },
                    "min_frequency": {
                        "type": "integer",
                        "description": "Minimum pattern frequency for promotion (default: 3)"
                    }
                },
                "required": ["action"]
            }),
            handler: handlers::tars_bridge,
        });

        // ── ix_ga_bridge ────────────────────────────────────────

        self.tools.push(Tool {
            name: "ix_ga_bridge",
            description: "Cross-repo bridge to GA. Converts GA music theory data into ML-ready feature matrices for ix pipelines. Provides data format specifications and example workflows for GA→ix analysis chains.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["chord_features", "progression_features", "scale_features", "workflow_guide"],
                        "description": "Bridge action: chord_features (chord→interval vector), progression_features (progression→feature matrix), scale_features (scale→binary pitch class set), workflow_guide (show GA→ix workflow examples)"
                    },
                    "chords": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Chord symbols to convert (e.g. ['Cmaj7', 'Am7', 'Dm7', 'G7'])"
                    },
                    "progression": {
                        "type": "string",
                        "description": "Chord progression string (e.g. 'C Am F G')"
                    }
                },
                "required": ["action"]
            }),
            handler: handlers::ga_bridge,
        });

        self.tools.push(Tool {
            name: "ix_demo",
            description: "Run curated real-world demo scenarios that chain multiple ix tools. \
                          Use action='list' to see available scenarios, action='describe' for \
                          details, or action='run' to execute a scenario end-to-end.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "run", "describe"],
                        "description": "list = catalog scenarios; run = execute; describe = show steps without executing"
                    },
                    "scenario": {
                        "type": "string",
                        "description": "Scenario id (required for run/describe, ignored for list)"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "RNG seed for reproducible data generation (default: 42)"
                    },
                    "verbosity": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                        "description": "0 = terse, 1 = normal (default), 2 = verbose"
                    }
                },
                "required": ["action"]
            }),
            handler: crate::demo::ix_demo,
        });

        self.tools.push(Tool {
            name: "ix_explain_algorithm",
            description: "Recommend an ix algorithm for a described problem. Designed to \
                          delegate selection to the client's LLM via MCP sampling \
                          (sampling/createMessage, spec 2025-06-18). Bidirectional \
                          JSON-RPC is not yet wired in the ix-mcp stdio dispatcher, so \
                          for now this returns a curated static algorithm catalog the \
                          calling LLM can pick from. Contract is stable — the sampling \
                          upgrade will be a drop-in swap.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "Natural-language description of the problem, data shape, scale, and constraints (e.g. '10,000 noisy 2D points, find clusters, unknown k')."
                    }
                },
                "required": ["problem"]
            }),
            handler: handlers::explain_algorithm,
        });

        self.tools.push(Tool {
            name: "ix_triage_session",
            description: "End-to-end harness triage: read recent session events, ask the \
                          client LLM (via MCP sampling) to propose up to max_actions ix tool \
                          invocations with hexavalent confidence labels, rank by the Demerzel \
                          tiebreak order (C>U>D>P>T>F), check plan-level escalation via \
                          HexavalentDistribution, and dispatch each action through the \
                          governed middleware chain. Optionally exports the resulting trace \
                          via the flywheel and re-ingests it via ix_trace_ingest for \
                          self-improvement statistics. Requires an installed SessionLog \
                          (set IX_SESSION_LOG env var or call install_session_log). \
                          Exercises all 7 harness primitives in one call.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Optional free-text hint describing what the triage should focus on (e.g. 'unblock the stats investigation')"
                    },
                    "max_actions": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 3,
                        "description": "Maximum number of actions the LLM may propose in the plan"
                    },
                    "learn": {
                        "type": "boolean",
                        "default": true,
                        "description": "If true, export the session log via the flywheel and invoke ix_trace_ingest on the result after dispatch"
                    }
                }
            }),
            handler: handlers::triage_session,
        });

        // ── ix_ast_query ─────────────────────────────────────────────────
        self.tools.push(Tool {
            name: "ix_ast_query",
            description: "Run an arbitrary tree-sitter S-expression query against source code \
                          or a file. Supports Rust, C#, TypeScript/JavaScript, and F#. Returns \
                          all captured node texts with 1-based line numbers. Use this to find \
                          method declarations, detect patterns, or extract any syntactic construct \
                          by writing a tree-sitter query (e.g. \"(function_item name: (identifier) \
                          @fn.name)\").",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Tree-sitter S-expression query string, e.g. \
                                       \"(function_item name: (identifier) @fn.name)\""
                    },
                    "source": {
                        "type": "string",
                        "description": "Source code to query (alternative to 'path')"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["rust", "csharp", "typescript", "javascript", "fsharp"],
                        "description": "Language (required when using 'source')"
                    },
                    "path": {
                        "type": "string",
                        "description": "File path to query; language auto-detected from extension"
                    }
                },
                "required": ["query"]
            }),
            handler: handlers::ast_query,
        });

        // ── ix_code_smells ───────────────────────────────────────────────
        self.tools.push(Tool {
            name: "ix_code_smells",
            description: "Detect code smells in source code, a single file, or a directory \
                          tree (recursive). Combines lexical checks (TODO/FIXME, magic numbers, \
                          long lines, language-specific patterns) with AST-based checks (deep \
                          nesting, excessive unsafe/any usage). Supports Rust, C#, TypeScript, \
                          JavaScript, F#, and 6 more languages for lexical checks.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Inline source code to analyse"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["rust", "python", "javascript", "typescript", "cpp", "java",
                                 "go", "csharp", "fsharp", "php", "ruby"],
                        "description": "Language (required with 'source')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to a single source file"
                    },
                    "dir": {
                        "type": "string",
                        "description": "Path to a directory; scans all recognised source files recursively"
                    },
                    "max_file_kb": {
                        "type": "integer",
                        "default": 256,
                        "description": "Skip files larger than this many kilobytes (dir mode only)"
                    }
                }
            }),
            handler: handlers::code_smells,
        });
    }
}
