//! Tool registry — defines MCP tools and dispatches calls.

use serde_json::{json, Value};
use std::collections::HashMap;

use crate::handlers;
use crate::registry_bridge;
use crate::schema::{object, object_with_additional, Prop};

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
fn substitute_refs(value: &Value, upstream: &HashMap<String, Value>) -> Result<Value, String> {
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

    /// List all tools as MCP tool definitions (full surface).
    ///
    /// Equivalent to `list_scoped(Scope::Default)`. Kept as the canonical
    /// entry point because every existing test + caller goes through it.
    pub fn list(&self) -> Value {
        self.list_scoped(crate::scopes::Scope::Default)
    }

    /// List tools advertised under the given scope.
    ///
    /// `Scope::Default` returns everything (backward compat). Named
    /// scopes return only the tools whitelisted in
    /// [`crate::scopes::SCOPES`]. The shape (`{ "tools": [...] }`) is
    /// identical regardless of scope.
    pub fn list_scoped(&self, scope: crate::scopes::Scope) -> Value {
        let tools: Vec<Value> = self
            .tools
            .iter()
            .filter(|t| scope.allows(t.name))
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

    /// Iterate every registered tool name. Used by scope coverage tests
    /// to assert that scoped subsets are strict subsets of the default
    /// surface.
    pub fn tool_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.tools.iter().map(|t| t.name)
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
                        dag.add_edge(dep_id, id)
                            .map_err(|e| format!("ix_pipeline_run: edge {dep_id} -> {id}: {e}"))?;
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
            let resolved = substitute_refs(&raw_args, &results)
                .map_err(|e| format!("ix_pipeline_run: step '{id}' arg substitution: {e}"))?;

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
            let result = self
                .call(&tool, resolved)
                .map_err(|e| format!("ix_pipeline_run: step '{id}' (tool '{tool}') failed: {e}"))?;
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
            let deps = depends_on.get(id).cloned().unwrap_or_default();
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
        let max_steps = args.get("max_steps").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
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

    /// Core manual tool registrations. Most former core tools (stats, distance,
    /// optimise, linear regression / k-means, FFT, Markov, search / decision,
    /// symbolic, structural) are now registry-backed and registered via
    /// [`Self::merge_registry_tools`]; only `ix_tsne` remains hand-registered.
    fn register_all_core(&mut self) {
        self.register_core_numerics();
    }

    /// The core tools still owned manually (only `ix_tsne` today).
    fn register_core_numerics(&mut self) {
        self.tools.push(Tool {
            name: "ix_tsne",
            description: "t-SNE dimensionality reduction (van der Maaten & Hinton 2008). Project a high-dim matrix to 2D or 3D for visualization. O(n²); keep n ≤ ~5000.",
            input_schema: object(
                vec![
                    (
                        "data",
                        Prop::num_matrix().desc("Data matrix (rows=samples, cols=features)"),
                    ),
                    (
                        "perplexity",
                        Prop::number()
                            .desc("Perplexity (default 30). Must be < (n-1)/3.")
                            .minimum(1),
                    ),
                    (
                        "n_iter",
                        Prop::integer().desc("Iterations (default 500)").minimum(50),
                    ),
                    (
                        "target_dim",
                        Prop::integer()
                            .desc("Output dimension (default 2)")
                            .minimum(1)
                            .maximum(3),
                    ),
                    (
                        "seed",
                        Prop::integer().desc("RNG seed for determinism (default 0)"),
                    ),
                ],
                &["data"],
            ),
            handler: handlers::tsne,
        });
    }

    /// Second half of the manual tool registrations: pipeline orchestration,
    /// catalogs, and source adapters. (Supervised learning, graph, and
    /// probabilistic sketches are now registry-backed.) Dispatches to
    /// sub-methods to keep per-function complexity bounded.
    fn register_all_advanced(&mut self) {
        self.register_advanced_pipelines();
        self.register_advanced_catalogs();
        self.register_advanced_misc();
    }

    /// Advanced sub-group 2: autograd execution and pipeline compile /
    /// run primitives (R7 Week 2 + ix_pipeline_* family).
    fn register_advanced_pipelines(&mut self) {
        self.tools.push(Tool {
            name: "ix_autograd_run",
            description: "R7 Week 2: run a differentiable tool (LinearRegressionTool, StatsVarianceTool) end-to-end, returning forward outputs AND per-input gradients in a single MCP call. Inputs are nested f64 arrays; shape is inferred from nesting depth. Used for pipeline-level gradient descent where the caller maintains an Adam/SGD loop outside the MCP boundary.",
            input_schema: object(
                vec![
                    (
                        "tool",
                        Prop::string()
                            .enum_of(&["linear_regression", "stats_variance"])
                            .desc("Name of the differentiable tool to run"),
                    ),
                    (
                        "inputs",
                        Prop::object_any()
                            .desc("Map of input name → nested f64 array (1-D, 2-D, or scalar). For linear_regression: x, w, b, y. For stats_variance: x.")
                            .additional_properties(true),
                    ),
                ],
                &["tool", "inputs"],
            ),
            handler: handlers::autograd_run,
        });

        self.tools.push(Tool {
            name: "ix_pipeline_run",
            description: "Execute a DAG pipeline end-to-end: topologically sorts steps, dispatches each step's tool with substituted upstream references, and returns per-step results + durations. Replaces hand-chaining of MCP calls. Reference upstream outputs in step arguments via the string `\"$step_id.field\"`. Handled by ToolRegistry::call_with_ctx; this entry exists only for tools/list discovery.",
            input_schema: object(
                vec![(
                    "steps",
                    Prop::array_of(Prop::object(
                        vec![
                            (
                                "id",
                                Prop::string()
                                    .desc("Step identifier, used for cross-step references"),
                            ),
                            (
                                "tool",
                                Prop::string()
                                    .desc("Name of the MCP tool to invoke, e.g. 'ix_stats'"),
                            ),
                            (
                                "arguments",
                                Prop::object_any().desc(
                                    "Arguments to pass to the tool; may contain $step_id.field references",
                                ),
                            ),
                            (
                                "depends_on",
                                Prop::str_array().desc(
                                    "IDs of prerequisite steps that must complete before this one",
                                ),
                            ),
                        ],
                        &["id", "tool"],
                    ))
                    .desc(
                        "Ordered or unordered set of pipeline steps; execution order is derived from depends_on",
                    ),
                )],
                &["steps"],
            ),
            handler: handlers::pipeline_run_placeholder,
        });

        self.tools.push(Tool {
            name: "ix_pipeline_compile",
            description: "Compile a natural-language sentence into a pipeline.json DAG via MCP sampling. The handler asks the client's LLM to emit a JSON {steps: [...]} spec using only registered ix tools, then validates the result against the registry (unknown tools, duplicate ids, unresolved depends_on, cycles). Returns status 'ok' when the spec is safe to pass directly to ix_pipeline_run. Handled by ToolRegistry::call_with_ctx; this entry exists only for tools/list discovery.",
            input_schema: object(
                vec![
                    (
                        "sentence",
                        Prop::string()
                            .desc("Natural-language description of the analysis you want to run"),
                    ),
                    (
                        "max_steps",
                        Prop::integer().desc(
                            "Upper bound on the number of steps the compiler is allowed to emit. Default 12.",
                        ),
                    ),
                    (
                        "context",
                        Prop::object_any()
                            .desc("Optional free-form context hints (data bindings, preferred tools, etc.) forwarded to the LLM prompt")
                            .additional_properties(true),
                    ),
                ],
                &["sentence"],
            ),
            handler: handlers::pipeline_compile_placeholder,
        });

        self.tools.push(Tool {
            name: "ix_nl_to_pipeline",
            description: "The IX \"thinking machine\": translate a natural-language request into a canonical PipelineSpec (ix.yaml), validate it with lower(), gate it through the Demerzel constitution (fail-closed), optionally execute it, and narrate the result back. Direct LLM-provider proposer with bounded self-repair and a two-tier coverage gate (refuses out-of-domain requests instead of confabulating). Prefer this over ix_pipeline_compile (which targets the legacy {steps:[…]} format via deprecated MCP sampling). Returns status one of: ok | compiled | out_of_domain | governance_rejected | translate_failed.",
            input_schema: object(
                vec![
                    (
                        "sentence",
                        Prop::string()
                            .desc("Natural-language description of the analysis/pipeline you want"),
                    ),
                    (
                        "run",
                        Prop::boolean().desc(
                            "Execute the compiled pipeline and narrate results (default false: compile + governance-gate only)",
                        ),
                    ),
                    (
                        "max_rounds",
                        Prop::integer()
                            .desc(
                                "Max self-repair rounds when the generated spec fails validation (default 3)",
                            )
                            .minimum(0),
                    ),
                ],
                &["sentence"],
            ),
            handler: handlers::nl_to_pipeline,
        });

        self.tools.push(Tool {
            name: "ix_thinker_hits",
            description: "Aggregate the IX thinking-machine's translation ledger (state/thinking-machine/hits.jsonl) into a yield metric paired with its refusal guardrails. `yield_rate` is the metric (fraction of requests that produced a runnable spec); `coverage_refusal_rate` / `governance_refusal_rate` / `translate_fail_rate` are guardrails. A rising yield with a FALLING coverage-refusal rate is the signature of the gate being loosened or the proposer confabulating out-of-domain specs — the pair makes Goodhart-style gaming visible (instrumenting a bare success rate would hide it). Read-only; rates are over unlabeled production outcomes.",
            input_schema: object_with_additional(vec![], &[], false),
            handler: handlers::thinker_hits,
        });
    }

    /// Advanced sub-group 3: catalog tools (code / grammar / RFC /
    /// meta) plus workspace introspection (cargo deps, git log).
    fn register_advanced_catalogs(&mut self) {
        self.tools.push(Tool {
            name: "ix_catalog_list",
            description: "Meta-tool: list every registered ix catalog (code_analysis, grammar, rfc, ...) with its name, scope, and entry count. Use this to discover what catalogs ix exposes before issuing a specific ix_*_catalog query.",
            input_schema: object(vec![], &[]),
            handler: handlers::catalog_list,
        });

        self.tools.push(Tool {
            name: "ix_grammar_catalog",
            description: "Query a curated catalog of ~30 real-world grammar sources across EBNF, ABNF, PEG, ANTLR G4, W3C EBNF, and BNF notations. Covers programming languages (Python, Go, ECMAScript, Rust, C11, ...), data formats (JSON, TOML, YAML, SQL-2016, GraphQL), IETF protocols (HTTP, TLS, DNS, SMTP, IMAP, OAuth, WebSockets), and meta-grammars (ABNF, ISO 14977 EBNF). Filter by language, format, or topic.",
            input_schema: object(
                vec![
                    (
                        "language",
                        Prop::string().desc("Case-insensitive language filter (e.g. 'python', 'http'). Meta-entries with language='many' always pass."),
                    ),
                    (
                        "format",
                        Prop::string()
                            .enum_of(&["ebnf", "w3c_ebnf", "abnf", "peg", "antlr_g4", "bnf"])
                            .desc("Filter by grammar notation format."),
                    ),
                    (
                        "topic",
                        Prop::string().desc("Case-insensitive substring match against topic tags (e.g. 'web', 'protocol', 'meta')."),
                    ),
                ],
                &[],
            ),
            handler: handlers::grammar_catalog,
        });

        self.tools.push(Tool {
            name: "ix_rfc_catalog",
            description: "Query a curated catalog of ~70 IETF RFCs covering the modern internet stack: IP, TCP/QUIC, HTTP/1.1/2/3, TLS 1.3, DNS+DNSSEC, SMTP/IMAP, OAuth/JOSE/JWT, SSH, SIP/RTP, JSON/CBOR/UUID, ABNF, and BCPs. Includes the obsolescence graph — passing current_standard=true filters out obsoleted entries, and obsolescence_chain=N walks both directions from the seed RFC (useful for 'what replaced RFC 2616' questions).",
            input_schema: object(
                vec![
                    (
                        "number",
                        Prop::integer().desc("Exact RFC number lookup (e.g. 9110)."),
                    ),
                    (
                        "topic",
                        Prop::string().desc("Topic substring (e.g. 'http', 'dns', 'tls', 'auth'). Case-insensitive."),
                    ),
                    (
                        "status",
                        Prop::string()
                            .enum_of(&[
                                "internet_standard",
                                "proposed_standard",
                                "draft_standard",
                                "experimental",
                                "informational",
                                "obsoleted",
                            ])
                            .desc("Filter by publication status."),
                    ),
                    (
                        "current_standard",
                        Prop::boolean().desc("When true, excludes obsoleted entries. Combine with topic to get 'the current spec for X'."),
                    ),
                    (
                        "obsolescence_chain",
                        Prop::integer().desc("Return the complete obsolescence chain (walked in both directions) for this RFC number. Overrides other filters."),
                    ),
                ],
                &[],
            ),
            handler: handlers::rfc_catalog,
        });

        self.tools.push(Tool {
            name: "ix_code_catalog",
            description: "Query a curated catalog of external mathematical tools for analysing programming-language repositories (static analysers, formal verifiers, safety / memory checkers, statistical + behavioural analysis tools, documentation generators, and numeric libraries). Filter by language, category, or technique substring. Use this to route users to the right specialist rather than over-stretching ix_code_analyze.",
            input_schema: object(
                vec![
                    (
                        "language",
                        Prop::string().desc("Case-insensitive language filter (e.g. 'rust', 'python'). Language-agnostic tools are always included."),
                    ),
                    (
                        "category",
                        Prop::string()
                            .enum_of(&[
                                "static_analysis",
                                "formal_verification",
                                "safety_memory",
                                "statistical_analysis",
                                "documentation",
                                "numeric_library",
                                "ml_framework",
                                "fuzzing",
                                "supply_chain",
                            ])
                            .desc("One of nine categories; omit to include all."),
                    ),
                    (
                        "technique",
                        Prop::string().desc("Substring match against the 'technique' field (e.g. 'cyclomatic', 'abstract interpretation', 'model checking')."),
                    ),
                ],
                &[],
            ),
            handler: handlers::code_catalog,
        });

        self.tools.push(Tool {
            name: "ix_cargo_deps",
            description: "P1.2 — walk a Rust workspace, parse every crates/<name>/Cargo.toml for intra-workspace ix-* dependencies, and emit a {nodes, edges, n_nodes} structure that ix_graph can consume directly. Each node records {id, name, sloc, file_count, dep_count}. Edges are [from_id, to_id, 1.0] triples. Default workspace_root is the process CWD.",
            input_schema: object(
                vec![(
                    "workspace_root",
                    Prop::string().desc("Absolute or relative path to the workspace root (the directory containing 'crates/'). Defaults to the process CWD."),
                )],
                &[],
            ),
            handler: handlers::cargo_deps,
        });

        self.tools.push(Tool {
            name: "ix_git_log",
            description: "P1.1 — shell out to `git log` on a repo-internal path and return a normalized commit cadence time series. Buckets commits into per-day or per-week dense arrays so downstream tools (ix_fft, ix_stats, ix_chaos_lyapunov) can consume the output directly. Every argument is passed through Command::arg(), not shell concatenation, and 'path' is whitelist-validated to reject '..', absolute prefixes, and shell metacharacters.",
            input_schema: object(
                vec![
                    (
                        "path",
                        Prop::string().desc("Relative repo-internal path to scope the git log to (e.g. 'crates/ix-agent'). Must not contain '..', absolute prefixes, or shell metacharacters."),
                    ),
                    (
                        "since_days",
                        Prop::integer().desc("Window size in days ending today. Default 90. Must be in 1..=3650."),
                    ),
                    (
                        "bucket",
                        Prop::string()
                            .enum_of(&["day", "week"])
                            .desc("Bucket size for the output time series. Default 'day'."),
                    ),
                    (
                        "repo_root",
                        Prop::string().desc("Optional absolute path to a git repository root. When provided, git runs as if invoked from that directory via `git -C <root>`. Use this when the MCP server's CWD is not the repo root."),
                    ),
                ],
                &["path"],
            ),
            handler: handlers::git_log,
        });

        self.tools.push(Tool {
            name: "ix_git_churn",
            description: "P1.3 — per-file change frequency over a window. Runs `git log --numstat` once and aggregates churn_count + lines_added + lines_deleted + last_changed per file, sorted by churn descending and truncated to `limit`. Pair with ix_git_log: log is 'how often does the repo change' (time series); churn is 'which files drive that change' (per-file ranking). Same hardening as ix_git_log (no shell concatenation, repo_root via `git -C`). Flat projections (paths/churn_counts/lines_added/lines_deleted) let pipeline $step.field substitution feed downstream stats/fft/kmeans tools directly.",
            input_schema: object(
                vec![
                    (
                        "since_days",
                        Prop::integer().desc("Window size in days ending today. Default 30. Must be in 1..=3650."),
                    ),
                    (
                        "limit",
                        Prop::integer().desc("Maximum number of files in the ranked output. Default 50. Must be in 1..=10000. 'total_files' always reports the un-truncated count."),
                    ),
                    (
                        "repo_root",
                        Prop::string().desc("Optional absolute path to a git repository root. Same semantics as ix_git_log."),
                    ),
                ],
                &[],
            ),
            handler: handlers::git_churn,
        });
    }

    /// Advanced sub-group 4: pipeline discovery / execution helpers
    /// and the in-memory cache.
    fn register_advanced_misc(&mut self) {
        self.tools.push(Tool {
            name: "ix_pipeline_list",
            description: "Discover canonical-showcase pipeline.json specs under a directory (default 'examples/canonical-showcase'). Returns metadata for each spec — name, description, step count, and the list of tools it uses. Companion to ix_pipeline_run for pipeline browsing.",
            input_schema: object(
                vec![(
                    "root",
                    Prop::string().desc("Directory to scan for '<subdir>/pipeline.json' specs. Relative paths resolve against CWD. Default: 'examples/canonical-showcase'."),
                )],
                &[],
            ),
            handler: handlers::pipeline_list,
        });

    }

    /// Third section: governance, federation bridges, and
    /// session / telemetry / explainer utilities. Dispatches to four
    /// sub-methods (P0.1 round 3) to keep per-function complexity
    /// bounded.
    fn register_all_governance_and_session(&mut self) {
        self.register_governance();
        self.register_ml_and_code();
        self.register_bridges_and_session();
    }

    /// Governance sub-group 1: the four Demerzel governance primitives
    /// (compliance check, persona loader, belief query, policy).
    fn register_governance(&mut self) {
        // ── ix_quality_gate_history ───────────────────────────────────────
        // Cross-repo quality-gate ledger query. Reads
        // state/quality/gate-ledger.jsonl (v1 entries) and returns recent
        // rows filtered by source / domain / since. See
        // docs/contracts/2026-05-24-quality-gate-ledger.contract.md.
        self.tools.push(Tool {
            name: "ix_quality_gate_history",
            description: "Query the unified quality-gate ledger (state/quality/gate-ledger.jsonl). Returns v1 entries filtered by source, domain, decision, and/or since-timestamp, sorted newest-first. Legacy v0 PR-shaped rows are excluded.",
            input_schema: object(
                vec![
                    (
                        "source",
                        Prop::string().desc("Producer id (e.g. 'ix-quality-trend', 'sentrux', 'chatbot-qa'). Omit for all."),
                    ),
                    (
                        "domain",
                        Prop::string().desc("Domain measured (e.g. 'structural', 'chatbot', 'tests'). Omit for all."),
                    ),
                    (
                        "decision",
                        Prop::string()
                            .enum_of(&["pass", "fail", "warn", "skip"])
                            .desc("Filter by decision."),
                    ),
                    (
                        "since",
                        Prop::string().desc("RFC3339 timestamp lower bound (inclusive). E.g. '2026-05-20T00:00:00Z'."),
                    ),
                    (
                        "limit",
                        Prop::integer().desc("Max rows to return (default 50)."),
                    ),
                    (
                        "ledger_path",
                        Prop::string().desc("Override ledger path. Default: state/quality/gate-ledger.jsonl (repo-relative)."),
                    ),
                ],
                &[],
            ),
            handler: handlers::quality_gate_history,
        });
    }

    /// Governance sub-group 2: higher-level ML (ml_pipeline, ml_predict)
    /// and source code analysis.
    fn register_ml_and_code(&mut self) {
        // ── ix_code_analyze ─────────────────────────────────────

        // ── ix_annotations_scan ─────────────────────────────────────

        self.tools.push(Tool {
            name: "ix_annotations_scan",
            description: "Scan a workspace for in-source @ai: annotations (invariant, assumption, hypothesis, contract, smell, decision, hint) with hexavalent truth values (T/P/U/D/F/C) and certainty markers. Runs the extractor + reconciler: matches annotations against test files, promotes contradictory same-line claims to C, flags stale ones, and emits weighted multi-source aggregates. See docs/contracts/2026-05-24-ai-annotation.contract.md.",
            input_schema: object(
                vec![
                    (
                        "workspace",
                        Prop::string().desc("Workspace root to scan. Default: current directory."),
                    ),
                    (
                        "stale_days",
                        Prop::integer().desc("Days after annotation update_at before file mtime flips the stale bit. Default: 7."),
                    ),
                    (
                        "test_files",
                        Prop::str_array().desc("Optional explicit list of test files (workspace-relative). If omitted, auto-discovered."),
                    ),
                ],
                &[],
            ),
            handler: handlers::annotations_scan,
        });
    }

    /// Governance sub-group 4: cross-repo bridges (TARS, GA) plus the
    /// session-facing utilities (demo, explain, triage).
    fn register_bridges_and_session(&mut self) {
        self.tools.push(Tool {
            name: "ix_optick_search",
            description: "Search the OPTIC-K voicing index by embedding similarity. Memory-mapped brute-force cosine search over 228-dim musical embeddings. Returns top-k most similar voicings with diagrams and metadata.",
            input_schema: object(
                vec![
                    (
                        "query",
                        Prop::num_array().desc("228-dim query embedding vector (will be L2-normalized internally)"),
                    ),
                    (
                        "instrument",
                        Prop::string()
                            .enum_of(&["guitar", "bass", "ukulele"])
                            .desc("Optional instrument filter"),
                    ),
                    (
                        "top_k",
                        Prop::integer().desc("Number of results to return (default: 10)"),
                    ),
                    (
                        "index_path",
                        Prop::string().desc("Path to optick.index file (default: state/voicings/optick.index)"),
                    ),
                ],
                &["query"],
            ),
            handler: handlers::optick_search,
        });

        self.tools.push(Tool {
            name: "ix_voicings_payload",
            description: "Returns a `voicings.payload.v1` JSON payload that tells GA's Prime Radiant (or any consumer) where to fetch the binary voicing-positions buffer and how to render it. Reuses the binary buffer pre-derived by `serve_viz` (`voicing-positions.bin` + `.meta.json`). The caller passes overrides; this tool does no I/O. See docs/plans/2026-05-02-voicings-in-prime-radiant.md.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "scene_offset": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "[x,y,z] translation for the cloud root. Default [200,0,0] keeps it clear of the governance graph at origin."
                    },
                    "default_spread": {
                        "type": "number",
                        "description": "Initial per-axis jitter to spread the densely-packed cluster knots. 0 = raw positions. Default 1.5."
                    },
                    "default_point_size": {
                        "type": "number",
                        "description": "Initial Three.js PointsMaterial.size. Default 0.3."
                    },
                    "serve_url": {
                        "type": "string",
                        "description": "Base URL where serve_viz is reachable. Default `http://127.0.0.1:8765`. Override when GA + ix run on different hosts (start serve_viz with --bind 0.0.0.0)."
                    }
                }
            }),
            handler: handlers::voicings_payload,
        });

        // ── ix_ga_bridge ────────────────────────────────────────

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

        // Governance verdict as a callable (ADR-0001). Feature-gated: pulls bundled DuckDB
        // via ix-duck, so it is absent from the default agent surface (and the parity test).
        #[cfg(feature = "maintain-gate")]
        self.tools.push(Tool {
            name: "ix_maintain_gate",
            description: "Evaluate one self-improvement iteration against the maintain-gate \
                          (the hexavalent T/P/U/D/F/C RSI oracle): fuse the externally-derived \
                          yield metric, the chatbot guardrail, and the convergence/drift lenses \
                          into one verdict. Returns the MaintainVerdict JSON (status, decision, \
                          signals, evidence, reason). Read-only — does not append to the ledger. \
                          Advisory until Phase-3b ledger write-isolation.",
            input_schema: object(
                vec![
                    (
                        "hits_path",
                        Prop::string().desc("Externally-derived hits.jsonl (the yield metric source)."),
                    ),
                    (
                        "corpus_dir",
                        Prop::string().desc("Chatbot guardrail baseline dir (chatbot-qa)."),
                    ),
                    (
                        "loops_dir",
                        Prop::string().desc("Optional loop-iteration ledger dir (convergence lens)."),
                    ),
                    (
                        "query_embeddings_dir",
                        Prop::string().desc("Optional query-embeddings dir (drift lens)."),
                    ),
                    (
                        "loop_id",
                        Prop::string().desc("Optional iteration scope: loop id (needs commit_sha + repo_dir)."),
                    ),
                    (
                        "commit_sha",
                        Prop::string().desc("Optional iteration scope: commit to verify against git."),
                    ),
                    (
                        "repo_dir",
                        Prop::string().desc("Optional iteration scope: repo to verify commit_sha in."),
                    ),
                    (
                        "run_at",
                        Prop::string().desc("Optional RFC3339 timestamp; defaults to now."),
                    ),
                ],
                &["hits_path", "corpus_dir"],
            ),
            handler: crate::maintain_gate::ix_maintain_gate,
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
            input_schema: object(
                vec![(
                    "problem",
                    Prop::string().desc("Natural-language description of the problem, data shape, scale, and constraints (e.g. '10,000 noisy 2D points, find clusters, unknown k')."),
                )],
                &["problem"],
            ),
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
            input_schema: object(
                vec![
                    (
                        "focus",
                        Prop::string().desc("Optional free-text hint describing what the triage should focus on (e.g. 'unblock the stats investigation')"),
                    ),
                    (
                        "max_actions",
                        Prop::integer()
                            .minimum(1)
                            .maximum(8)
                            .default(3)
                            .desc("Maximum number of actions the LLM may propose in the plan"),
                    ),
                    (
                        "learn",
                        Prop::boolean()
                            .default(true)
                            .desc("If true, export the session log via the flywheel and invoke ix_trace_ingest on the result after dispatch"),
                    ),
                ],
                &[],
            ),
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
            input_schema: object(
                vec![
                    (
                        "query",
                        Prop::string().desc(
                            "Tree-sitter S-expression query string, e.g. \
                             \"(function_item name: (identifier) @fn.name)\"",
                        ),
                    ),
                    (
                        "source",
                        Prop::string().desc("Source code to query (alternative to 'path')"),
                    ),
                    (
                        "language",
                        Prop::string()
                            .enum_of(&["rust", "csharp", "typescript", "javascript", "fsharp"])
                            .desc("Language (required when using 'source')"),
                    ),
                    (
                        "path",
                        Prop::string().desc("File path to query; language auto-detected from extension"),
                    ),
                ],
                &["query"],
            ),
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
            input_schema: object(
                vec![
                    (
                        "source",
                        Prop::string().desc("Inline source code to analyse"),
                    ),
                    (
                        "language",
                        Prop::string()
                            .enum_of(&[
                                "rust", "python", "javascript", "typescript", "cpp", "java", "go",
                                "csharp", "fsharp", "php", "ruby",
                            ])
                            .desc("Language (required with 'source')"),
                    ),
                    (
                        "path",
                        Prop::string().desc("Path to a single source file"),
                    ),
                    (
                        "dir",
                        Prop::string().desc("Path to a directory; scans all recognised source files recursively"),
                    ),
                    (
                        "max_file_kb",
                        Prop::integer()
                            .default(256)
                            .desc("Skip files larger than this many kilobytes (dir mode only)"),
                    ),
                ],
                &[],
            ),
            handler: handlers::code_smells,
        });

        self.tools.push(Tool {
            name: "ix_grothendieck_delta",
            description: "Compute the signed Grothendieck ICV delta between two pitch-class sets (target − source in ℤ⁶). Returns the 6-component delta, L1/L2 norms, and both source/target ICVs. PC-sets are arrays of pitch classes 0..11 (values reduced mod 12).",
            input_schema: object(
                vec![
                    (
                        "source",
                        Prop::int_array().desc("Source PC-set as an array of pitch classes (0..11)"),
                    ),
                    (
                        "target",
                        Prop::int_array().desc("Target PC-set as an array of pitch classes (0..11)"),
                    ),
                ],
                &["source", "target"],
            ),
            handler: handlers::grothendieck_delta,
        });

        self.tools.push(Tool {
            name: "ix_grothendieck_nearby",
            description: "Find pitch-class sets within an L1 Grothendieck-distance budget of the source ICV. Orbit-aware — uses the 224 D₁₂ prime-form representatives for ~18× fewer ICV computations than a brute-force 4096-set scan. Results are sorted by ascending cost.",
            input_schema: object(
                vec![
                    (
                        "source",
                        Prop::int_array().desc("Source PC-set as an array of pitch classes (0..11)"),
                    ),
                    (
                        "max_l1",
                        Prop::integer()
                            .minimum(0)
                            .desc("Maximum L1 norm of the ICV delta (0 returns the source orbit only)"),
                    ),
                    (
                        "limit",
                        Prop::integer()
                            .minimum(1)
                            .desc("Optional cap on the number of returned (set, delta, cost) triples after sorting by cost"),
                    ),
                ],
                &["source", "max_l1"],
            ),
            handler: handlers::grothendieck_nearby,
        });

        self.tools.push(Tool {
            name: "ix_grothendieck_path",
            description: "Find the shortest harmonic path between two PC-sets of equal cardinality using A* (admissible heuristic L1(δ)/2). Drop-in replacement for GA's BFS FindShortestPath: same path output, fewer nodes expanded. Returns an empty path when no route exists within max_steps.",
            input_schema: object(
                vec![
                    (
                        "source",
                        Prop::int_array().desc("Source PC-set as an array of pitch classes (0..11)"),
                    ),
                    (
                        "target",
                        Prop::int_array().desc("Target PC-set as an array of pitch classes (0..11). Must have the same cardinality as source."),
                    ),
                    (
                        "max_steps",
                        Prop::integer()
                            .minimum(0)
                            .default(5)
                            .desc("Maximum number of edges in the path (matches GA's path.Count >= maxSteps + 1 cutoff)"),
                    ),
                ],
                &["source", "target"],
            ),
            handler: handlers::grothendieck_path,
        });

        self.tools.push(Tool {
            name: "ix_autoresearch_run",
            description: "Run a Karpathy-style edit-eval-iterate loop against an IX subsystem. v1 ships only the 'grammar' target (smoke test). Iterations are capped at 10000 over MCP; use the CLI binary `ix-autoresearch run` for larger runs. Returns the run id, best reward, log path, and a cost ledger.",
            input_schema: object(
                vec![
                    (
                        "target",
                        Prop::string()
                            .enum_of(&["grammar"])
                            .default("grammar")
                            .desc("Adapter to invoke. Only 'grammar' in v1; Phase 4/5 add 'chatbot' and 'optick'."),
                    ),
                    (
                        "iterations",
                        Prop::integer()
                            .minimum(1)
                            .maximum(10000)
                            .desc("Number of perturb→eval→decide cycles (MCP cap is 10000)."),
                    ),
                    (
                        "strategy",
                        Prop::string()
                            .enum_of(&["greedy", "sa", "random"])
                            .default("greedy")
                            .desc("Acceptance strategy. 'sa' = simulated annealing; pair with 'initial_temperature' + 'cooling_rate'."),
                    ),
                    (
                        "initial_temperature",
                        Prop::number()
                            .exclusive_min(0)
                            .desc("SA initial temperature; omit to trigger Ben-Ameur 2004 calibration on first 10 random samples."),
                    ),
                    (
                        "cooling_rate",
                        Prop::number()
                            .minimum(0.0)
                            .maximum(1.0)
                            .default(0.95)
                            .desc("SA geometric cooling rate (T_{n+1} = cooling_rate · T_n)."),
                    ),
                    (
                        "soft_seconds",
                        Prop::number()
                            .exclusive_min(0)
                            .default(300)
                            .desc("Per-iteration soft deadline (eval honors as a hint)."),
                    ),
                    (
                        "hard_seconds",
                        Prop::number()
                            .exclusive_min(0)
                            .desc("Optional per-iteration hard timeout (kernel watchdog kills the worker)."),
                    ),
                    (
                        "seed",
                        Prop::integer()
                            .minimum(0)
                            .default(42)
                            .desc("Deterministic RNG seed."),
                    ),
                    (
                        "state_dir",
                        Prop::string()
                            .default("state/autoresearch")
                            .desc("Root for runs/ and milestones/ subdirs."),
                    ),
                ],
                &["iterations"],
            ),
            handler: handlers::autoresearch_run,
        });

        // ── ix_sentrux_annotate ────────────────────────────────────
        // Bridge that drives `sentrux.exe mcp` and converts each
        // structural-rule violation into an ai-annotation-v1 record.
        // Closes the claim -> verify -> promote/demote loop with sentrux
        // as the machine ground-truth verifier (see PRs #54/#55/#56 and
        // crate `ix-sentrux-annotations`).
        //
        // The `emit_untested` arg adds a second pass: sentrux `test_gaps`
        // -> intersect with `@ai:business-value` files -> emit one
        // `@ai:smell "no test coverage detected by sentrux"` per
        // intersection file. Default off — without it, behavior is
        // unchanged from PR #61.
        self.tools.push(Tool {
            name: "ix_sentrux_annotate",
            description: "Run sentrux structural-rule checks against a workspace and emit one ai-annotation-v1 record per violation (truth_value=F, certainty=detected-by-sentrux, source.author=sentrux). Default mode is `dry-run` (counts only, no file mutation). Use `sidecar` to write the JSONL stream consumed by the reconciler; use `inline` to patch source files with `// @ai:smell` comments. Set `emit_untested=true` to additionally call sentrux `test_gaps` and emit one untested-smell annotation per file in the intersection of (untested files) ∩ (files with `@ai:business-value` annotations).",
            input_schema: object(
                vec![
                    (
                        "workspace",
                        Prop::string().desc("Repo root passed to sentrux's `scan` tool (default `.`)."),
                    ),
                    (
                        "mode",
                        Prop::string()
                            .enum_of(&["sidecar", "inline", "dry-run"])
                            .default("dry-run")
                            .desc("Emit mode. `sidecar` writes JSONL, `inline` patches sources, `dry-run` counts without writing."),
                    ),
                    (
                        "out",
                        Prop::string().desc("Override sidecar output path (default `<workspace>/state/quality/ai-annotations-sentrux.jsonl`)."),
                    ),
                    (
                        "sentrux_exe",
                        Prop::string().desc("Override sentrux binary path (default C:/Users/spare/bin/sentrux.exe)."),
                    ),
                    (
                        "timeout_secs",
                        Prop::integer()
                            .minimum(1)
                            .default(60)
                            .desc("Timeout for the JSON-RPC handshake."),
                    ),
                    (
                        "emit_untested",
                        Prop::boolean()
                            .default(false)
                            .desc("Additionally call sentrux `test_gaps` and emit `@ai:smell` annotations for files in the intersection of (untested files) ∩ (files with `@ai:business-value` annotations). Off by default."),
                    ),
                    (
                        "untested_limit",
                        Prop::integer()
                            .minimum(1)
                            .default(100)
                            .desc("Top-N untested-file cap passed through to sentrux `test_gaps.limit`. Only meaningful when emit_untested=true."),
                    ),
                    (
                        "untested_out",
                        Prop::string().desc("Override sidecar path for the untested-smell JSONL stream (default `<workspace>/state/quality/ai-annotations-sentrux-untested.jsonl`). Only meaningful when emit_untested=true."),
                    ),
                ],
                &[],
            ),
            handler: handlers::sentrux_annotate,
        });
    }
}
