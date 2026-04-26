# ix — User Manual

> Canonical entry point. Start here before any other doc.

This manual is the single page that tells you what ix is, how to run it, and which of the 180+ existing docs to read next. It does not try to be an encyclopedia — every topic links out to the authoritative doc when one exists.

---

## 1. What ix is — one-paragraph answer

**ix** is a Rust workspace of composable ML, math, and governance primitives exposed as **MCP tools** for agent consumption. It ships 64 crates and 64 MCP tools behind a single `ix-agent` server, plus a CLI (`ix-skill`), a DAG pipeline runner (`ix_pipeline_run`), a natural-language pipeline compiler (`ix_pipeline_compile`), and integration with the Demerzel governance framework. It is part of the [GuitarAlchemist](https://github.com/GuitarAlchemist) ecosystem alongside [tars](https://github.com/GuitarAlchemist/tars) (F# cognition) and [ga](https://github.com/GuitarAlchemist/ga) (C# music theory). ix is used by LLM agents as a callable toolbox for stats, clustering, classification, optimization, signal processing, topology, adversarial ML, and cross-repo analysis — all subject to constitutional governance.

---

## 2. Audiences and reading paths

### Human developers using ix crates from Rust code

You want to consume a crate like `ix-math`, `ix-supervised`, or `ix-pipeline` from your own Rust project. Start with this manual's **§3 Install and first run** to confirm the build works, then jump to the top-level [`README.md`](../README.md) for the per-crate maturity matrix (Stable vs Beta vs Experimental — only Stable crates are safe to depend on downstream). For end-to-end worked examples, read [`docs/INDEX.md`](INDEX.md) which is a 60+ tutorial curriculum. For any graph-theoretic task specifically, read [`docs/guides/graph-theory-in-ix.md`](guides/graph-theory-in-ix.md) **before** adding any dep — ix already has most primitives covered.

### LLM agents calling ix tools via MCP

You are a language model wired to ix via the MCP protocol. Read **§3** for the `.mcp.json` snippet that registers the server, **§4** for the categorized tool inventory, **§5** for the pipeline format that lets you submit a DAG of tool calls in one request, and **§6** for `ix_pipeline_compile` which lets you turn a natural-language brief into a validated pipeline without hand-writing JSON. The five canonical showcase pipelines under [`examples/canonical-showcase/0*/pipeline.json`](../examples/canonical-showcase/) are concrete reference shapes.

### Contributors extending ix itself

You want to add a tool, fix a handler, or refactor a crate. Start with [`CLAUDE.md`](../CLAUDE.md) (project conventions), then this manual's **§8 Extending ix — adding a new tool** for the step-by-step. The roadmap at [`examples/canonical-showcase/ix-roadmap-plan-v1.md`](../examples/canonical-showcase/ix-roadmap-plan-v1.md) is the canonical statement of what's planned, what's shipped, and what's deferred. The [`examples/canonical-showcase/05-adversarial-refactor-oracle/FINDINGS.md`](../examples/canonical-showcase/05-adversarial-refactor-oracle/FINDINGS.md) is an actionable P0/P1/P2 list of concrete improvements the demo surfaced.

---

## 3. Install and first run

### Prerequisites

- Rust 1.80+ (MSRV, due to wgpu 28)
- `git` on the PATH (required by `ix_git_log`)
- Optional: a running Claude Code or other MCP client for the full agent experience

### Build

```bash
git clone https://github.com/GuitarAlchemist/ix.git
cd ix
git submodule update --init --recursive   # pulls governance/demerzel
cargo build --workspace
```

First build takes 5-10 minutes. Subsequent incremental builds are under 10 seconds.

### Run the MCP server directly

```bash
cargo run -p ix-agent
```

The server speaks MCP JSON-RPC over stdio. It is designed to be spawned by an MCP client, not invoked interactively. To confirm it comes up cleanly, send one line of JSON-RPC and kill it:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  | cargo run -p ix-agent 2>/dev/null | head -c 200
```

### Register with Claude Code

Add the following to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "ix": {
      "command": "cargo",
      "args": ["run", "--release", "-p", "ix-agent"]
    }
  }
}
```

Use `--release` — the debug build is slow enough that tool calls noticeably lag. For federation with tars and ga, see [`docs/FEDERATION.md`](FEDERATION.md) for the three-server config.

### Run one canonical showcase end-to-end

The simplest demo is `01-cost-anomaly-hunter` — 3 tools (`ix_stats` → `ix_fft` → `ix_kmeans`) over 90 days of baked cloud spend data. It exists as a testable integration harness:

```bash
cargo test -p ix-agent --test showcase_cost_anomaly
```

Expected output:

```
running 2 tests
test cost_anomaly_hunter_replays_via_pipeline_run ... ok
test cost_anomaly_hunter_second_run_hits_cache ... ok

test result: ok. 2 passed; 0 failed; ...
```

Two tests: the first runs the 3-step pipeline end-to-end and asserts the three injected anomaly days (23, 52, 71) are flagged; the second runs it a second time and asserts every step hits the cache (confirms the R2 Phase 2 content-addressed cache is live). If those pass, your setup is working.

For the most impressive demo, run the 14-tool live-data Adversarial Refactor Oracle with full narration:

```bash
cargo test -p ix-agent --test showcase_refactor_oracle \
  run_refactor_oracle_with_narration -- --ignored --nocapture
```

This runs `ix` against `ix`'s own source tree (via `ix_cargo_deps` + `ix_git_log`), clusters every crate by health profile, trains a random forest on the labels, and prints each tool's output with narration.

---

## 4. The 64 MCP tools — by category

Full schemas live in [`crates/ix-agent/src/tools.rs`](../crates/ix-agent/src/tools.rs). The table below is a navigation aid, not a reference. Every tool accepts JSON input and returns JSON output via `ToolRegistry::call` or `ToolRegistry::call_with_ctx`.

| Category | Representative tools | Where to look |
|---|---|---|
| **Core math & stats** | `ix_stats`, `ix_distance`, `ix_fft` | [`ix-math`](../crates/ix-math), [`ix-signal`](../crates/ix-signal) |
| **Supervised ML** | `ix_linear_regression`, `ix_random_forest`, `ix_gradient_boosting`, `ix_supervised` | [`ix-supervised`](../crates/ix-supervised), [`ix-ensemble`](../crates/ix-ensemble) |
| **Unsupervised ML** | `ix_kmeans`, `ix_ml_pipeline`, `ix_ml_predict` | [`ix-unsupervised`](../crates/ix-unsupervised) |
| **Neural + autograd** | `ix_nn_forward`, `ix_autograd_run` | [`ix-nn`](../crates/ix-nn), [`ix-autograd`](../crates/ix-autograd) |
| **Signal & chaos** | `ix_fft`, `ix_chaos_lyapunov` | [`ix-signal`](../crates/ix-signal), [`ix-chaos`](../crates/ix-chaos) |
| **Graph & topology** | `ix_graph`, `ix_markov`, `ix_viterbi`, `ix_topo` | [`ix-graph`](../crates/ix-graph), [`ix-topo`](../crates/ix-topo), [graph theory guide](guides/graph-theory-in-ix.md) |
| **Search** | `ix_search` (A*, Q*, MCTS, minimax, BFS/DFS) | [`ix-search`](../crates/ix-search) |
| **Adversarial & evolution** | `ix_adversarial_fgsm`, `ix_evolution`, `ix_bandit` | [`ix-adversarial`](../crates/ix-adversarial), [`ix-evolution`](../crates/ix-evolution), [`ix-rl`](../crates/ix-rl) |
| **Game theory** | `ix_game_nash` | [`ix-game`](../crates/ix-game) |
| **Advanced math** | `ix_category`, `ix_rotation`, `ix_sedenion`, `ix_fractal`, `ix_number_theory` | respective crates |
| **Probabilistic structures** | `ix_bloom_filter`, `ix_hyperloglog` | [`ix-probabilistic`](../crates/ix-probabilistic) |
| **Cache, optimization & search indexes** | `ix_cache`, `ix_optimize`, `ix_optick_search` | [`ix-cache`](../crates/ix-cache), [`ix-optimize`](../crates/ix-optimize), [`ix-optick`](../crates/ix-optick) |
| **Grammar** | `ix_grammar_evolve`, `ix_grammar_search`, `ix_grammar_weights` | [`ix-grammar`](../crates/ix-grammar) |
| **Source adapters (P1)** | `ix_git_log`, `ix_cargo_deps`, `ix_code_analyze`, `ix_code_catalog`, `ix_ast_query`, `ix_code_smells` | [`ix-agent/src/handlers.rs`](../crates/ix-agent/src/handlers.rs), [code-analysis tools guide](guides/code-analysis-tools.md) |
| **Catalogs (queryable indexes)** | `ix_catalog_list`, `ix_code_catalog`, `ix_grammar_catalog`, `ix_rfc_catalog` | [`ix-catalog-core`](../crates/ix-catalog-core), [grammar catalog guide](guides/grammar-catalog.md), [rfc catalog guide](guides/rfc-catalog.md) |
| **Pipeline orchestration** | `ix_pipeline_run`, `ix_pipeline_compile`, `ix_pipeline_list`, `ix_pipeline` | [`ix-agent/src/tools.rs`](../crates/ix-agent/src/tools.rs), [`ix-pipeline`](../crates/ix-pipeline) |
| **Governance** | `ix_governance_check`, `ix_governance_persona`, `ix_governance_policy`, `ix_governance_belief`, `ix_governance_graph` | [`ix-governance`](../crates/ix-governance), [`governance/demerzel`](../governance/demerzel) |
| **Federation** | `ix_federation_discover`, `ix_tars_bridge`, `ix_ga_bridge`, `ix_trace_ingest` | [`docs/FEDERATION.md`](FEDERATION.md) |
| **Session & telemetry** | `ix_triage_session`, `ix_explain_algorithm`, `ix_demo`, `ix_session_flywheel_export` | [`ix-agent`](../crates/ix-agent) |

To dump the live schema for any tool, send `tools/list` over MCP and grep by name. The `input_schema` field on each `Tool` struct is authoritative.

---

## 5. Pipelines — the second-class surface

A **pipeline** is a DAG of MCP tool calls submitted as one request to `ix_pipeline_run`. Instead of hand-chaining N tool calls from a client, you submit the whole graph, get back the per-step results + an execution order + a content-addressed cache key per step + a lineage DAG.

### Pipeline spec shape

```json
{
  "steps": [
    {
      "id": "<unique step id>",
      "tool": "<mcp tool name>",
      "asset_name": "<optional content-addressing key>",
      "depends_on": ["<other step id>", ...],
      "arguments": { ...tool input schema... }
    }
  ]
}
```

- **`id`** — required, must be unique across the pipeline. Used for cross-step references.
- **`tool`** — required, must match a tool in the live registry (validated before execution).
- **`arguments`** — the JSON passed to the tool's handler. Can contain `"$step_id.field"` references that resolve to the upstream step's output at execution time.
- **`asset_name`** — optional but strongly recommended. When set, the runner computes a blake3 content hash over `(asset_name, tool, canonical_arguments)` and caches the result. Subsequent runs of the same spec on unchanged inputs hit the cache in zero ms.
- **`depends_on`** — optional. Declares execution order. Omitted edges still get topologically sorted.

### Cross-step references

The substitution layer supports object-field walks and numeric array indexing:

- `"$s01_stats.mean"` → resolves to the `mean` field of step `s01_stats`'s output
- `"$s00_cargo_deps.features.0"` → the first row of the `features` matrix emitted by `s00_cargo_deps`

It does **not** support arithmetic, slicing, or projection (`$s.nodes[*].sloc` is not a legal reference). If you need a reshaped value, emit it explicitly from the producing tool — `ix_cargo_deps` does exactly this by emitting denormalized `sloc` / `file_counts` / `features` vectors alongside the raw `nodes` array.

### Running a pipeline

```rust
use ix_agent::tools::ToolRegistry;
use serde_json::json;

let reg = ToolRegistry::new();
let spec = json!({
    "steps": [
        {
            "id": "baseline",
            "tool": "ix_stats",
            "asset_name": "demo.baseline",
            "arguments": { "data": [1.0, 2.0, 3.0, 50.0, 4.0] }
        },
        {
            "id": "clusters",
            "tool": "ix_kmeans",
            "asset_name": "demo.clusters",
            "depends_on": ["baseline"],
            "arguments": {
                "data": [[1.0], [2.0], [3.0], [50.0], [4.0]],
                "k": 2,
                "max_iter": 100
            }
        }
    ]
});
let result = reg.call("ix_pipeline_run", spec)?;
```

The result is a JSON object with `results`, `execution_order`, `durations_ms`, `cache_hits`, `cache_keys`, and `lineage` fields. See `ix_pipeline_run`'s doc comment in [`tools.rs`](../crates/ix-agent/src/tools.rs) for the full response shape.

### The five canonical showcases

Under [`examples/canonical-showcase/`](../examples/canonical-showcase/):

| # | Folder | Tools | Theme |
|---|---|---|---|
| 01 | [`01-cost-anomaly-hunter`](../examples/canonical-showcase/01-cost-anomaly-hunter/) | 3 | FinOps: stats + FFT + k-means over 90 days of AWS cost data, finds the 3 injected anomaly days |
| 02 | [`02-chaos-detective`](../examples/canonical-showcase/02-chaos-detective/) | 4 | Signal: proves a logistic-map series is deterministic chaos via Lyapunov + persistent homology |
| 03 | [`03-governance-gauntlet`](../examples/canonical-showcase/03-governance-gauntlet/) | 5 | Governance: multi-action constitutional audit chain |
| 04 | [`04-sprint-oracle`](../examples/canonical-showcase/04-sprint-oracle/) | 4 | Forecasting: stats + linreg + Markov + Thompson bandit on sprint velocity data |
| 05 | [`05-adversarial-refactor-oracle`](../examples/canonical-showcase/05-adversarial-refactor-oracle/) | 14 | Self-audit: ix reads its own workspace via `ix_cargo_deps` + `ix_git_log`, clusters crates, attacks the classifier with FGSM, proposes refactors via GA, audits via governance. See its [`FINDINGS.md`](../examples/canonical-showcase/05-adversarial-refactor-oracle/FINDINGS.md) for the actionable P0/P1/P2 improvements the demo surfaced. |

Each folder has its own `pipeline.json` — the load-bearing artifact — plus any reports. The 05 folder also ships a FINDINGS.md that is itself a valuable read.

---

## 6. The natural-language compiler

`ix_pipeline_compile` turns a sentence into a validated `pipeline.json`. The handler builds a prompt containing the live tool registry summary plus two worked examples, asks the client LLM via MCP sampling to emit a JSON spec, strips any markdown fencing, and runs the spec through the shape validator before returning. A `status: "ok"` result is guaranteed to parse, reference only real tools, have unique step ids, resolve every `depends_on` edge, and pass topological-sort cycle detection.

Input:

```json
{
  "sentence": "Cluster 5 numbers into 2 groups and report the stats baseline first",
  "max_steps": 12,
  "context": { /* optional free-form hints */ }
}
```

Output:

```json
{
  "status": "ok" | "invalid" | "parse_error",
  "sentence": "<echoed>",
  "spec": { "steps": [...] },
  "validation": { "errors": [...], "warnings": [...] },
  "raw_llm_response": "<unparsed text for debugging>"
}
```

**Use it when** you have a recurring analysis you'd otherwise hand-write as JSON, or when you want an agent to compose tool chains on the fly. **Skip it when** you need bit-identical reproducibility (hand-write the spec and check it into a showcase folder) or when the task needs a specific step order the LLM keeps reshuffling.

The validator is also public as `ToolRegistry::validate_pipeline_spec` — use it to shape-check any hand-written pipeline spec before calling `ix_pipeline_run`.

---

## 7. Governance integration

ix participates in the **Demerzel** governance framework — a constitution (11 articles), 12 behavioural personas, and a tetravalent logic (True / False / Unknown / Contradictory) for uncertainty-aware reasoning. Governance is not a bolted-on audit layer: it runs inside the MCP surface as first-class tools and inside the pipeline runner as lineage metadata.

### Tools

- **`ix_governance_check`** — check a proposed action against the Demerzel constitution. Optionally accepts a `lineage` field (the one `ix_pipeline_run` emits) and surfaces it as `lineage_audit` in the response, giving auditors a single hop from a governance verdict to the upstream content-addressed cache keys of every asset that fed into it.
- **`ix_governance_persona`** — load one of the 12 Demerzel personas by name (e.g. `skeptical-auditor`, `kaizen-optimizer`) and return its capabilities, constraints, voice, and interaction patterns.
- **`ix_governance_policy`** — query a named policy (e.g. `alignment`, `rollback`, `self-modification`) for its rules and escalation triggers.
- **`ix_governance_belief`** — manage belief states under tetravalent logic for uncertainty-aware reasoning.
- **`ix_governance_graph`** / **`ix_governance_graph_rescan`** — walk and refresh the governance relationship graph.

### Lineage audit trail (R2 Phase 2)

Every `ix_pipeline_run` response includes a `lineage` object: for each step, the tool name, the declared `asset_name`, the content-addressed `cache_key`, the `depends_on` edges, and the `upstream_cache_keys` of its parents. Passing this object to `ix_governance_check` via the `lineage` input produces a `lineage_audit` field in the verdict that an auditor can walk back to concrete cache keys — which are blake3 hashes of the exact inputs that produced each result. See [`tests/showcase_r1_migrations.rs::governance_check_consumes_pipeline_lineage`](../crates/ix-agent/tests/showcase_r1_migrations.rs) for the round-trip pattern.

### Galactic Protocol and knowledge packages

ix is a participant repo in the Demerzel federation and consumes directives / knowledge packages / policies from the governance submodule. The belief state lives in `state/` (belief snapshots, PDCA cycles, knowledge transfers, reconnaissance snapshots). Cross-repo communication flows through [`docs/FEDERATION.md`](FEDERATION.md). The upstream submodule at [`governance/demerzel/`](../governance/demerzel) is the authoritative constitution source; see its `constitutions/` tree for the full set of articles.

---

## 8. Extending ix — adding a new tool

The pattern below is what `ix_git_log` and `ix_cargo_deps` followed when they were added during the P1 source-adapter work. The full diff is in commit `a36dc78` if you want to trace it.

### Step 1 — Write the handler

Add a `pub fn <name>(params: Value) -> Result<Value, String>` in [`crates/ix-agent/src/handlers.rs`](../crates/ix-agent/src/handlers.rs). Parse arguments defensively — every missing or wrong-typed field must return a message that names the field, the expected type, and the offending value. The `parse_str`, `parse_usize`, `parse_f64_array`, and `parse_f64_matrix` helpers at the top of the file cover most cases.

If the handler needs shell access (like `ix_git_log` spawning `git`), pass every argument through `Command::arg()`, never `arg_line()` or format-string concatenation. Whitelist-validate any path or identifier input — see `is_safe_git_path` as the reference pattern.

### Step 2 — Register the tool in `tools.rs`

In [`crates/ix-agent/src/tools.rs`](../crates/ix-agent/src/tools.rs), add a new entry inside `register_all`:

```rust
self.tools.push(Tool {
    name: "ix_your_tool",
    description: "One-sentence summary of what this tool does and when to use it.",
    input_schema: json!({
        "type": "object",
        "properties": {
            "required_field": { "type": "string", "description": "..." },
            "optional_field": { "type": "integer", "description": "..." }
        },
        "required": ["required_field"]
    }),
    handler: handlers::your_tool,
});
```

Keep the description LLM-friendly — agents see this text when selecting a tool, and it's the only thing distinguishing your tool from the ~56 others.

### Step 3 — Add a smoke test

Create `crates/ix-agent/tests/your_tool_smoke.rs`. Run the tool against live inputs (the real workspace state for source adapters, a deterministic fixture for pure computations). Assert both happy-path shape and every rejection case you care about. The [`git_log_smoke.rs`](../crates/ix-agent/tests/git_log_smoke.rs) test is a good template — 7 tests covering happy path, bucket consistency, and 5 distinct rejection modes.

### Step 4 — Update the parity allowlist

Add the tool name to the sorted `EXPECTED` array in [`crates/ix-agent/tests/parity.rs`](../crates/ix-agent/tests/parity.rs) and bump the `EXPECTED.len()` assertion by one. The parity test is an intentional rate-limiter that forces every surface change to be reviewed in the commit that adds it.

### Step 5 — Document the tool

Add one row to the category table in this manual's §4. Do not write a separate reference doc — the `description` + `input_schema` on the Tool struct is the authoritative reference, and the schema flows through `tools/list` to every MCP client.

### Step 6 — Run the sweep

```bash
cargo test -p ix-agent                      # all tests green
cargo clippy -p ix-agent --tests            # zero new warnings
```

`cargo clippy` with a zero-new-warnings bar is the commit gate for the ix-agent crate.

---

## 9. The learning path

If you want the full curriculum — foundations, core algorithms, advanced topics, 60+ tutorials with runnable Rust code — start at [`docs/INDEX.md`](INDEX.md) and work top to bottom. A French translation is available at [`docs/fr/INDEX.md`](fr/INDEX.md). This manual does not duplicate the learning path; it is the entry to ix, not the textbook.

---

## 10. Troubleshooting — known gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `cargo test` suddenly fails with "os error 4551" on Windows mid-session | Windows Defender Application Control (WDAC) is blocking the unsigned test binary | Rebuild with a fresh target dir: `cargo clean -p ix-agent && cargo test -p ix-agent`. Run a smoke `cargo test` at session start so you know the environment is clean before investing in changes. |
| `ix_graph` returns `"Invalid 'from'"` on a `[[f64; 3]; N]` edge list | Fixed in commit `651e316` (P0.2). Integer-valued floats now parse as node indices. | Upgrade to HEAD. If you must stay on an older commit, pass edges as `json!([[0, 1, 1.0], ...])` so `from`/`to` serialise as integers. |
| `cargo build` fails with `E0463: can't find crate for zerocopy` | Cargo.lock is stale after `zerocopy 0.8.42` upgrade | Run `cargo update -p zerocopy` then rebuild. |
| Codex CLI dispatch emits `rmcp ERROR` lines during an otherwise-successful run | Codex logs the MCP init handshake at ERROR level but still completes successfully | `rmcp ERROR` is a warning, not a failure. Grep past it when parsing the log. Use `</dev/null` for stdin, redirect `2>&1`, and allow up to 180s before treating the run as hung. |
| `ix_pipeline_compile` returns `parse_error` even though the LLM "looked right" | Client wrapped the JSON in a markdown fence despite the system prompt saying not to | The handler already strips ` ```json ... ``` ` fences via `strip_markdown_fence`. If it still fails, inspect `raw_llm_response` in the result — that's the unedited LLM output. |
| MCP sampling call times out after 30s | Client is slow or unresponsive | `SAMPLING_TIMEOUT` in [`server_context.rs`](../crates/ix-agent/src/server_context.rs) is 30 seconds. If your client is genuinely slow (e.g. local model on CPU), raise the constant. If it's hanging, the MCP client likely crashed — restart it. |
| `$step.field` substitution fails on `$s.features[0]` or `$s.a + $s.b` | The substitute_refs layer only supports object-field walks and numeric array indexing (`$s.features.0`). Arithmetic and slicing are not supported. | Reshape at the producer side. `ix_cargo_deps` emits denormalized `sloc`, `file_counts`, `features` vectors alongside the raw `nodes` array for exactly this reason. For arithmetic, do it inside a custom handler. |
| `ix_pipeline_run` step fails with `"missing field 'n_nodes'"` but your spec has it | The field is a `$ref` to a step that hasn't run yet, or the upstream step's output doesn't contain that field name | Check `execution_order` in the response — a missing upstream means the depends_on edge wasn't declared. Check the upstream tool's output shape in `handlers.rs` — field names must match exactly. |
| `ix_git_log` returns `commits: 0` when run from `crates/<x>` | `cargo test` CWD is the crate directory, and git resolves paths relative to the CWD of its process | Pass the `repo_root` parameter: `{"path": "crates/ix-agent", "repo_root": "/abs/path/to/repo"}`. The handler uses `git -C <root>` when set. |
| `ix_cargo_deps` misses a workspace crate named without the `ix-` prefix | Fixed. The extractor now filters deps against the known crate set (every directory under `crates/`), not a hard-coded prefix. | Upgrade to HEAD. If you see it on older code, the prefix was the bug. |
| First run of a fresh clone fails with "Failed to load constitution" | `governance/demerzel` submodule isn't initialized | `git submodule update --init --recursive` |

For any issue not listed here, the first diagnostic step is to run `cargo test -p ix-agent --test parity` — the parity test is designed to catch 90% of "something broke" drift before you waste time on deeper debugging.

---

## 11. Roadmap pointers

Current state, in dependency order:

- **R1** (`ix_pipeline_run` MCP tool) — **shipped**. Every canonical showcase is now reproducible from one MCP call.
- **R2 Phase 1** (content-addressed cache) — **shipped**. Every asset-backed step hits cache on replay.
- **R2 Phase 2** (lineage DAG + governance audit wiring) — **shipped**. `ix_governance_check` can consume pipeline lineage as an audit trail.
- **R3** (Buf-style registry-check CI) — **shipped**. PRs touching `capability-registry.json` are gated on breaking-change detection.
- **R7 Week 1** (ix-autograd scaffold, primitive ops, finite-diff verifier, linreg + variance tools) — **shipped**, gate passed at 242× speedup.
- **R7 Week 2** (`ix_autograd_run` MCP tool, `StatsMeanTool`, `MseLossTool`) — **shipped**.
- **NL compiler** (`ix_pipeline_compile`) — **shipped**. Validator + fake-client tests green.
- **P0/P1 from the refactor oracle FINDINGS** — **shipped**: ix_graph error messages fixed, registry allowlist populated, `ix_git_log` + `ix_cargo_deps` source adapters built, refactor oracle graduated from baked constants to live workspace data.
- **R4** (meta-MCP gateway) — planned, Phase 3.
- **R5** (Arrow IPC side-channel for oversized outputs) — planned, Phase 3.
- **R6** (adversarial pipelines, Levels 1–3) — Level 1 preview shipped in the refactor oracle; Levels 2–3 planned.
- **R8** (QED-MCP, Lean 4 / Kani formal verification) — planned, Phase 4.
- **R9a/R9b** (PyO3 bridge, WASM surface) — planned, Phase 4.

The authoritative roadmap with dependency graph, phase breakdown, week-by-week plan, risk register, and definition-of-done gates lives at [`examples/canonical-showcase/ix-roadmap-plan-v1.md`](../examples/canonical-showcase/ix-roadmap-plan-v1.md). The P0/P1/P2 priority list specific to ix's own structural health is at [`examples/canonical-showcase/05-adversarial-refactor-oracle/FINDINGS.md`](../examples/canonical-showcase/05-adversarial-refactor-oracle/FINDINGS.md).

---

## 12. References

**Top-level**
- [`README.md`](../README.md) — crate maturity matrix, quick-start commands, dependency versions
- [`CLAUDE.md`](../CLAUDE.md) — project conventions for Claude Code and other agent sessions
- [`.mcp.json`](../.mcp.json) — live MCP server configuration (ix + tars + ga)

**Docs tree**
- [`docs/INDEX.md`](INDEX.md) — 60+ tutorial learning path (foundations → advanced)
- [`docs/fr/INDEX.md`](fr/INDEX.md) — French translation of the learning path
- [`docs/FEDERATION.md`](FEDERATION.md) — three-repo federation (ix + tars + ga), bridge skills, federated pipelines
- [`docs/MIRROR-TO-ECOSYSTEM.md`](MIRROR-TO-ECOSYSTEM.md) — how ix's capability registry is mirrored across the ecosystem
- [`docs/guides/graph-theory-in-ix.md`](guides/graph-theory-in-ix.md) — **read this before adding any graph dep**
- [`docs/guides/code-analysis-tools.md`](guides/code-analysis-tools.md) — curated catalog of mathematical tools for analysing programming-language repositories, served live by `ix_code_catalog`
- [`docs/guides/grammar-catalog.md`](guides/grammar-catalog.md) — ~30 real-world EBNF / ABNF / PEG grammar sources, served live by `ix_grammar_catalog`
- [`docs/guides/rfc-catalog.md`](guides/rfc-catalog.md) — ~70 curated IETF RFCs covering the modern internet stack with obsolescence graph, served live by `ix_rfc_catalog`

**Canonical showcases**
- [`examples/canonical-showcase/README.md`](../examples/canonical-showcase/README.md) — showcase inventory
- [`examples/canonical-showcase/01-cost-anomaly-hunter/pipeline.json`](../examples/canonical-showcase/01-cost-anomaly-hunter/pipeline.json) — simplest 3-tool demo
- [`examples/canonical-showcase/02-chaos-detective/pipeline.json`](../examples/canonical-showcase/02-chaos-detective/pipeline.json)
- [`examples/canonical-showcase/03-governance-gauntlet/pipeline.json`](../examples/canonical-showcase/03-governance-gauntlet/pipeline.json)
- [`examples/canonical-showcase/04-sprint-oracle/pipeline.json`](../examples/canonical-showcase/04-sprint-oracle/pipeline.json)
- [`examples/canonical-showcase/05-adversarial-refactor-oracle/pipeline.json`](../examples/canonical-showcase/05-adversarial-refactor-oracle/pipeline.json) — 14-tool self-audit, live data
- [`examples/canonical-showcase/05-adversarial-refactor-oracle/FINDINGS.md`](../examples/canonical-showcase/05-adversarial-refactor-oracle/FINDINGS.md) — actionable P0/P1/P2 roadmap the oracle surfaced
- [`examples/canonical-showcase/ix-roadmap-plan-v1.md`](../examples/canonical-showcase/ix-roadmap-plan-v1.md) — authoritative R1-R9 roadmap

**Source**
- [`crates/ix-agent/src/tools.rs`](../crates/ix-agent/src/tools.rs) — MCP tool registry (authoritative for schemas)
- [`crates/ix-agent/src/handlers.rs`](../crates/ix-agent/src/handlers.rs) — handler implementations (authoritative for output shapes)
- [`crates/ix-pipeline/src/dag.rs`](../crates/ix-pipeline/src/dag.rs) — DAG substrate
- Per-crate rustdoc — `cargo doc --workspace --no-deps --open`

**Governance**
- [`governance/demerzel/constitutions/`](../governance/demerzel/constitutions/) — the constitution and policies
- [`governance/demerzel/schemas/capability-registry.json`](../governance/demerzel/schemas/capability-registry.json) — cross-repo tool inventory
- [`state/`](../state/) — local belief state, PDCA cycles, knowledge packages

---

*If anything in this manual is out of date or wrong, the source tree is authoritative. Open a PR to fix it at the same time as the underlying code change — stale docs are worse than missing ones.*
