---
title: "feat: ML pipeline builder — ephemeral and persistent pipelines via skill + MCP tool"
type: feat
status: active
date: 2026-03-15
origin: docs/brainstorms/2026-03-15-ix-ml-builder-brainstorm.md
---

# feat: ML Pipeline Builder

## Enhancement Summary

**Deepened on:** 2026-03-15
**Research agents used:** architecture-strategist, security-sentinel, performance-oracle, ML-best-practices

### Key Improvements from Review
1. **Security**: Add file path validation (allowlist, traversal check, extension check, size limit) before CSV loading
2. **Architecture**: Extract pipeline logic into `ml_pipeline.rs` module — keep handler thin (10-20 lines)
3. **Schema fixes**: Add required `target_column`, `model_params`, `return_predictions`, `max_rows` fields
4. **Performance**: Enforce row/feature limits (default 100k rows, 500 features) to prevent OOM
5. **Auto-detection gap**: CSV reader produces all-f64 — use cardinality threshold (< 20 unique values → classify) instead of "categorical" detection
6. **Cache safety**: Prefix all model keys with `ix_ml:model:` namespace, validate key format
7. **Auditability**: Log all pipeline executions to stderr (Article 7/8 compliance)
8. **Response size**: Add `return_predictions: false` default — return metrics only, predictions opt-in

### New Considerations Discovered
- DecisionTree is the slowest model (~500ms-2s for 10k rows) — document this
- Pipeline will hold 3-4 copies of dataset in memory — peak ~30MB for 10k×100
- Serde JSON deserialization is safe (no code execution) but recursive tree models need depth limits
- Existing `ix-pipeline` DAG executor could be used but flat sequential is simpler for Phase 1

---

## Overview

Two-layer ML pipeline builder: (A) a Claude Code skill that teaches orchestration of existing ix MCP tools into complete ML workflows, and (B) an `ix_ml_pipeline` MCP tool that executes a full pipeline in a single server-side call. Supports ephemeral (one-shot) and persistent (cached across sessions) pipelines. (see brainstorm: docs/brainstorms/2026-03-15-ix-ml-builder-brainstorm.md)

## Problem Statement / Motivation

Currently, building an ML pipeline in ix requires 5-6 sequential MCP tool calls (load data, preprocess, split, train, predict, evaluate) with manual data wiring between each step. Users must know which crate provides which algorithm and what data format each expects. The ML builder skill eliminates this friction.

## Proposed Solution

### Layer A: Skill (`ix-ml-builder`)

A SKILL.md that teaches Claude Code how to:
1. Detect the ML task type from user intent (classify/regress/cluster/reduce/anomaly)
2. Select the appropriate model using governance-aware heuristics (Article 4: Proportionality)
3. Compose the right sequence of tool calls
4. Cache results for persistent pipelines

### Layer B: MCP Tool (`ix_ml_pipeline`)

A single MCP tool that accepts a pipeline JSON spec and returns results — data loading through evaluation in one call. Implemented as a handler in ix-agent that internally uses ix-io, ix-math, ix-supervised, ix-unsupervised, and ix-cache.

## Technical Approach

### Phase 1: Preprocessing Helpers in ix-math

Add normalization/standardization and train/test split to `crates/ix-math/src/`.

**Files to create/modify:**

- [ ] `crates/ix-math/src/preprocessing.rs` — Normalization and scaling utilities

```rust
pub struct StandardScaler {
    pub means: Array1<f64>,
    pub stds: Array1<f64>,
}

impl StandardScaler {
    pub fn fit(x: &Array2<f64>) -> Self;
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64>;
    pub fn fit_transform(x: &Array2<f64>) -> (Self, Array2<f64>);
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Array2<f64>;
}

pub struct MinMaxScaler {
    pub mins: Array1<f64>,
    pub maxs: Array1<f64>,
}

impl MinMaxScaler {
    pub fn fit(x: &Array2<f64>) -> Self;
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64>;
    pub fn fit_transform(x: &Array2<f64>) -> (Self, Array2<f64>);
}

/// Split data into train and test sets
pub fn train_test_split(
    x: &Array2<f64>,
    y: &Array1<f64>,
    test_ratio: f64,
    seed: Option<u64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>);

/// Drop rows containing NaN values
pub fn drop_nan_rows(x: &Array2<f64>) -> Array2<f64>;
```

- [ ] `crates/ix-math/src/lib.rs` — Add `pub mod preprocessing;`
- [ ] Tests: fit/transform roundtrip, split ratio correctness, NaN handling, empty input, zero-variance columns

### Research Insights (Phase 1)

**StandardScaler edge cases (from scikit-learn, smartcore, linfa patterns):**
- Zero-variance columns (std=0) → replace std with 1.0, result is 0 after mean subtraction
- Single-row data → std is 0 for all columns with ddof=1, guard with `if n <= 1 { ones }`
- All-NaN columns → add `validate()` method that scans for non-finite values upfront
- Empty arrays → return error from `fit()` if nrows == 0 or ncols == 0
- Add `Serialize, Deserialize` derives on scalers for pipeline persistence

**MinMaxScaler:**
- Support configurable `feature_range: (f64, f64)` (default 0.0-1.0)
- Constant columns → range is 0, use 1.0 to avoid NaN (same as zero-variance)

**Train/test split (from linfa DatasetBase patterns):**
- `train_test_split(x, y, test_ratio, seed)` → simple shuffle + slice
- `stratified_split(x, y, test_ratio, seed)` → group by class, split each proportionally
- Guarantee at least 1 sample per class in each split: `split.max(1).min(n-1)`
- Use `ChaCha8Rng::seed_from_u64(seed)` for reproducibility (consistent with ix conventions)
- For very small datasets (< 20 rows), warn that split may not be representative

**Preprocessing as standalone:**
- Implement in `ix-math` so it's reusable across crates
- Consider exposing as `ix_preprocess` MCP tool in a future phase

**References:**
- [scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [linfa LinearScaler](https://rust-ml.github.io/linfa/rustdocs/linfa_preprocessing/linear_scaling/struct.LinearScaler.html)
- [linfa DatasetBase split](https://docs.rs/linfa/latest/linfa/dataset/struct.DatasetBase.html)

### Phase 2: Model Serialization

Add `to_json` / `from_json` to supervised and unsupervised models for cache persistence.

**Files to modify:**

- [ ] `crates/ix-supervised/src/linear_regression.rs` — Add `Serialize, Deserialize` derives, `to_json()` / `from_json()` methods
- [ ] `crates/ix-supervised/src/decision_tree.rs` — Same (serialize tree structure)
- [ ] `crates/ix-supervised/src/knn.rs` — Same (serialize training data + k)
- [ ] `crates/ix-unsupervised/src/kmeans.rs` — Same (serialize centroids)
- [ ] `crates/ix-unsupervised/src/pca.rs` — Same (serialize components + explained variance)

Pattern — use a `ModelEnvelope` wrapper for version-safe serialization:
```rust
/// Every serialized model carries a version tag + preprocessing state
#[derive(Serialize, Deserialize)]
pub struct ModelEnvelope {
    pub version: String,        // semver of the format
    pub algorithm: String,      // "linear_regression", "kmeans", etc.
    pub params: serde_json::Value,  // model-specific state
    pub preprocessing: Option<serde_json::Value>,  // scaler params
    pub feature_names: Option<Vec<String>>,
    pub trained_at: String,     // ISO 8601 timestamp
}
```

Per-model state pattern:
```rust
#[derive(Serialize, Deserialize)]
pub struct LinearRegressionState {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub n_features: usize,
}

impl LinearRegression {
    pub fn save_state(&self) -> Option<LinearRegressionState>;
    pub fn load_state(state: &LinearRegressionState) -> Self;
}
```

### Research Insights (Phase 2)

**Serialization format: JSON** (not bincode/msgpack) because:
- Models are small relative to data — size/speed barely matters
- Human-readability aids debugging
- Schema evolution with `#[serde(default)]` and `Option<T>` is trivial
- Integrates naturally with existing `serde_json::Value` pipeline

**Version migration strategy:**
- Use `#[serde(default)]` on all new optional fields so old JSON still deserializes
- If breaking changes needed, use tagged enums: `#[serde(tag = "version")]`
- Store crate version that produced the model for migration code

**DecisionTree serialization caution:**
- Recursive `Node` enum — crafted JSON with extreme depth could stack overflow
- Use flat node array with index-based children, or set `serde_json::Deserializer` max depth = 128

**References:**
- [Rust serde versioning patterns](https://siedentop.dev/posts/rust-serde-versioning/)
- [serde-evolve crate](https://docs.rs/serde-evolve/latest/serde_evolve/)

### Phase 3: MCP Tool Handler (`ix_ml_pipeline`)

**Files to modify:**

- [ ] `crates/ix-agent/Cargo.toml` — Ensure ix-math, ix-supervised, ix-unsupervised, ix-io, ix-cache deps
- [ ] `crates/ix-agent/src/tools.rs` — Register `ix_ml_pipeline` tool with JSON schema:

```json
{
  "name": "ix_ml_pipeline",
  "description": "Execute a complete ML pipeline — load data, preprocess, train, evaluate, optionally persist",
  "inputSchema": {
    "type": "object",
    "required": ["source", "task"],
    "properties": {
      "source": {
        "type": "object",
        "properties": {
          "type": { "enum": ["csv", "json", "inline"] },
          "path": { "type": "string" },
          "data": { "type": "array" },
          "has_header": { "type": "boolean", "default": true },
          "target_column": { "type": ["string", "integer"] }
        }
      },
      "task": { "enum": ["classify", "regress", "cluster", "reduce", "auto"] },
      "model": { "enum": ["linear_regression", "logistic_regression", "decision_tree", "knn", "naive_bayes", "svm", "random_forest", "kmeans", "dbscan", "pca", "tsne", "gmm", "auto"] },
      "params": { "type": "object" },
      "preprocess": {
        "type": "object",
        "properties": {
          "normalize": { "type": "boolean", "default": false },
          "drop_nan": { "type": "boolean", "default": true },
          "pca_components": { "type": "integer" }
        }
      },
      "split": {
        "type": "object",
        "properties": {
          "test_ratio": { "type": "number", "default": 0.2 },
          "seed": { "type": "integer", "default": 42 }
        }
      },
      "model_params": { "type": "object", "description": "Model-specific hyperparameters (k, max_depth, etc.)" },
      "persist": { "type": "boolean", "default": false },
      "persist_key": { "type": "string" },
      "return_predictions": { "type": "boolean", "default": false },
      "max_rows": { "type": "integer", "default": 100000 },
      "max_features": { "type": "integer", "default": 500 }
    }
  }
}
```

### Research Insights (Phase 3)

**Security hardening (must-fix):**
- Validate file paths: reject `..`, UNC paths, non-csv/json extensions
- Size check before loading: `std::fs::metadata(path)?.len()` — reject > 50MB
- Enforce `max_rows` and `max_features` with early termination during parsing
- Prefix all cache keys with `ix_ml:model:` to avoid namespace collision
- Validate `persist_key` format: alphanumeric + underscore + hyphen, max 128 chars

**Architecture (should-fix):**
- Extract orchestration into `crates/ix-agent/src/ml_pipeline.rs` with typed structs:
  ```rust
  pub struct PipelineConfig { source: Source, task: Task, model: Model, ... }
  pub struct PipelineResult { metrics: Metrics, model_info: ModelInfo, timing: Timing, ... }
  pub fn run_pipeline(config: PipelineConfig) -> Result<PipelineResult, String>
  ```
- Handler becomes thin: deserialize → `run_pipeline()` → serialize response

**Auto-detection heuristic (refined from H2O AutoML + AutoGluon patterns):**
```rust
fn infer_task(y: &[f64]) -> Task {
    let unique: HashSet<u64> = y.iter().map(|v| v.to_bits()).collect();
    let n_unique = unique.len();
    let all_integer = y.iter().all(|v| *v == v.floor() && v.is_finite());
    let ratio = n_unique as f64 / y.len() as f64;

    if all_integer && n_unique == 2 { BinaryClassification }
    else if all_integer && n_unique <= 20 && ratio < 0.05 { MulticlassClassification }
    else { Regression }
}
```
- **Cardinality threshold = 20** (H2O uses 10, but 20 covers more multiclass cases)
- **Integer check** — if all values are whole numbers, classification is more likely
- **Ratio check** — 10 unique values out of 1M rows is clearly categorical (ratio < 5%)
- No target column → cluster (no supervised task possible)
- Log the auto-detection decision for transparency (Article 2)

**Auditability (Article 7/8):**
- Log to stderr: `[ix-ml] pipeline: {file}, {rows}x{cols}, model={model}, task={task}, duration={ms}ms`
- When `persist: true`, log: `[ix-ml] cached: key={key}, model_size={bytes}`
- When overwriting cached model, log warning
```

- [ ] `crates/ix-agent/src/handlers.rs` — Add `ml_pipeline` handler function

Handler logic:
1. Parse source → load data via ix-io (`load_csv_xy` or `load_csv_matrix`)
2. If `task == "auto"`: detect from target column (continuous → regress, categorical → classify, missing → cluster)
3. If `model == "auto"`: select based on data size + task (see brainstorm heuristic table)
4. Preprocess: drop NaN, normalize if requested, PCA if requested
5. If supervised: train/test split → fit → predict → compute metrics
6. If unsupervised: fit_predict → compute cluster stats
7. If `persist`: serialize model state + scaler params to ix-cache
8. Return JSON with: predictions, metrics, model info, preprocessing stats, timing

### Phase 4: Skill File

- [ ] `.claude/skills/ix-ml-builder/SKILL.md` — The orchestration skill

Content covers:
- **When to use**: User has data and wants ML analysis
- **Task detection table**: Maps user intent → task type → default algorithm
- **Model selection heuristic**: Data size × features → recommendation (with governance check)
- **Ephemeral workflow**: Sequential tool calls with data wiring
- **Persistent workflow**: Same + `persist: true` flag + cache key naming
- **Prediction from cached model**: Load model state, apply to new data
- **Governance integration**: Check proportionality before training, self-modification policy before persisting
- **Example conversations**: 3 complete examples (classify CSV, cluster data, regression with persistence)

### Phase 5: Predict from Cached Model

- [ ] `crates/ix-agent/src/handlers.rs` — Add `ml_predict` handler (or extend `ml_pipeline`)

```json
{
  "name": "ix_ml_predict",
  "description": "Load a persisted model and predict on new data",
  "inputSchema": {
    "properties": {
      "persist_key": { "type": "string", "description": "Cache key of the saved model" },
      "data": { "type": "array", "description": "New data points to predict" }
    }
  }
}
```

## System-Wide Impact

### Interaction Graph
- `ix_ml_pipeline` handler calls into: ix-io (CSV loading), ix-math (preprocessing), ix-supervised OR ix-unsupervised (model), ix-cache (persistence)
- No new cross-crate trait dependencies — handler orchestrates at the JSON level
- ix-cache snapshots persist to disk via existing `save_snapshot` / `load_snapshot`

### Error & Failure Propagation
- CSV parse errors → "Failed to load data: {detail}"
- Shape mismatches (e.g., target col out of range) → "Invalid target column: {detail}"
- Model training errors → "Training failed: {detail}"
- Cache errors → "Failed to persist model: {detail}" (non-fatal, results still returned)

### State Lifecycle Risks
- Cached models may become stale if training data changes → skill advises marking as Unknown (tetravalent)
- Scaler params must be cached alongside model weights (otherwise predictions on unscaled data are wrong)
- Cache TTL should be set for persistent models (default: no TTL, user manages lifecycle)

### API Surface Parity
- **MCP tool**: `ix_ml_pipeline` (primary), `ix_ml_predict` (secondary)
- **Skill**: `ix-ml-builder` (Claude Code orchestration)
- **Rust API**: No new public crate API (handler-only, uses existing crate APIs internally)
- **CLI**: Not wired into ix-skill for v1 (future enhancement)

## Acceptance Criteria

### Functional Requirements
- [ ] `ix_ml_pipeline` with `task: "classify"` trains a model and returns accuracy, predictions
- [ ] `ix_ml_pipeline` with `task: "cluster"` returns cluster labels and cluster sizes
- [ ] `ix_ml_pipeline` with `task: "auto"` correctly detects task type from data
- [ ] `ix_ml_pipeline` with `model: "auto"` selects appropriate model for data size
- [ ] `ix_ml_pipeline` with `persist: true` saves model to ix-cache
- [ ] `ix_ml_predict` loads cached model and returns predictions
- [ ] Preprocessing (normalize, drop_nan, PCA) works correctly
- [ ] Train/test split is reproducible with seed

### Non-Functional Requirements
- [ ] Single `ix_ml_pipeline` call completes in < 5s for datasets under 10k rows
- [ ] Cached models survive across Claude Code sessions (via ix-cache snapshot)
- [ ] Error messages are actionable (not stack traces)

### Quality Gates
- [ ] Every new public function has a `#[test]`
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] Skill includes 3 end-to-end example conversations

## Dependencies & Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model serialization breaks on struct changes | Low | Medium | Version the serialized state format |
| Preprocessing stats not cached with model | Medium | High | Always serialize scaler alongside model |
| Auto-detection guesses wrong task type | Medium | Low | User can always override with explicit `task` |
| Large datasets slow in single tool call | Low | Medium | Enforce max_rows=100k, max_features=500 |
| **File path injection via CSV path** | **Medium** | **High** | **Validate paths: reject .., UNC, non-csv/json, >50MB** |
| **OOM on large CSV** | **Low** | **High** | **Enforce row/feature limits, size check before loading** |
| **DecisionTree training timeout** | **Medium** | **Medium** | **Document ~2s for 10k rows, suggest simpler models for large data** |
| **Recursive tree deserialization stack overflow** | **Low** | **Medium** | **Set serde max_depth=128 or use flat node array** |

## Sources & References

### Origin
- **Brainstorm document:** [docs/brainstorms/2026-03-15-ix-ml-builder-brainstorm.md](docs/brainstorms/2026-03-15-ix-ml-builder-brainstorm.md) — Key decisions: A+B approach (skill + MCP tool), ephemeral/persistent modes, task auto-detection, governance-aware model selection.

### Internal References
- MCP handler pattern: `crates/ix-agent/src/handlers.rs:1179` (supervised handler)
- Pipeline executor: `crates/ix-pipeline/src/executor.rs` (DAG execution)
- CSV loading: `crates/ix-io/src/csv_io.rs` (load_csv_xy, load_csv_matrix)
- Model traits: `crates/ix-supervised/src/traits.rs` (Regressor, Classifier)
- Cache persistence: `crates/ix-cache/src/persist.rs` (save_snapshot, load_snapshot)
- Stats functions: `crates/ix-math/src/stats.rs` (mean, std_dev, min_max)
