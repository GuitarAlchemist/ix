---
date: 2026-03-15
topic: ix-ml-builder
---

# ix-ml-builder: ML Pipeline Orchestrator

## What We're Building

A two-layer ML pipeline builder:
- **(A) Skill layer**: SKILL.md that teaches Claude Code how to compose existing ix MCP tools into complete ML workflows — data loading, preprocessing, model selection, training, evaluation, and caching
- **(B) MCP tool layer**: `ix_ml_pipeline` tool that accepts a pipeline definition as JSON and executes it server-side, eliminating multi-tool round-trips

## Why This Approach

**Skill-first (A)** gives us immediate value with zero Rust code. Claude learns the patterns, we see what users actually ask for. **MCP tool (B)** then hardcodes the most common patterns for speed — a single tool call replaces 5-6 sequential calls.

We explicitly skip a full Rust DSL crate (option C) because:
- YAGNI — the skill + tool covers all practical use cases
- A Rust DSL would duplicate what Claude already does (pipeline composition)
- If we later need a crate, the MCP tool's pipeline JSON schema becomes the API spec

## Key Decisions

### Pipeline types
- **Ephemeral**: In-memory, one-shot analysis. No persistence. Default mode.
- **Persistent**: Trained model params + preprocessing stats cached via ix-cache. Survives across sessions. User opts in with `"persist": true`.

### Task auto-detection
The skill infers the task type from context:

| Signal | Task | Default Algorithm |
|--------|------|-------------------|
| Target column is continuous | Regression | LinearRegression |
| Target column is categorical / few unique values | Classification | DecisionTree |
| No target column specified | Clustering | KMeans (k=3) |
| User says "reduce dimensions" / "visualize" | Dimensionality reduction | PCA |
| User says "anomaly" / "outlier" | Anomaly detection | DBSCAN |

### Model selection heuristic
Governance-aware (Article 4: Proportionality — don't use a complex model when a simple one suffices):

| Data size | Features | Recommendation |
|-----------|----------|----------------|
| < 100 rows | Any | LinearRegression / KNN |
| 100-10k | < 20 | DecisionTree / KMeans |
| 100-10k | 20+ | PCA first, then model |
| 10k+ | Any | Random forest / GMM |

### Data flow through pipeline

```
ix_io (load CSV/JSON)
  → Array2<f64> + column names
    → Preprocessing (normalize, handle NaN)
      → Train/test split (80/20 default)
        → Model.fit(X_train, y_train)
          → Model.predict(X_test)
            → Metrics (accuracy/MSE/silhouette)
              → Optional: ix_cache (persist model state)
```

### Pipeline JSON schema (for MCP tool)

```json
{
  "source": {
    "type": "csv",
    "path": "data.csv",
    "has_header": true,
    "target_column": "label"
  },
  "preprocess": {
    "normalize": true,
    "drop_nan": true,
    "pca_components": null
  },
  "task": "classify",
  "model": "decision_tree",
  "params": {},
  "split": { "test_ratio": 0.2, "seed": 42 },
  "persist": false,
  "persist_key": null
}
```

### Persistence format (ix-cache)

```
cache key: "ml:pipeline:{name}"
cache value: {
  "model_type": "linear_regression",
  "weights": [0.5, -0.3, 1.2],
  "bias": 0.1,
  "feature_means": [5.0, 3.5],
  "feature_stds": [1.2, 0.8],
  "feature_names": ["age", "income"],
  "trained_at": "2026-03-15T...",
  "metrics": { "r_squared": 0.85, "mse": 0.12 }
}
```

### What the skill does vs what the tool does

| Concern | Skill (A) | MCP Tool (B) |
|---------|-----------|--------------|
| Task detection | Claude infers from user request | Explicit `"task"` field |
| Model selection | Claude picks based on heuristics | Explicit `"model"` field or `"auto"` |
| Data loading | Claude calls `ix_io` tool | Tool loads internally |
| Preprocessing | Claude calls `ix_stats` tool | Tool normalizes internally |
| Training | Claude calls `ix_supervised`/`ix_unsupervised` | Tool trains internally |
| Evaluation | Claude calls metrics tools | Tool returns metrics in response |
| Governance | Claude runs `ix_governance_check` | Tool checks proportionality internally |
| Multi-step reasoning | Claude chains tool calls | Single tool call |
| Customization | Claude adapts to user feedback | Fixed pipeline steps |

### Governance integration

- **Before training**: Check proportionality — is the chosen model appropriate for the data size?
- **Before persisting**: Check self-modification policy — are we within the session's modification budget?
- **On prediction with stale model**: Mark belief as Unknown — model may be outdated

## Open Questions

1. **Should `ix_ml_pipeline` return intermediate results?** (e.g., preprocessing stats, not just final metrics) — Leaning yes, verbose by default.
2. **Should we support pipeline chaining?** (output of one pipeline feeds into another) — Leaning no for v1, YAGNI.
3. **Ensemble support?** (run multiple models, compare) — The skill can do this by calling the tool multiple times. No special support needed.

## Next Steps

→ `/ce:plan` for implementation:
1. Create `.claude/skills/ix-ml-builder/SKILL.md` (skill layer A)
2. Add `ix_ml_pipeline` handler to ix-agent (MCP tool layer B)
3. Add preprocessing helpers to ix-math (normalize, standardize, train/test split)
4. Add model serialization to ix-supervised/ix-unsupervised (for persistence)
