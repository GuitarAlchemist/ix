---
name: ix-ml-builder
description: Build ephemeral or persistent ML pipelines — auto-detects task type, selects models, handles preprocessing, evaluation, and caching
---

# ML Pipeline Builder

Build complete ML pipelines from data to predictions in one step.

## When to Use
When the user has data (CSV, JSON, or inline) and wants ML analysis — classification, regression, clustering, dimensionality reduction, or anomaly detection.

## Quick Start

**Single tool call (Layer B):**
```json
ix_ml_pipeline({
  "source": { "type": "csv", "path": "data.csv", "target_column": "label" },
  "task": "auto",
  "model": "auto",
  "preprocess": { "normalize": true }
})
```

**Multi-step orchestration (Layer A):**
1. Load data → `ix_stats` or direct CSV
2. Preprocess → normalize, drop NaN
3. Split → train/test (80/20)
4. Train → `ix_supervised` or `ix_unsupervised`
5. Evaluate → metrics
6. Optionally persist → `ix_cache`

## Task Auto-Detection

| Signal | Inferred Task | Default Model |
|--------|---------------|---------------|
| Target has 2 unique integer values | Binary classification | LogisticRegression |
| Target has 3-20 unique integers, ratio < 5% | Multiclass classification | DecisionTree |
| Target is continuous (> 20 unique or non-integer) | Regression | LinearRegression |
| No target column specified | Clustering | KMeans (k=3) |
| User says "reduce", "visualize", "embed" | Dimensionality reduction | PCA |
| User says "anomaly", "outlier" | Anomaly detection | DBSCAN |

## Model Selection (Governance-Aware)

Article 4 (Proportionality) — don't use complex models when simple ones suffice:

| Data Size | Features | Recommended | Why |
|-----------|----------|-------------|-----|
| < 100 rows | Any | LinearRegression / KNN | Small data → simple model |
| 100-10k | < 20 | DecisionTree / KMeans | Good interpretability |
| 100-10k | 20+ | PCA → then model | Reduce dimensions first |
| 10k+ | Any | RandomForest / GradientBoosting / GMM | Enough data for complexity |
| 10k+ | Sequence data | **Transformer** | Attention captures long-range patterns |

**Override:** User can always specify `model` explicitly to bypass the heuristic.

### Transformer Model
For sequence or high-dimensional data, use `"model": "transformer"` with params:
```json
"model_params": { "d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 128, "epochs": 50, "learning_rate": 0.001 }
```
- Full end-to-end backprop through attention, LayerNorm, FFN
- GPU-accelerated attention via WGPU (auto-fallback to CPU)
- `seq_len`: auto-detected from features (n_features / d_model) or specify explicitly

## Pipeline JSON Schema

```json
{
  "source": {
    "type": "csv|json|inline",
    "path": "path/to/data.csv",
    "data": [[1,2,3], [4,5,6]],
    "has_header": true,
    "target_column": "label_or_index"
  },
  "task": "classify|regress|cluster|reduce|auto",
  "model": "linear_regression|logistic_regression|decision_tree|knn|naive_bayes|svm|random_forest|transformer|kmeans|dbscan|pca|tsne|gmm|auto",
  "model_params": { "k": 5, "max_depth": 10 },
  "preprocess": {
    "normalize": false,
    "drop_nan": true,
    "pca_components": null
  },
  "split": { "test_ratio": 0.2, "seed": 42 },
  "persist": false,
  "persist_key": "my_model",
  "return_predictions": false,
  "max_rows": 100000,
  "max_features": 500
}
```

## Ephemeral Pipeline (default)

One-shot analysis — results returned, nothing cached.

```
User: "Classify iris.csv using the species column"

→ ix_ml_pipeline({
    "source": { "type": "csv", "path": "iris.csv", "target_column": "species" },
    "task": "classify",
    "model": "auto"
  })

→ Response: {
    "task": "multiclass_classification",
    "model": "decision_tree",
    "model_params": { "max_depth": 10 },
    "metrics": { "accuracy": 0.97, "f1": 0.96 },
    "n_train": 120, "n_test": 30,
    "timing_ms": 45
  }
```

## Persistent Pipeline

Train once, predict later. Model + scaler params cached.

```
User: "Train a classifier on customers.csv and save it"

→ ix_ml_pipeline({
    "source": { "type": "csv", "path": "customers.csv", "target_column": "churn" },
    "task": "classify",
    "model": "random_forest",
    "preprocess": { "normalize": true },
    "persist": true,
    "persist_key": "churn_model"
  })

→ Response: { "persisted": true, "key": "ix_ml:model:churn_model", "metrics": {...} }
```

**Later, predict on new data:**
```
→ ix_ml_predict({
    "persist_key": "churn_model",
    "data": [[45, 2, 50000], [28, 1, 30000]]
  })

→ Response: { "predictions": [1, 0], "model": "random_forest" }
```

## Governance Integration

**Before training:**
- Check proportionality — is the model appropriate for the data size?
- Log to stderr: `[ix-ml] pipeline: file=data.csv, rows=10000, cols=50, model=decision_tree`

**Before persisting:**
- Check self-modification policy — within session's modification budget?
- Prefix cache keys with `ix_ml:model:` namespace

**On stale model:**
- If data has changed since training, mark belief as Unknown (tetravalent)
- Recommend retraining

## Preprocessing Options

| Option | What It Does | When to Use |
|--------|-------------|-------------|
| `normalize: true` | StandardScaler (zero mean, unit variance) | Features on different scales |
| `drop_nan: true` | Remove rows with any NaN | Missing data (default: on) |
| `pca_components: N` | Reduce to N dimensions via PCA | High-dimensional data (> 50 features) |

## Limits

- **Max rows**: 100,000 (configurable via `max_rows`)
- **Max features**: 500 (configurable via `max_features`)
- **Max file size**: 50MB
- **File types**: `.csv` and `.json` only
- **Performance**: ~100ms regression, ~500ms-2.5s classification for 10k rows

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| "Failed to load data" | Bad path or format | Check file exists and is valid CSV/JSON |
| "Invalid target column" | Column name/index not found | List columns with `ix_stats` first |
| "Training failed" | Degenerate data (all same value, etc.) | Check data quality |
| "File too large" | Exceeds max_rows or file size | Sample data or increase limits |

## MCP Tools

| Tool | Purpose |
|------|---------|
| `ix_ml_pipeline` | Full pipeline: load → preprocess → train → evaluate → persist |
| `ix_ml_predict` | Load cached model and predict on new data |
