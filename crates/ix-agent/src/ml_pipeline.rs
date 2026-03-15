//! ML Pipeline orchestration — end-to-end train/predict workflows.
//!
//! Provides `run_pipeline` (load → preprocess → train → evaluate) and
//! `run_predict` (load persisted model → predict on new data).

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use ix_cache::{Cache, CacheConfig};
use ix_math::preprocessing::{self, InferredTask, StandardScaler};
use ix_supervised::serialization::ModelEnvelope;

use std::sync::OnceLock;

/// Global cache instance shared across tool calls.
fn global_cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Cache::new(CacheConfig::default()))
}

// ── Config types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub source: SourceConfig,
    #[serde(default = "default_task")]
    pub task: String,
    #[serde(default = "default_model")]
    pub model: String,
    pub model_params: Option<Value>,
    #[serde(default)]
    pub preprocess: PreprocessConfig,
    #[serde(default)]
    pub split: SplitConfig,
    #[serde(default)]
    pub persist: bool,
    pub persist_key: Option<String>,
    #[serde(default)]
    pub return_predictions: bool,
    #[serde(default = "default_max_rows")]
    pub max_rows: usize,
    #[serde(default = "default_max_features")]
    pub max_features: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    #[serde(rename = "type")]
    pub source_type: String,
    pub path: Option<String>,
    pub data: Option<Vec<Vec<f64>>>,
    pub has_header: Option<bool>,
    pub target_column: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    #[serde(default)]
    pub normalize: bool,
    #[serde(default = "default_true")]
    pub drop_nan: bool,
    pub pca_components: Option<usize>,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            normalize: false,
            drop_nan: true,
            pca_components: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConfig {
    #[serde(default = "default_test_ratio")]
    pub test_ratio: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            test_ratio: default_test_ratio(),
            seed: default_seed(),
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_test_ratio() -> f64 {
    0.2
}
fn default_seed() -> u64 {
    42
}
fn default_task() -> String {
    "auto".into()
}
fn default_model() -> String {
    "auto".into()
}
fn default_max_rows() -> usize {
    50_000
}
fn default_max_features() -> usize {
    500
}

// ── Pipeline execution ────────────────────────────────────────────

pub fn run_pipeline(config: PipelineConfig) -> Result<Value, String> {
    let start = Instant::now();

    // 1. Load data
    let (matrix, col_names) = load_data(&config.source)?;

    // 2. Validate limits
    let (nrows, ncols) = matrix.dim();
    if nrows > config.max_rows {
        return Err(format!(
            "Data has {} rows, exceeding max_rows={}",
            nrows, config.max_rows
        ));
    }
    if ncols > config.max_features {
        return Err(format!(
            "Data has {} columns, exceeding max_features={}",
            ncols, config.max_features
        ));
    }
    if nrows == 0 {
        return Err("Dataset is empty after loading".into());
    }

    // 3. Split into X and y (if supervised)
    let target_col = resolve_target_column(&config.source.target_column, &col_names, ncols);
    let (mut x, y_opt) = if let Some(tc) = target_col {
        if tc >= ncols {
            return Err(format!(
                "target_column index {} is out of range (data has {} columns)",
                tc, ncols
            ));
        }
        let y = matrix.column(tc).to_owned();
        // Remove target column from X
        let x_cols: Vec<usize> = (0..ncols).filter(|&c| c != tc).collect();
        let x = matrix.select(Axis(1), &x_cols);
        (x, Some(y))
    } else {
        (matrix, None)
    };

    // 4. Preprocess
    let mut nan_rows_dropped: usize = 0;
    let mut scaler: Option<StandardScaler> = None;
    let mut y_opt = y_opt;

    if config.preprocess.drop_nan {
        let pre_rows = x.nrows();
        if let Some(ref y) = y_opt {
            // Combine X and y to drop rows with NaN in either
            let combined = ndarray::concatenate(
                Axis(1),
                &[x.view(), y.clone().insert_axis(Axis(1)).view()],
            )
            .map_err(|e| format!("concat error: {e}"))?;
            let clean = preprocessing::drop_nan_rows(&combined);
            if clean.nrows() == 0 {
                return Err("All rows contain NaN values".into());
            }
            let yc = clean.ncols() - 1;
            x = clean.slice(ndarray::s![.., ..yc]).to_owned();
            y_opt = Some(clean.column(yc).to_owned());
        } else {
            x = preprocessing::drop_nan_rows(&x);
            if x.nrows() == 0 {
                return Err("All rows contain NaN values".into());
            }
        }
        nan_rows_dropped = pre_rows - x.nrows();
    }

    if config.preprocess.normalize {
        let (sc, transformed) =
            StandardScaler::fit_transform(&x).map_err(|e| format!("Scaler error: {e}"))?;
        x = transformed;
        scaler = Some(sc);
    }

    if let Some(n_comp) = config.preprocess.pca_components {
        if n_comp > 0 && n_comp < x.ncols() {
            use ix_unsupervised::pca::PCA;
            use ix_unsupervised::traits::DimensionReducer;
            let mut pca = PCA::new(n_comp);
            x = pca.fit_transform(&x);
        }
    }

    let data_shape = json!({ "rows": x.nrows(), "features": x.ncols() });

    // 5. Detect task
    let task = if config.task == "auto" {
        if let Some(ref y) = y_opt {
            match preprocessing::infer_task_type(y.as_slice().unwrap(), 20) {
                InferredTask::BinaryClassification
                | InferredTask::MulticlassClassification { .. } => "classify".to_string(),
                InferredTask::Regression => "regress".to_string(),
            }
        } else {
            "cluster".to_string()
        }
    } else {
        config.task.clone()
    };

    // 6. Select model
    let model_name = if config.model == "auto" {
        auto_select_model(&task, x.nrows(), x.ncols())
    } else {
        config.model.clone()
    };

    eprintln!(
        "[ix-ml] pipeline: rows={}, cols={}, task={}, model={}",
        x.nrows(),
        x.ncols(),
        task,
        model_name
    );

    // 7. Train & evaluate
    let result = match task.as_str() {
        "classify" => run_classification(
            &x,
            y_opt.as_ref().ok_or("Classification requires a target column")?,
            &model_name,
            &config.model_params,
            &config.split,
            config.return_predictions,
        )?,
        "regress" => run_regression(
            &x,
            y_opt.as_ref().ok_or("Regression requires a target column")?,
            &model_name,
            &config.split,
            config.return_predictions,
        )?,
        "cluster" => run_clustering(
            &x,
            &model_name,
            &config.model_params,
            config.return_predictions,
        )?,
        _ => return Err(format!("Unknown task: '{}'. Use classify, regress, cluster, or auto", task)),
    };

    // 8. Persist
    let persisted = if config.persist {
        let key = config
            .persist_key
            .clone()
            .unwrap_or_else(|| format!("pipeline_{}", chrono_now_stub()));
        let cache_key = format!("ix_ml:model:{}", key);

        let preprocessing_state = scaler.as_ref().map(|sc| {
            json!({
                "type": "standard_scaler",
                "means": sc.means.to_vec(),
                "stds": sc.stds.to_vec(),
            })
        });

        let envelope = ModelEnvelope {
            version: "0.1.0".to_string(),
            algorithm: model_name.clone(),
            params: result
                .get("model_state")
                .cloned()
                .unwrap_or(json!(null)),
            preprocessing: preprocessing_state,
            feature_names: col_names,
            trained_at: timestamp_now_stub(),
        };

        let envelope_json =
            serde_json::to_string(&envelope).map_err(|e| format!("Serialize error: {e}"))?;
        global_cache().set_str(&cache_key, &envelope_json);
        true
    } else {
        false
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;

    // 9. Build response
    let mut resp = json!({
        "task": task,
        "model": model_name,
        "data_shape": data_shape,
        "preprocessing": {
            "normalized": config.preprocess.normalize,
            "nan_rows_dropped": nan_rows_dropped,
        },
        "persisted": persisted,
        "timing_ms": elapsed_ms,
    });

    // Merge in metrics / predictions from the task result
    if let Some(obj) = resp.as_object_mut() {
        if let Some(metrics) = result.get("metrics") {
            obj.insert("metrics".into(), metrics.clone());
        }
        if let Some(split) = result.get("split") {
            obj.insert("split".into(), split.clone());
        }
        if let Some(preds) = result.get("predictions") {
            obj.insert("predictions".into(), preds.clone());
        }
        if let Some(cluster_info) = result.get("cluster_info") {
            obj.insert("cluster_info".into(), cluster_info.clone());
        }
        if let Some(mp) = result.get("model_params") {
            obj.insert("model_params".into(), mp.clone());
        }
        if persisted {
            let key = config
                .persist_key
                .unwrap_or_else(|| format!("pipeline_{}", chrono_now_stub()));
            obj.insert("persist_key".into(), json!(key));
        }
    }

    Ok(resp)
}

pub fn run_predict(persist_key: &str, data: &[Vec<f64>]) -> Result<Value, String> {
    let cache_key = format!("ix_ml:model:{}", persist_key);
    let envelope_json: String = global_cache()
        .get_str(&cache_key)
        .ok_or_else(|| format!("No persisted model found for key '{}'", persist_key))?;

    let envelope: ModelEnvelope =
        serde_json::from_str(&envelope_json).map_err(|e| format!("Deserialize error: {e}"))?;

    // Convert input data to Array2
    let x = vecs_to_array2(data)?;

    // Apply saved preprocessing
    let x = if let Some(ref prep) = envelope.preprocessing {
        if prep.get("type").and_then(|v| v.as_str()) == Some("standard_scaler") {
            let means_vec: Vec<f64> = prep
                .get("means")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .ok_or("Missing scaler means in envelope")?;
            let stds_vec: Vec<f64> = prep
                .get("stds")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .ok_or("Missing scaler stds in envelope")?;
            let scaler = StandardScaler {
                means: Array1::from_vec(means_vec),
                stds: Array1::from_vec(stds_vec),
            };
            scaler.transform(&x)
        } else {
            x
        }
    } else {
        x
    };

    // Predict based on algorithm
    let predictions = match envelope.algorithm.as_str() {
        "linear_regression" => {
            use ix_supervised::linear_regression::{LinearRegression, LinearRegressionState};
            use ix_supervised::traits::Regressor;
            let state: LinearRegressionState =
                serde_json::from_value(envelope.params.clone())
                    .map_err(|e| format!("Deserialize model state: {e}"))?;
            let model = LinearRegression::load_state(&state);
            let preds = model.predict(&x);
            json!(preds.to_vec())
        }
        "decision_tree" => {
            use ix_supervised::decision_tree::{DecisionTree, DecisionTreeState};
            use ix_supervised::traits::Classifier;
            let state: DecisionTreeState =
                serde_json::from_value(envelope.params.clone())
                    .map_err(|e| format!("Deserialize model state: {e}"))?;
            let model = DecisionTree::load_state(&state);
            let preds = model.predict(&x);
            json!(preds.to_vec())
        }
        "kmeans" => {
            use ix_unsupervised::kmeans::{KMeans, KMeansState};
            use ix_unsupervised::traits::Clusterer;
            let state: KMeansState =
                serde_json::from_value(envelope.params.clone())
                    .map_err(|e| format!("Deserialize model state: {e}"))?;
            let model = KMeans::load_state(&state);
            let preds = model.predict(&x);
            json!(preds.to_vec())
        }
        other => return Err(format!("Prediction not supported for algorithm '{}'", other)),
    };

    Ok(json!({
        "persist_key": persist_key,
        "algorithm": envelope.algorithm,
        "n_samples": data.len(),
        "predictions": predictions,
    }))
}

// ── Internal helpers ──────────────────────────────────────────────

fn load_data(source: &SourceConfig) -> Result<(Array2<f64>, Option<Vec<String>>), String> {
    match source.source_type.as_str() {
        "inline" => {
            let data = source
                .data
                .as_ref()
                .ok_or("source.type='inline' requires source.data")?;
            let matrix = vecs_to_array2(data)?;
            Ok((matrix, None))
        }
        "csv" => {
            let path_str = source
                .path
                .as_ref()
                .ok_or("source.type='csv' requires source.path")?;

            validate_file_path(path_str)?;

            let path = Path::new(path_str);
            let has_header = source.has_header.unwrap_or(true);
            let (matrix, names) = ix_io::csv_io::load_csv_matrix(path, has_header)
                .map_err(|e| format!("CSV load error: {e}"))?;
            Ok((matrix, names))
        }
        "json" => Err("JSON source type is not yet supported; use 'csv' or 'inline'".into()),
        other => Err(format!(
            "Unknown source type '{}'. Use 'csv', 'json', or 'inline'",
            other
        )),
    }
}

fn validate_file_path(path: &str) -> Result<(), String> {
    if path.contains("..") {
        return Err("File path must not contain '..'".into());
    }
    let lower = path.to_lowercase();
    if !(lower.ends_with(".csv") || lower.ends_with(".json")) {
        return Err("Only .csv and .json file extensions are allowed".into());
    }
    // Check file size (50 MB limit)
    let metadata = std::fs::metadata(path).map_err(|e| format!("Cannot access file: {e}"))?;
    if metadata.len() > 50 * 1024 * 1024 {
        return Err("File exceeds 50 MB size limit".into());
    }
    Ok(())
}

fn resolve_target_column(
    target: &Option<Value>,
    col_names: &Option<Vec<String>>,
    ncols: usize,
) -> Option<usize> {
    let tc = target.as_ref()?;
    if let Some(idx) = tc.as_u64() {
        return Some(idx as usize);
    }
    if let Some(idx) = tc.as_i64() {
        if idx >= 0 {
            return Some(idx as usize);
        }
        // Negative index: count from end
        let resolved = ncols as i64 + idx;
        if resolved >= 0 {
            return Some(resolved as usize);
        }
        return None;
    }
    if let Some(name) = tc.as_str() {
        if let Some(names) = col_names {
            return names.iter().position(|n| n == name);
        }
    }
    None
}

fn auto_select_model(task: &str, nrows: usize, ncols: usize) -> String {
    match task {
        "classify" => {
            if nrows < 100 {
                "knn".to_string()
            } else if nrows <= 10_000 && ncols < 20 {
                "decision_tree".to_string()
            } else {
                "random_forest".to_string()
            }
        }
        "regress" => "linear_regression".to_string(),
        "cluster" => "kmeans".to_string(),
        _ => "linear_regression".to_string(),
    }
}

fn run_classification(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model_name: &str,
    model_params: &Option<Value>,
    split: &SplitConfig,
    return_predictions: bool,
) -> Result<Value, String> {
    // Convert y from f64 to usize labels
    let y_usize: Array1<usize> = y.mapv(|v| v.round() as usize);
    let n_classes = *y_usize.iter().max().unwrap_or(&0) + 1;

    // Split
    let split_result = preprocessing::train_test_split(x, y, split.test_ratio, split.seed)
        .map_err(|e| format!("Split error: {e}"))?;

    let y_train_usize: Array1<usize> = split_result.y_train.mapv(|v| v.round() as usize);
    let y_test_usize: Array1<usize> = split_result.y_test.mapv(|v| v.round() as usize);

    let (predictions, model_state, mp) = match model_name {
        "knn" => {
            use ix_supervised::knn::KNN;
            use ix_supervised::traits::Classifier;
            let k = model_params
                .as_ref()
                .and_then(|p| p.get("k"))
                .and_then(|v| v.as_u64())
                .unwrap_or(5) as usize;
            let mut model = KNN::new(k);
            model.fit(&split_result.x_train, &y_train_usize);
            let preds = model.predict(&split_result.x_test);
            (preds, json!(null), json!({ "k": k }))
        }
        "decision_tree" => {
            use ix_supervised::decision_tree::DecisionTree;
            use ix_supervised::traits::Classifier;
            let max_depth = model_params
                .as_ref()
                .and_then(|p| p.get("max_depth"))
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;
            let mut model = DecisionTree::new(max_depth);
            model.fit(&split_result.x_train, &y_train_usize);
            let preds = model.predict(&split_result.x_test);
            let state = model.save_state().map(|s| json!(s)).unwrap_or(json!(null));
            (preds, state, json!({ "max_depth": max_depth }))
        }
        "random_forest" => {
            use ix_ensemble::random_forest::RandomForest;
            use ix_ensemble::traits::EnsembleClassifier;
            let n_trees = model_params
                .as_ref()
                .and_then(|p| p.get("n_trees"))
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;
            let max_depth = model_params
                .as_ref()
                .and_then(|p| p.get("max_depth"))
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;
            let mut model = RandomForest::new(n_trees, max_depth).with_seed(split.seed);
            model.fit(&split_result.x_train, &y_train_usize);
            let preds = model.predict(&split_result.x_test);
            // RandomForest doesn't have save_state; store null
            (preds, json!(null), json!({ "n_trees": n_trees, "max_depth": max_depth }))
        }
        "transformer" => {
            use ix_nn::classifier::{TransformerClassifier, TransformerConfig};
            use ix_supervised::traits::Classifier;
            let d_model = model_params.as_ref().and_then(|p| p.get("d_model")).and_then(|v| v.as_u64()).unwrap_or(32) as usize;
            let n_heads = model_params.as_ref().and_then(|p| p.get("n_heads")).and_then(|v| v.as_u64()).unwrap_or(4) as usize;
            let n_layers = model_params.as_ref().and_then(|p| p.get("n_layers")).and_then(|v| v.as_u64()).unwrap_or(2) as usize;
            let d_ff = model_params.as_ref().and_then(|p| p.get("d_ff")).and_then(|v| v.as_u64()).unwrap_or(128) as usize;
            let epochs = model_params.as_ref().and_then(|p| p.get("epochs")).and_then(|v| v.as_u64()).unwrap_or(50) as usize;
            let lr = model_params.as_ref().and_then(|p| p.get("learning_rate")).and_then(|v| v.as_f64()).unwrap_or(0.001);
            let seq_len = model_params.as_ref().and_then(|p| p.get("seq_len")).and_then(|v| v.as_u64()).map(|v| v as usize);
            let config = TransformerConfig { d_model, n_heads, n_layers, d_ff, seq_len, epochs, learning_rate: lr, seed: split.seed };
            let mut model = TransformerClassifier::new(config);
            model.fit(&split_result.x_train, &y_train_usize);
            let preds = model.predict(&split_result.x_test);
            (preds, json!(null), json!({ "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff, "epochs": epochs }))
        }
        other => return Err(format!("Unknown classification model: '{}'", other)),
    };

    // Metrics: macro-average across classes
    let acc = ix_supervised::metrics::accuracy(&y_test_usize, &predictions);
    let (avg_p, avg_r, avg_f1) = if n_classes > 0 {
        let mut sum_p = 0.0;
        let mut sum_r = 0.0;
        let mut sum_f1 = 0.0;
        for c in 0..n_classes {
            sum_p += ix_supervised::metrics::precision(&y_test_usize, &predictions, c);
            sum_r += ix_supervised::metrics::recall(&y_test_usize, &predictions, c);
            sum_f1 += ix_supervised::metrics::f1_score(&y_test_usize, &predictions, c);
        }
        let nc = n_classes as f64;
        (sum_p / nc, sum_r / nc, sum_f1 / nc)
    } else {
        (0.0, 0.0, 0.0)
    };

    let mut result = json!({
        "metrics": {
            "accuracy": acc,
            "precision": avg_p,
            "recall": avg_r,
            "f1": avg_f1,
        },
        "split": {
            "train": split_result.x_train.nrows(),
            "test": split_result.x_test.nrows(),
        },
        "model_state": model_state,
        "model_params": mp,
    });

    if return_predictions {
        result
            .as_object_mut()
            .unwrap()
            .insert("predictions".into(), json!(predictions.to_vec()));
    }

    Ok(result)
}

fn run_regression(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model_name: &str,
    split: &SplitConfig,
    return_predictions: bool,
) -> Result<Value, String> {
    let split_result = preprocessing::train_test_split(x, y, split.test_ratio, split.seed)
        .map_err(|e| format!("Split error: {e}"))?;

    let (predictions, model_state) = match model_name {
        "linear_regression" => {
            use ix_supervised::linear_regression::LinearRegression;
            use ix_supervised::traits::Regressor;
            let mut model = LinearRegression::new();
            model.fit(&split_result.x_train, &split_result.y_train);
            let preds = model.predict(&split_result.x_test);
            let state = model.save_state().map(|s| json!(s)).unwrap_or(json!(null));
            (preds, state)
        }
        "transformer" => {
            use ix_nn::classifier::{TransformerRegressor, TransformerConfig};
            use ix_supervised::traits::Regressor;
            let config = TransformerConfig {
                d_model: 32, n_heads: 4, n_layers: 2, d_ff: 128,
                seq_len: None, epochs: 50, learning_rate: 0.001, seed: split.seed,
            };
            let mut model = TransformerRegressor::new(config);
            model.fit(&split_result.x_train, &split_result.y_train);
            let preds = model.predict(&split_result.x_test);
            (preds, json!(null))
        }
        other => return Err(format!("Unknown regression model: '{}'", other)),
    };

    let mse_val = ix_supervised::metrics::mse(&split_result.y_test, &predictions);
    let rmse_val = ix_supervised::metrics::rmse(&split_result.y_test, &predictions);
    let r2_val = ix_supervised::metrics::r_squared(&split_result.y_test, &predictions);

    let mut result = json!({
        "metrics": {
            "mse": mse_val,
            "rmse": rmse_val,
            "r_squared": r2_val,
        },
        "split": {
            "train": split_result.x_train.nrows(),
            "test": split_result.x_test.nrows(),
        },
        "model_state": model_state,
        "model_params": { "model": model_name },
    });

    if return_predictions {
        result
            .as_object_mut()
            .unwrap()
            .insert("predictions".into(), json!(predictions.to_vec()));
    }

    Ok(result)
}

fn run_clustering(
    x: &Array2<f64>,
    model_name: &str,
    model_params: &Option<Value>,
    return_predictions: bool,
) -> Result<Value, String> {
    match model_name {
        "kmeans" => {
            use ix_unsupervised::kmeans::KMeans;
            use ix_unsupervised::traits::Clusterer;

            let k = model_params
                .as_ref()
                .and_then(|p| p.get("k"))
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize;

            let mut model = KMeans::new(k);
            let labels = model.fit_predict(x);

            // Compute cluster sizes
            let mut sizes: HashMap<usize, usize> = HashMap::new();
            for &label in labels.iter() {
                *sizes.entry(label).or_insert(0) += 1;
            }
            let cluster_sizes: Vec<Value> = (0..k)
                .map(|c| json!({ "cluster": c, "size": sizes.get(&c).copied().unwrap_or(0) }))
                .collect();

            let model_state = model
                .save_state()
                .map(|s| json!(s))
                .unwrap_or(json!(null));

            let mut result = json!({
                "cluster_info": {
                    "n_clusters": k,
                    "cluster_sizes": cluster_sizes,
                },
                "model_state": model_state,
                "model_params": { "k": k },
            });

            if return_predictions {
                result
                    .as_object_mut()
                    .unwrap()
                    .insert("predictions".into(), json!(labels.to_vec()));
            }

            Ok(result)
        }
        other => Err(format!("Unknown clustering model: '{}'", other)),
    }
}

fn vecs_to_array2(rows: &[Vec<f64>]) -> Result<Array2<f64>, String> {
    if rows.is_empty() {
        return Err("Empty data".into());
    }
    let ncols = rows[0].len();
    if rows.iter().any(|r| r.len() != ncols) {
        return Err("Inconsistent row lengths in data".into());
    }
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), ncols), flat)
        .map_err(|e| format!("Matrix shape error: {e}"))
}

/// Stub: returns a simple epoch-based identifier.
fn chrono_now_stub() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Stub: returns an ISO-8601-ish timestamp.
fn timestamp_now_stub() -> String {
    let secs = chrono_now_stub();
    format!("1970-01-01T00:00:00Z+{secs}s")
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn iris_inline_config(task: &str, model: &str) -> PipelineConfig {
        // Simple linearly separable 2-class dataset
        let data = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.2, 0.0],
            vec![0.3, 0.1, 0.0],
            vec![0.2, 0.3, 0.0],
            vec![5.0, 5.0, 1.0],
            vec![5.1, 5.2, 1.0],
            vec![5.3, 5.1, 1.0],
            vec![5.2, 5.3, 1.0],
            vec![0.4, 0.1, 0.0],
            vec![4.9, 5.1, 1.0],
        ];
        PipelineConfig {
            source: SourceConfig {
                source_type: "inline".into(),
                path: None,
                data: Some(data),
                has_header: None,
                target_column: Some(json!(2)),
            },
            task: task.into(),
            model: model.into(),
            model_params: None,
            preprocess: PreprocessConfig {
                normalize: false,
                drop_nan: true,
                pca_components: None,
            },
            split: SplitConfig {
                test_ratio: 0.3,
                seed: 42,
            },
            persist: false,
            persist_key: None,
            return_predictions: true,
            max_rows: 50_000,
            max_features: 500,
        }
    }

    #[test]
    fn test_classification_knn() {
        let config = iris_inline_config("classify", "knn");
        let result = run_pipeline(config).unwrap();
        assert_eq!(result["task"], "classify");
        assert_eq!(result["model"], "knn");
        assert!(result["metrics"]["accuracy"].as_f64().unwrap() >= 0.0);
        assert!(result["predictions"].is_array());
    }

    #[test]
    fn test_classification_decision_tree() {
        let config = iris_inline_config("classify", "decision_tree");
        let result = run_pipeline(config).unwrap();
        assert_eq!(result["task"], "classify");
        assert_eq!(result["model"], "decision_tree");
    }

    #[test]
    fn test_regression() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
            vec![5.0, 10.0],
            vec![6.0, 12.0],
            vec![7.0, 14.0],
            vec![8.0, 16.0],
            vec![9.0, 18.0],
            vec![10.0, 20.0],
        ];
        let config = PipelineConfig {
            source: SourceConfig {
                source_type: "inline".into(),
                path: None,
                data: Some(data),
                has_header: None,
                target_column: Some(json!(1)),
            },
            task: "regress".into(),
            model: "linear_regression".into(),
            model_params: None,
            preprocess: PreprocessConfig::default(),
            split: SplitConfig {
                test_ratio: 0.2,
                seed: 42,
            },
            persist: false,
            persist_key: None,
            return_predictions: true,
            max_rows: 50_000,
            max_features: 500,
        };
        let result = run_pipeline(config).unwrap();
        assert_eq!(result["task"], "regress");
        assert!(result["metrics"]["r_squared"].as_f64().unwrap() > 0.9);
    }

    #[test]
    fn test_clustering() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![5.2, 5.0],
        ];
        let config = PipelineConfig {
            source: SourceConfig {
                source_type: "inline".into(),
                path: None,
                data: Some(data),
                has_header: None,
                target_column: None,
            },
            task: "cluster".into(),
            model: "kmeans".into(),
            model_params: Some(json!({ "k": 2 })),
            preprocess: PreprocessConfig::default(),
            split: SplitConfig::default(),
            persist: false,
            persist_key: None,
            return_predictions: true,
            max_rows: 50_000,
            max_features: 500,
        };
        let result = run_pipeline(config).unwrap();
        assert_eq!(result["task"], "cluster");
        assert_eq!(result["cluster_info"]["n_clusters"], 2);
        assert!(result["predictions"].is_array());
    }

    #[test]
    fn test_auto_task_detection() {
        let config = iris_inline_config("auto", "auto");
        let result = run_pipeline(config).unwrap();
        // With integer labels 0/1, should auto-detect as classify
        assert_eq!(result["task"], "classify");
    }

    #[test]
    fn test_persist_and_predict() {
        let mut config = iris_inline_config("classify", "decision_tree");
        config.persist = true;
        config.persist_key = Some("test_persist_model".into());

        let result = run_pipeline(config).unwrap();
        assert!(result["persisted"].as_bool().unwrap());

        // Now predict with new data
        let new_data = vec![vec![0.1, 0.1], vec![5.0, 5.0]];
        let pred_result = run_predict("test_persist_model", &new_data).unwrap();
        assert_eq!(pred_result["algorithm"], "decision_tree");
        assert!(pred_result["predictions"].is_array());
    }

    #[test]
    fn test_reject_path_traversal() {
        let config = PipelineConfig {
            source: SourceConfig {
                source_type: "csv".into(),
                path: Some("../../etc/passwd.csv".into()),
                data: None,
                has_header: None,
                target_column: None,
            },
            task: "auto".into(),
            model: "auto".into(),
            model_params: None,
            preprocess: PreprocessConfig::default(),
            split: SplitConfig::default(),
            persist: false,
            persist_key: None,
            return_predictions: false,
            max_rows: 50_000,
            max_features: 500,
        };
        let result = run_pipeline(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains(".."));
    }

    #[test]
    fn test_max_rows_exceeded() {
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        let config = PipelineConfig {
            source: SourceConfig {
                source_type: "inline".into(),
                path: None,
                data: Some(data),
                has_header: None,
                target_column: Some(json!(1)),
            },
            task: "regress".into(),
            model: "auto".into(),
            model_params: None,
            preprocess: PreprocessConfig::default(),
            split: SplitConfig::default(),
            persist: false,
            persist_key: None,
            return_predictions: false,
            max_rows: 5, // intentionally low
            max_features: 500,
        };
        let result = run_pipeline(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max_rows"));
    }
}
