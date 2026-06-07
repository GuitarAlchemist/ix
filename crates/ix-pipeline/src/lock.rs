//! `ix.lock` — a reproducibility manifest written next to `ix.yaml`.
//!
//! Phase 1 is **write-only**: the lock is regenerated from scratch on every
//! `ix pipeline run` to provide an audit trail. Hash verification (refusing
//! to run when `ix.lock` doesn't match current `ix.yaml` stages) arrives in
//! Phase 2.
//!
//! Format (YAML):
//!
//! ```yaml
//! schema: ix-lock/v1
//! generated: 2026-04-05T14:32:11Z
//! stages:
//!   load:
//!     skill: stats
//!     args_hash: sha256:3f2c8d9a…
//!     deps: []
//!     duration_ms: 3
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::executor::PipelineResult;
use crate::spec::PipelineSpec;

pub const LOCK_SCHEMA: &str = "ix-lock/v2";
/// Accepted on read: v2 is a strict additive superset of v1's field set.
const LOCK_SCHEMA_V1: &str = "ix-lock/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFile {
    pub schema: String,
    pub generated: String,
    /// Unique-per-run id (content+time derived) — keys the append-only
    /// `provenance.jsonl` trail. NEW in v2; defaulted when reading v1.
    #[serde(default)]
    pub run_id: String,
    /// sha256 of the canonicalized `PipelineSpec`. NEW in v2.
    #[serde(default)]
    pub spec_hash: String,
    pub stages: BTreeMap<String, LockedStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockedStage {
    pub skill: String,
    /// fnv1a64 over the canonicalized **template** args (v1 — unchanged).
    pub args_hash: String,
    pub deps: Vec<String>,
    pub duration_ms: u64,
    pub cache_hit: bool,
    /// fnv1a64 over the canonicalized **resolved** args (post `{"from"}`
    /// resolution) — the inputs the skill actually ran with. NEW in v2.
    #[serde(default)]
    pub resolved_args_hash: String,
    /// sha256 of the canonical-JSON stage output. NEW in v2.
    #[serde(default)]
    pub output_hash: String,
    /// The cross-stage edges that actually flowed into this stage. NEW in v2.
    #[serde(default)]
    pub inputs: Vec<InputBinding>,
    /// RESERVED for chain-of-evidence: a hexavalent certainty for this output.
    /// Unpopulated in v1 of the mechanism (no reusable propagation algebra yet —
    /// see docs/plans/2026-06-07-chain-of-evidence-v1.md §Deferred).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certainty: Option<Value>,
}

/// One consumed cross-stage reference (`{"from": "stage.key"}`), linked to the
/// exact upstream output it carried.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputBinding {
    /// The consumer arg name the reference appeared under.
    pub name: String,
    /// The reference's stage token — the text before the first `.` in the
    /// `{"from"}` target. For a producing stage this is its id (and
    /// `upstream_output_hash` links to it); for a seed-input ref it is the seed
    /// key and `upstream_output_hash` is empty (no producing stage this run).
    pub from_stage: String,
    /// Output key within the producer's output (`*` = whole output).
    pub from_key: String,
    /// sha256 of the producing stage's output — equals that stage's
    /// `output_hash` in the same run (the verifiable link). Empty for seed
    /// inputs (no producing stage in this run).
    pub upstream_output_hash: String,
}

impl LockFile {
    /// Build an `ix-lock/v2` provenance record from a successfully-executed
    /// pipeline. Every v2 field is derived from `(spec, result, initial_inputs)`
    /// alone — the per-stage outputs live in `result.node_results`, and the
    /// resolved args are reconstructed by replaying `{"from"}` resolution against
    /// those outputs (deterministic, identical to what ran), so no executor or
    /// closure changes are needed. `initial_inputs` are the seed values handed to
    /// `execute()` (empty for inline-data specs).
    ///
    /// CALLER CONTRACT: `initial_inputs` MUST be the same seed map passed to
    /// `execute()` for this run. Re-resolution looks refs up against (those
    /// seeds + stage outputs); if a caller seeds `execute()` but hands `from_run`
    /// a different/empty map, a seed-backed ref fails to resolve and
    /// `resolved_args_hash` silently falls back to the template hash. The sole
    /// caller (`ix pipeline run`) passes one shared, empty map, so this holds today.
    // @ai:assumption from_run's `initial_inputs` is the SAME seed map passed to execute() this run; else a seed-backed resolved_args_hash silently degrades to the template hash [U:uncertain conf:0.6 src:ix-skill pipeline.rs run() passes one shared empty map to both — unenforced precondition]
    pub fn from_run(
        spec: &PipelineSpec,
        result: &PipelineResult,
        initial_inputs: &std::collections::HashMap<String, Value>,
    ) -> Self {
        // {stage_id -> output} (plus any seed inputs by their own keys), for
        // re-resolving `{"from"}` refs and linking upstream output hashes.
        let mut outputs: std::collections::HashMap<String, Value> = initial_inputs.clone();
        for (id, r) in &result.node_results {
            outputs.insert(id.clone(), r.output.clone());
        }
        // Per-stage output hashes, computed once so InputBinding can link to them.
        let output_hashes: BTreeMap<String, String> = result
            .node_results
            .iter()
            .map(|(id, r)| (id.clone(), sha256_json(&r.output)))
            .collect();

        let mut stages = BTreeMap::new();
        for (id, stage) in &spec.stages {
            let node_result = result.node_results.get(id);
            let (duration_ms, cache_hit) = node_result
                .map(|r| (r.duration.as_millis() as u64, r.cache_hit))
                .unwrap_or((0, false));
            // Resolved args: replay {"from"} resolution; fall back to the template
            // if a ref can't resolve (e.g. a seed not present) — never panic.
            let resolved = crate::lower::resolve_from_refs(&stage.args, &outputs)
                .unwrap_or_else(|_| stage.args.clone());
            let inputs = collect_input_bindings(&stage.args, &output_hashes);
            stages.insert(
                id.clone(),
                LockedStage {
                    skill: stage.skill.clone(),
                    args_hash: hash_json(&stage.args),
                    deps: stage.deps.clone(),
                    duration_ms,
                    cache_hit,
                    resolved_args_hash: hash_json(&resolved),
                    output_hash: output_hashes.get(id).cloned().unwrap_or_default(),
                    inputs,
                    certainty: None,
                },
            );
        }
        let spec_hash = sha256_str(&canonicalize(
            &serde_json::to_value(spec).unwrap_or(Value::Null),
        ));
        let generated = current_iso_timestamp();
        // Content+time run id: distinct per run, stable to re-parse. 16 hex chars
        // of sha256 over (timestamp, spec hash, stage count) — drop the
        // "sha256:" prefix so the id reads `run:<hex>`.
        let run_digest = sha256_str(&format!("{generated}|{spec_hash}|{}", stages.len()));
        let run_hex = run_digest.strip_prefix("sha256:").unwrap_or(&run_digest);
        let run_id = format!("run:{}", &run_hex[..16.min(run_hex.len())]);
        LockFile {
            schema: LOCK_SCHEMA.into(),
            generated,
            run_id,
            spec_hash,
            stages,
        }
    }

    /// Verify the evidence chain: every `InputBinding.upstream_output_hash` must
    /// equal the `output_hash` recorded for its `from_stage` in this run (seed
    /// inputs, which have no producing stage, are exempt). Returns the list of
    /// broken edges (empty = the chain holds). This *records* integrity; v1 does
    /// not gate execution on it.
    pub fn verify_chain(&self) -> Vec<String> {
        let mut broken = Vec::new();
        for (stage_id, st) in &self.stages {
            for b in &st.inputs {
                if b.upstream_output_hash.is_empty() {
                    continue; // seed input — no producing stage this run
                }
                match self.stages.get(&b.from_stage) {
                    Some(up) if up.output_hash == b.upstream_output_hash => {}
                    Some(up) => broken.push(format!(
                        "stage '{stage_id}' input '{}' claims {}={} but '{}' produced {}",
                        b.name, b.from_stage, b.upstream_output_hash, b.from_stage, up.output_hash
                    )),
                    None => broken.push(format!(
                        "stage '{stage_id}' input '{}' references unknown producing stage '{}'",
                        b.name, b.from_stage
                    )),
                }
            }
        }
        broken
    }

    /// Serialize to YAML.
    pub fn to_yaml_string(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    /// One-line JSON for the append-only `provenance.jsonl` trail.
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Load a lock from disk. Accepts both `ix-lock/v2` and the legacy
    /// `ix-lock/v1` (v2 fields default to empty when reading v1).
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, LockError> {
        let text = std::fs::read_to_string(path)?;
        let lf: LockFile = serde_yaml::from_str(&text)?;
        if lf.schema != LOCK_SCHEMA && lf.schema != LOCK_SCHEMA_V1 {
            return Err(LockError::SchemaMismatch {
                expected: LOCK_SCHEMA.into(),
                actual: lf.schema,
            });
        }
        Ok(lf)
    }
}

/// Collect the cross-stage references in a stage's `args` as `InputBinding`s,
/// attributing each to the arg name it appeared under and linking it to the
/// producer's output hash. Mirrors `lower::collect_from_refs`/`resolve_from_refs`
/// ref semantics (a ref is a single-key `{"from": "stage[.key]"}` object).
fn collect_input_bindings(
    args: &Value,
    output_hashes: &BTreeMap<String, String>,
) -> Vec<InputBinding> {
    fn walk(
        value: &Value,
        name: &str,
        hashes: &BTreeMap<String, String>,
        out: &mut Vec<InputBinding>,
    ) {
        match value {
            Value::Object(map) => {
                if map.len() == 1 {
                    if let Some(Value::String(target)) = map.get("from") {
                        let (from_stage, from_key) = match target.split_once('.') {
                            Some((s, k)) => (s.to_string(), k.to_string()),
                            None => (target.clone(), "*".to_string()),
                        };
                        let upstream_output_hash =
                            hashes.get(&from_stage).cloned().unwrap_or_default();
                        out.push(InputBinding {
                            name: name.to_string(),
                            from_stage,
                            from_key,
                            upstream_output_hash,
                        });
                        return;
                    }
                }
                for (k, v) in map {
                    walk(v, k, hashes, out);
                }
            }
            Value::Array(items) => {
                for v in items {
                    walk(v, name, hashes, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(args, "", output_hashes, &mut out);
    out
}

/// sha256 hex of a string, `sha256:` prefixed (the provenance integrity hash).
fn sha256_str(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("sha256:{:x}", h.finalize())
}

/// sha256 of a JSON value over its canonical form (key-sorted), `sha256:` prefixed.
fn sha256_json(v: &Value) -> String {
    sha256_str(&canonicalize(v))
}

#[derive(Debug, thiserror::Error)]
pub enum LockError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("schema mismatch: expected {expected}, got {actual}")]
    SchemaMismatch { expected: String, actual: String },
}

/// Stable content hash of a JSON `Value`. Uses the serialized form so that
/// structurally-equivalent JSON produces the same hash regardless of key
/// insertion order (BTreeMap ordering from serde_json).
fn hash_json(v: &Value) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let serialized = canonicalize(v);
    let mut hasher = DefaultHasher::new();
    serialized.hash(&mut hasher);
    format!("fnv1a64:{:016x}", hasher.finish())
}

fn canonicalize(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| k.as_str());
            let parts: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{k:?}:{}", canonicalize(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(canonicalize).collect();
            format!("[{}]", parts.join(","))
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn current_iso_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let (year, month, day) = days_to_ymd(days);
    let h = (secs % 86_400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(mut days: i64) -> (i64, u32, u32) {
    days += 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = (days - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::StageSpec;
    use serde_json::json;

    #[test]
    fn hash_is_stable_across_key_ordering() {
        let a = json!({ "x": 1, "y": 2 });
        let b = json!({ "y": 2, "x": 1 });
        assert_eq!(hash_json(&a), hash_json(&b));
    }

    #[test]
    fn hash_differs_with_different_values() {
        let a = json!({ "x": 1 });
        let b = json!({ "x": 2 });
        assert_ne!(hash_json(&a), hash_json(&b));
    }

    #[test]
    fn lock_roundtrips_through_yaml() {
        let mut stages = BTreeMap::new();
        stages.insert(
            "load".to_string(),
            StageSpec {
                skill: "stats".into(),
                args: json!({"data": [1.0, 2.0]}),
                deps: vec![],
                cache: None,
            },
        );
        let spec = PipelineSpec {
            version: "1".into(),
            params: BTreeMap::new(),
            stages,
            x_editor: Value::Null,
        };

        // Empty PipelineResult stand-in
        let result = PipelineResult {
            node_results: Default::default(),
            total_duration: std::time::Duration::ZERO,
            cache_hits: 0,
            execution_order: vec![],
        };

        let lock = LockFile::from_run(&spec, &result, &std::collections::HashMap::new());
        assert_eq!(lock.schema, LOCK_SCHEMA);
        assert_eq!(lock.stages.len(), 1);
        assert!(lock.stages["load"].args_hash.starts_with("fnv1a64:"));

        let yaml = lock.to_yaml_string().unwrap();
        let back: LockFile = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.stages["load"].skill, "stats");
        assert_eq!(back.stages["load"].args_hash, lock.stages["load"].args_hash);
    }

    use crate::executor::NodeResult;
    use std::collections::HashMap;
    use std::time::Duration;

    fn two_stage_run() -> (PipelineSpec, PipelineResult) {
        let mut stages = BTreeMap::new();
        stages.insert(
            "reduce".to_string(),
            StageSpec {
                skill: "pca".into(),
                args: json!({"data": [[1.0, 2.0]], "n_components": 1}),
                deps: vec![],
                cache: None,
            },
        );
        stages.insert(
            "cluster".to_string(),
            StageSpec {
                skill: "kmeans".into(),
                args: json!({"data": {"from": "reduce.transformed"}, "k": 2}),
                deps: vec!["reduce".into()],
                cache: None,
            },
        );
        let spec = PipelineSpec {
            version: "1".into(),
            params: BTreeMap::new(),
            stages,
            x_editor: Value::Null,
        };
        let mut node_results = HashMap::new();
        node_results.insert(
            "reduce".to_string(),
            NodeResult {
                node_id: "reduce".into(),
                output: json!({"transformed": [[0.5]], "explained_variance_ratio": [1.0]}),
                duration: Duration::ZERO,
                cache_hit: false,
            },
        );
        node_results.insert(
            "cluster".to_string(),
            NodeResult {
                node_id: "cluster".into(),
                output: json!({"labels": [0], "k": 2}),
                duration: Duration::ZERO,
                cache_hit: false,
            },
        );
        let result = PipelineResult {
            node_results,
            total_duration: Duration::ZERO,
            cache_hits: 0,
            execution_order: vec![],
        };
        (spec, result)
    }

    // Criterion 1 + 4: v2 record materializes with hashes + a verifiable edge.
    #[test]
    fn v2_records_hashes_and_links_the_edge() {
        let (spec, result) = two_stage_run();
        let lock = LockFile::from_run(&spec, &result, &HashMap::new());

        assert_eq!(lock.schema, "ix-lock/v2");
        assert!(lock.run_id.starts_with("run:"));
        assert!(lock.spec_hash.starts_with("sha256:"));

        let red = &lock.stages["reduce"];
        let clu = &lock.stages["cluster"];
        assert!(red.output_hash.starts_with("sha256:"), "output hashed");
        assert!(clu.resolved_args_hash.starts_with("fnv1a64:"));

        // The cross-stage edge is captured AND linked to the producer's hash.
        assert_eq!(clu.inputs.len(), 1);
        let edge = &clu.inputs[0];
        assert_eq!(edge.name, "data");
        assert_eq!(edge.from_stage, "reduce");
        assert_eq!(edge.from_key, "transformed");
        assert_eq!(edge.upstream_output_hash, red.output_hash);

        // Criterion 4: the chain verifies clean.
        assert!(lock.verify_chain().is_empty());
    }

    // Criterion 4 (negative): a tampered upstream hash is detected.
    #[test]
    fn v2_verify_chain_detects_a_broken_edge() {
        let (spec, result) = two_stage_run();
        let mut lock = LockFile::from_run(&spec, &result, &HashMap::new());
        lock.stages.get_mut("cluster").unwrap().inputs[0].upstream_output_hash =
            "sha256:deadbeef".into();
        let broken = lock.verify_chain();
        assert_eq!(broken.len(), 1, "one broken edge: {broken:?}");
        assert!(broken[0].contains("cluster"));
    }

    // Criterion 2 + 3: output hashing is deterministic and value-sensitive.
    #[test]
    fn v2_output_hash_is_deterministic_and_change_sensitive() {
        let a = sha256_json(&json!({"transformed": [[0.5]]}));
        let b = sha256_json(&json!({"transformed": [[0.5]]}));
        let c = sha256_json(&json!({"transformed": [[0.6]]}));
        assert_eq!(a, b, "same value → same hash");
        assert_ne!(a, c, "changed value → changed hash");
        assert!(a.starts_with("sha256:"));
    }

    // Criterion 6: a v1 lock still deserializes (v2 fields default).
    #[test]
    fn v1_lock_still_loads() {
        let v1 = r#"schema: ix-lock/v1
generated: "2026-01-01T00:00:00Z"
stages:
  load:
    skill: stats
    args_hash: "fnv1a64:0000000000000000"
    deps: []
    duration_ms: 1
    cache_hit: false
"#;
        let lf: LockFile = serde_yaml::from_str(v1).unwrap();
        assert_eq!(lf.stages["load"].skill, "stats");
        assert!(
            lf.stages["load"].output_hash.is_empty(),
            "v2 field defaults"
        );
        assert!(lf.run_id.is_empty());
    }
}
