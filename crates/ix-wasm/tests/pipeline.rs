//! `wasm.run` as stages of a real `ix.yaml`: lowered by ix-pipeline, run by its executor.

use ix_pipeline::executor::{execute, NoCache};
use ix_pipeline::lower::lower;
use ix_pipeline::spec::PipelineSpec;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::OnceLock;
// Links ix-wasm, so its `#[ix_tool]` entry reaches the registry.
use ix_wasm as _;

/// Echoes stdin to stdout.
const ECHO: &str = r#"(module
  (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const 4096))
    (drop (call $read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 16)))
    (i32.store (i32.const 8) (i32.const 64))
    (i32.store (i32.const 12) (i32.load (i32.const 16)))
    (drop (call $write (i32.const 1) (i32.const 8) (i32.const 1) (i32.const 20)))))"#;

/// One cache folder for the whole test binary: the skill reads `IX_WASM_CACHE`.
fn cache() {
    static DIR: OnceLock<tempfile::TempDir> = OnceLock::new();
    let dir = DIR.get_or_init(|| tempfile::tempdir().unwrap());
    std::env::set_var("IX_WASM_CACHE", dir.path());
}

fn run(spec: Value) -> HashMap<String, Value> {
    cache();
    let spec: PipelineSpec = serde_json::from_value(spec).unwrap();
    let dag = lower(&spec).unwrap();
    let result = execute(&dag, &HashMap::new(), &NoCache).unwrap();
    spec.stages
        .keys()
        .map(|id| (id.clone(), result.output(id).cloned().unwrap()))
        .collect()
}

#[test]
fn wasm_stages_chain_fan_out_and_join() {
    let out = run(json!({
        "version": "1",
        "stages": {
            "load":  { "skill": "wasm.run", "args": { "wat": ECHO, "input": { "data": [3, 1, 4] } } },
            "left":  { "skill": "wasm.run", "args": { "wat": ECHO, "input": { "from": "load.output" } } },
            "right": { "skill": "wasm.run", "args": { "stage": { "from": "load.stage" },
                                                       "input": { "from": "load.output.data" } } },
            "join":  { "skill": "wasm.run", "args": { "wat": ECHO, "input": {
                "left": { "from": "left.output" }, "right": { "from": "right.output" } } } }
        }
    }));
    assert_eq!(
        out["join"]["output"],
        json!({ "left": { "data": [3, 1, 4] }, "right": [3, 1, 4] })
    );
    // One module, one id; its id flowed from `load` into `right`.
    assert_eq!(out["right"]["stage"], out["load"]["stage"]);
    assert_eq!(out["left"]["fuel_used"], out["load"]["fuel_used"]);
}

#[test]
fn a_failing_stage_fails_the_pipeline_with_its_reason() {
    cache();
    let spec: PipelineSpec = serde_json::from_value(json!({
        "version": "1",
        "stages": {
            "spin": { "skill": "wasm.run", "args": {
                "wat": r#"(module (func (export "_start") (loop $l (br $l))))"#, "fuel": 100000 } }
        }
    }))
    .unwrap();
    let e = execute(&lower(&spec).unwrap(), &HashMap::new(), &NoCache).unwrap_err();
    assert!(e.to_string().contains("out of fuel"), "{e}");
}

#[test]
fn the_skill_is_registered_with_its_schemas() {
    let d = ix_registry::by_name("wasm.run").expect("wasm.run is registered");
    assert_eq!(d.domain, "wasm");
    assert!((d.json_schema)()["properties"]["go"].is_object());
    assert!((d.output_schema)()["properties"]["fuel_used"].is_object());
}

const STATS: &str = r#"package main

import (
	"encoding/json"
	"math"
	"os"
)

func main() {
	var x struct{ Data []float64 `json:"data"` }
	json.NewDecoder(os.Stdin).Decode(&x)
	n := float64(len(x.Data))
	mean, std := 0.0, 0.0
	for _, v := range x.Data {
		mean += v / n
	}
	for _, v := range x.Data {
		std += (v - mean) * (v - mean) / n
	}
	json.NewEncoder(os.Stdout).Encode(map[string]float64{"mean": mean, "std": math.Sqrt(std)})
}
"#;

const ZSCORE: &str = r#"package main

import (
	"encoding/json"
	"os"
)

func main() {
	var x struct {
		Data  []float64 `json:"data"`
		Stats struct {
			Mean float64 `json:"mean"`
			Std  float64 `json:"std"`
		} `json:"stats"`
	}
	json.NewDecoder(os.Stdin).Decode(&x)
	z := make([]float64, len(x.Data))
	for i, v := range x.Data {
		z[i] = (v - x.Stats.Mean) / x.Stats.Std
	}
	json.NewEncoder(os.Stdout).Encode(z)
}
"#;

#[test]
#[ignore = "needs the Go toolchain"]
fn go_stages_compose_in_a_pipeline() {
    let out = run(json!({
        "version": "1",
        "stages": {
            "stats": { "skill": "wasm.run", "args": { "go": STATS, "input": { "data": [2, 4, 4, 4, 5, 5, 7, 9] } } },
            "z": { "skill": "wasm.run", "args": { "go": ZSCORE, "input": {
                "data": [2, 4, 4, 4, 5, 5, 7, 9], "stats": { "from": "stats.output" } } } }
        }
    }));
    // Go writes 5.0 as 5, which JSON reads back as an integer.
    assert_eq!(out["stats"]["output"], json!({ "mean": 5, "std": 2 }));
    assert_eq!(
        out["z"]["output"],
        json!([-1.5, -0.5, -0.5, -0.5, 0, 0, 1, 2])
    );
}
