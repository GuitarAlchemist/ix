//! `wasm` nodes of `stream.run`: each value runs a sandboxed WASI program. Run with
//! `--features wasm`; the Go stages need Go 1.21+ on `PATH` (or `IX_GO`), so they are ignored by
//! default: `cargo test -p ix-stream --features wasm -- --include-ignored`.

use ix_stream::skill::stream_run;
use serde_json::{json, Value};

/// A source, one stage, and a sink, the stage being `stage`.
fn chain(stage: Value, max_events: u64) -> Value {
    json!({
        "graph": {
            "nodes": [{ "name": "src", "source": true }, stage, { "name": "sink" }],
            "edges": [{ "from": 0, "to": 1 }, { "from": 1, "to": 2 }]
        },
        "max_events": max_events
    })
}

#[cfg(not(feature = "wasm"))]
#[test]
fn without_the_feature_a_wasm_node_is_refused_before_the_run() {
    let stage = json!({ "name": "w", "wasm": true, "program": { "wat": "(module)" } });
    let e = stream_run(chain(stage, 4)).unwrap_err();
    assert!(e.contains("`wasm` feature"), "{e}");
}

#[cfg(feature = "wasm")]
mod with_wasm {
    use super::*;

    /// Doubles `i`: `{ "i": n }` in, `{ "i": 2n }` out.
    const DOUBLE: &str = r#"package main

import (
	"encoding/json"
	"os"
)

func main() {
	var v struct {
		I int64 `json:"i"`
	}
	if err := json.NewDecoder(os.Stdin).Decode(&v); err != nil {
		os.Stderr.WriteString("bad input: " + err.Error())
		os.Exit(2)
	}
	json.NewEncoder(os.Stdout).Encode(map[string]int64{"i": 2 * v.I})
}
"#;

    /// Passes `{ "i": n }` on when `n` is even, and exits 3 otherwise.
    const EVEN: &str = r#"package main

import (
	"encoding/json"
	"os"
)

func main() {
	var v struct {
		I int64 `json:"i"`
	}
	if err := json.NewDecoder(os.Stdin).Decode(&v); err != nil || v.I%2 != 0 {
		os.Exit(3)
	}
	json.NewEncoder(os.Stdout).Encode(v)
}
"#;

    const GO_FUEL: u64 = 500_000_000;

    /// A source, a Go stage that doubles, a Go stage that refuses odd numbers, and a sink. The
    /// source sends `i` from 0, so the second stage passes everything only if the first doubled.
    fn doubling(max_events: u64) -> Value {
        json!({
            "graph": {
                "nodes": [
                    { "name": "src", "source": true },
                    { "name": "double", "wasm": true, "program": { "go": DOUBLE }, "fuel": GO_FUEL,
                      "cap": 4, "workers": 2 },
                    { "name": "even", "wasm": true, "program": { "go": EVEN }, "fuel": GO_FUEL },
                    { "name": "sink" }
                ],
                "edges": [{ "from": 0, "to": 1 }, { "from": 1, "to": 2 }, { "from": 2, "to": 3 }]
            },
            "max_events": max_events
        })
    }

    fn node<'a>(out: &'a Value, name: &str) -> &'a Value {
        out["receipts"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["name"] == name)
            .unwrap()
    }

    #[test]
    #[ignore = "needs the Go toolchain"]
    fn a_go_stage_doubles_each_value_on_its_way_to_the_sink() {
        let out = stream_run(doubling(20)).unwrap();
        assert_eq!(out["stopped_by"], "max_events", "{out:#}");
        assert_eq!(
            (out["events"].as_u64(), out["steps"].as_u64()),
            (Some(20), Some(60))
        );
        for name in ["double", "even"] {
            let n = node(&out, name);
            assert_eq!(
                (n["received"].as_u64(), n["emitted"].as_u64()),
                (Some(20), Some(20))
            );
            let w = &n["wasm"];
            assert_eq!(w["out"], 20, "{n}");
            assert!(w["stage"].as_str().unwrap().starts_with("go-"), "{n}");
            let fuel = w["fuel"].as_u64().unwrap();
            assert!(fuel > 0 && fuel <= 20 * GO_FUEL, "{n}");
            eprintln!("{name}: {fuel} fuel for 20 values");
        }
        assert_eq!(node(&out, "sink")["received"], 20);
    }

    #[test]
    #[ignore = "needs the Go toolchain"]
    fn three_runs_give_the_same_answer_and_spend_the_same_fuel() {
        let runs: Vec<Value> = (0..3).map(|_| stream_run(doubling(12)).unwrap()).collect();
        assert_eq!(runs[0]["stopped_by"], "max_events", "{:#}", runs[0]);
        assert_eq!(runs[0], runs[1]);
        assert_eq!(runs[1], runs[2]);
        eprintln!(
            "fuel over 3 runs: double {}, even {}",
            node(&runs[0], "double")["wasm"]["fuel"],
            node(&runs[0], "even")["wasm"]["fuel"]
        );
    }

    #[test]
    fn a_stage_out_of_fuel_stops_the_stream_with_an_error() {
        let spin = r#"(module (memory (export "memory") 1)
                        (func (export "_start") (loop $l (br $l))))"#;
        let stage = json!({ "name": "spin", "wasm": true, "program": { "wat": spin }, "fuel": 100000, "cap": 1 });
        let out = stream_run(chain(stage, 1000)).unwrap();
        assert_eq!(out["stopped_by"], "error", "{out:#}");
        assert_eq!(
            (&out["error"]["node"], &out["error"]["kind"]),
            (&json!("spin"), &json!("fuel"))
        );
        assert!(
            out["error"]["message"]
                .as_str()
                .unwrap()
                .contains("out of fuel"),
            "{out:#}"
        );
        // The stop reached the source long before its 1000 values.
        assert!(out["events"].as_u64().unwrap() < 100, "{out:#}");
        let w = &node(&out, "spin")["wasm"];
        // wasmi traps when its next charge would pass the cap, so a run stops just short of it.
        assert!(
            w["fuel"].as_u64().unwrap() >= 90_000 && w["out"] == 0,
            "{w}"
        );
    }

    #[test]
    fn a_stage_that_opens_a_file_is_refused_and_stops_the_stream() {
        // path_open from the first descriptor past stdio, then exit with its answer.
        let open = r#"
(module
  (import "wasi_snapshot_preview1" "path_open"
    (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 64) "secret.txt")
  (func (export "_start")
    (call $exit (call $open (i32.const 3) (i32.const 0) (i32.const 64) (i32.const 10)
      (i32.const 0) (i64.const -1) (i64.const -1) (i32.const 0) (i32.const 32)))))
"#;
        let stage =
            json!({ "name": "peek", "wasm": true, "program": { "wat": open }, "fuel": 1000000 });
        let out = stream_run(chain(stage, 50)).unwrap();
        assert_eq!(out["stopped_by"], "error", "{out:#}");
        assert_eq!(out["error"]["kind"], "refused", "{out:#}");
        assert!(
            out["error"]["message"]
                .as_str()
                .unwrap()
                .contains("path_open"),
            "{out:#}"
        );
    }

    #[test]
    #[ignore = "needs the Go toolchain"]
    fn a_go_stage_that_reads_a_file_is_refused_and_stops_the_stream() {
        let src = r#"package main

import (
	"fmt"
	"os"
)

func main() {
	b, err := os.ReadFile("C:/Windows/win.ini")
	fmt.Printf("{\"bytes\": %d, \"failed\": %t}\n", len(b), err != nil)
}
"#;
        let stage =
            json!({ "name": "peek", "wasm": true, "program": { "go": src }, "fuel": GO_FUEL });
        let out = stream_run(chain(stage, 5)).unwrap();
        assert_eq!(out["stopped_by"], "error", "{out:#}");
        assert_eq!(out["error"]["kind"], "refused", "{out:#}");
    }

    #[test]
    fn a_wasm_node_names_its_program_and_only_a_wasm_node_has_one() {
        let e = stream_run(chain(json!({ "name": "w", "wasm": true }), 4)).unwrap_err();
        assert!(e.contains("`program`"), "{e}");
        let e = stream_run(chain(
            json!({ "name": "w", "program": { "wat": "(module)" } }),
            4,
        ))
        .unwrap_err();
        assert!(e.contains("wasm: true"), "{e}");
    }
}
