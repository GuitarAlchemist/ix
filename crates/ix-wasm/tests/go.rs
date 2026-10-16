//! A Go stage, built and run. Needs Go 1.21+ on `PATH` (or `IX_GO`), so it is ignored by
//! default: `cargo test -p ix-wasm -- --ignored`.

use ix_wasm::build::{is_id, Toolchain};
use ix_wasm::{Limits, Program};
use std::time::{Duration, Instant};

const STATS: &str = r#"package main

import (
	"encoding/json"
	"math"
	"os"
	"sort"
)

type in struct {
	Data []float64 `json:"data"`
}

type out struct {
	N      int     `json:"n"`
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	Std    float64 `json:"std"`
}

func main() {
	var x in
	if err := json.NewDecoder(os.Stdin).Decode(&x); err != nil {
		os.Stderr.WriteString("bad input: " + err.Error() + "\n")
		os.Exit(2)
	}
	n := len(x.Data)
	o := out{N: n}
	if n > 0 {
		s := append([]float64(nil), x.Data...)
		sort.Float64s(s)
		for _, v := range s {
			o.Mean += v
		}
		o.Mean /= float64(n)
		for _, v := range s {
			o.Std += (v - o.Mean) * (v - o.Mean)
		}
		o.Std = math.Sqrt(o.Std / float64(n))
		o.Median = s[n/2]
		if n%2 == 0 {
			o.Median = (s[n/2-1] + s[n/2]) / 2
		}
	}
	json.NewEncoder(os.Stdout).Encode(o)
}
"#;

/// A toolchain caching into a folder that lives as long as the guard.
fn go() -> (tempfile::TempDir, Toolchain) {
    let cache = tempfile::tempdir().unwrap();
    let go = std::env::var_os("IX_GO").unwrap_or_else(|| "go".into());
    let build = Toolchain::go(go, cache.path());
    (cache, build)
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(300)
}

#[test]
#[ignore = "needs the Go toolchain"]
fn a_go_stage_is_built_once_and_runs_in_the_sandbox() {
    let (_cache, go) = go();
    let first = go.build(STATS, deadline()).unwrap();
    assert!(is_id(&first.id) && !first.cached, "{first:?}");
    let again = go.build(STATS, deadline()).unwrap();
    assert_eq!((again.id.as_str(), again.cached), (first.id.as_str(), true));

    let loaded = Instant::now();
    let p = Program::new(&std::fs::read(&first.path).unwrap()).unwrap();
    let load_ms = loaded.elapsed().as_secs_f64() * 1e3;
    let ran = Instant::now();
    let out = p
        .run(br#"{"data":[3,1,4,1,5,9,2,6]}"#, &Limits::default())
        .unwrap();
    let run_ms = ran.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "go stage: load {load_ms:.1} ms, run {run_ms:.1} ms, fuel {}",
        out.fuel_used
    );
    assert_eq!(out.exit, 0, "{}", String::from_utf8_lossy(&out.stderr));
    // The Go runtime reads the clock and the random source; both are the sandbox's.
    let twice = p
        .run(br#"{"data":[3,1,4,1,5,9,2,6]}"#, &Limits::default())
        .unwrap();
    assert_eq!(twice, out);
    assert_eq!(
        String::from_utf8(out.stdout).unwrap().trim(),
        r#"{"n":8,"mean":3.875,"median":3.5,"std":2.5708704751503917}"#
    );
    let bad = p.run(b"not json", &Limits::default()).unwrap();
    assert_eq!(bad.exit, 2);
}

#[test]
#[ignore = "needs the Go toolchain"]
fn a_go_stage_that_does_not_compile_answers_the_compiler() {
    let (_cache, go) = go();
    let e = go
        .build("package main\n\nfunc main() { undefined() }\n", deadline())
        .unwrap_err();
    assert!(e.contains("undefined"), "{e}");
}

#[test]
#[ignore = "needs the Go toolchain"]
fn a_go_stage_cannot_fetch_a_module() {
    let src =
        "package main\n\nimport \"github.com/example/nothing\"\n\nfunc main() { nothing.X() }\n";
    let (_cache, go) = go();
    let e = go.build(src, deadline()).unwrap_err();
    assert!(e.contains("github.com/example/nothing"), "{e}");
}

#[test]
#[ignore = "needs the Go toolchain"]
fn a_go_stage_that_opens_a_file_reads_nothing() {
    let src = r#"package main

import "os"

func main() {
	b, _ := os.ReadFile("C:/Windows/win.ini")
	os.Stdout.Write(b)
}
"#;
    let (_cache, go) = go();
    let built = go.build(src, deadline()).unwrap();
    let p = Program::new(&std::fs::read(&built.path).unwrap()).unwrap();
    assert!(
        p.closed().iter().any(|f| f == "path_open"),
        "{:?}",
        p.closed()
    );
    let out = p.run(b"", &Limits::default()).unwrap();
    assert_eq!((out.exit, out.stdout.len()), (0, 0));
}
