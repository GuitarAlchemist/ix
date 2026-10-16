//! Stages in other languages, run from a loop. WAT stages need no toolchain; the Go test is
//! ignored unless asked for (`cargo test -p ix-loop -- --ignored`).

use ix_loop::wasm::WASM_FUEL_PER_FUEL;
use ix_loop::{Grant, Host, Scripted, Stages, Store};
use serde_json::{json, Value as Json};
use std::rc::Rc;

/// Echoes stdin to stdout.
const ECHO: &str = r#"
(module
  (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const 4096))
    (drop (call $read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 16)))
    (i32.store (i32.const 8) (i32.const 64))
    (i32.store (i32.const 12) (i32.load (i32.const 16)))
    (drop (call $write (i32.const 1) (i32.const 8) (i32.const 1) (i32.const 20)))))
"#;

const SPIN: &str = r#"(module (func (export "_start") (loop $l (br $l))))"#;

fn host(dir: &tempfile::TempDir, wasm: bool) -> Rc<Host> {
    let host = Host::new(Rc::new(Scripted::always("hi")), Store::at(dir.path()));
    let go = std::env::var_os("IX_GO").unwrap_or_else(|| "go".into());
    Rc::new(if wasm {
        host.with_wasm(Stages::new(dir.path().join("wasm"), go))
    } else {
        host
    })
}

fn grant(fuel: i64) -> Grant {
    Grant {
        calls: 0,
        fuel,
        seconds: 120.0,
    }
}

fn run(host: &Rc<Host>, src: &str, fuel: i64) -> (Json, i64) {
    let src = format!("local ECHO, SPIN = {ECHO:?}, {SPIN:?}\n{src}");
    let out = host.run(&src, "test", &grant(fuel));
    let v = out.value.unwrap_or_else(|e| panic!("the loop failed: {e}"));
    (v, out.fuel_used)
}

#[test]
fn stages_are_off_unless_the_person_turns_them_on() {
    let dir = tempfile::tempdir().unwrap();
    let (v, _) = run(
        &host(&dir, false),
        r#"local ok, why = wasm.wat(ECHO) return { ok, why }"#,
        1_000_000,
    );
    assert_eq!(
        v,
        json!([false, "wasm: stages are off (run ix-loop with --wasm)"])
    );
}

#[test]
fn a_loop_builds_a_stage_and_runs_it_on_json() {
    let dir = tempfile::tempdir().unwrap();
    let (v, _) = run(
        &host(&dir, true),
        r#"
        local ok, id = wasm.wat(ECHO)
        assert(ok, id)
        local ran, out, stats = wasm.run(id, { data = { 3, 1, 4 }, note = "x" })
        return { id = id, ran = ran, out = out, exit = stats.exit, fuel = stats.fuel_used > 0 }
    "#,
        1_000_000,
    );
    assert_eq!(v["ran"], json!(true));
    assert_eq!(v["out"], json!({ "data": [3, 1, 4], "note": "x" }));
    assert_eq!(
        (v["exit"].clone(), v["fuel"].clone()),
        (json!(0), json!(true))
    );
    let id = v["id"].as_str().unwrap();
    assert!(dir.path().join("wasm").join(format!("{id}.wat")).is_file());
}

#[test]
fn a_stage_id_is_never_a_path() {
    let dir = tempfile::tempdir().unwrap();
    let (v, _) = run(
        &host(&dir, true),
        r#"
        local out = {}
        for _, id in ipairs({ "../../etc/passwd", "go-zz", "wat-0123456789abcdef0123456789abcdef" }) do
          local ok, why = wasm.run(id, {})
          out[#out + 1] = { ok, why }
        end
        return out
    "#,
        1_000_000,
    );
    assert_eq!(
        v[0],
        json!([false, "wasm: not a stage id: ../../etc/passwd"])
    );
    assert_eq!(v[1], json!([false, "wasm: not a stage id: go-zz"]));
    assert_eq!(
        v[2],
        json!([false, "wasm: no stage wat-0123456789abcdef0123456789abcdef"])
    );
}

#[test]
fn a_stage_that_never_ends_is_charged_to_the_loop_and_stops() {
    let dir = tempfile::tempdir().unwrap();
    let loop_fuel = 2_000_000;
    let (v, used) = run(
        &host(&dir, true),
        r#"
        local _, id = wasm.wat(SPIN)
        local ok, why = wasm.run(id, {}, { fuel = 8000000 })
        return { ok, why }
    "#,
        loop_fuel,
    );
    assert_eq!(v[0], json!(false));
    assert!(
        v[1].as_str().unwrap().starts_with("wasm: out of fuel"),
        "{v}"
    );
    // 8e6 wasm fuel is 5e5 of the loop's, at 16 to one.
    let charged = 8_000_000 / WASM_FUEL_PER_FUEL as i64;
    assert!(
        used >= charged && used < charged + 100_000,
        "the loop spent {used}"
    );
}

#[test]
fn a_stage_gets_no_more_fuel_than_the_loop_can_pay() {
    let dir = tempfile::tempdir().unwrap();
    let (v, _) = run(
        &host(&dir, true),
        r#"
        local _, id = wasm.wat(SPIN)
        local ok, why, stats = wasm.run(id, {})
        local after = budget().fuel
        local again, why2 = wasm.run(id, {})
        return { ok, why, after, again, why2 }
    "#,
        200_000,
    );
    assert_eq!(v[0], json!(false));
    assert!(
        v[1].as_str().unwrap().starts_with("wasm: out of fuel"),
        "{v}"
    );
    assert_eq!(v[3], json!(false));
    assert!(
        v[4].as_str().unwrap().starts_with("budget: out of fuel"),
        "{v}"
    );
}

#[test]
#[ignore = "needs the Go toolchain"]
fn a_loop_writes_a_go_stage_and_a_compile_error_comes_back_as_data() {
    let dir = tempfile::tempdir().unwrap();
    let (v, _) = run(
        &host(&dir, true),
        r#"
        local bad_ok, report = wasm.go("package main\nfunc main() { nope() }\n")
        local ok, id, info = wasm.go([[
package main

import (
	"encoding/json"
	"os"
)

func main() {
	var x struct{ Words []string `json:"words"` }
	json.NewDecoder(os.Stdin).Decode(&x)
	n := 0
	for _, w := range x.Words {
		n += len(w)
	}
	json.NewEncoder(os.Stdout).Encode(map[string]int{"letters": n})
}
]])
        assert(ok, id)
        local ran, out = wasm.run(id, { words = { "plier", "une", "grue" } })
        local again, id2, info2 = wasm.go([[
package main

import (
	"encoding/json"
	"os"
)

func main() {
	var x struct{ Words []string `json:"words"` }
	json.NewDecoder(os.Stdin).Decode(&x)
	n := 0
	for _, w := range x.Words {
		n += len(w)
	}
	json.NewEncoder(os.Stdout).Encode(map[string]int{"letters": n})
}
]])
        return { bad_ok = bad_ok, undefined = report:find("undefined") ~= nil,
                 ran = ran, out = out, cached = info2.cached, same = id2 == id }
    "#,
        10_000_000,
    );
    assert_eq!(
        v,
        json!({ "bad_ok": false, "undefined": true, "ran": true, "out": { "letters": 12 },
                "cached": true, "same": true })
    );
}
