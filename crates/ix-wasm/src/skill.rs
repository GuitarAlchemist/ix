//! `wasm.run`, the ix-pipeline skill: one stage of an `ix.yaml` that runs a WASI module.
//!
//! ```yaml
//! stages:
//!   stats:
//!     skill: wasm.run
//!     args: { go: "package main ...", input: { data: [3, 1, 4] } }
//!   report:
//!     skill: wasm.run
//!     args: { stage: "go-…", input: { from: "stats.output" } }
//! ```
//!
//! The module is named by exactly one of `stage` (an id from the cache), `go` or `zig` (source,
//! built or reused) or `wat` (text). Its answer is `{ output, exit, fuel_used, stage }`, where
//! `output` is stdout as JSON (as text when it does not parse); a non-zero exit is an error.
//!
//! Stages are cached in `IX_WASM_CACHE`, else `%LOCALAPPDATA%\ix-wasm` (`~/.cache/ix-wasm`
//! elsewhere). Go is `IX_GO`, else `go` on `PATH`; Zig is `IX_ZIG`, else `zig`.

use crate::build::{id_of, is_id, Toolchain};
use crate::{Limits, Program};
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long a Go build may take inside a pipeline stage.
const BUILD_SECONDS: u64 = 300;

fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("IX_WASM_CACHE") {
        return dir.into();
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local).join("ix-wasm");
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".cache").join("ix-wasm")
}

fn go() -> &'static Toolchain {
    static GO: OnceLock<Toolchain> = OnceLock::new();
    GO.get_or_init(|| {
        let exe = std::env::var_os("IX_GO").unwrap_or_else(|| "go".into());
        Toolchain::go(exe, cache_dir())
    })
}

fn zig() -> &'static Toolchain {
    static ZIG: OnceLock<Toolchain> = OnceLock::new();
    ZIG.get_or_init(|| {
        let exe = std::env::var_os("IX_ZIG").unwrap_or_else(|| "zig".into());
        Toolchain::zig(exe, cache_dir())
    })
}

/// Builds `source` with `toolchain`, or reuses what an earlier build left.
fn built(toolchain: &Toolchain, source: &str) -> Result<(String, Arc<Program>), String> {
    let deadline = Instant::now() + Duration::from_secs(BUILD_SECONDS);
    let built = toolchain.build(source, deadline)?;
    if let Some(p) = recall(&built.id) {
        return Ok((built.id, p));
    }
    let bytes = std::fs::read(&built.path).map_err(|e| e.to_string())?;
    let program = Program::new(&bytes).map_err(|e| e.to_string())?;
    Ok((built.id.clone(), remember(&built.id, program)))
}

/// Loaded modules, shared by every stage of every pipeline in the process.
fn loaded() -> &'static Mutex<HashMap<String, Arc<Program>>> {
    static LOADED: OnceLock<Mutex<HashMap<String, Arc<Program>>>> = OnceLock::new();
    LOADED.get_or_init(Mutex::default)
}

fn remember(id: &str, program: Program) -> Arc<Program> {
    let program = Arc::new(program);
    loaded()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id.to_string(), program.clone());
    program
}

fn recall(id: &str) -> Option<Arc<Program>> {
    loaded()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(id)
        .cloned()
}

/// The module `params` names with exactly one of `stage`, `go`, `zig` or `wat`, loaded (and
/// built first, for source), and its id. ix-stream's `wasm` nodes name theirs the same way.
pub fn program_of(params: &Value) -> Result<(String, Arc<Program>), String> {
    let text = |k: &str| params.get(k).and_then(Value::as_str);
    let named = ["stage", "go", "zig", "wat"]
        .iter()
        .filter(|k| params.get(**k).is_some())
        .count();
    if named != 1 {
        return Err("name the module with exactly one of `stage`, `go`, `zig` or `wat`".into());
    }
    match (text("stage"), text("go"), text("zig"), text("wat")) {
        (Some(id), ..) => {
            if !is_id(id) {
                return Err(format!("not a stage id: {id}"));
            }
            if let Some(p) = recall(id) {
                return Ok((id.to_string(), p));
            }
            let ext = if id.starts_with("wat-") {
                "wat"
            } else {
                "wasm"
            };
            let bytes = std::fs::read(cache_dir().join(format!("{id}.{ext}")))
                .map_err(|_| format!("no stage {id} in {}", cache_dir().display()))?;
            let program = Program::new(&bytes).map_err(|e| e.to_string())?;
            Ok((id.to_string(), remember(id, program)))
        }
        (_, Some(source), ..) => built(go(), source),
        (_, _, Some(source), _) => built(zig(), source),
        (_, _, _, Some(wat)) => {
            let id = id_of("wat", "", wat.as_bytes());
            if let Some(p) = recall(&id) {
                return Ok((id, p));
            }
            let program = Program::new(wat.as_bytes()).map_err(|e| e.to_string())?;
            let dir = cache_dir();
            std::fs::create_dir_all(&dir)
                .and_then(|()| std::fs::write(dir.join(format!("{id}.wat")), wat))
                .map_err(|e| e.to_string())?;
            Ok((id.clone(), remember(&id, program)))
        }
        _ => Err("`stage`, `go`, `zig` and `wat` are text".into()),
    }
}

pub fn wasm_run_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "stage": { "type": "string", "description": "The id of a cached stage (go-…, wat-…)." },
            "go": { "type": "string", "description": "Go source of a package main, built for wasip1 or reused from the cache." },
            "zig": { "type": "string", "description": "Zig source with a pub fn main, built for wasm32-wasi or reused from the cache." },
            "wat": { "type": "string", "description": "A module in WAT text." },
            "input": { "description": "Any JSON value: the stage reads it on stdin." },
            "fuel": { "type": "integer", "minimum": 0, "description": "Wasm fuel, about one unit per instruction (default 2e9, about 1.5 s)." }
        }
    })
}

pub fn wasm_run_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "output": { "description": "The stage's stdout, as JSON when it parses, else as text." },
            "exit": { "type": "integer" },
            "fuel_used": { "type": "integer", "description": "The same on every run of one stage on one input." },
            "stage": { "type": "string", "description": "The stage's id: give it as `stage` to reuse it." },
            "closed": { "type": "array", "items": { "type": "string" }, "description": "WASI functions the module imports that only answer an error (files, sockets)." }
        }
    })
}

/// Runs a WASI module as a pipeline stage, sandboxed: JSON in on stdin, JSON out on stdout, no
/// files, no network, a seeded clock and random source, bounded by fuel and memory.
#[ix_skill(
    domain = "wasm",
    name = "wasm.run",
    governance = "safety,deterministic",
    schema_fn = "crate::skill::wasm_run_schema",
    output_schema_fn = "crate::skill::wasm_run_output_schema"
)]
pub fn wasm_run(params: Value) -> Result<Value, String> {
    let (id, program) = program_of(&params)?;
    let input = params.get("input").cloned().unwrap_or(Value::Null);
    let stdin = serde_json::to_vec(&input).map_err(|e| e.to_string())?;
    let mut limits = Limits::default();
    if let Some(fuel) = params.get("fuel").and_then(Value::as_u64) {
        limits.fuel = fuel;
    }
    let out = program.run(&stdin, &limits).map_err(|e| e.to_string())?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if out.exit != 0 {
        return Err(format!("{id} exited {}: {}", out.exit, stderr.trim()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let output = serde_json::from_str(text.trim()).unwrap_or_else(|_| Value::String(text.into()));
    Ok(json!({
        "output": output, "exit": out.exit, "fuel_used": out.fuel_used, "stage": id,
        "closed": program.closed(),
    }))
}
