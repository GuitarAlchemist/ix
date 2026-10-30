//! Stages in other languages, run sandboxed: `wasm.go`, `wasm.zig`, `wasm.wat`, `wasm.run`.
//!
//! A loop hands Go or Zig source (or WAT text) to the host, which builds it into a WASI module in a
//! cache keyed by content and answers its id. `wasm.run(id, input)` runs that module in
//! [`ix_wasm`]'s sandbox: the input goes in as JSON on stdin, the output comes back as JSON
//! when stdout parses, else as text. The module sees no files, no network, a seeded clock.
//!
//! Off unless the person turns it on (`ix-loop run --wasm`): building spawns a toolchain.
//!
//! A run's wasm fuel is charged to the loop's fuel at [`WASM_FUEL_PER_FUEL`] to one, about the
//! same wall time as a Lua instruction, so the root budget still bounds the whole tree.

use crate::budget::Budget;
use ix_wasm::build::{id_of, is_id, Toolchain};
use ix_wasm::{Limits, Program};
use serde_json::{json, Value as Json};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

/// Wasm fuel per unit of a loop's fuel.
pub const WASM_FUEL_PER_FUEL: u64 = 16;
/// The wasm fuel one run gets when the loop asks for none: about 1.5 s.
pub const DEFAULT_WASM_FUEL: u64 = 2_000_000_000;
/// Builds one root run may ask for (reusing a cached build counts too).
pub const MAX_BUILDS_PER_RUN: u32 = 50;
/// The most bytes of JSON one run takes as input.
pub const MAX_INPUT_BYTES: usize = 8 << 20;

pub const WASM_OFF: &str = "wasm: stages are off (run ix-loop with --wasm)";

/// What one `wasm.run` gave back.
#[derive(Debug, Clone, PartialEq)]
pub struct Ran {
    /// The output as JSON, or as text when stdout does not parse.
    pub value: Result<Json, String>,
    pub exit: i32,
    pub fuel_used: u64,
    pub stderr: String,
}

/// The cache of stages, and the toolchains that fill it.
pub struct Stages {
    cache: PathBuf,
    go: Toolchain,
    zig: Toolchain,
    loaded: RefCell<HashMap<String, Rc<Program>>>,
    builds: Cell<u32>,
}

impl Stages {
    /// Stages cached in `cache`, built with the toolchains `go` and `zig`.
    pub fn new(cache: impl Into<PathBuf>, go: impl Into<PathBuf>, zig: impl Into<PathBuf>) -> Self {
        let cache = cache.into();
        Self {
            go: Toolchain::go(go, &cache),
            zig: Toolchain::zig(zig, &cache),
            cache,
            loaded: RefCell::default(),
            builds: Cell::new(0),
        }
    }

    fn count_build(&self) -> Result<(), String> {
        if self.builds.get() >= MAX_BUILDS_PER_RUN {
            return Err("wasm: too many builds in one run".into());
        }
        self.builds.set(self.builds.get() + 1);
        Ok(())
    }

    /// Builds `lang` source (`go` or `zig`) into a stage; its id, and whether an earlier build
    /// was reused.
    pub fn build(
        &self,
        budget: &Budget,
        lang: &str,
        source: &str,
    ) -> Result<(String, bool), String> {
        let toolchain = match lang {
            "go" => &self.go,
            "zig" => &self.zig,
            other => return Err(format!("wasm: no toolchain for {other}")),
        };
        self.count_build()?;
        std::fs::create_dir_all(&self.cache).map_err(|e| format!("wasm: {e}"))?;
        let built = toolchain.build(source, budget.deadline())?;
        Ok((built.id, built.cached))
    }

    /// Checks WAT text and stores it as a stage; its id.
    pub fn wat(&self, text: &str) -> Result<String, String> {
        self.count_build()?;
        let program = Program::new(text.as_bytes()).map_err(|e| e.to_string())?;
        let id = id_of("wat", "", text.as_bytes());
        std::fs::create_dir_all(&self.cache).map_err(|e| format!("wasm: {e}"))?;
        std::fs::write(self.cache.join(format!("{id}.wat")), text)
            .map_err(|e| format!("wasm: {e}"))?;
        self.loaded
            .borrow_mut()
            .insert(id.clone(), Rc::new(program));
        Ok(id)
    }

    fn program(&self, id: &str) -> Result<Rc<Program>, String> {
        if !is_id(id) {
            return Err(format!("wasm: not a stage id: {}", crate::clip(id, 80)));
        }
        if let Some(p) = self.loaded.borrow().get(id) {
            return Ok(p.clone());
        }
        let ext = if id.starts_with("wat-") {
            "wat"
        } else {
            "wasm"
        };
        let path = self.cache.join(format!("{id}.{ext}"));
        let bytes = std::fs::read(&path).map_err(|_| format!("wasm: no stage {id}"))?;
        let program = Rc::new(Program::new(&bytes).map_err(|e| e.to_string())?);
        self.loaded
            .borrow_mut()
            .insert(id.to_string(), program.clone());
        Ok(program)
    }

    /// Runs stage `id` on `input`, with at most `fuel` wasm fuel (and no more than `budget` can
    /// pay), and charges what it spent to `budget`.
    pub fn run(
        &self,
        budget: &Budget,
        id: &str,
        input: &Json,
        fuel: Option<u64>,
    ) -> Result<Ran, String> {
        if let Some(why) = budget.exhausted() {
            return Err(why.into());
        }
        if budget.fuel() == 0 {
            return Err("budget: out of fuel (Lua instructions)".into());
        }
        let program = self.program(id)?;
        let stdin = serde_json::to_vec(input).map_err(|e| e.to_string())?;
        if stdin.len() > MAX_INPUT_BYTES {
            return Err(format!(
                "wasm: {} bytes of input; a run takes at most {MAX_INPUT_BYTES}",
                stdin.len()
            ));
        }
        let affordable = (budget.fuel() as u64).saturating_mul(WASM_FUEL_PER_FUEL);
        let limits = Limits {
            fuel: fuel.unwrap_or(DEFAULT_WASM_FUEL).min(affordable),
            ..Limits::default()
        };
        let out = program.run(&stdin, &limits);
        let spent = match &out {
            Ok(o) => o.fuel_used,
            Err(ix_wasm::Error::OutOfFuel { fuel_used }) => *fuel_used,
            Err(_) => 0,
        };
        let _ = budget.burn(spent.div_ceil(WASM_FUEL_PER_FUEL) as i64);
        let out = out.map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&out.stdout);
        let stderr = crate::clip(String::from_utf8_lossy(&out.stderr).trim(), 2000);
        let value = if out.exit == 0 {
            Ok(serde_json::from_str(text.trim()).unwrap_or_else(|_| Json::String(text.into())))
        } else {
            Err(format!("exit {}: {}", out.exit, crate::clip(&stderr, 600)))
        };
        Ok(Ran {
            value,
            exit: out.exit,
            fuel_used: out.fuel_used,
            stderr,
        })
    }
}

/// What a run reports besides its value.
pub fn stats(ran: &Ran) -> Json {
    json!({ "exit": ran.exit, "fuel_used": ran.fuel_used, "stderr": ran.stderr })
}
