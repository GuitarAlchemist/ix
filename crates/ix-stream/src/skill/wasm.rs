//! A `wasm` node of `stream.run`: each value it takes runs a WASI program in ix-wasm, with the
//! value as JSON on stdin, and the one JSON value the program writes on stdout goes on every link.
//!
//! The program is named as `wasm.run` names it, `{ stage | go | zig | wat }`, and is loaded (and
//! built, for source) once, before the run. `fuel` caps each value's run. A run that spends all
//! its fuel, calls a file or socket function the sandbox refuses, exits non-zero, traps, or
//! writes anything but one JSON value fails the node, and the failure stops the stream.

use ix_wasm::{Error, Limits, Program};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The most of a failed run's stderr a failure quotes.
const MAX_STDERR_CHARS: usize = 400;

/// A loaded program and what its runs spent.
pub(super) struct Wasm {
    stage: String,
    program: Arc<Program>,
    limits: Limits,
    /// Values the program answered.
    out: AtomicU64,
    fuel: AtomicU64,
}

impl Wasm {
    /// The node `spec` describes: `{ program: { stage | go | zig | wat }, fuel }`.
    pub(super) fn new(name: &str, spec: &Value) -> Result<Self, String> {
        let program = spec.get("program").ok_or_else(|| {
            format!("{name}: a wasm node names its `program`, {{ stage | go | zig | wat }}")
        })?;
        let (stage, program) =
            ix_wasm::skill::program_of(program).map_err(|e| format!("{name}: {e}"))?;
        let mut limits = Limits::default();
        match spec.get("fuel") {
            None | Some(Value::Null) => {}
            Some(v) => {
                limits.fuel = v.as_u64().filter(|&f| f > 0).ok_or_else(|| {
                    format!("{name}: `fuel` must be a whole number above 0, got {v}")
                })?
            }
        }
        Ok(Self {
            stage,
            program,
            limits,
            out: AtomicU64::new(0),
            fuel: AtomicU64::new(0),
        })
    }

    /// Runs the program on `v`: the value it answers, or the kind and the reason of its failure.
    pub(super) fn run(&self, v: &Value) -> Result<Value, (&'static str, String)> {
        let stdin = serde_json::to_vec(v).map_err(|e| ("input", e.to_string()))?;
        let ran = self.program.run(&stdin, &self.limits);
        let spent = match &ran {
            Ok(out) => out.fuel_used,
            Err(Error::OutOfFuel { fuel_used }) => *fuel_used,
            Err(_) => 0,
        };
        self.fuel.fetch_add(spent, Ordering::Relaxed);
        let out = ran.map_err(|e| match e {
            Error::OutOfFuel { fuel_used } => (
                "fuel",
                format!(
                    "{} ran out of fuel: spent {fuel_used} of the {} a value may",
                    self.stage, self.limits.fuel
                ),
            ),
            e => ("trap", format!("{}: {e}", self.stage)),
        })?;
        if !out.refused.is_empty() {
            return Err((
                "refused",
                format!(
                    "{} called {}, which the sandbox refuses: no files, no network",
                    self.stage,
                    out.refused.join(", ")
                ),
            ));
        }
        if out.exit != 0 {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stderr: String = stderr.trim().chars().take(MAX_STDERR_CHARS).collect();
            return Err((
                "exit",
                format!("{} exited {}: {stderr}", self.stage, out.exit),
            ));
        }
        let answer = serde_json::from_slice(&out.stdout).map_err(|e| {
            (
                "output",
                format!("{} did not write one JSON value: {e}", self.stage),
            )
        })?;
        self.out.fetch_add(1, Ordering::Relaxed);
        Ok(answer)
    }

    /// Its receipt: the program, the values it answered, and the fuel all its runs spent.
    pub(super) fn json(&self) -> Value {
        json!({
            "stage": self.stage,
            "fuel_cap": self.limits.fuel,
            "out": self.out.load(Ordering::Relaxed),
            "fuel": self.fuel.load(Ordering::Relaxed),
        })
    }
}
