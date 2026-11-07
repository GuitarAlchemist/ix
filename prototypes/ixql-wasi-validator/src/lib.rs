//! PROTOTYPE: an IXQL capability backed by a WASI validator.
//!
//! The pieces it joins already exist and are tested on their own:
//! - `ix-wasm` runs a WASI **preview 1** command module in wasmi with no files, no network, a
//!   seeded clock and fuel/memory/output caps, and builds Go source with `GOOS=wasip1`;
//! - `ix-ixql` dispatches `name(args)` to a registered [`Capability`] and routes a returned
//!   [`Verdict`] through `→ when …:` gates.
//!
//! What is new here is only the seam between them: [`WasiValidator`] feeds the call's first
//! argument (or the piped value) to the module as JSON on stdin, reads the module's report
//! (`{"valid": bool, "errors": [...]}`) from stdout, and attaches `T 1.0` or `F 1.0`.
//! Anything else — a trap, running out of fuel, a non-zero exit, output that is not a report —
//! is an `Err`, which fails the IXQL run instead of becoming a verdict.

use std::sync::Arc;

use ix_ixql::{CallArgs, Capability, Produced, Verdict};
use ix_types::Hexavalent;
use ix_wasm::{Limits, Program};
use serde_json::{json, Value};

/// A loaded validator module and the limits every call runs under.
pub struct WasiValidator {
    program: Arc<Program>,
    limits: Limits,
}

impl WasiValidator {
    pub fn new(program: Arc<Program>, limits: Limits) -> Self {
        Self { program, limits }
    }
}

impl Capability for WasiValidator {
    fn call(&self, args: CallArgs) -> Result<Produced, String> {
        let input = args
            .positional
            .into_iter()
            .next()
            .or(args.piped)
            .ok_or("go.validate needs a value: an argument or a piped value")?;
        let stdin = serde_json::to_vec(&input).map_err(|e| e.to_string())?;
        let out = self
            .program
            .run(&stdin, &self.limits)
            .map_err(|e| e.to_string())?;
        if out.exit != 0 {
            return Err(format!(
                "validator exited {}: {}",
                out.exit,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let report: Value = serde_json::from_slice(&out.stdout)
            .map_err(|e| format!("validator output is not JSON: {e}"))?;
        let valid = report
            .get("valid")
            .and_then(Value::as_bool)
            .ok_or("validator output has no boolean `valid`")?;
        let truth = if valid {
            Hexavalent::True
        } else {
            Hexavalent::False
        };
        // The run's cost and what it was refused travel with the report, for the record.
        let value = json!({
            "report": report,
            "fuel_used": out.fuel_used,
            "refused": out.refused,
        });
        Ok(Produced::judged(
            value,
            Verdict {
                truth,
                confidence: 1.0,
            },
        ))
    }
}
