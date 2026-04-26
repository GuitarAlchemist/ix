//! `ix run <skill>` — invoke a registered skill with JSON input.
//!
//! Input sources (priority order):
//!   1. `--input-file <path>` — read JSON from file
//!   2. `--input <literal>` — inline JSON string
//!   3. stdin if not a TTY

use crate::output::{self, Format};
use ix_types::Value as IxValue;
use serde_json::Value;
use std::io::{self, IsTerminal, Read};

pub fn run(
    skill_name: &str,
    input_file: Option<&str>,
    input_literal: Option<&str>,
    format: Format,
) -> Result<(), String> {
    let descriptor = ix_registry::by_name(skill_name)
        .ok_or_else(|| format!("skill not found: {skill_name}\n\nTry `ix list skills`."))?;

    // Resolve input
    let params_json: Value = match (input_file, input_literal) {
        (Some(path), _) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
            serde_json::from_str(&text).map_err(|e| format!("parsing JSON from {path}: {e}"))?
        }
        (None, Some(lit)) => {
            serde_json::from_str(lit).map_err(|e| format!("parsing --input JSON: {e}"))?
        }
        (None, None) => {
            if io::stdin().is_terminal() {
                // No input provided and no piped stdin — use empty object.
                Value::Object(Default::default())
            } else {
                let mut buf = String::new();
                io::stdin()
                    .read_to_string(&mut buf)
                    .map_err(|e| format!("reading stdin: {e}"))?;
                if buf.trim().is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&buf).map_err(|e| format!("parsing stdin JSON: {e}"))?
                }
            }
        }
    };

    // Registry skills built via batch1/batch2 take a single Value::Json arg.
    let args = [IxValue::Json(params_json)];
    let result = (descriptor.fn_ptr)(&args).map_err(|e| e.to_string())?;

    // Unwrap to plain JSON for display.
    let out_json = match result {
        IxValue::Json(j) => j,
        other => serde_json::to_value(other).unwrap_or(Value::Null),
    };

    output::emit(&out_json, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}
