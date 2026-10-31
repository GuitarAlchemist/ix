//! `ix demo` — run and explore curated demo scenarios.

use crate::output::{self, Format};
use ix_agent::demo;
use serde_json::json;

/// List all available demo scenarios.
pub fn list(fmt: Format) -> Result<(), String> {
    let result = demo::ix_demo(json!({"action": "list"}))?;
    output::emit(&result, fmt).map_err(|e| e.to_string())
}

/// Show scenario details and steps without executing.
pub fn describe(scenario: &str, fmt: Format) -> Result<(), String> {
    let result = demo::ix_demo(json!({"action": "describe", "scenario": scenario}))?;
    output::emit(&result, fmt).map_err(|e| e.to_string())
}

/// Execute a demo scenario end-to-end.
pub fn run(scenario: &str, seed: u64, verbosity: u8, fmt: Format) -> Result<(), String> {
    let result = demo::ix_demo(json!({
        "action": "run",
        "scenario": scenario,
        "seed": seed,
        "verbosity": verbosity,
    }))?;
    output::emit(&result, fmt).map_err(|e| e.to_string())
}
