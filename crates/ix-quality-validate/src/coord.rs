//! Validator for the PR coordination room `coord` block.
//!
//! Contract: `docs/contracts/2026-10-17-pr-coordination-room.contract.md`.
//! The schema checks the parsed key/value pairs; [`parse_block`] does the
//! parsing the contract describes (one `key: value` per line, split at the
//! first `: `, every value a string).

use jsonschema::Validator;
use serde_json::{Map, Value};

/// The coord block schema (draft 2020-12), embedded at compile time.
pub const COORD_SCHEMA_JSON: &str =
    include_str!("../../../docs/contracts/2026-10-17-pr-coordination-room.schema.json");

/// Compile the coord block schema.
pub fn build_coord_validator() -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_str(COORD_SCHEMA_JSON).map_err(|e| format!("coord schema parse: {e}"))?;
    jsonschema::draft202012::new(&schema).map_err(|e| format!("coord schema compile: {e}"))
}

/// Parse the first element of a comment: a fenced block whose info string is
/// `coord`. Text after the closing fence is free text and is ignored.
///
/// Returns every pair as a string value. A line without `: `, a repeated key,
/// a missing block, or an unclosed block is an error (the reader ignores such a
/// comment, contract rule 6).
pub fn parse_block(comment: &str) -> Result<Map<String, Value>, String> {
    let mut lines = comment.lines();
    if lines.next().map(str::trim_end) != Some("```coord") {
        return Err("comment does not start with a ```coord block".into());
    }
    let mut pairs = Map::new();
    for line in lines {
        if line.trim_end() == "```" {
            return Ok(pairs);
        }
        let (key, value) = line
            .split_once(": ")
            .ok_or_else(|| format!("line without `: `: {line:?}"))?;
        if pairs
            .insert(key.to_string(), Value::String(value.to_string()))
            .is_some()
        {
            return Err(format!("repeated key {key:?}"));
        }
    }
    Err("coord block is not closed".into())
}

/// Parse a comment and validate its block against the schema.
///
/// `Err` carries every reason: a parse failure, or one message per violated
/// schema rule.
pub fn validate_comment(validator: &Validator, comment: &str) -> Result<(), Vec<String>> {
    let pairs = parse_block(comment).map_err(|e| vec![e])?;
    let value = Value::Object(pairs);
    let errors: Vec<String> = validator
        .iter_errors(&value)
        .map(|e| e.to_string())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
