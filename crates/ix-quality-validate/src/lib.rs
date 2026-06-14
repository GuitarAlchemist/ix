//! `ix-quality-validate` — JSON-schema validator for the canonical dashboard
//! envelope written to `state/quality/<domain>/*.json`.
//!
//! The schema lives at `docs/contracts/quality-snapshot.schema.json` and is
//! embedded into the binary at compile time via `include_str!`, so the
//! validator is a single self-contained artifact suitable for CI use.
//!
//! Public surface:
//! - [`SCHEMA_JSON`]       — the embedded schema text.
//! - [`build_validator`]   — compiles the schema into a reusable validator.
//! - [`validate_value`]    — validates one in-memory JSON value.
//! - [`validate_path`]     — reads a single `*.json` file from disk and validates it.
//! - [`walk_and_validate`] — recursively walks a directory, validating every `*.json`.

use std::path::{Path, PathBuf};

use jsonschema::Validator;
use serde_json::Value;

/// The canonical dashboard-envelope schema (draft 2020-12), embedded at compile time.
pub const SCHEMA_JSON: &str = include_str!("../../../docs/contracts/quality-snapshot.schema.json");

/// Outcome of validating a single snapshot file.
#[derive(Debug)]
pub struct FileReport {
    pub path: PathBuf,
    pub status: FileStatus,
}

#[derive(Debug)]
pub enum FileStatus {
    Pass,
    Fail(Vec<String>),
    /// Could not read or parse the file at all (I/O or JSON-syntax error).
    Unreadable(String),
}

impl FileStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, FileStatus::Pass)
    }
}

/// Build a reusable validator from the embedded schema.
pub fn build_validator() -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_str(SCHEMA_JSON).map_err(|e| format!("embedded schema parse: {e}"))?;
    jsonschema::draft202012::new(&schema).map_err(|e| format!("schema compile: {e}"))
}

/// Validate a single in-memory JSON value against the dashboard envelope schema.
///
/// Returns `Ok(())` on success, or a list of human-readable error messages.
pub fn validate_value(validator: &Validator, value: &Value) -> Result<(), Vec<String>> {
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|e| {
            let loc = e.instance_path.to_string();
            if loc.is_empty() {
                format!("{e}")
            } else {
                format!("{e} (at `{loc}`)")
            }
        })
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate a single `*.json` file on disk.
pub fn validate_path(validator: &Validator, path: &Path) -> FileReport {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            return FileReport {
                path: path.to_path_buf(),
                status: FileStatus::Unreadable(format!("read: {e}")),
            }
        }
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return FileReport {
                path: path.to_path_buf(),
                status: FileStatus::Unreadable(format!("json parse: {e}")),
            }
        }
    };
    match validate_value(validator, &value) {
        Ok(()) => FileReport {
            path: path.to_path_buf(),
            status: FileStatus::Pass,
        },
        Err(errors) => FileReport {
            path: path.to_path_buf(),
            status: FileStatus::Fail(errors),
        },
    }
}

/// Recursively walk `root` and validate every `*.json` file found.
///
/// Filenames beginning with `_` (e.g. `_schema.json`) are skipped — by
/// convention those are sidecar/meta files governed by their own schemas.
pub fn walk_and_validate(root: &Path, validator: &Validator) -> Vec<FileReport> {
    let mut out = Vec::new();
    walk(root, validator, &mut out);
    out
}

fn walk(dir: &Path, validator: &Validator, out: &mut Vec<FileReport>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            out.push(FileReport {
                path: dir.to_path_buf(),
                status: FileStatus::Unreadable(format!("read_dir: {e}")),
            });
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, validator, out);
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            if name.starts_with('_') {
                continue;
            }
        }
        out.push(validate_path(validator, &path));
    }
}
