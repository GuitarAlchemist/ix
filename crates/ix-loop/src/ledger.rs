//! The ledger: one JSON line per call, child run, proposal and judgement, appended as they happen.

use serde_json::{json, Value as Json};
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct Ledger {
    path: Option<PathBuf>,
}

impl Ledger {
    /// A ledger that records nothing.
    pub fn none() -> Self {
        Self { path: None }
    }

    /// A ledger appended to the JSONL file at `path`.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
        }
    }

    /// Appends `{ ts_ms, entry }`. A ledger that cannot be written never stops a loop.
    pub fn record(&self, entry: Json) {
        let Some(path) = &self.path else { return };
        let line = json!({ "ts_ms": crate::store::now_ms(), "entry": entry }).to_string();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{line}");
        }
    }
}
