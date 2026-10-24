//! Sources: a DuckDB query read while it runs, JSON-lines files, and iterators.
//!
//! [`duckdb`] runs the `duckdb` CLI with `-jsonlines` and sends each row as it is read. When
//! the graph is full the source stops reading, the pipe fills, and DuckDB waits on its write:
//! the back-pressure reaches the SQL engine. When the graph stops, the process is killed.

use crate::Out;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// What a source sent, and whether it stopped because the graph did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sent {
    pub rows: u64,
    pub cancelled: bool,
}

/// `IX_DUCKDB`, else `duckdb` on the path.
pub fn duckdb_exe() -> PathBuf {
    std::env::var_os("IX_DUCKDB")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("duckdb"))
}

/// Runs `sql` in an in-memory DuckDB and sends each row, as a JSON object, on port 0.
pub fn duckdb(sql: &str, out: &Out) -> Result<Sent, String> {
    let exe = duckdb_exe();
    let mut child = Command::new(&exe)
        .args(["-jsonlines", "-c", sql])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("duckdb: cannot start {}: {e}", exe.display()))?;
    let mut err = child.stderr.take().expect("piped stderr");
    let err_t = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        s
    });
    let stdout = child.stdout.take().expect("piped stdout");
    let mut rows = 0;
    for line in BufReader::new(stdout).lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("duckdb: {e}"));
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("duckdb: row {} is not JSON: {e}", rows + 1));
            }
        };
        if out.emit(0, row).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(Sent {
                rows,
                cancelled: true,
            });
        }
        rows += 1;
    }
    let status = child.wait().map_err(|e| format!("duckdb: {e}"))?;
    let stderr = err_t.join().unwrap_or_default();
    if !status.success() {
        return Err(format!("duckdb: {}", stderr.trim()));
    }
    Ok(Sent {
        rows,
        cancelled: false,
    })
}

/// Sends each line of a JSON-lines file on port 0.
pub fn jsonl(path: &Path, out: &Out) -> Result<Sent, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = 0;
    for line in BufReader::new(f).lines() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .map_err(|e| format!("{}: line {}: {e}", path.display(), rows + 1))?;
        if out.emit(0, row).is_err() {
            return Ok(Sent {
                rows,
                cancelled: true,
            });
        }
        rows += 1;
    }
    Ok(Sent {
        rows,
        cancelled: false,
    })
}

/// Sends each value on port 0, until the graph stops.
pub fn iter(values: impl IntoIterator<Item = Value>, out: &Out) -> Sent {
    let mut rows = 0;
    for v in values {
        if out.emit(0, v).is_err() {
            return Sent {
                rows,
                cancelled: true,
            };
        }
        rows += 1;
    }
    Sent {
        rows,
        cancelled: false,
    }
}
