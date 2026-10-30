//! SQL in a stream: DuckDB over each window of values, and a sink that keeps a stream in
//! Parquet.
//!
//! SQL works on tables, a stream brings values one at a time, so the bridge is the window:
//! [`Window`] holds values back and, when the window closes, runs a query over them as the
//! table `batch` and sends every row it answers. A window closes when it holds `rows` values,
//! when a value arrives `every` after the window opened, and at the end of the stream; it does
//! not close on a timer while nothing arrives.
//!
//! Each window costs one `duckdb` process (tens of milliseconds), so windows should hold many
//! values: SQL per value is the wrong grain. The values pass through a JSON-lines file in the
//! node's folder, which DuckDB reads with `read_json`.
//!
//! DuckDB writes a `HUGEINT` (what `sum` of integers answers) as a JSON string: cast it
//! (`sum(i)::BIGINT`) when the next node wants a number.

use crate::source::duckdb_exe;
use crate::{Batch, Out, Signal};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// `path` as a DuckDB string literal.
fn literal(path: &Path) -> String {
    format!(
        "'{}'",
        path.display()
            .to_string()
            .replace('\\', "/")
            .replace('\'', "''")
    )
}

fn write_lines(path: &Path, rows: &[Value]) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut w = BufWriter::new(f);
    for r in rows {
        writeln!(w, "{r}").map_err(|e| format!("{}: {e}", path.display()))?;
    }
    w.flush().map_err(|e| format!("{}: {e}", path.display()))
}

/// Runs `sql` in an in-memory DuckDB and answers its rows.
pub fn query(sql: &str) -> Result<Vec<Value>, String> {
    let out = Command::new(duckdb_exe())
        .args(["-jsonlines", "-c", sql])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("duckdb: cannot start: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "duckdb: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    BufReader::new(&out.stdout[..])
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).map_err(|e| format!("duckdb: a row is not JSON: {e}")))
        .collect()
}

/// Runs `sql` over `rows`, which it reads as the table `batch`, through the file `scratch`.
pub fn query_batch(rows: &[Value], sql: &str, scratch: &Path) -> Result<Vec<Value>, String> {
    write_lines(scratch, rows)?;
    query(&format!(
        "CREATE TEMP TABLE batch AS SELECT * FROM read_json({}, format = 'newline_delimited'); {sql}",
        literal(scratch)
    ))
}

/// [`query_batch`] for a query that must reach no file and no network: once `batch` is loaded,
/// DuckDB's external access and extension autoloading are turned off, and a statement cannot
/// turn them back on.
pub fn query_sealed(rows: &[Value], sql: &str, scratch: &Path) -> Result<Vec<Value>, String> {
    query_batch(
        rows,
        &format!(
            "SET enable_external_access = false; SET autoinstall_known_extensions = false; \
             SET autoload_known_extensions = false; SET lock_configuration = true; {sql}"
        ),
        scratch,
    )
}

/// A window of values and the query run over it. Give it to [`crate::Graph::batch`].
pub struct Window {
    name: String,
    sql: String,
    rows: usize,
    every: Option<Duration>,
    scratch: PathBuf,
    held: Vec<Value>,
    opened: Option<Instant>,
    windows: u64,
}

impl Window {
    /// A window of at most `rows` values over which `sql` runs, reading them as `batch`. Its
    /// file is kept in `dir`.
    pub fn new(name: &str, sql: &str, rows: usize, dir: &Path) -> Self {
        Self {
            name: name.to_string(),
            sql: sql.to_string(),
            rows: rows.max(1),
            every: None,
            scratch: dir.join(format!("{name}.window.jsonl")),
            held: Vec::new(),
            opened: None,
            windows: 0,
        }
    }

    /// Also closes the window when a value arrives `every` after it opened.
    pub fn every(mut self, every: Duration) -> Self {
        self.every = Some(every);
        self
    }

    fn close(&mut self, out: &Out) {
        if self.held.is_empty() {
            return;
        }
        let rows = std::mem::take(&mut self.held);
        self.opened = None;
        self.windows += 1;
        match query_batch(&rows, &self.sql, &self.scratch) {
            Ok(answer) => {
                for mut r in answer {
                    if let Value::Object(m) = &mut r {
                        m.insert("window".into(), json!(self.windows));
                    }
                    if out.emit(0, r).is_err() {
                        return;
                    }
                }
            }
            Err(e) => out.signal(&Signal::pain(&self.name, 1.0, &e)),
        }
    }
}

impl Batch for Window {
    fn take(&mut self, v: Value, out: &Out) {
        let now = Instant::now();
        let opened = *self.opened.get_or_insert(now);
        self.held.push(v);
        let late = self.every.is_some_and(|e| now.duration_since(opened) >= e);
        if self.held.len() >= self.rows || late {
            self.close(out);
        }
    }

    fn end(&mut self, out: &Out) {
        self.close(out);
    }
}

/// A sink that keeps every value it takes and, when the stream ends, writes them to a
/// Parquet file with DuckDB. Values go to a JSON-lines file beside it as they come, so a
/// stream cut short still leaves that file.
pub struct ParquetSink {
    name: String,
    path: PathBuf,
    lines: PathBuf,
    w: Option<BufWriter<std::fs::File>>,
    rows: u64,
}

impl ParquetSink {
    pub fn new(name: &str, path: &Path) -> Result<Self, String> {
        let lines = path.with_extension("jsonl");
        let f = std::fs::File::create(&lines).map_err(|e| format!("{}: {e}", lines.display()))?;
        Ok(Self {
            name: name.to_string(),
            path: path.to_path_buf(),
            lines,
            w: Some(BufWriter::new(f)),
            rows: 0,
        })
    }
}

impl Batch for ParquetSink {
    fn take(&mut self, v: Value, out: &Out) {
        if let Some(w) = &mut self.w {
            if let Err(e) = writeln!(w, "{v}") {
                out.signal(&Signal::pain(&self.name, 1.0, &format!("write: {e}")));
                return;
            }
            self.rows += 1;
        }
    }

    fn end(&mut self, out: &Out) {
        if let Some(mut w) = self.w.take() {
            let _ = w.flush();
        }
        if self.rows == 0 {
            return;
        }
        let sql = format!(
            "COPY (SELECT * FROM read_json({}, format = 'newline_delimited')) TO {} (FORMAT parquet)",
            literal(&self.lines),
            literal(&self.path)
        );
        match query(&sql) {
            Ok(_) => out.bus().publish(
                "sink",
                json!({ "node": self.name, "rows": self.rows, "path": self.path.display().to_string() }),
            ),
            Err(e) => out.signal(&Signal::pain(&self.name, 1.0, &e)),
        }
    }
}
