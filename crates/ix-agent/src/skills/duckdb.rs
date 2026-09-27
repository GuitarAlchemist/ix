//! `ix_duckdb_query` — DuckDB SQL over rows the caller supplies.
//!
//! The pipeline editor and agents can put a SQL step between IX tools: earlier
//! steps' outputs arrive as `tables` (arrays of JSON objects), and the query's
//! rows come back as JSON. This is the "expose DuckDB as an MCP tool" seam of
//! `docs/adr/0001-ixql-duckdb-integration-via-mcp-seam.md`, without linking
//! DuckDB into `ix-agent`: it runs the operator-installed DuckDB 1.x CLI
//! (`IX_DUCKDB_BIN`, else `duckdb` on `PATH`).
//!
//! The CLI runs in memory, in safe mode: no file, extension, `ATTACH`,
//! network or environment access, the configuration locked, and the file and
//! shell dot commands refused. Before safe mode is entered, memory is capped at
//! 512 MiB with no spilling to disk and the query gets 2 threads. That, a
//! wall-clock timeout and output caps make it a pure, bounded computation over
//! the request, so it is classified Tier 1.
//!
//! Tables are loaded without touching disk: one CLI call infers each table's
//! `json_structure`, a second loads it with `from_json` + `unnest` (one level
//! deep, so nested objects stay STRUCT columns) and then runs the query.

use ix_skill_macros::ix_skill;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MAX_SQL_BYTES: usize = 64 * 1024;
const MAX_TABLES: usize = 16;
const MAX_TABLE_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_ROWS: usize = 1_000;
const MAX_ROWS_CEILING: usize = 10_000;
const TIMEOUT: Duration = Duration::from_secs(30);

/// Prefixes every script. `-safe` on the command line would lock the
/// configuration before any statement could bound memory, so the limits are
/// set first and `.safe_mode` then enters the same safe mode, locking them.
/// An empty `temp_directory` makes a query over the limit fail instead of
/// spilling to disk.
const PREAMBLE: &str =
    "SET memory_limit = '512MiB';\nSET threads = 2;\nSET temp_directory = '';\n.safe_mode\n";

fn duckdb_query_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "DuckDB SQL. The result of the last statement is returned. Supplied tables are in scope by name. Files, extensions, ATTACH and network are unavailable (DuckDB -safe mode); dot commands are refused."
            },
            "tables": {
                "type": "object",
                "description": "Input tables: name -> non-empty array of JSON objects (one per row). Column types are inferred with json_structure. Names match [A-Za-z_][A-Za-z0-9_]* (at most 64 chars).",
                "additionalProperties": { "type": "array", "items": { "type": "object" } }
            },
            "max_rows": {
                "type": "integer",
                "description": "Rows returned at most (1-10000, default 1000); `truncated` says when more were produced."
            }
        },
        "required": ["sql"]
    })
}

fn duckdb_query_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "rows": { "type": "array", "items": { "type": "object" }, "description": "Rows of the last statement's result" },
            "row_count": { "type": "integer", "description": "Rows the last statement produced, before truncation" },
            "truncated": { "type": "boolean" },
            "columns": { "type": "array", "items": { "type": "string" }, "description": "Column names of the result, sorted by name (rows are JSON objects, which do not keep SQL column order)" },
            "tables": { "type": "object", "description": "Loaded input tables: name -> inferred structure" },
            "duckdb": { "type": "string", "description": "DuckDB CLI version" }
        }
    })
}

/// Run DuckDB SQL over caller-supplied tables and return the rows as JSON.
///
/// Needs the DuckDB 1.x CLI (`IX_DUCKDB_BIN`, else `duckdb` on `PATH`). The
/// query runs in memory in DuckDB's safe mode, bounded to 512 MiB, 2 threads
/// and 30 s per run (a query whose result is empty runs twice, the second
/// time to read its column names).
#[ix_skill(
    domain = "duckdb",
    name = "duckdb.query",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::duckdb::duckdb_query_schema",
    output_schema_fn = "crate::skills::duckdb::duckdb_query_output_schema"
)]
pub fn duckdb_query(params: Value) -> Result<Value, String> {
    let sql = params
        .get("sql")
        .and_then(Value::as_str)
        .ok_or("`sql` must be a string")?;
    check_sql(sql)?;
    let max_rows = match params.get("max_rows") {
        None | Some(Value::Null) => DEFAULT_MAX_ROWS,
        Some(v) => v
            .as_u64()
            .filter(|n| (1..=MAX_ROWS_CEILING as u64).contains(n))
            .ok_or_else(|| {
                format!("max_rows must be an integer from 1 to {MAX_ROWS_CEILING}, got {v}")
            })? as usize,
    };
    let tables = read_tables(params.get("tables"))?;
    let bin = std::env::var("IX_DUCKDB_BIN").unwrap_or_else(|_| "duckdb".to_string());

    // Phase 1: each table's structure, one statement per table, in order.
    let mut structures = Vec::with_capacity(tables.len());
    if !tables.is_empty() {
        let script: String = tables
            .iter()
            .map(|(_, text)| {
                format!(
                    "SELECT json_structure({}::JSON)::VARCHAR AS s;\n",
                    literal(text)
                )
            })
            .collect();
        let outputs = run(&bin, &script)?;
        if outputs.len() != tables.len() {
            return Err(format!(
                "expected {} structures from DuckDB, got {}",
                tables.len(),
                outputs.len()
            ));
        }
        for out in outputs {
            let s = out
                .get(0)
                .and_then(|row| row.get("s"))
                .and_then(Value::as_str)
                .ok_or("DuckDB returned no json_structure")?;
            structures.push(s.to_string());
        }
    }

    // Phase 2: load every table, then the caller's query.
    let mut script = String::new();
    for ((name, text), structure) in tables.iter().zip(&structures) {
        script.push_str(&format!(
            "CREATE TABLE \"{name}\" AS SELECT r.* FROM (SELECT unnest(from_json({}, {})) AS r);\n",
            literal(text),
            literal(structure)
        ));
    }
    script.push_str(sql);
    script.push('\n');
    let outputs = run(&bin, &script)?;
    let all_rows = match outputs.last() {
        Some(Value::Array(rows)) => rows.clone(),
        Some(other) => return Err(format!("unexpected DuckDB output: {other}")),
        None => Vec::new(),
    };
    let row_count = all_rows.len();
    let rows: Vec<Value> = all_rows.into_iter().take(max_rows).collect();
    let mut columns: Vec<String> = rows
        .first()
        .and_then(Value::as_object)
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
    if row_count == 0 && !outputs.is_empty() {
        // `-json` prints an empty result as `[]`, without its names. The script
        // runs again in CSV mode, where an empty result still prints its
        // header; it is in memory and cut off from everything else, so running
        // it twice changes nothing but the time taken.
        columns = csv_header_of_empty_result(&bin, &script)?;
        columns.sort();
    }

    Ok(json!({
        "rows": rows,
        "row_count": row_count,
        "truncated": row_count > max_rows,
        "columns": columns,
        "tables": tables.iter().zip(&structures).map(|((n, _), s)| (n.clone(), Value::String(s.clone()))).collect::<serde_json::Map<_, _>>(),
        "duckdb": version(&bin),
    }))
}

fn check_sql(sql: &str) -> Result<(), String> {
    if sql.trim().is_empty() {
        return Err("`sql` is empty".to_string());
    }
    if sql.len() > MAX_SQL_BYTES {
        return Err(format!(
            "`sql` is {} bytes; the limit is {MAX_SQL_BYTES}",
            sql.len()
        ));
    }
    // The CLI reads its script from stdin, where a line starting with '.' is a
    // dot command. -safe refuses the ones touching files or the shell; the
    // rest (.mode, .headers, ...) would only corrupt the JSON output.
    if sql.lines().any(|l| l.trim_start().starts_with('.')) {
        return Err("dot commands are not allowed in `sql`".to_string());
    }
    Ok(())
}

fn valid_table_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && name.len() <= 64
}

/// Table name -> compact JSON text of its rows, in name order.
fn read_tables(raw: Option<&Value>) -> Result<Vec<(String, String)>, String> {
    let map = match raw {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Object(m)) => m,
        Some(_) => return Err("`tables` must be an object of name -> rows".to_string()),
    };
    if map.len() > MAX_TABLES {
        return Err(format!("at most {MAX_TABLES} tables, got {}", map.len()));
    }
    let mut total = 0;
    let mut out = Vec::with_capacity(map.len());
    for (name, rows) in map {
        if !valid_table_name(name) {
            return Err(format!(
                "table name {name:?} must match [A-Za-z_][A-Za-z0-9_]* (at most 64 chars)"
            ));
        }
        let arr = rows
            .as_array()
            .ok_or_else(|| format!("table {name} must be an array of objects"))?;
        if arr.is_empty() {
            return Err(format!(
                "table {name} is empty; its column types could not be inferred"
            ));
        }
        if !arr.iter().all(Value::is_object) {
            return Err(format!("every row of table {name} must be a JSON object"));
        }
        let text = serde_json::to_string(rows).map_err(|e| e.to_string())?;
        total += text.len();
        if total > MAX_TABLE_JSON_BYTES {
            return Err(format!(
                "tables exceed {MAX_TABLE_JSON_BYTES} bytes of JSON"
            ));
        }
        out.push((name.clone(), text));
    }
    Ok(out)
}

/// A SQL string literal: DuckDB treats backslashes literally, so doubling
/// single quotes is the whole escape. Compact JSON has no raw newlines.
fn literal(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

fn version(bin: &str) -> String {
    Command::new(bin)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Runs `PREAMBLE` then a script in `duckdb -bail -json` (in memory) and
/// returns the JSON value each result-producing statement printed, in order.
fn run(bin: &str, script: &str) -> Result<Vec<Value>, String> {
    let out = execute(bin, script)?;
    let values = serde_json::Deserializer::from_slice(&out)
        .into_iter::<Value>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("could not read DuckDB JSON output: {e}"))?;
    // A row read into a JSON object keeps only the last of two same-named
    // columns (`SELECT t.*, u.*`), so the returned result's names are read in
    // output order first, and a repeat is refused rather than dropped.
    if let Some(Ok(ResultColumns(names))) = serde_json::Deserializer::from_slice(&out)
        .into_iter::<ResultColumns>()
        .last()
    {
        let mut seen = std::collections::HashSet::new();
        if let Some(dup) = names.iter().find(|n| !seen.insert(n.as_str())) {
            return Err(format!(
                "the result has more than one column named {dup:?}; rows are JSON objects, so give each column a distinct name with AS"
            ));
        }
    }
    Ok(values)
}

/// The column names of a script whose last result is empty, from its CSV
/// header (the last record printed).
fn csv_header_of_empty_result(bin: &str, script: &str) -> Result<Vec<String>, String> {
    let out = execute(bin, &format!(".mode csv\n.headers on\n{script}"))?;
    Ok(last_csv_record(&String::from_utf8_lossy(&out)))
}

/// The last record of DuckDB's CSV output: fields split on `,`, quoted with
/// `"` (doubled inside quotes), records ended by a newline outside quotes.
fn last_csv_record(text: &str) -> Vec<String> {
    let (mut last, mut record, mut field) = (Vec::new(), Vec::new(), String::new());
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => record.push(std::mem::take(&mut field)),
            '\r' if !quoted => {}
            '\n' if !quoted => {
                record.push(std::mem::take(&mut field));
                last = std::mem::take(&mut record);
            }
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        last = record;
    }
    last
}

/// Runs `PREAMBLE` then a script in `duckdb -bail -json` (in memory) and
/// returns what it printed.
// @ai:invariant a query run here cannot read, write or attach a file, install an extension or re-enable external access [P:test conf:0.6 src:duckdb_query::cannot_read_write_or_attach_files] — the test skips where the CLI is absent, which includes default CI, so the binding is live only under IX_REQUIRE_DUCKDB=1
// @ai:invariant a query run here holds at most 512 MiB of DuckDB memory on 2 threads, never spills to disk, and cannot raise either limit [P:test conf:0.6 src:duckdb_query::memory_and_threads_are_bounded_and_locked] — same binding as above: live only under IX_REQUIRE_DUCKDB=1
fn execute(bin: &str, script: &str) -> Result<Vec<u8>, String> {
    let mut child = Command::new(bin)
        .args(["-bail", "-json", "-no-init"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                format!("DuckDB CLI not found ({bin}); install DuckDB 1.x or set IX_DUCKDB_BIN")
            }
            _ => format!("could not start DuckDB ({bin}): {e}"),
        })?;

    let mut stdin = child.stdin.take().ok_or("no stdin for DuckDB")?;
    let script = format!("{PREAMBLE}{script}");
    let writer = std::thread::spawn(move || {
        // A write error means DuckDB already stopped (-bail); its stderr says why.
        let _ = stdin.write_all(script.as_bytes());
    });
    let read_capped = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe
                .by_ref()
                .take(MAX_OUTPUT_BYTES as u64 + 1)
                .read_to_end(&mut buf);
            // Drain the rest so DuckDB is never blocked on a full pipe.
            let _ = std::io::copy(&mut pipe, &mut std::io::sink());
            buf
        })
    };
    let stdout = read_capped(Box::new(child.stdout.take().ok_or("no stdout for DuckDB")?));
    let stderr = read_capped(Box::new(child.stderr.take().ok_or("no stderr for DuckDB")?));

    let deadline = Instant::now() + TIMEOUT;
    let status = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "DuckDB query exceeded {} s and was stopped",
                    TIMEOUT.as_secs()
                ));
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    };
    let _ = writer.join();
    let out = stdout.join().map_err(|_| "DuckDB stdout reader panicked")?;
    let err = stderr.join().map_err(|_| "DuckDB stderr reader panicked")?;

    if !status.success() {
        let msg = String::from_utf8_lossy(&err);
        return Err(format!(
            "DuckDB: {}",
            msg.trim().chars().take(2_000).collect::<String>()
        ));
    }
    if out.len() > MAX_OUTPUT_BYTES {
        return Err(format!(
            "DuckDB output exceeds {MAX_OUTPUT_BYTES} bytes; select fewer rows or columns"
        ));
    }
    Ok(out)
}

/// The column names of one DuckDB `-json` result (an array of row objects),
/// in output order, read from its first row.
struct ResultColumns(Vec<String>);

impl<'de> serde::Deserialize<'de> for ResultColumns {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};

        struct Row(Vec<String>);
        impl<'de> serde::Deserialize<'de> for Row {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                struct Names;
                impl<'de> Visitor<'de> for Names {
                    type Value = Row;
                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        f.write_str("a result row")
                    }
                    fn visit_map<A: MapAccess<'de>>(self, mut row: A) -> Result<Row, A::Error> {
                        let mut names = Vec::new();
                        while let Some((name, IgnoredAny)) =
                            row.next_entry::<String, IgnoredAny>()?
                        {
                            names.push(name);
                        }
                        Ok(Row(names))
                    }
                }
                d.deserialize_map(Names)
            }
        }

        struct FirstRow;
        impl<'de> Visitor<'de> for FirstRow {
            type Value = ResultColumns;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a DuckDB result")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut rows: A) -> Result<ResultColumns, A::Error> {
                let names = rows.next_element::<Row>()?.map(|r| r.0).unwrap_or_default();
                while rows.next_element::<IgnoredAny>()?.is_some() {}
                Ok(ResultColumns(names))
            }
        }
        d.deserialize_seq(FirstRow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_bad_input_before_starting_duckdb() {
        assert!(duckdb_query(json!({ "sql": "  " }))
            .unwrap_err()
            .contains("empty"));
        assert!(duckdb_query(json!({ "sql": "SELECT 1;\n.mode csv" }))
            .unwrap_err()
            .contains("dot commands"));
        assert!(duckdb_query(json!({ "sql": "SELECT 1", "max_rows": 0 }))
            .unwrap_err()
            .contains("max_rows"));
        let bad_name =
            duckdb_query(json!({ "sql": "SELECT 1", "tables": { "a b": [{ "x": 1 }] } }));
        assert!(bad_name.unwrap_err().contains("must match"));
        let empty = duckdb_query(json!({ "sql": "SELECT 1", "tables": { "t": [] } }));
        assert!(empty.unwrap_err().contains("empty"));
        let scalar_rows = duckdb_query(json!({ "sql": "SELECT 1", "tables": { "t": [1, 2] } }));
        assert!(scalar_rows.unwrap_err().contains("JSON object"));
    }

    #[test]
    fn result_columns_keep_repeated_names_in_order() {
        let cols =
            serde_json::from_str::<ResultColumns>(r#"[{"a":1,"b":2,"a":3},{"a":4,"b":5,"a":6}]"#)
                .unwrap();
        assert_eq!(cols.0, ["a", "b", "a"]);
        assert!(serde_json::from_str::<ResultColumns>("[]")
            .unwrap()
            .0
            .is_empty());
    }

    #[test]
    fn last_csv_record_reads_duckdb_quoting() {
        // Bytes DuckDB 1.5.3 printed for `SELECT 7 AS x;` then an empty
        // result with columns id, "a,b", "q""x" and "multi<newline>line".
        let out = "x\r\n7\r\nid,\"a,b\",\"q\"\"x\",\"multi\nline\"\r\n";
        assert_eq!(last_csv_record(out), ["id", "a,b", "q\"x", "multi\nline"]);
        assert!(last_csv_record("").is_empty());
    }

    #[test]
    fn literal_doubles_single_quotes_only() {
        assert_eq!(
            literal(r#"[{"b":"it's \\ fine"}]"#),
            r#"'[{"b":"it''s \\ fine"}]'"#
        );
    }
}
