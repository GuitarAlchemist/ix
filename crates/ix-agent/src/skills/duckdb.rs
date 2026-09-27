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
                "description": "DuckDB SQL. The result of the last statement is returned, so a script ending with a statement that returns no rows (CREATE, INSERT, SET, ...) is refused. Supplied tables are in scope by name. Files, extensions, ATTACH and network are unavailable (DuckDB safe mode); a line starting with '.' is a CLI dot command and is refused."
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
    governance = "safety",
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
        for ((name, _), out) in tables.iter().zip(outputs) {
            let s = out
                .get(0)
                .and_then(|row| row.get("s"))
                .and_then(Value::as_str)
                .ok_or("DuckDB returned no json_structure")?;
            let rows = params.get("tables").and_then(|t| t.get(name));
            structures.push(match rows {
                Some(rows) if !has_wide_integer(rows) => integers_as_bigint(s)?,
                _ => s.to_string(),
            });
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
        // runs again in HTML mode, where an empty result still prints its
        // header row; it is in memory and cut off from everything else, so
        // running it twice changes nothing but the time taken. A query whose
        // result differs between runs (over `random()`, say) can have rows the
        // second time: then the names stay unknown and `columns` stays empty.
        if let Some(mut names) = header_of_empty_result(&bin, &script)? {
            refuse_repeated(&names)?;
            names.sort();
            columns = names;
        }
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
    // dot command; an indented one is SQL. Safe mode refuses the ones touching
    // files or the shell; the rest (.mode, .headers, ...) would only corrupt
    // the JSON output.
    if sql.lines().any(|l| l.starts_with('.')) {
        return Err(
            "lines starting with '.' are CLI dot commands, which are not allowed in `sql`; indent a line that is SQL (a `.5` literal, say)"
                .to_string(),
        );
    }
    // `-json` prints nothing for a statement without rows, so a script ending
    // with one would return the rows of the query before it as the last
    // statement's result.
    let last = last_statement(sql);
    let keyword = last
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_ascii_uppercase();
    let returning = matches!(keyword.as_str(), "INSERT" | "UPDATE" | "DELETE")
        && last.to_ascii_lowercase().contains("returning");
    if NO_ROWS.contains(&keyword.as_str()) && !returning {
        return Err(format!(
            "the result of the last statement is returned, but the last statement of `sql` ({keyword} ...) returns no rows; end `sql` with a query such as SELECT"
        ));
    }
    Ok(())
}

/// First keywords of the statements that succeed in safe mode without
/// printing rows (INSERT, UPDATE and DELETE print them with RETURNING).
/// Checked against DuckDB 1.5.3; the ones safe mode refuses (COPY, ATTACH,
/// INSTALL, a configuration SET, ...) fail anyway, and any other word, a
/// typo included, is left to DuckDB.
// @ai:assumption each statement kind in NO_ROWS succeeds in safe mode without printing rows under -json [P:test conf:0.5 src:duckdb_query::returns_the_last_statement_or_refuses_the_script] — the test exercises CREATE only; the rest of the list was probed by hand on DuckDB 1.5.3, and it holds only under IX_REQUIRE_DUCKDB=1
const NO_ROWS: &[&str] = &[
    "CREATE",
    "DROP",
    "ALTER",
    "INSERT",
    "UPDATE",
    "DELETE",
    "TRUNCATE",
    "SET",
    "RESET",
    "USE",
    "LOAD",
    "BEGIN",
    "START",
    "COMMIT",
    "END",
    "ROLLBACK",
    "ABORT",
    "PREPARE",
    "DEALLOCATE",
    "COMMENT",
    "VACUUM",
    "ANALYZE",
];

/// The last statement of a script: the text after its last `;` outside a
/// string, quoted name, comment or dollar-quoted string, when more than
/// whitespace and comments follows it. It only reads the script, which runs
/// as sent, so a misreading can refuse a script or let one through but never
/// change what runs.
fn last_statement(sql: &str) -> &str {
    let b = sql.as_bytes();
    let name_byte = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'$';
    let (mut i, mut start, mut after_semicolon) = (0, 0, true);
    while i < b.len() {
        let rest = &b[i..];
        if rest.starts_with(b"--") {
            i += rest.iter().position(|&c| c == b'\n').unwrap_or(rest.len());
            continue;
        }
        if rest.starts_with(b"/*") {
            i += rest
                .windows(2)
                .position(|w| w == b"*/")
                .map_or(rest.len(), |p| p + 2);
            continue;
        }
        let c = b[i];
        if c == b';' || c.is_ascii_whitespace() {
            after_semicolon |= c == b';';
            i += 1;
            continue;
        }
        if after_semicolon {
            (start, after_semicolon) = (i, false);
        }
        i += match c {
            // A quoted string or name ends at its unpaired closing quote;
            // E'...' strings also take backslash escapes.
            b'\'' | b'"' => {
                let escapes = c == b'\''
                    && i > 0
                    && matches!(b[i - 1], b'e' | b'E')
                    && (i < 2 || !name_byte(b[i - 2]));
                let mut j = 1;
                while j < rest.len() {
                    match rest[j] {
                        b'\\' if escapes => j += 2,
                        q if q == c && rest.get(j + 1) == Some(&c) => j += 2,
                        q if q == c => break,
                        _ => j += 1,
                    }
                }
                j + 1
            }
            // $$...$$ or $tag$...$tag$, but not $1 or a name containing '$'.
            b'$' if i == 0 || !name_byte(b[i - 1]) => {
                match rest[1..]
                    .iter()
                    .position(|&c| !(c.is_ascii_alphanumeric() || c == b'_'))
                {
                    Some(n) if rest[1 + n] == b'$' && !rest[1].is_ascii_digit() => {
                        let tag = &rest[..n + 2];
                        let body = &rest[n + 2..];
                        n + 2
                            + body
                                .windows(tag.len())
                                .position(|w| w == tag)
                                .map_or(body.len(), |p| p + tag.len())
                    }
                    _ => 1,
                }
            }
            _ => 1,
        };
    }
    &sql[start..]
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

/// Whether a JSON value holds an integer beyond i64 (a u64 above i64::MAX).
fn has_wide_integer(v: &Value) -> bool {
    match v {
        Value::Number(n) => n.is_u64() && n.as_i64().is_none(),
        Value::Array(items) => items.iter().any(has_wide_integer),
        Value::Object(fields) => fields.values().any(has_wide_integer),
        _ => false,
    }
}

/// `json_structure` infers UBIGINT for non-negative integers and HUGEINT for
/// mixed signs, and DuckDB prints both as JSON strings, so a supplied integer
/// would come back as "1". For a table whose integers all fit an i64, those
/// types become BIGINT, which prints as a number. Only type names change:
/// they are the structure's string values, never its keys (column names).
/// Re-serialising lists keys in name order, the order the rows themselves
/// are sent in (serde_json sorts them), so columns load in name order.
fn integers_as_bigint(structure: &str) -> Result<String, String> {
    fn narrow(v: &mut Value) {
        match v {
            Value::String(t) if t == "UBIGINT" || t == "HUGEINT" => *t = "BIGINT".to_string(),
            Value::Array(items) => items.iter_mut().for_each(narrow),
            Value::Object(fields) => fields.values_mut().for_each(narrow),
            _ => {}
        }
    }
    let mut v: Value = serde_json::from_str(structure)
        .map_err(|e| format!("could not read json_structure {structure:?}: {e}"))?;
    narrow(&mut v);
    Ok(v.to_string())
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
/// The last is refused if reading it would drop a column or change a number.
fn run(bin: &str, script: &str) -> Result<Vec<Value>, String> {
    let out = execute(bin, script)?;
    let mut stream = serde_json::Deserializer::from_slice(&out).into_iter::<Value>();
    let (mut values, mut last) = (Vec::new(), &out[..0]);
    loop {
        let start = stream.byte_offset();
        let Some(value) = stream.next() else { break };
        values.push(value.map_err(|e| format!("could not read DuckDB JSON output: {e}"))?);
        last = &out[start..stream.byte_offset()];
    }
    // A row read into a JSON object keeps only the last of two same-named
    // columns (`SELECT t.*, u.*`), so the returned result's names are read in
    // output order first, and a repeat is refused rather than dropped.
    if let Ok(ResultColumns(names)) = serde_json::from_slice(last) {
        refuse_repeated(&names)?;
    }
    // A JSON number is read as a 64-bit integer or a double. DuckDB prints
    // HUGEINT, UBIGINT and DECIMAL values as strings, but as numbers inside a
    // LIST or STRUCT: one that would come back changed is refused instead.
    if let Some(num) = inexact_number(last) {
        return Err(format!(
            "the result holds the number {num}, which would lose digits as a JSON double; cast it to VARCHAR in SQL"
        ));
    }
    Ok(values)
}

/// Refuses two columns of one name: a row read into a JSON object would keep
/// only the last of them.
fn refuse_repeated(names: &[String]) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    match names.iter().find(|n| !seen.insert(n.as_str())) {
        Some(dup) => Err(format!(
            "the result has more than one column named {dup:?}; rows are JSON objects, so give each column a distinct name with AS"
        )),
        None => Ok(()),
    }
}

/// The first number in a JSON text that reading would change: an integer
/// outside 64 bits, or a decimal other than the shortest decimal of the
/// double nearest to it (the way DuckDB prints a DOUBLE).
fn inexact_number(json: &[u8]) -> Option<&str> {
    let (mut i, mut in_string) = (0, false);
    while i < json.len() {
        let c = json[i];
        if in_string {
            match c {
                b'\\' => i += 1,
                b'"' => in_string = false,
                _ => {}
            }
        } else if c == b'"' {
            in_string = true;
        } else if c == b'-' || c.is_ascii_digit() {
            // Outside a string, only a number has these bytes.
            let start = i;
            while i < json.len()
                && matches!(json[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
            {
                i += 1;
            }
            let num = std::str::from_utf8(&json[start..i]).unwrap_or_default();
            let exact = if num.contains(['.', 'e', 'E']) {
                num.parse::<f64>()
                    .is_ok_and(|f| decimal(&format!("{f:e}")) == decimal(num))
            } else {
                num.parse::<i64>().is_ok() || num.parse::<u64>().is_ok()
            };
            if !exact {
                return Some(num);
            }
            continue;
        }
        i += 1;
    }
    None
}

/// A JSON number's value as (negative, significant digits, power of ten), so
/// that equal values compare equal: `0.10` and `1e-1` are both `(false, "1", -1)`.
fn decimal(num: &str) -> (bool, String, i64) {
    let (negative, num) = match num.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, num),
    };
    let (mantissa, exponent) = num.split_once(['e', 'E']).unwrap_or((num, "0"));
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{int}{frac}");
    let kept = digits.trim_end_matches('0');
    let power = exponent.parse::<i64>().unwrap_or(0) - frac.len() as i64
        + (digits.len() - kept.len()) as i64;
    match kept.trim_start_matches('0') {
        "" => (false, String::new(), 0),
        significant => (negative, significant.to_string(), power),
    }
}

/// Runs a script again in HTML mode and returns the names of its last result
/// if that result is empty; `None` if it has rows this time.
fn header_of_empty_result(bin: &str, script: &str) -> Result<Option<Vec<String>>, String> {
    let out = execute(bin, &format!(".mode html\n.headers on\n{script}"))?;
    Ok(last_html_header(
        &String::from_utf8_lossy(&out).replace("\r\n", "\n"),
    ))
}

/// The cells of the last `<tr>` DuckDB's HTML mode printed when it is a
/// header row (`<th>` cells), unescaped; `None` for a data row (`<td>`) or
/// no row. Cell text is escaped, so a name cannot fake either tag.
fn last_html_header(html: &str) -> Option<Vec<String>> {
    let row = &html[html.rfind("<tr>")?..];
    if row.contains("<td>") {
        return None;
    }
    let unescape = |s: &str| {
        s.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&amp;", "&")
    };
    Some(
        row.split("<th>")
            .skip(1)
            .map(|cell| unescape(cell.split("</th>").next().unwrap_or_default()))
            .collect(),
    )
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
    fn check_sql_refuses_a_last_statement_that_returns_no_rows() {
        // `-json` prints nothing for these, so the rows of the query before
        // them would have been returned as the last statement's result.
        for sql in [
            "SELECT 1 AS stale; CREATE VIEW v AS SELECT 2",
            "SELECT 1;\nINSERT INTO t VALUES (1); -- trailing comment",
            "SELECT 1; set threads = 1;",
        ] {
            let err = check_sql(sql).unwrap_err();
            assert!(err.contains("last statement"), "{sql}: {err}");
        }
        for sql in [
            "CREATE VIEW v AS SELECT 2 AS x; SELECT * FROM v",
            "SELECT 'a;CREATE' AS s",
            "SELECT 'it''s;' AS s; select 2",
            r#"SELECT 1 AS "x;y""#,
            "SELECT 1 -- ; CREATE\n",
            "SELECT 1 /* ; CREATE */",
            "SELECT $$a;CREATE$$ AS s",
            "SELECT $q$a;CREATE$q$ AS s",
            r"SELECT E'it\'s; CREATE' AS s",
            "WITH t AS (SELECT 1) SELECT * FROM t;",
            "(SELECT 1) UNION ALL (SELECT 2)",
            "FROM range(3)",
            "INSERT INTO t VALUES (1) RETURNING a",
            "SELEC 1",
        ] {
            assert!(check_sql(sql).is_ok(), "{sql}: {:?}", check_sql(sql));
        }
    }

    #[test]
    fn check_sql_refuses_dot_commands_only_at_line_start() {
        // The CLI reads a dot command only from a line starting with '.'.
        assert!(check_sql("SELECT 1;\n.mode csv\nSELECT 2")
            .unwrap_err()
            .contains("dot command"));
        assert!(check_sql("SELECT\n  .5 AS ratio").is_ok());
        assert!(check_sql("SELECT 1,\n\t.5 AS ratio").is_ok());
    }

    #[test]
    fn is_not_tagged_deterministic() {
        // SQL can call random(), uuid() or now(): the same arguments need not
        // give the same rows.
        let desc = ix_registry::by_name("duckdb.query").expect("registered");
        assert!(
            !desc.governance_tags.contains(&"deterministic"),
            "{:?}",
            desc.governance_tags
        );
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
    fn last_html_header_reads_only_a_header_row() {
        // What DuckDB 1.5.3 printed (newlines normalised) for `SELECT 7 AS x;`
        // then an empty result with columns id, "a<b&c", "q""x", "multi
        // line".
        let out = "<tr><th>x</th>\n</tr>\n<tr><td>7</td>\n</tr>\n<tr><th>id</th>\n\
                   <th>a&lt;b&amp;c</th>\n<th>q&quot;x</th>\n<th>multi\nline</th>\n</tr>\n";
        assert_eq!(
            last_html_header(out).unwrap(),
            ["id", "a<b&c", "q\"x", "multi\nline"]
        );
        // The same script, when its last result has a row: no names.
        assert_eq!(
            last_html_header("<tr><th>id</th>\n</tr>\n<tr><td>1</td>\n</tr>\n"),
            None
        );
        assert_eq!(last_html_header(""), None);
    }

    #[test]
    fn inexact_number_finds_what_reading_would_change() {
        let exact = br#"[{"h":"18446744073709551617","q":"a\"1e999","n":[1,-2,0.1,4.0,-0.0,1e+20,0.30000000000000004,1.0715660391465826e-75,-9223372036854775808,18446744073709551615]}]"#;
        assert_eq!(inexact_number(exact), None);
        assert_eq!(
            inexact_number(br#"[{"l":[1,18446744073709551616]}]"#),
            Some("18446744073709551616")
        );
        assert_eq!(
            inexact_number(br#"[{"s":{"x":12345678901234567890.12}}]"#),
            Some("12345678901234567890.12")
        );
        // Reads as the double 0.1, whose shortest decimal is 0.1.
        assert_eq!(
            inexact_number(b"[0.1000000000000000055511151231257827]"),
            Some("0.1000000000000000055511151231257827")
        );
        assert_eq!(decimal("0.10"), decimal("1e-1"));
        assert_eq!(decimal("-0.0"), decimal("0"));
    }

    #[test]
    fn literal_doubles_single_quotes_only() {
        assert_eq!(
            literal(r#"[{"b":"it's \\ fine"}]"#),
            r#"'[{"b":"it''s \\ fine"}]'"#
        );
    }
}
