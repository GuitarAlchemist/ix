//! `ix_repl` — an interactive SQL prompt over the in-process DuckDB "analyst
//! bench" with every IX UDF (`ix_cosine`, `ix_euclidean`, …) registered and
//! callable from SQL.
//!
//! This is the tier-1 way to "invoke IX inside DuckDB": the standalone
//! `duckdb.exe` CLI runs in its own process and cannot see UDFs registered by a
//! Rust embedder, so to get `ix_cosine(...)` at a real prompt you run *this*
//! binary instead — it embeds DuckDB, registers the UDFs, and loops on stdin.
//!
//! Usage:
//!   cargo run -p ix-duck --example ix_repl --features duck
//!   cargo run -p ix-duck --example ix_repl --features duck -- <path/to/some.duckdb>
//!
//! With a path, that database is ATTACHed READ-ONLY as schema `q` and made
//! current (`USE q`), so its tables are visible *and* the IX UDFs still resolve
//! (scalar functions are connection-global). Without a path it's a scratch
//! in-memory bench — query on-disk data via `read_json_auto('…')` /
//! `read_parquet('…')`.
//!
//! Dot-commands: `.tables`, `.udf`, `.help`, `.quit` / `.exit`. Everything else
//! is SQL; statements are accumulated until a line ends with `;`.

#[cfg(not(feature = "duck"))]
fn main() {
    eprintln!("ix_repl requires the `duck` feature: cargo run -p ix-duck --example ix_repl --features duck");
    std::process::exit(2);
}

#[cfg(feature = "duck")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self, Write};

    let conn = ix_duck::open_bench()?;

    // Optional on-disk DB attached read-only so its tables join the bench.
    let attached = std::env::args().nth(1);
    if let Some(path) = attached.as_deref() {
        let safe = path.replace('\\', "/").replace('\'', "''");
        conn.execute_batch(&format!("ATTACH '{safe}' AS q (READ_ONLY); USE q;"))?;
    }

    println!("ix-duck bench — DuckDB with IX UDFs registered.");
    match attached.as_deref() {
        Some(p) => println!("attached: {p}  (READ-ONLY, schema `q`, current)"),
        None => println!("in-memory scratch bench (read files via read_json_auto('…'))"),
    }
    println!("IX UDFs: ix_cosine(a, b), ix_euclidean(a, b)  — args are DOUBLE[]");
    println!("dot-commands: .tables  .udf  .help  .quit\n");

    let stdin = io::stdin();
    let mut buf = String::new();
    loop {
        // Prompt: `ix> ` fresh, `..> ` for a continued statement.
        print!("{}", if buf.is_empty() { "ix> " } else { "..> " });
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            println!();
            break; // EOF (Ctrl-Z / Ctrl-D)
        }
        let trimmed = line.trim();

        // Dot-commands only at the start of a fresh statement.
        if buf.is_empty() && trimmed.starts_with('.') {
            match trimmed {
                ".quit" | ".exit" => break,
                ".help" => {
                    println!(".tables  — list tables/views\n.udf     — list IX scalar functions\n.quit    — exit\nanything else is SQL (terminate with `;`)");
                }
                ".tables" => run_and_print(
                    &conn,
                    "SELECT table_schema, table_name FROM information_schema.tables ORDER BY 1, 2",
                ),
                ".udf" => run_and_print(
                    &conn,
                    "SELECT function_name, parameter_types, return_type \
                     FROM duckdb_functions() WHERE function_name LIKE 'ix\\_%' ESCAPE '\\' ORDER BY 1",
                ),
                other => println!("unknown command: {other}  (try .help)"),
            }
            continue;
        }

        buf.push_str(&line);
        // Execute once the accumulated statement is terminated.
        if buf.trim_end().ends_with(';') {
            let sql = std::mem::take(&mut buf);
            run_and_print(&conn, sql.trim());
        }
    }
    Ok(())
}

/// Execute `sql` and print the result set as a simple aligned table, or the
/// rowcount for statements that return none. Errors print to stderr and never
/// abort the loop.
#[cfg(feature = "duck")]
fn run_and_print(conn: &duckdb::Connection, sql: &str) {
    if sql.is_empty() {
        return;
    }
    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return;
        }
    };

    // `Statement::column_names()` panics before the statement has stepped, so
    // names are taken from the first executed row (`Row::as_ref()` hands back
    // the now-stepped statement). Empty result set → no header, just "(0 rows)".
    let mut rows = match stmt.query([]) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return;
        }
    };

    let mut cols: Vec<String> = Vec::new();
    let mut table: Vec<Vec<String>> = Vec::new();
    loop {
        match rows.next() {
            Ok(Some(row)) => {
                if cols.is_empty() {
                    cols = row.as_ref().column_names();
                }
                let mut rec = Vec::with_capacity(cols.len());
                for i in 0..cols.len() {
                    rec.push(match row.get_ref(i) {
                        Ok(v) => fmt_value(v),
                        Err(_) => String::new(),
                    });
                }
                table.push(rec);
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("error: {e}");
                return;
            }
        }
    }

    if cols.is_empty() {
        println!("(0 rows)");
        return;
    }

    // Column widths from header + cells.
    let mut widths: Vec<usize> = cols.iter().map(|c| c.len()).collect();
    for rec in &table {
        for (i, cell) in rec.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    let sep = |w: &[usize]| {
        w.iter()
            .map(|n| "-".repeat(n + 2))
            .collect::<Vec<_>>()
            .join("+")
    };
    let fmt_row = |r: &[String], w: &[usize]| {
        r.iter()
            .enumerate()
            .map(|(i, c)| format!(" {:width$} ", c, width = w[i]))
            .collect::<Vec<_>>()
            .join("|")
    };

    println!("{}", fmt_row(&cols, &widths));
    println!("{}", sep(&widths));
    for rec in &table {
        println!("{}", fmt_row(rec, &widths));
    }
    println!(
        "({} row{})",
        table.len(),
        if table.len() == 1 { "" } else { "s" }
    );
}

/// Render a `ValueRef` for display. Common scalar types get a clean form; the
/// long tail (lists, structs, timestamps, …) falls back to Debug so nothing
/// ever panics on an unexpected type.
#[cfg(feature = "duck")]
fn fmt_value(v: duckdb::types::ValueRef<'_>) -> String {
    use duckdb::types::ValueRef::*;
    match v {
        Null => "NULL".to_string(),
        Boolean(b) => b.to_string(),
        TinyInt(n) => n.to_string(),
        SmallInt(n) => n.to_string(),
        Int(n) => n.to_string(),
        BigInt(n) => n.to_string(),
        HugeInt(n) => n.to_string(),
        UTinyInt(n) => n.to_string(),
        USmallInt(n) => n.to_string(),
        UInt(n) => n.to_string(),
        UBigInt(n) => n.to_string(),
        Float(x) => x.to_string(),
        Double(x) => x.to_string(),
        Text(bytes) => String::from_utf8_lossy(bytes).into_owned(),
        other => format!("{other:?}"),
    }
}
