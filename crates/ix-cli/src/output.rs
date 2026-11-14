//! Output formatting — table / json / jsonl / yaml / csv modes.

use serde_json::Value;
use std::io::{self, IsTerminal, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// Auto: table on TTY, json on pipe.
    Auto,
    Table,
    Json,
    Jsonl,
    Yaml,
}

impl Format {
    /// Resolve `Auto` based on whether stdout is a TTY.
    pub fn resolve(self) -> Self {
        match self {
            Format::Auto => {
                if io::stdout().is_terminal() {
                    Format::Table
                } else {
                    Format::Json
                }
            }
            other => other,
        }
    }
}

/// Emit a single JSON value using the resolved format.
pub fn emit(value: &Value, fmt: Format) -> io::Result<()> {
    let bytes = render(value, fmt)?;
    io::stdout().lock().write_all(&bytes)
}

/// The bytes [`emit`] prints for `value`.
pub fn render(value: &Value, fmt: Format) -> io::Result<Vec<u8>> {
    let resolved = fmt.resolve();
    let mut out = Vec::new();
    match resolved {
        Format::Json => {
            serde_json::to_writer_pretty(&mut out, value)?;
            writeln!(out)?;
        }
        Format::Jsonl => {
            serde_json::to_writer(&mut out, value)?;
            writeln!(out)?;
        }
        Format::Yaml => {
            let s = serde_yaml::to_string(value).unwrap_or_else(|_| "---\n".into());
            out.write_all(s.as_bytes())?;
        }
        Format::Table => emit_table(value, &mut out)?,
        Format::Auto => unreachable!("resolved above"),
    }
    Ok(out)
}

/// Best-effort table rendering. Objects become 2-column key/value tables;
/// arrays of objects become N-column tables; scalars print as-is.
fn emit_table(value: &Value, out: &mut dyn Write) -> io::Result<()> {
    match value {
        Value::Object(map) => {
            let key_width = map.keys().map(|k| k.len()).max().unwrap_or(0);
            for (k, v) in map {
                writeln!(out, "  {k:<width$}  {}", short(v), width = key_width)?;
            }
        }
        Value::Array(items) if items.first().is_some_and(|v| v.is_object()) => {
            // Rows as objects — print first N keys as columns.
            let first = items[0].as_object().unwrap();
            let cols: Vec<&String> = first.keys().collect();
            for c in &cols {
                write!(out, "{c:<16}")?;
            }
            writeln!(out)?;
            writeln!(out, "{}", "-".repeat(cols.len() * 16))?;
            for row in items {
                if let Some(obj) = row.as_object() {
                    for c in &cols {
                        let v = obj.get(c.as_str()).unwrap_or(&Value::Null);
                        write!(out, "{:<16}", short(v))?;
                    }
                    writeln!(out)?;
                }
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                writeln!(out, "  [{i:>3}] {}", short(v))?;
            }
        }
        scalar => writeln!(out, "{}", short(scalar))?,
    }
    Ok(())
}

fn short(v: &Value) -> String {
    match v {
        Value::Null => "—".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if s.len() > 60 {
                format!("{}…", &s[..60])
            } else {
                s.clone()
            }
        }
        Value::Array(a) => format!("[{} items]", a.len()),
        Value::Object(o) => format!("{{{} keys}}", o.len()),
    }
}
