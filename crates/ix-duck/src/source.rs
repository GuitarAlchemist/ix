//! Artifact source — the deep module a [`lens`](crate) reads through.
//!
//! Every lens used to re-implement the same three things by hand: an `Io | Duck`
//! error enum, file enumeration + a DuckDB list literal, and the
//! `CREATE OR REPLACE TABLE … AS SELECT … FROM read_json_auto(…)` materialization.
//! That last one carries a load-bearing invariant — the *safe projection* — and
//! re-applying it by hand is exactly how the absence-as-zero / struct-field
//! bind-crash defect class kept recurring (routing #126, chatbot/loops #127). See
//! `docs/solutions/feature-implementations/2026-06-19-duckdb-absence-as-zero-and-struct-bind-crash.md`.
//!
//! This module concentrates all three behind one seam. A lens declares *which files*
//! ([`Files`]) and *which columns* (a flat [`Col`] spec); [`materialize`] owns the
//! rest and guarantees the invariant:
//!
//! - **Optional / nested fields go through `json_extract(to_json(root), path)`**, never
//!   struct-field access — so a field absent from *every* file yields `NULL`, not a
//!   bind-time "Could not find key" error.
//! - **Absence is `NULL`, never `coalesce(…, 0)`** — a missing metric can't masquerade
//!   as a real zero and fake a trend.
//! - **An absent/empty directory yields an empty table _with the declared schema_**, so
//!   downstream lens queries still bind (graceful degrade, never an error).
//!
//! Lenses whose shape isn't a flat projection (e.g. routing's `perIntent` map-explode)
//! keep a custom `SELECT` but still reuse [`select_files`], [`sql_list`], [`READ_FLAGS`]
//! and [`SourceError`] — so only the genuinely-different projection stays bespoke.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use duckdb::Connection;

/// `read_json_auto` flags every artifact read uses: per-file `filename`, schema union
/// across ragged files, and type inference over *all* rows (`sample_size=-1`).
pub const READ_FLAGS: &str = "filename=true, union_by_name=true, sample_size=-1";

/// Error from an artifact source: directory I/O vs DuckDB. Shared by every lens
/// (replaces the per-lens `RoutingError`/`ChatbotError`/… copies).
#[derive(Debug)]
pub enum SourceError {
    Io(std::io::Error),
    Duck(duckdb::Error),
}
impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceError::Io(e) => write!(f, "artifact source I/O error: {e}"),
            SourceError::Duck(e) => write!(f, "duckdb error: {e}"),
        }
    }
}
impl std::error::Error for SourceError {}
impl From<std::io::Error> for SourceError {
    fn from(e: std::io::Error) -> Self {
        SourceError::Io(e)
    }
}
impl From<duckdb::Error> for SourceError {
    fn from(e: duckdb::Error) -> Self {
        SourceError::Duck(e)
    }
}

/// A file selector: a directory and a filename predicate. The predicate is a plain
/// `fn` (lens patterns capture nothing — prefix/suffix/extension matches).
#[derive(Clone, Copy)]
pub struct Files<'a> {
    pub dir: &'a Path,
    pub matches: fn(&str) -> bool,
}

/// How one output column is produced from the parsed JSON.
#[derive(Clone, Copy)]
pub enum ColSource {
    /// A top-level field/expression `read_json_auto` infers; emitted as-is. `union_by_name`
    /// NULL-fills it across ragged files, so direct access is safe. e.g. `"generatedAt::VARCHAR"`.
    Direct(&'static str),
    /// An optional / nested field: `json_extract(to_json(root), path)` cast to the column
    /// type. NULL on whole-corpus absence (never a bind error), never coalesced to zero.
    /// `path` is a JSONPath (`"$.Accuracy"`) or JSON-Pointer (`"/key/F1"`).
    Extract {
        root: &'static str,
        path: &'static str,
    },
}

/// One output column of an artifact source: its name, SQL type (used for the
/// empty-table schema and the cast), and how it's projected.
#[derive(Clone, Copy)]
pub struct Col {
    pub name: &'static str,
    pub ty: &'static str,
    pub source: ColSource,
}
impl Col {
    /// A top-level field/expression, emitted directly.
    pub const fn direct(name: &'static str, ty: &'static str, expr: &'static str) -> Self {
        Col { name, ty, source: ColSource::Direct(expr) }
    }
    /// An optional/nested field read safely via `json_extract` (NULL on absence).
    pub const fn extract(name: &'static str, ty: &'static str, root: &'static str, path: &'static str) -> Self {
        Col { name, ty, source: ColSource::Extract { root, path } }
    }

    /// `name ty` for the empty-table schema.
    fn schema_decl(&self) -> String {
        format!("{} {}", self.name, self.ty)
    }

    /// The SELECT projection for a populated table — this is where the safe-read
    /// invariant lives.
    fn projection(&self) -> String {
        match self.source {
            ColSource::Direct(expr) => format!("{expr} AS {}", self.name),
            // VARCHAR comes back JSON-quoted from json_extract, so use json_extract_string;
            // everything else round-trips through TRY_CAST (NULL, never an error, on a bad/absent value).
            ColSource::Extract { root, path } if self.ty.eq_ignore_ascii_case("VARCHAR") => {
                format!("json_extract_string(to_json({root}), '{path}') AS {}", self.name)
            }
            ColSource::Extract { root, path } => {
                format!("TRY_CAST(json_extract(to_json({root}), '{path}') AS {}) AS {}", self.ty, self.name)
            }
        }
    }
}

/// Files in `sel.dir` matching `sel.matches`, sorted. An **absent** directory →
/// empty (the lens degrades to "no data"); any other read error on a present dir is
/// surfaced.
pub fn select_files(sel: Files) -> Result<Vec<PathBuf>, SourceError> {
    let rd = match std::fs::read_dir(sel.dir) {
        Ok(r) => r,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut out = Vec::new();
    for entry in rd {
        let p = entry?.path();
        if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
            if (sel.matches)(n) {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// A DuckDB list literal of POSIX-slashed, quote-escaped paths.
pub fn sql_list(paths: &[PathBuf]) -> String {
    let items: Vec<String> = paths
        .iter()
        .map(|p| format!("'{}'", p.to_string_lossy().replace('\\', "/").replace('\'', "''")))
        .collect();
    format!("[{}]", items.join(", "))
}

/// Materialize `table` from a precomputed file list using the flat column `spec`.
/// Empty list → an empty table with the declared schema. Returns the row count.
pub fn materialize_files(
    conn: &Connection,
    table: &str,
    files: &[PathBuf],
    spec: &[Col],
) -> Result<usize, SourceError> {
    if files.is_empty() {
        let decls: Vec<String> = spec.iter().map(Col::schema_decl).collect();
        conn.execute_batch(&format!("CREATE OR REPLACE TABLE {table} ({});", decls.join(", ")))?;
        return Ok(0);
    }
    let proj: Vec<String> = spec.iter().map(Col::projection).collect();
    let list = sql_list(files);
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE {table} AS SELECT {} FROM read_json_auto({list}, {READ_FLAGS});",
        proj.join(", ")
    ))?;
    let n: i64 = conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))?;
    Ok(n as usize)
}

/// Select files and materialize `table` in one step — the common lens path.
pub fn materialize(
    conn: &Connection,
    table: &str,
    sel: Files,
    spec: &[Col],
) -> Result<usize, SourceError> {
    let files = select_files(sel)?;
    materialize_files(conn, table, &files, spec)
}

/// A flat artifact lens: the *whole* shape of "read a dir of matching files → project a
/// fixed set of columns → a named table" captured as one value, so a lens declares it
/// once as a `const` and the read path is a single `.materialize(conn, dir)` call.
///
/// This makes the implicit lens shape — directory + file predicate + column spec →
/// table — explicit and reusable. The flat lenses (`loops`, `ood`) and routing's
/// top-line `routing_overall` are exactly this. Lenses whose projection isn't flat
/// keep a bespoke `SELECT` (routing's `perIntent` map-explode, chatbot's nested
/// `golden-traces` + grounding/signature shapes) and reuse [`select_files`] /
/// [`sql_list`] / [`READ_FLAGS`] directly instead.
#[derive(Clone, Copy)]
pub struct ArtifactLens {
    /// The DuckDB table this lens materializes into.
    pub table: &'static str,
    /// Filename predicate selecting the artifact files inside the lens directory.
    pub matches: fn(&str) -> bool,
    /// The flat column projection (owns the absence-as-NULL-safe read invariant).
    pub spec: &'static [Col],
}

impl ArtifactLens {
    /// Materialize the lens's table from `dir`. Absent/empty dir → an empty table with
    /// the declared schema (graceful degrade). Returns the row count.
    pub fn materialize(&self, conn: &Connection, dir: &Path) -> Result<usize, SourceError> {
        materialize(conn, self.table, Files { dir, matches: self.matches }, self.spec)
    }
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;
    use std::io::Write;

    // A spec mirroring routing_overall: two Direct columns + four optional Extract metrics.
    const SPEC: &[Col] = &[
        Col::direct("generated_at", "VARCHAR", "generatedAt::VARCHAR"),
        Col::direct("day", "VARCHAR", "(generatedAt::VARCHAR)[1:10]"),
        Col::extract("accuracy", "DOUBLE", "overall", "$.Accuracy"),
        Col::extract("oos_decline_rate", "DOUBLE", "overall", "$.OosDeclineRate"),
        Col::extract("label", "VARCHAR", "overall", "$.Label"),
    ];

    fn matches_eval(n: &str) -> bool {
        n.starts_with("eval-") && n.ends_with(".json")
    }

    fn write(dir: &Path, name: &str, body: &str) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    #[test]
    fn absent_dir_yields_empty_table_with_schema() {
        let conn = crate::open_bench().unwrap();
        let n = materialize(
            &conn,
            "t",
            Files { dir: Path::new("/no/such/dir"), matches: matches_eval },
            SPEC,
        )
        .unwrap();
        assert_eq!(n, 0);
        // Downstream queries must still bind against the declared schema.
        let rows: i64 = conn
            .query_row("SELECT count(*) FROM (SELECT accuracy, label, oos_decline_rate FROM t)", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0);
    }

    #[test]
    fn optional_field_absent_corpuswide_is_null_not_crash() {
        // Every file predates the OosDeclineRate / Label fields → both absent from the
        // whole corpus. Struct-field access would fail at bind time; json_extract yields
        // NULL. This is the defect-class guard, now in ONE place for all lenses.
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "eval-2026-05-11.json", r#"{"generatedAt":"2026-05-11T00:00:00Z","overall":{"Accuracy":0.875}}"#);
        write(dir.path(), "eval-2026-05-12.json", r#"{"generatedAt":"2026-05-12T00:00:00Z","overall":{"Accuracy":0.9}}"#);
        let conn = crate::open_bench().unwrap();
        let n = materialize(&conn, "t", Files { dir: dir.path(), matches: matches_eval }, SPEC).unwrap();
        assert_eq!(n, 2);
        let (acc, oos, label): (Option<f64>, Option<f64>, Option<String>) = conn
            .query_row(
                "SELECT accuracy, oos_decline_rate, label FROM t ORDER BY generated_at LIMIT 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(acc, Some(0.875), "present field reads its value");
        assert_eq!(oos, None, "absent numeric field is NULL, not 0.0");
        assert_eq!(label, None, "absent string field is NULL");
    }

    #[test]
    fn present_fields_read_through_including_string_and_derived() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "eval-2026-06-02.json",
            r#"{"generatedAt":"2026-06-02T12:00:00Z","overall":{"Accuracy":0.95,"OosDeclineRate":0.5,"Label":"green"}}"#,
        );
        let conn = crate::open_bench().unwrap();
        materialize(&conn, "t", Files { dir: dir.path(), matches: matches_eval }, SPEC).unwrap();
        let (day, oos, label): (String, Option<f64>, Option<String>) = conn
            .query_row("SELECT day, oos_decline_rate, label FROM t", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!(day, "2026-06-02", "Direct derived expr (slice) works");
        assert_eq!(oos, Some(0.5));
        assert_eq!(label.as_deref(), Some("green"), "VARCHAR extract is unquoted via json_extract_string");
    }

    #[test]
    fn non_matching_files_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "eval-2026-06-02.json", r#"{"generatedAt":"2026-06-02T00:00:00Z","overall":{"Accuracy":0.9}}"#);
        write(dir.path(), "README.md", "not an eval");
        let conn = crate::open_bench().unwrap();
        let n = materialize(&conn, "t", Files { dir: dir.path(), matches: matches_eval }, SPEC).unwrap();
        assert_eq!(n, 1, "only eval-*.json is read");
    }
}
