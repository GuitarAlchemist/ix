//! `ix_duckdb_query` against the real DuckDB CLI.
//!
//! These need `duckdb` on PATH (or `IX_DUCKDB_BIN`). Where it is missing they
//! skip loudly, and under `IX_REQUIRE_DUCKDB=1` they fail instead, the same
//! rule as `crates/ix-duck/tests/common/mod.rs`: a test that cannot reach the
//! CLI must not read as a pass where the CLI is required.

use ix_agent::skills::duckdb::duckdb_query;
use serde_json::json;

fn duckdb_available() -> bool {
    let bin = std::env::var("IX_DUCKDB_BIN").unwrap_or_else(|_| "duckdb".into());
    let ok = std::process::Command::new(&bin)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    let required = std::env::var_os("IX_REQUIRE_DUCKDB").is_some_and(|v| !v.is_empty() && v != "0");
    assert!(
        ok || !required,
        "IX_REQUIRE_DUCKDB is set but the DuckDB CLI ({bin}) cannot run"
    );
    if !ok {
        eprintln!("SKIPPED: DuckDB CLI ({bin}) not found; ix_duckdb_query unchecked. Set IX_REQUIRE_DUCKDB=1 to make this fail.");
    }
    ok
}

#[test]
fn queries_supplied_tables_with_inferred_types() {
    if !duckdb_available() {
        return;
    }
    let out = duckdb_query(json!({
        "sql": "SELECT t.b, sum(t.a) AS total, any_value(u.label) AS label FROM t JOIN u ON t.b = u.b GROUP BY t.b ORDER BY t.b",
        "tables": {
            "t": [{ "a": 1, "b": "x" }, { "a": 2.5, "b": "x" }, { "a": 4, "b": "it's" }],
            "u": [{ "b": "x", "label": "ex" }, { "b": "it's", "label": "quote" }]
        }
    }))
    .expect("query runs");
    assert_eq!(
        out["rows"],
        json!([{ "b": "it's", "total": 4.0, "label": "quote" }, { "b": "x", "total": 3.5, "label": "ex" }])
    );
    assert_eq!(out["columns"], json!(["b", "label", "total"]));
    assert_eq!(out["truncated"], json!(false));
    assert_eq!(
        out["tables"]["t"],
        json!(r#"[{"a":"DOUBLE","b":"VARCHAR"}]"#)
    );
}

#[test]
fn truncates_to_max_rows_and_reports_it() {
    if !duckdb_available() {
        return;
    }
    let out =
        duckdb_query(json!({ "sql": "SELECT * FROM range(25) r(i)", "max_rows": 10 })).unwrap();
    assert_eq!(out["rows"].as_array().unwrap().len(), 10);
    assert_eq!(out["row_count"], json!(25));
    assert_eq!(out["truncated"], json!(true));
}

#[test]
fn cannot_read_write_or_attach_files() {
    if !duckdb_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let secret = dir.path().join("secret.csv");
    std::fs::write(&secret, "k\nSECRET-VALUE\n").unwrap();
    let p = secret.to_str().unwrap().replace('\\', "/");
    let out_path = dir
        .path()
        .join("out.csv")
        .to_str()
        .unwrap()
        .replace('\\', "/");
    for sql in [
        format!("SELECT * FROM read_csv('{p}')"),
        format!("COPY (SELECT 1) TO '{out_path}'"),
        format!("ATTACH '{p}.db'"),
        "INSTALL httpfs".to_string(),
        "SET enable_external_access = true".to_string(),
    ] {
        let err = duckdb_query(json!({ "sql": sql })).expect_err(&sql);
        assert!(
            !err.contains("SECRET-VALUE"),
            "{sql} leaked the file: {err}"
        );
    }
    assert!(!dir.path().join("out.csv").exists(), "COPY wrote a file");
}

#[test]
fn reports_sql_errors() {
    if !duckdb_available() {
        return;
    }
    let err = duckdb_query(json!({ "sql": "SELEC 1" })).unwrap_err();
    assert!(
        err.starts_with("DuckDB:") && err.contains("syntax error"),
        "{err}"
    );
}
