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
        format!("COPY (SELECT 1) TO '{out_path}'; SELECT 1 AS x"),
        format!("ATTACH '{p}.db'; SELECT 1 AS x"),
        "INSTALL httpfs; SELECT 1 AS x".to_string(),
        "LOAD httpfs; SELECT 1 AS x".to_string(),
        "SET enable_external_access = true; SELECT 1 AS x".to_string(),
        "SELECT getenv('PATH') AS p".to_string(),
    ] {
        // Each ends with a query, so DuckDB's safe mode refuses it, not the
        // last-statement check.
        let err = duckdb_query(json!({ "sql": sql })).expect_err(&sql);
        assert!(err.starts_with("DuckDB:"), "{sql}: {err}");
        assert!(
            !err.contains("SECRET-VALUE"),
            "{sql} leaked the file: {err}"
        );
    }
    assert!(!dir.path().join("out.csv").exists(), "COPY wrote a file");
}

#[test]
fn memory_and_threads_are_bounded_and_locked() {
    if !duckdb_available() {
        return;
    }
    let out = duckdb_query(json!({
        "sql": "SELECT current_setting('memory_limit') AS m, current_setting('threads') AS t, current_setting('temp_directory') AS d"
    }))
    .unwrap();
    assert_eq!(out["rows"], json!([{ "m": "512.0 MiB", "t": 2, "d": "" }]));
    let err =
        duckdb_query(json!({ "sql": "SET memory_limit = '8GB'; SELECT 1 AS x" })).unwrap_err();
    assert!(err.contains("locked"), "{err}");
    // Materialises 200M BIGINTs (1.6 GB): refused at the limit, not spilled to disk.
    let err = duckdb_query(json!({ "sql": "SELECT len(list(i)) AS n FROM range(200000000) r(i)" }))
        .unwrap_err();
    assert!(err.contains("Out of Memory"), "{err}");
}

#[test]
fn refuses_results_with_repeated_column_names() {
    if !duckdb_available() {
        return;
    }
    let err = duckdb_query(json!({ "sql": "SELECT 1 AS a, 2 AS a" })).unwrap_err();
    assert!(err.contains("more than one column named \"a\""), "{err}");
    let err = duckdb_query(json!({
        "sql": "SELECT t.*, u.* FROM t JOIN u ON t.b = u.b",
        "tables": {
            "t": [{ "a": 1, "b": "x" }],
            "u": [{ "b": "x", "label": "ex" }]
        }
    }))
    .unwrap_err();
    assert!(err.contains("more than one column named \"b\""), "{err}");
    // An empty result's names come from a second run, and are checked too.
    let err = duckdb_query(json!({ "sql": "SELECT 1 AS a, 2 AS a WHERE false" })).unwrap_err();
    assert!(err.contains("more than one column named \"a\""), "{err}");
}

#[test]
fn returns_numbers_exactly_or_refuses_them() {
    if !duckdb_available() {
        return;
    }
    // DuckDB prints HUGEINT, UBIGINT and DECIMAL values as strings, but as
    // numbers inside a LIST or STRUCT, and a DOUBLE as its shortest decimal.
    let out = duckdb_query(json!({
        "sql": "SELECT 18446744073709551617::HUGEINT AS h, 12345678901234567890.12::DECIMAL(38,2) AS d, 9007199254740993::BIGINT AS b, [18446744073709551615::UBIGINT] AS l, {'x': 0.1::DECIMAL(3,1)} AS s, 0.1::DOUBLE + 0.2::DOUBLE AS f, 1.0715660391465826e-75::DOUBLE AS g"
    }))
    .unwrap();
    assert_eq!(
        out["rows"],
        json!([{
            "h": "18446744073709551617",
            "d": "12345678901234567890.12",
            "b": 9007199254740993_u64,
            "l": [18446744073709551615_u64],
            "s": { "x": 0.1 },
            "f": 0.30000000000000004,
            "g": 1.0715660391465826e-75
        }])
    );
    for sql in [
        "SELECT [18446744073709551617::HUGEINT] AS l",
        "SELECT {'x': 12345678901234567890.12::DECIMAL(38,2)} AS s",
    ] {
        let err = duckdb_query(json!({ "sql": sql })).expect_err(sql);
        assert!(err.contains("would lose digits"), "{sql}: {err}");
    }
    // Only the returned result is read.
    let out = duckdb_query(json!({
        "sql": "SELECT [18446744073709551617::HUGEINT] AS l; SELECT 1 AS ok"
    }))
    .unwrap();
    assert_eq!(out["rows"], json!([{ "ok": 1 }]));
}

#[test]
fn returns_supplied_integers_as_numbers() {
    if !duckdb_available() {
        return;
    }
    // json_structure infers UBIGINT (or HUGEINT, for mixed signs), which
    // DuckDB prints as strings: a step's integer output would come back as
    // "1" to the next step.
    let out = duckdb_query(json!({
        "sql": "SELECT a, a + 1 AS b FROM t ORDER BY a",
        "tables": { "t": [{ "a": 2 }, { "a": -1 }] }
    }))
    .unwrap();
    assert_eq!(
        out["rows"],
        json!([{ "a": -1, "b": 0 }, { "a": 2, "b": 3 }])
    );
    assert_eq!(out["tables"]["t"], json!(r#"[{"a":"BIGINT"}]"#));
    // An integer beyond i64 keeps its inferred type, printed exactly.
    let out = duckdb_query(json!({
        "sql": "SELECT a FROM t",
        "tables": { "t": [{ "a": 18446744073709551615_u64 }] }
    }))
    .unwrap();
    assert_eq!(out["rows"], json!([{ "a": "18446744073709551615" }]));
}

#[test]
fn keeps_nested_objects_as_struct_columns() {
    if !duckdb_available() {
        return;
    }
    let out = duckdb_query(json!({
        "sql": "SELECT typeof(meta) AS tm, meta.score::INTEGER AS score, id::INTEGER AS id FROM t",
        "tables": { "t": [{ "id": 1, "meta": { "score": 2, "id": 9 } }] }
    }))
    .expect("`meta` is a top-level column");
    let row = &out["rows"][0];
    assert!(row["tm"].as_str().unwrap().starts_with("STRUCT("), "{row}");
    assert_eq!((&row["score"], &row["id"]), (&json!(2), &json!(1)));
}

#[test]
fn reports_the_columns_of_an_empty_result() {
    if !duckdb_available() {
        return;
    }
    let out = duckdb_query(json!({
        "sql": "SELECT 7 AS earlier; SELECT 1 AS id, 'x' AS \"a,b\", 2 AS \"q\"\"x\" WHERE false"
    }))
    .unwrap();
    assert_eq!(out["rows"], json!([]));
    assert_eq!(out["row_count"], json!(0));
    assert_eq!(out["columns"], json!(["a,b", "id", "q\"x"]));
}

#[test]
fn never_reports_a_data_row_as_column_names() {
    if !duckdb_available() {
        return;
    }
    // Empty on one run and not on the next about a quarter of the time: the
    // run that reads an empty result's names must not take a row for them.
    for _ in 0..20 {
        let out = duckdb_query(json!({ "sql": "SELECT 1 AS id WHERE random() < 0.5" })).unwrap();
        let columns = &out["columns"];
        assert!(*columns == json!(["id"]) || *columns == json!([]), "{out}");
    }
}

#[test]
fn returns_the_last_statement_or_refuses_the_script() {
    if !duckdb_available() {
        return;
    }
    let err =
        duckdb_query(json!({ "sql": "SELECT 1 AS stale; CREATE VIEW v AS SELECT 2" })).unwrap_err();
    assert!(err.contains("last statement"), "{err}");
    let out =
        duckdb_query(json!({ "sql": "CREATE VIEW v AS SELECT 2 AS x; SELECT * FROM v" })).unwrap();
    assert_eq!(out["rows"], json!([{ "x": 2 }]));
    // A write after a CTE list, or with 'returning' only in a string, prints
    // nothing, so it is refused too; with a RETURNING clause it prints rows.
    for sql in [
        "CREATE TABLE t(a INT); SELECT 99 AS stale; WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x",
        "CREATE TABLE t(s VARCHAR); SELECT 99 AS stale; INSERT INTO t VALUES ('returning')",
        "CREATE TABLE t(a INT); PREPARE ins AS INSERT INTO t VALUES (1); SELECT 99 AS stale; EXECUTE ins",
        "CREATE TABLE t(s VARCHAR); SELECT 99 AS stale; INSERT INTO t VALUES ($é$; SELECT 1$é$)",
        "SELECT 99 AS stale; -- c\rCREATE VIEW v AS SELECT 2",
    ] {
        let err = duckdb_query(json!({ "sql": sql })).unwrap_err();
        assert!(err.contains("last statement"), "{sql}: {err}");
    }
    let out = duckdb_query(json!({ "sql": "CREATE TABLE t(a INT); SELECT 99 AS stale; WITH x AS (SELECT 1 AS a) INSERT INTO t SELECT * FROM x RETURNING a" })).unwrap();
    assert_eq!(out["rows"], json!([{ "a": 1 }]));
    let out = duckdb_query(json!({ "sql": "CREATE TABLE t(a INT); MERGE INTO t USING (SELECT 1 AS a) AS s ON t.a = s.a WHEN NOT MATCHED THEN INSERT VALUES (s.a) RETURNING a" })).unwrap();
    assert_eq!(out["rows"], json!([{ "a": 1 }]));
    // Each kind of statement accepted last prints its own result, even when
    // it has no rows, so the query before it is never returned in its place.
    for last in [
        "SELECT 1 AS a",
        "FROM t",
        "WITH x AS (SELECT 1 AS a) (SELECT * FROM x)",
        "VALUES (1)",
        "TABLE t",
        "SHOW TABLES",
        "DESCRIBE t",
        "SUMMARIZE t",
        "PIVOT (SELECT 1 AS a, 'x' AS s) ON s USING count(*)",
        "UNPIVOT (SELECT 1 AS a, 2 AS b) ON a, b",
        "CALL range(0)",
        "PRAGMA table_info('t')",
        "PRAGMA enable_checkpoint_on_shutdown",
        "EXECUTE q",
        "INSERT INTO t VALUES (1) RETURNING a",
    ] {
        let sql = format!(
            "CREATE TABLE t(a INT); PREPARE q AS SELECT 7 AS a; SELECT 99 AS stale; {last}"
        );
        let out = duckdb_query(json!({ "sql": sql })).unwrap_or_else(|e| panic!("{last}: {e}"));
        assert_ne!(out["rows"], json!([{ "stale": 99 }]), "{last}");
    }
    // DuckDB reads these statement boundaries the same way.
    let out = duckdb_query(json!({ "sql": "SELECT $é$a;b$é$ AS s" })).unwrap();
    assert_eq!(out["rows"], json!([{ "s": "a;b" }]));
    let out = duckdb_query(json!({ "sql": "SELECT 1 AS stale; -- c\rSELECT 2 AS x" })).unwrap();
    assert_eq!(out["rows"], json!([{ "x": 2 }]));
    // A leading-dot literal on an indented line is SQL.
    let out = duckdb_query(json!({ "sql": "SELECT\n  .5::DOUBLE AS ratio" })).unwrap();
    assert_eq!(out["rows"], json!([{ "ratio": 0.5 }]));
    // An indented dot command is SQL too, so a syntax error, never a mode change.
    let err =
        duckdb_query(json!({ "sql": "SELECT 1 AS a;\n  .mode csv; SELECT 2 AS b" })).unwrap_err();
    assert!(err.contains("syntax error"), "{err}");
}

#[test]
fn reports_sql_errors() {
    if !duckdb_available() {
        return;
    }
    let err = duckdb_query(json!({ "sql": "SELECT 1 FROM" })).unwrap_err();
    assert!(
        err.starts_with("DuckDB:") && err.contains("syntax error"),
        "{err}"
    );
}
