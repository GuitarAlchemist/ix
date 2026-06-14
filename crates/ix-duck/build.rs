//! Build script: on Windows, the bundled DuckDB (`libduckdb-sys`) references the
//! Restart Manager API (`RmStartSession`/`RmEndSession`/`RmRegisterResources`/
//! `RmGetList`, used by DuckDB's `AdditionalLockInfo` file-lock diagnostics) but
//! does not emit the link directive for it. Without this, the lib compiles but
//! linking any binary (tests, examples) fails with LNK2019 unresolved externals.
//! Only needed when the `duck` feature pulls bundled DuckDB on a Windows target.

fn main() {
    let duck = std::env::var("CARGO_FEATURE_DUCK").is_ok();
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if duck && windows {
        println!("cargo:rustc-link-lib=dylib=rstrtmgr");
    }
}
