#!/usr/bin/env pwsh
# build.ps1 — compile ix-duck-ext, append the DuckDB metadata footer, emit
# `ix.duckdb_extension`, and (optionally) smoke-test it against a real duckdb.exe.
#
# This crate is EXCLUDED from the ix workspace (see root Cargo.toml), so the
# default/CI `cargo build --workspace` never compiles DuckDB. Run this explicitly
# when you want the loadable extension.
#
# Plan: docs/plans (GA) 2026-06-15-tools-ix-duckdb-loadable-extension-plan.md
#
#   pwsh crates/ix-duck-ext/build.ps1                 # build + footer
#   pwsh crates/ix-duck-ext/build.ps1 -SmokeTest      # + LOAD into duckdb.exe and assert
#
# Requires: cargo, python (for append_extension_metadata.py), and duckdb.exe on
# PATH (for the platform string + smoke test).
[CmdletBinding()]
param(
    [string]$ExtName       = 'ix',
    [string]$ExtVersion    = '0.1.0',
    # C-API version encoded in the footer (C_STRUCT ABI). The extension loads into
    # any engine >= this. Keep <= the oldest engine you must support.
    [string]$CApiVersion   = 'v1.0.0',
    [switch]$SmokeTest
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Push-Location $root
try {
    Write-Host '== cargo build --release ==' -ForegroundColor Cyan
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }

    $dll = Get-ChildItem "$root/target/release" -Filter '*.dll' |
        Where-Object { $_.Name -like '*ix_duck_ext*' } | Select-Object -First 1
    if (-not $dll) { throw 'cdylib not found in target/release (expected ix_duck_ext*.dll)' }
    Write-Host " - cdylib: $($dll.FullName)"

    # Resolve the DuckDB platform string from the CLI itself so the footer matches.
    $platform = (& duckdb -noheader -list -c 'PRAGMA platform;').Trim()
    if (-not $platform) { throw 'could not resolve duckdb platform (is duckdb.exe on PATH?)' }
    Write-Host " - platform: $platform"

    $out = Join-Path $root "$ExtName.duckdb_extension"
    Write-Host '== append metadata footer (C_STRUCT) ==' -ForegroundColor Cyan
    python "$root/append_extension_metadata.py" `
        -l $dll.FullName -n $ExtName `
        -dv $CApiVersion -p $platform -ev $ExtVersion `
        --abi-type C_STRUCT -o $out
    if ($LASTEXITCODE -ne 0) { throw "metadata append failed ($LASTEXITCODE)" }
    Write-Host " - emitted: $out" -ForegroundColor Green

    if ($SmokeTest) {
        Write-Host '== smoke test (LOAD into duckdb.exe) ==' -ForegroundColor Cyan
        $loadPath = $out -replace '\\', '/'

        # Scalars: ix_cosine / ix_euclidean -> 1.0|0.0|5.0
        $scalarSql = "LOAD '$loadPath';
            SELECT ix_cosine([1.0,2.0,3.0]::DOUBLE[],[1.0,2.0,3.0]::DOUBLE[]) AS a,
                   ix_cosine([1.0,0.0]::DOUBLE[],[0.0,1.0]::DOUBLE[])         AS b,
                   ix_euclidean([0.0,0.0]::DOUBLE[],[3.0,4.0]::DOUBLE[])      AS c;"
        $scalar = (& duckdb -unsigned -noheader -list -c $scalarSql).Trim()
        Write-Host " - scalars: $scalar"
        if ($scalar -notmatch '^1(\.0)?\|0(\.0)?\|5(\.0)?$') {
            throw "SMOKE TEST FAILED (scalars) — expected '1.0|0.0|5.0', got '$scalar'"
        }

        # Table fns: ix_pca_project row count (4) | ix_silhouette mean rounded (1)
        $tableSql = "LOAD '$loadPath';
            SELECT (SELECT count(*) FROM ix_pca_project('[[1,2,3],[2,4,6],[3,6,9],[4,8,12]]', 1)) AS n,
                   (SELECT round(avg(silhouette)) FROM ix_silhouette('[[0,0],[0,1],[10,10],[10,11]]','[0,0,1,1]')) AS s;"
        $table = (& duckdb -unsigned -noheader -list -c $tableSql).Trim()
        Write-Host " - table fns: $table"
        if ($table -notmatch '^4\|1(\.0)?$') {
            throw "SMOKE TEST FAILED (table fns) — expected '4|1.0', got '$table'"
        }
        Write-Host ' - PASS: scalar + table UDFs correct via LOAD' -ForegroundColor Green
    }
}
finally {
    Pop-Location
}
