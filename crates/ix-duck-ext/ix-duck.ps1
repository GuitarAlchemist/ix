#!/usr/bin/env pwsh
# ix-duck.ps1 — launch duckdb.exe with the IX extension pre-loaded.
#
# Saves typing the `-unsigned` flag and the `LOAD '<abs path>'` every session.
# Builds the extension first if it's missing.
#
#   pwsh ix-duck.ps1                       # interactive CLI, ix_* ready
#   pwsh ix-duck.ps1 -Ui                   # browser UI (http://localhost:4213)
#   pwsh ix-duck.ps1 -Database work.duckdb # open/persist a database file
#   pwsh ix-duck.ps1 -Sql "SELECT ix_cosine([1,0]::DOUBLE[],[1,0]::DOUBLE[])"
#
# Tip: add this folder to PATH (or make an alias) so it's a one-word command:
#   Set-Alias ixduck 'C:\Users\spare\source\repos\ix\crates\ix-duck-ext\ix-duck.ps1'
[CmdletBinding()]
param(
    [switch]$Ui,                       # also start the browser UI
    [string]$Database = '',            # optional database file (default: in-memory)
    [string]$Sql = '',                 # run one SQL statement and exit (non-interactive)
    [Parameter(ValueFromRemainingArguments = $true)]
    $Rest                              # any extra duckdb args, passed through
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# duckdb.exe must be on PATH.
if (-not (Get-Command duckdb -ErrorAction SilentlyContinue)) {
    throw "duckdb.exe not found on PATH. Install DuckDB or add it to PATH."
}

# Build the extension on first use (or if it was cleaned away).
$ext = Join-Path $root 'ix.duckdb_extension'
if (-not (Test-Path $ext)) {
    Write-Host "ix.duckdb_extension not found — building it…" -ForegroundColor Yellow
    & (Join-Path $root 'build.ps1')
    if (-not (Test-Path $ext)) { throw "build did not produce $ext" }
}
$extFwd = $ext -replace '\\', '/'   # LOAD wants forward slashes

# -unsigned is required (local unsigned extension) and must be set at launch.
# -cmd runs the LOAD before the prompt/UI/-c, so ix_* is ready immediately.
$duckArgs = @('-unsigned', '-cmd', "LOAD '$extFwd'")
if ($Ui)        { $duckArgs += '-ui' }
if ($Sql)       { $duckArgs += @('-c', $Sql) }
if ($Database)  { $duckArgs += $Database }   # positional db path
if ($Rest)      { $duckArgs += $Rest }

& duckdb @duckArgs
exit $LASTEXITCODE
