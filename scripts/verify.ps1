$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

# rustfmt is ADVISORY, not a gate. Several crates intentionally use a terser
# hand style (e.g. ix-duck — see CLAUDE.md), so `cargo fmt --all --check` reports
# diffs repo-wide and would fail EVERY run. Run it for visibility but never block
# on it; the hard gate is the test suite below. (Before this, the Agent-Blackbox
# `risk-report` verdict failed on every PR purely from this fmt skew.)
& cargo fmt --all --check
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'rustfmt --check reported diffs (advisory; not blocking — intentional terse style, see CLAUDE.md).'
}

Invoke-Checked 'cargo' @('test', '--workspace')

Write-Host '[verify] running supervised-loop preflight regression harness'
Invoke-Checked 'pwsh' @('-NoProfile', '-File', (Join-Path $PSScriptRoot 'test-supervised-loop-preflight.ps1'))
