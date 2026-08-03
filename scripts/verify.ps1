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

# Clippy IS a hard gate, and the flags below must stay byte-identical to the
# `Clippy lint` step in .github/workflows/ci.yml. CI blocks on it, so local
# verify blocks on it too — otherwise `verify.ps1` green does not imply CI green.
Invoke-Checked 'cargo' @('clippy', '--workspace', '--all-targets', '--', '-D', 'warnings')

Invoke-Checked 'cargo' @('test', '--workspace')

# ix#185: single pre-PR health command. Detects skill/tool inventory drift
# against the committed state/registry/skills.snapshot.json (no hand-edited
# magic number — the snapshot is generated from the live registry via
# `ix check doctor --write-snapshot`) plus governance/environment checks.
# Emits actionable JSON (which skill was added/removed, and the exact fix
# command) instead of raw test noise. Exit 0 (T) or 1 (P/warnings-only) pass;
# anything else (F/false) blocks.
Write-Host '[verify] running ix check doctor (skill/tool inventory + environment health)'
& cargo run -q -p ix-skill -- --format json check doctor
$doctorExit = $LASTEXITCODE
if ($doctorExit -ne 0 -and $doctorExit -ne 1) {
    throw "ix check doctor failed (exit ${doctorExit} — see the JSON 'checks' array above for which check failed and how to fix it; re-run 'cargo run -p ix-skill -- check doctor' locally to reproduce)"
}

Write-Host '[verify] running supervised-loop preflight regression harness'
Invoke-Checked 'pwsh' @('-NoProfile', '-File', (Join-Path $PSScriptRoot 'test-supervised-loop-preflight.ps1'))
