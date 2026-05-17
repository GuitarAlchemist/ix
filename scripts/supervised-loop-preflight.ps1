# Supervised Autonomous Loop Preflight — ix
#
# Deterministic gate that runs BEFORE the supervised-loop skill enters its
# cycle. Exits 0 if the repo is safe to loop, 1 otherwise. Always prints a
# final line of the form `LOOP_READY=true|false` followed by a one-line reason.
#
# Mirrors the hard refusals documented in AGENTS.md and docs/agent-blackbox/install.md.

[CmdletBinding()]
param(
    [int]$OverseerMaxAgeHours = 24,
    [string]$OverseerPath = "state/governance/dev-process-overseer.json",
    [string]$LoopPolicyPath = "agent-blackbox.loop-policy.json",
    [string]$RiskPolicyPath = "agent-blackbox.policy.json",
    [string]$BaselinePath = "state/quality/ix-harness/baseline.json",
    [string]$StopMarkerPath = ".STOP",
    [string]$HaltAllPath = "$HOME/.demerzel/HALT-ALL",
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function Emit-Result {
    param(
        [Parameter(Mandatory = $true)][bool]$Ready,
        [Parameter(Mandatory = $true)][string]$Reason
    )

    $verdict = if ($Ready) { 'true' } else { 'false' }
    Write-Host "LOOP_READY=$verdict reason=$Reason"
    if ($Ready) { exit 0 } else { exit 1 }
}

Write-Host "[preflight] supervised-loop preflight starting"
Write-Host "[preflight] root: $root"

# 1. Halt markers (global + per-repo) take precedence over everything.
if (Test-Path -LiteralPath $HaltAllPath) {
    Emit-Result -Ready:$false -Reason "halt_all_active"
}
if (Test-Path -LiteralPath $StopMarkerPath) {
    Emit-Result -Ready:$false -Reason "repo_stop_marker"
}

# 2. Loop policy must exist and parse.
$loopPolicyFull = Join-Path $root $LoopPolicyPath
if (-not (Test-Path -LiteralPath $loopPolicyFull)) {
    Emit-Result -Ready:$false -Reason "loop_policy_missing"
}
try {
    $loopPolicy = Get-Content -LiteralPath $loopPolicyFull -Raw | ConvertFrom-Json
} catch {
    Emit-Result -Ready:$false -Reason "loop_policy_invalid_json"
}
if (-not $loopPolicy.allow_edit) {
    Emit-Result -Ready:$false -Reason "loop_policy_missing_allow_edit"
}
if (-not $loopPolicy.protected_paths) {
    Emit-Result -Ready:$false -Reason "loop_policy_missing_protected"
}

# 3. PR risk policy must exist (the loop and the CI gate share this file).
$riskPolicyFull = Join-Path $root $RiskPolicyPath
if (-not (Test-Path -LiteralPath $riskPolicyFull)) {
    Emit-Result -Ready:$false -Reason "risk_policy_missing"
}

# 4. Baseline must exist (so the loop has an oracle command + kill switches).
$baselineFull = Join-Path $root $BaselinePath
if (-not (Test-Path -LiteralPath $baselineFull)) {
    Emit-Result -Ready:$false -Reason "baseline_missing"
}

# 5. Overseer artifact must exist and be fresh.
$overseerFull = Join-Path $root $OverseerPath
if (-not (Test-Path -LiteralPath $overseerFull)) {
    Emit-Result -Ready:$false -Reason "overseer_missing"
}
$age = (Get-Date) - (Get-Item -LiteralPath $overseerFull).LastWriteTime
if ($age.TotalHours -gt $OverseerMaxAgeHours) {
    Emit-Result -Ready:$false -Reason "overseer_stale"
}

try {
    $overseer = Get-Content -LiteralPath $overseerFull -Raw | ConvertFrom-Json
} catch {
    Emit-Result -Ready:$false -Reason "overseer_invalid_json"
}
if ($overseer.workflowMode -ne 'loop-eligible') {
    Emit-Result -Ready:$false -Reason "overseer_not_loop_eligible"
}
if ($overseer.counts.blocks -gt 0) {
    Emit-Result -Ready:$false -Reason "overseer_blocks_present"
}

# 6. Optional verify run (off by default — CI runs it).
if (-not $SkipVerify) {
    Write-Host "[preflight] skipping verify run; pass -SkipVerify to silence this notice"
}

Emit-Result -Ready:$true -Reason "loop_ready"
