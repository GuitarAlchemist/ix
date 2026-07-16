$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$preflight = Join-Path $PSScriptRoot 'supervised-loop-preflight.ps1'
$tempName = "preflight-regression-$([guid]::NewGuid().ToString('N'))"
$tempRelative = "dist/$tempName"
$temp = Join-Path $root $tempRelative

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

function Write-Json {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value
    )
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-TestPreflight {
    param(
        [Parameter(Mandatory = $true)][string]$VerifyScript,
        [switch]$SkipVerify
    )

    $arguments = @(
        '-NoProfile', '-File', $preflight,
        '-OverseerPath', "$tempRelative/overseer.json",
        '-LoopPolicyPath', "$tempRelative/loop-policy.json",
        '-RiskPolicyPath', "$tempRelative/risk-policy.json",
        '-BaselinePath', "$tempRelative/baseline.json",
        '-StopMarkerPath', "$tempRelative/no-stop",
        '-HaltAllPath', "$tempRelative/no-halt",
        '-VerifyScriptPath', "$tempRelative/$VerifyScript"
    )
    if ($SkipVerify) {
        $arguments += '-SkipVerify'
    }

    $output = & pwsh @arguments 2>&1 | Out-String
    $code = $LASTEXITCODE
    return [pscustomobject]@{
        Code = $code
        Output = $output
    }
}

try {
    New-Item -ItemType Directory -Force -Path $temp | Out-Null

    Write-Json -Path (Join-Path $temp 'loop-policy.json') -Value ([ordered]@{
        allow_edit = $true
        protected_paths = @('never/**')
    })
    Write-Json -Path (Join-Path $temp 'risk-policy.json') -Value ([ordered]@{
        version = 'test'
    })
    Write-Json -Path (Join-Path $temp 'baseline.json') -Value ([ordered]@{
        oracle = 'test'
    })
    Write-Json -Path (Join-Path $temp 'overseer.json') -Value ([ordered]@{
        workflowMode = 'loop-eligible'
        counts = [ordered]@{ blocks = 0 }
    })

    Set-Content -LiteralPath (Join-Path $temp 'verify-pass.ps1') -Value 'exit 0' -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $temp 'verify-fail.ps1') -Value 'exit 7' -Encoding UTF8

    $defaultPass = Invoke-TestPreflight -VerifyScript 'verify-pass.ps1'
    Assert-True ($defaultPass.Code -eq 0) "Default preflight should pass when verify passes. Output: $($defaultPass.Output)"
    Assert-True ($defaultPass.Output -match 'running verify oracle') 'Default preflight did not execute the verify oracle.'
    Assert-True ($defaultPass.Output -match 'LOOP_READY=true reason=loop_ready') 'Default pass did not emit loop_ready.'

    $defaultFail = Invoke-TestPreflight -VerifyScript 'verify-fail.ps1'
    Assert-True ($defaultFail.Code -eq 1) "Default preflight should fail when verify fails. Output: $($defaultFail.Output)"
    Assert-True ($defaultFail.Output -match 'LOOP_READY=false reason=verify_failed') 'Verify failure was not propagated as verify_failed.'

    $explicitSkip = Invoke-TestPreflight -VerifyScript 'verify-fail.ps1' -SkipVerify
    Assert-True ($explicitSkip.Code -eq 0) "-SkipVerify should bypass the failing oracle explicitly. Output: $($explicitSkip.Output)"
    Assert-True ($explicitSkip.Output -match 'verify run SKIPPED') '-SkipVerify did not emit an explicit skipped message.'
    Assert-True ($explicitSkip.Output -match 'LOOP_READY=true reason=loop_ready') '-SkipVerify did not continue to loop_ready.'

    Write-Host 'PASS supervised-loop preflight regression: default verify, failure propagation, explicit bypass'
}
finally {
    if (Test-Path -LiteralPath $temp) {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
}
