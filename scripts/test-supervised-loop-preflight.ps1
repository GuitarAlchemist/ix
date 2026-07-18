$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$sourcePreflight = Join-Path $PSScriptRoot 'supervised-loop-preflight.ps1'
$tempName = "preflight-regression-$([guid]::NewGuid().ToString('N'))"
$temp = Join-Path $root "dist/$tempName"
$fixtureRelative = 'fixtures'
$fixturePath = Join-Path $temp $fixtureRelative
$isolatedScripts = Join-Path $temp 'scripts'
$preflight = Join-Path $isolatedScripts 'supervised-loop-preflight.ps1'
$verifyScript = Join-Path $isolatedScripts 'verify.ps1'

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
        [Parameter(Mandatory = $true)][int]$VerifyExitCode,
        [switch]$SkipVerify
    )

    Set-Content -LiteralPath $verifyScript -Value "exit $VerifyExitCode" -Encoding UTF8

    $arguments = @(
        '-NoProfile', '-File', $preflight,
        '-OverseerPath', "$fixtureRelative/overseer.json",
        '-LoopPolicyPath', "$fixtureRelative/loop-policy.json",
        '-RiskPolicyPath', "$fixtureRelative/risk-policy.json",
        '-BaselinePath', "$fixtureRelative/baseline.json",
        '-StopMarkerPath', "$fixtureRelative/no-stop",
        '-HaltAllPath', "$fixtureRelative/no-halt"
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
    New-Item -ItemType Directory -Force -Path $fixturePath, $isolatedScripts | Out-Null
    Copy-Item -LiteralPath $sourcePreflight -Destination $preflight

    Write-Json -Path (Join-Path $fixturePath 'loop-policy.json') -Value ([ordered]@{
        allow_edit = $true
        protected_paths = @('never/**')
    })
    Write-Json -Path (Join-Path $fixturePath 'risk-policy.json') -Value ([ordered]@{
        version = 'test'
    })
    Write-Json -Path (Join-Path $fixturePath 'baseline.json') -Value ([ordered]@{
        oracle = 'test'
    })
    Write-Json -Path (Join-Path $fixturePath 'overseer.json') -Value ([ordered]@{
        workflowMode = 'loop-eligible'
        counts = [ordered]@{ blocks = 0 }
    })

    $defaultPass = Invoke-TestPreflight -VerifyExitCode 0
    Assert-True ($defaultPass.Code -eq 0) "Default preflight should pass when verify passes. Output: $($defaultPass.Output)"
    Assert-True ($defaultPass.Output -match 'running verify oracle') 'Default preflight did not execute the verify oracle.'
    Assert-True ($defaultPass.Output -match 'LOOP_READY=true reason=loop_ready') 'Default pass did not emit loop_ready.'

    $defaultFail = Invoke-TestPreflight -VerifyExitCode 7
    Assert-True ($defaultFail.Code -eq 1) "Default preflight should fail when verify fails. Output: $($defaultFail.Output)"
    Assert-True ($defaultFail.Output -match 'LOOP_READY=false reason=verify_failed') 'Verify failure was not propagated as verify_failed.'

    $explicitSkip = Invoke-TestPreflight -VerifyExitCode 7 -SkipVerify
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
