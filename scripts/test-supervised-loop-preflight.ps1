$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$sourcePreflight = Join-Path $PSScriptRoot 'supervised-loop-preflight.ps1'
$sourceOverseer = Join-Path $PSScriptRoot 'dev-process-overseer.ps1'
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

# Runs dev-process-overseer.ps1 against a hermetic fixture repo (canonical loop
# policy + a baseline variant) and returns the parsed finding codes. Covers the
# scope-boundary resolution path that feeds the preflight via counts.blocks.
function Invoke-TestOverseer {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][object]$Baseline,
        [object]$LoopPolicy
    )

    $repo = Join-Path $temp "overseer-$Name"
    $domainDir = Join-Path $repo 'state/quality/ix-harness'
    New-Item -ItemType Directory -Force -Path $domainDir | Out-Null
    if ($LoopPolicy) {
        Write-Json -Path (Join-Path $repo 'agent-blackbox.loop-policy.json') -Value $LoopPolicy
    }
    Write-Json -Path (Join-Path $domainDir 'baseline.json') -Value $Baseline

    $raw = & pwsh -NoProfile -File $sourceOverseer -RepoRoot $repo -Domain 'ix-harness' -Json -NoEmit | Out-String
    $report = $raw | ConvertFrom-Json
    return @($report.findings.code)
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

    # --- Overseer scope-boundary resolution (ix#228 P1-5 consolidation) ---
    $canonPolicy = [ordered]@{
        version         = '0.2.0'
        allow_edit      = @('docs/**', 'scripts/**')
        protected_paths = @('crates/**', 'src/**', 'Cargo.toml')
    }

    $matchFindings = Invoke-TestOverseer -Name 'version-match' -LoopPolicy $canonPolicy -Baseline ([ordered]@{
        schema_version     = 1
        domain             = 'ix-harness'
        scope_boundary_ref = [ordered]@{ policy = 'agent-blackbox.loop-policy.json'; policy_version = '0.2.0' }
    })
    Assert-True ($matchFindings -notcontains 'loop-policy-version-drift') "Matched version must not report drift. Findings: $($matchFindings -join ',')"
    Assert-True ($matchFindings -notcontains 'loop-policy-unresolvable') "Matched version must resolve the policy. Findings: $($matchFindings -join ',')"
    Assert-True ($matchFindings -notcontains 'inline-scope-boundary-deprecated') "Ref baseline must not trip the inline-deprecated block. Findings: $($matchFindings -join ',')"

    $driftFindings = Invoke-TestOverseer -Name 'version-drift' -LoopPolicy $canonPolicy -Baseline ([ordered]@{
        schema_version     = 1
        domain             = 'ix-harness'
        scope_boundary_ref = [ordered]@{ policy = 'agent-blackbox.loop-policy.json'; policy_version = '9.9.9' }
    })
    Assert-True ($driftFindings -contains 'loop-policy-version-drift') "Pinned-version mismatch must emit loop-policy-version-drift. Findings: $($driftFindings -join ',')"

    $inlineFindings = Invoke-TestOverseer -Name 'inline-scope' -LoopPolicy $canonPolicy -Baseline ([ordered]@{
        schema_version = 1
        domain         = 'ix-harness'
        scope_boundary = [ordered]@{ allow_edit = @('**'); protected_paths = @('Cargo.lock') }
    })
    Assert-True ($inlineFindings -contains 'inline-scope-boundary-deprecated') "Inline scope_boundary must emit inline-scope-boundary-deprecated. Findings: $($inlineFindings -join ',')"

    $unresolvableFindings = Invoke-TestOverseer -Name 'unresolvable' -Baseline ([ordered]@{
        schema_version     = 1
        domain             = 'ix-harness'
        scope_boundary_ref = [ordered]@{ policy = 'agent-blackbox.loop-policy.json'; policy_version = '0.2.0' }
    })
    Assert-True ($unresolvableFindings -contains 'loop-policy-unresolvable') "Missing canonical policy must emit loop-policy-unresolvable. Findings: $($unresolvableFindings -join ',')"

    Write-Host 'PASS supervised-loop preflight regression: default verify, failure propagation, explicit bypass'
    Write-Host 'PASS overseer scope-boundary resolution: version-match clean, version-drift block, inline-deprecated block, unresolvable block'
}
finally {
    if (Test-Path -LiteralPath $temp) {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
}
