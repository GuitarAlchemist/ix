# scripts/test-manifest-sync.ps1
# Regression harness for the agent-facing manifest scorecard renderer
# (scripts/manifest-sync.ps1 -> .claude/manifest-bootstrap.md, injected into every
# session by .claude/hooks/sessionstart-digest.sh).
#
# Covers ix#244: the scorecard must distinguish "ran and it is bad" (fresh non-ok
# evidence -> red) from "not evaluated here" (stale evidence -> neutral), and must
# never launder unknown freshness into a healthy-looking line.
#
# Offline and deterministic: the renderer is driven through its -FixturePath seam,
# so no HTTP call is made. Ages are computed relative to the real current time, so
# a 40h-old snapshot is stale and a 1h-old snapshot is fresh on every run.

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$syncScript = Join-Path $PSScriptRoot 'manifest-sync.ps1'
$tempName = "manifest-sync-regression-$([guid]::NewGuid().ToString('N'))"
$tempRelative = "dist/$tempName"
$temp = Join-Path $root $tempRelative

# The 36h threshold is not invented here: it mirrors GA's isMaintainStale default
# (ga/src/dev-data/parsers.ts), so the scorecard and the Prime Radiant tile agree.
$staleAfterHours = 36

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

function New-Domain {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$EmittedAt,
        [switch]$OmitEmittedAt
    )
    $data = [ordered]@{}
    if (-not $OmitEmittedAt) {
        $data['emitted_at'] = $EmittedAt
    }
    $data['metric_value'] = 0.0
    $data['oracle_status'] = $Status
    $data['summary'] = 'metric evidence missing - cannot decide'
    return [ordered]@{ source = 'last.json'; data = $data }
}

function Get-StatusLabel {
    param(
        [Parameter(Mandatory = $true)][string]$Markdown,
        [Parameter(Mandatory = $true)][string]$Domain
    )
    $pattern = '(?m)^### ' + [regex]::Escape($Domain) + ' \((.+)\)\s*$'
    $m = [regex]::Match($Markdown, $pattern)
    Assert-True $m.Success "No scorecard heading rendered for domain '$Domain'."
    return $m.Groups[1].Value
}

function Get-RegressionBlock {
    param([Parameter(Mandatory = $true)][string]$Markdown)
    $m = [regex]::Match($Markdown, '(?s)ACTIVE REGRESSIONS(.*?)(?=\n## )')
    if (-not $m.Success) { return '' }
    return $m.Groups[1].Value
}

function Invoke-Renderer {
    param(
        [Parameter(Mandatory = $true)][string]$FixturePath,
        [Parameter(Mandatory = $true)][string]$OutputRelative
    )
    $output = & pwsh -NoProfile -File $syncScript -FixturePath $FixturePath -OutputPath $OutputRelative 2>&1 | Out-String
    $code = $LASTEXITCODE
    $rendered = Join-Path $root $OutputRelative
    $markdown = if (Test-Path -LiteralPath $rendered) { Get-Content -LiteralPath $rendered -Raw } else { '' }
    return [pscustomobject]@{
        Code     = $code
        Output   = $output
        Markdown = $markdown
    }
}

function Get-AgoStamp {
    param(
        [Parameter(Mandatory = $true)][datetimeoffset]$Now,
        [Parameter(Mandatory = $true)][double]$Hours
    )
    return $Now.AddHours(-$Hours).ToString('o')
}

try {
    New-Item -ItemType Directory -Force -Path $temp | Out-Null

    $now = [datetimeoffset]::UtcNow

    # One manifest, nine domains, one rendering pass. Domain names double as the
    # regression-entry prefixes emitted by ga's gatherQuality().
    $domains = [ordered]@{
        'fresh-warn'     = New-Domain -Status 'warn' -EmittedAt (Get-AgoStamp -Now $now -Hours 1)
        'stale-warn'     = New-Domain -Status 'warn' -EmittedAt (Get-AgoStamp -Now $now -Hours 40)
        'fresh-ok'       = New-Domain -Status 'ok'   -EmittedAt (Get-AgoStamp -Now $now -Hours 1)
        'stale-ok'       = New-Domain -Status 'ok'   -EmittedAt (Get-AgoStamp -Now $now -Hours 40)
        'missing-ts'     = New-Domain -Status 'warn' -OmitEmittedAt
        'malformed-ts'   = New-Domain -Status 'warn' -EmittedAt 'not-a-timestamp'
        'boundary-fresh' = New-Domain -Status 'warn' -EmittedAt (Get-AgoStamp -Now $now -Hours ($staleAfterHours - 1))
        'boundary-stale' = New-Domain -Status 'warn' -EmittedAt (Get-AgoStamp -Now $now -Hours ($staleAfterHours + 1))
        # Byte-for-byte the frozen ga/state/quality/maintain-gate/last.json value:
        # 9 fractional digits, which must still parse rather than fall to "unknown".
        'maintain-gate'  = New-Domain -Status 'warn' -EmittedAt '2026-07-20T10:37:32.623554221+00:00'
    }

    $manifest = [ordered]@{
        repo         = 'GuitarAlchemist/ga'
        generated_at = $now.ToString('o')
        quality      = [ordered]@{
            domains     = $domains
            regressions = @($domains.Keys | ForEach-Object { "${_}: oracle_status=warn" })
        }
        services     = @()
        activity     = @()
    }

    $fixture = Join-Path $temp 'manifest.json'
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $fixture -Encoding UTF8

    $result = Invoke-Renderer -FixturePath $fixture -OutputRelative "$tempRelative/manifest-bootstrap.md"
    Assert-True ($result.Code -eq 0) "Offline render should exit 0. Exit=$($result.Code) Output: $($result.Output)"
    Assert-True ($result.Markdown.Length -gt 0) "Renderer produced no markdown. Output: $($result.Output)"
    Assert-True ($result.Output -notmatch 'Fetching live manifest') '-FixturePath must not touch the network.'

    $md = $result.Markdown
    $regressionBlock = Get-RegressionBlock -Markdown $md

    # --- Case 1: fresh non-ok evidence stays red and stays an active regression ---
    $freshWarn = Get-StatusLabel -Markdown $md -Domain 'fresh-warn'
    Assert-True ($freshWarn -match 'DEGRADED') "Fresh warn must stay DEGRADED, got '$freshWarn'."
    Assert-True ($freshWarn -notmatch 'NOT EVALUATED') "Fresh warn must not be neutralised, got '$freshWarn'."
    Assert-True ($regressionBlock -match 'fresh-warn: oracle_status=warn') 'Fresh warn must remain in ACTIVE REGRESSIONS.'

    # --- Case 2: stale non-ok evidence becomes the third, neutral state ---
    $staleWarn = Get-StatusLabel -Markdown $md -Domain 'stale-warn'
    Assert-True ($staleWarn -match 'NOT EVALUATED') "Stale warn must render NOT EVALUATED, got '$staleWarn'."
    Assert-True ($staleWarn -match 'stale') "Stale warn label must say why it is neutral, got '$staleWarn'."
    Assert-True ($staleWarn -match '\d+\s*[hd]') "Stale warn label must carry the evidence age, got '$staleWarn'."
    Assert-True ($staleWarn -notmatch 'DEGRADED') "Stale warn must not stay red, got '$staleWarn'."
    Assert-True ($staleWarn -notmatch '\bOK\b') "Stale warn must never read as healthy, got '$staleWarn'."
    Assert-True ($regressionBlock -notmatch 'stale-warn: oracle_status=warn') 'Stale domain must be excluded from ACTIVE REGRESSIONS.'

    # --- Case 3: ok stays green (fresh and stale alike) ---
    $freshOk = Get-StatusLabel -Markdown $md -Domain 'fresh-ok'
    Assert-True ($freshOk -match '\bOK\b') "Fresh ok must stay OK, got '$freshOk'."
    $staleOk = Get-StatusLabel -Markdown $md -Domain 'stale-ok'
    Assert-True ($staleOk -match '\bOK\b') "ok must remain green regardless of age, got '$staleOk'."

    # --- Case 4: missing timestamp fails honestly (never silently healthy) ---
    $missing = Get-StatusLabel -Markdown $md -Domain 'missing-ts'
    Assert-True ($missing -match 'DEGRADED') "Missing emitted_at on a warn must stay DEGRADED, got '$missing'."
    Assert-True ($missing -notmatch 'NOT EVALUATED') "Missing emitted_at must not be laundered into NOT EVALUATED, got '$missing'."
    Assert-True ($missing -match 'unknown') "Missing emitted_at must be labelled unknown freshness, got '$missing'."
    Assert-True ($regressionBlock -match 'missing-ts: oracle_status=warn') 'Unknown-freshness domain must remain an active regression.'

    # --- Case 5: malformed timestamp fails honestly too ---
    $malformed = Get-StatusLabel -Markdown $md -Domain 'malformed-ts'
    Assert-True ($malformed -match 'DEGRADED') "Unparseable emitted_at on a warn must stay DEGRADED, got '$malformed'."
    Assert-True ($malformed -notmatch 'NOT EVALUATED') "Unparseable emitted_at must not be laundered, got '$malformed'."
    Assert-True ($malformed -match 'unknown') "Unparseable emitted_at must be labelled unknown freshness, got '$malformed'."
    Assert-True ($regressionBlock -match 'malformed-ts: oracle_status=warn') 'Unparseable-freshness domain must remain an active regression.'

    # --- Case 6: the threshold is exactly GA's 36h precedent, pinned from both sides ---
    $boundaryFresh = Get-StatusLabel -Markdown $md -Domain 'boundary-fresh'
    Assert-True ($boundaryFresh -match 'DEGRADED') "$staleAfterHours h minus 1h must still be fresh, got '$boundaryFresh'."
    $boundaryStale = Get-StatusLabel -Markdown $md -Domain 'boundary-stale'
    Assert-True ($boundaryStale -match 'NOT EVALUATED') "$staleAfterHours h plus 1h must be stale, got '$boundaryStale'."

    # --- Case 7: the real frozen maintain-gate snapshot shape (nanosecond precision) ---
    $maintain = Get-StatusLabel -Markdown $md -Domain 'maintain-gate'
    Assert-True ($maintain -match 'NOT EVALUATED') "Frozen maintain-gate snapshot must render NOT EVALUATED, got '$maintain'."
    Assert-True ($maintain -match '\d+d') "A month-old snapshot should report its age in days, got '$maintain'."
    Assert-True ($regressionBlock -notmatch 'maintain-gate: oracle_status=warn') 'Frozen maintain-gate must not populate ACTIVE REGRESSIONS.'

    # --- Case 8: suppression is visible, not silent ---
    Assert-True ($md -match '(?i)stale evidence') 'Excluded stale domains must be disclosed in the rendered scorecard.'

    Write-Host 'PASS manifest scorecard freshness: fresh-warn red, stale-warn neutral+excluded, ok green, missing/malformed timestamps fail closed, 36h boundary pinned'
}
finally {
    if (Test-Path -LiteralPath $temp) {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
}
