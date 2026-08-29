# Scripts/manifest-sync.ps1
# Pulls the active dev-data manifest from GuitarAlchemist dev servers and boots/aligns project context.

param(
    [string]$Url = "https://demos.guitaralchemist.com/dev-data/manifest",
    [string]$LocalUrl = "http://localhost:5176/dev-data/manifest",
    [string]$OutputPath = ".claude/manifest-bootstrap.md",
    # Offline seam: render a manifest read from disk instead of fetching one.
    # Exercised by scripts/test-manifest-sync.ps1; no network call is made.
    [string]$FixturePath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== Syncing Ecosystem Manifest ===" -ForegroundColor Cyan

function Get-SafeProperty {
    param($obj, $propName)
    if ($obj -and $obj.psobject.Properties[$propName]) {
        return $obj.$propName
    }
    return $null
}

# Freshness threshold for quality snapshots, in hours. Deliberately the same
# number as GA's isMaintainStale default (ga/src/dev-data/parsers.ts), so the
# agent-facing scorecard and the human-facing Prime Radiant tile agree about
# what "stale" means. Do not diverge without changing both.
$StaleAfterHours = 36

function Format-EvidenceAge {
    param([Parameter(Mandatory = $true)][timespan]$Age)
    $hours = [math]::Max(0, [math]::Floor($Age.TotalHours))
    if ($hours -lt 48) { return "${hours}h" }
    return "$([math]::Floor($Age.TotalDays))d"
}

# Classifies one quality domain from its own emitted_at.
#   ok                              -> healthy, whatever its age
#   non-ok + fresh evidence         -> degraded (a real, current failure)
#   non-ok + evidence older than    -> not evaluated (the producer skipped here;
#     $StaleAfterHours                 neutral, and not an active regression)
#   non-ok + missing/unparseable    -> degraded, freshness unknown (fail closed:
#     emitted_at                       unknown freshness must never suppress a
#                                      failure or read as healthy)
function Get-DomainFreshness {
    param($data)

    $status = Get-SafeProperty $data "oracle_status"
    $emittedRaw = Get-SafeProperty $data "emitted_at"

    $age = $null
    $ageText = $null
    $emittedUtc = $null
    $parseNote = $null
    if (-not $emittedRaw) {
        $parseNote = "no emitted_at in snapshot"
    } elseif ($emittedRaw -is [datetimeoffset]) {
        # Invoke-RestMethod / ConvertFrom-Json hydrate ISO-8601 timestamps into
        # date objects rather than strings. Use them directly: round-tripping
        # them through [string] loses the offset and skews the age by the local
        # UTC offset.
        $emittedUtc = $emittedRaw.ToUniversalTime()
    } elseif ($emittedRaw -is [datetime]) {
        $emittedUtc = ([datetimeoffset]$emittedRaw).ToUniversalTime()
    } else {
        $parsed = [datetimeoffset]::MinValue
        if ([datetimeoffset]::TryParse(
                [string]$emittedRaw,
                [cultureinfo]::InvariantCulture,
                [System.Globalization.DateTimeStyles]::AssumeUniversal,
                [ref]$parsed)) {
            $emittedUtc = $parsed.ToUniversalTime()
        } else {
            $parseNote = "unparseable emitted_at '$emittedRaw'"
        }
    }

    if ($null -ne $emittedUtc) {
        $age = [datetimeoffset]::UtcNow - $emittedUtc
        $ageText = Format-EvidenceAge -Age $age
    }

    if ($status -eq 'ok') {
        $label = "🟢 OK"
        $stale = $false
    } elseif ($null -eq $age) {
        $label = "🔴 DEGRADED, freshness unknown"
        $stale = $false
    } elseif ($age.TotalHours -gt $StaleAfterHours) {
        $label = "⚪ NOT EVALUATED, stale $ageText"
        $stale = $true
    } else {
        $label = "🔴 DEGRADED"
        $stale = $false
    }

    if ($ageText) {
        $freshness = "emitted $($emittedUtc.ToString('yyyy-MM-ddTHH:mm:ssZ')) ($ageText ago)"
    } else {
        $freshness = "unknown - $parseNote"
    }

    return [pscustomobject]@{
        Label     = $label
        Stale     = $stale
        AgeText   = $ageText
        Freshness = $freshness
    }
}

# 1. Fetch JSON manifest
$manifest = $null
if ($FixturePath) {
    if (-not (Test-Path -LiteralPath $FixturePath)) {
        Write-Host "  ERROR: Fixture manifest not found at $FixturePath." -ForegroundColor Red
        exit 1
    }
    Write-Host "Reading offline manifest fixture from $FixturePath..." -ForegroundColor Gray
    $manifest = Get-Content -LiteralPath $FixturePath -Raw | ConvertFrom-Json
} else {
    try {
        Write-Host "Fetching live manifest from $Url..." -ForegroundColor Gray
        $resp = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 10
        $manifest = $resp
        Write-Host "  Successfully fetched live manifest." -ForegroundColor Green
    } catch {
        Write-Host "  Live server fetch failed, trying local Vite dev server at $LocalUrl..." -ForegroundColor Yellow
        try {
            $resp = Invoke-RestMethod -Uri $LocalUrl -Method Get -TimeoutSec 5
            $manifest = $resp
            Write-Host "  Successfully fetched local dev manifest." -ForegroundColor Green
        } catch {
            Write-Host "  ERROR: Failed to connect to both live and local dev-data endpoints. Ensure Vite dev server is running." -ForegroundColor Red
            exit 1
        }
    }
}

if (-not $manifest) {
    Write-Host "  ERROR: Fetched manifest is empty." -ForegroundColor Red
    exit 1
}

# 2. Extract values
$repo = Get-SafeProperty $manifest "repo"
$generatedAt = Get-SafeProperty $manifest "generated_at"
$backlog = Get-SafeProperty $manifest "backlog"
$quality = Get-SafeProperty $manifest "quality"
$activity = Get-SafeProperty $manifest "activity"
$services = Get-SafeProperty $manifest "services"

# Check for regressions or failures
$regressions = @()
if ($quality -and $quality.psobject.Properties["regressions"]) {
    $regressions = $quality.regressions
}

# 3. Construct a beautiful Markdown report
$md = @"
# Ecosystem Manifest Bootstrap

**Ecosystem:** GuitarAlchemist
**Source Repository:** $repo
**Fetched At:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
**Manifest Generation Time:** $generatedAt

---

## 🚦 System Health & Quality Scorecard

"@

# Add quality domains
$staleDomains = @()
if ($quality -and $quality.psobject.Properties["domains"]) {
    foreach ($domainName in $quality.domains.psobject.properties.name) {
        $domain = $quality.domains.$domainName
        $source = Get-SafeProperty $domain "source"
        $data = Get-SafeProperty $domain "data"

        $freshness = Get-DomainFreshness $data
        if ($freshness.Stale) {
            $staleDomains += [pscustomobject]@{ Name = $domainName; AgeText = $freshness.AgeText }
        }

        $md += "### $domainName ($($freshness.Label))`n"
        $md += "- **Source:** $source`n"
        $md += "- **Freshness:** $($freshness.Freshness)`n"

        $metricVal = Get-SafeProperty $data "metric_value"
        if ($metricVal -ne $null) {
            $md += "- **Metric Value:** $metricVal`n"
        }
        
        $totalFailures = Get-SafeProperty $data "totalFailures"
        $totalWarnings = Get-SafeProperty $data "totalWarnings"
        if ($totalFailures -ne $null -or $totalWarnings -ne $null) {
            $md += "- **Failures/Warnings:** $($totalFailures ?? 0) fail(s), $($totalWarnings ?? 0) warn(s)`n"
        }
        
        $summary = Get-SafeProperty $data "summary"
        if ($summary) {
            $md += "- **Summary:** $summary`n"
        }
        $md += "`n"
    }
}

# Add Regressions.
# A regression entry is "<domain>: <detail>". Entries whose domain rendered as
# NOT EVALUATED are dropped: stale evidence is not an active regression, it is an
# absence of evidence. The suppression is disclosed below so it is never silent.
$activeRegressions = @()
foreach ($reg in $regressions) {
    $regText = [string]$reg
    $isStale = $false
    foreach ($sd in $staleDomains) {
        if ($regText.StartsWith("$($sd.Name):")) { $isStale = $true; break }
    }
    if (-not $isStale) { $activeRegressions += $regText }
}

if ($activeRegressions.Count -gt 0) {
    $md += "### ⚠️ ACTIVE REGRESSIONS`n"
    foreach ($reg in $activeRegressions) {
        $md += "- $reg`n"
    }
    $md += "`n"
} else {
    $md += "### 🟢 Regressions: None detected`n`n"
}

if ($staleDomains.Count -gt 0) {
    $excluded = ($staleDomains | ForEach-Object { "$($_.Name) ($($_.AgeText))" }) -join ', '
    $md += "_Not counted as regressions - stale evidence, older than ${StaleAfterHours}h: $excluded._`n`n"
}

# Add Service Ports
$md += @"
## 🌐 Active Services & Dev Ports

| Service | Port | Public Path | Expected Behavior |
|---|---|---|---|
"@
if ($services) {
    foreach ($svc in $services) {
        $name = Get-SafeProperty $svc "name"
        $port = Get-SafeProperty $svc "port"
        $path = Get-SafeProperty $svc "public_path"
        $exp = Get-SafeProperty $svc "expected"
        $md += "`n| $name | $port | $path | $exp |"
    }
}

$md += "`n`n"

# Add Backlog Progress
if ($backlog) {
    $progress = Get-SafeProperty $backlog "overall_progress_pct"
    $totalShipped = Get-SafeProperty $backlog "total_shipped"
    $totalItems = Get-SafeProperty $backlog "total_items"
    $totalEpics = Get-SafeProperty $backlog "total_epics"
    
    $md += @"
## 📋 Project Backlog Progress

**Overall Progress:** $progress% Shipped ($totalShipped of $totalItems items across $totalEpics epics)

| Epic | Shipped | Active | Backlog | Progress |
|---|---|---|---|---|
"@

    $epics = Get-SafeProperty $backlog "epics"
    if ($epics) {
        foreach ($epic in $epics) {
            $title = Get-SafeProperty $epic "title"
            $shipped = Get-SafeProperty $epic "shipped"
            $active = Get-SafeProperty $epic "active"
            $bklg = Get-SafeProperty $epic "backlog"
            $prog = Get-SafeProperty $epic "progress_pct"
            $md += "`n| $title | $shipped | $active | $bklg | $prog% |"
        }
    }
}

$md += "`n`n"

# Add Recent Activity
$md += @"
## 🕒 Recent Commit Activity

| Commit | Author | Date | Subject |
|---|---|---|---|
"@

if ($activity) {
    foreach ($act in $activity) {
        $sha = Get-SafeProperty $act "short_sha"
        $author = Get-SafeProperty $act "author"
        $date = Get-SafeProperty $act "date"
        $sub = Get-SafeProperty $act "subject"
        $md += "`n| $sha | $author | $date | $sub |"
    }
}

# Save output file
$outputPathFull = Join-Path (Split-Path -Parent $PSScriptRoot) $OutputPath
$outputDir = Split-Path -Parent $outputPathFull
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$md | Set-Content -Path $outputPathFull -Encoding UTF8
Write-Host "Manifest synced and saved to $outputPathFull" -ForegroundColor Green
Write-Host "=== Coordination Sync Complete ===" -ForegroundColor Cyan
