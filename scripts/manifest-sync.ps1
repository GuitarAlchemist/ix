# Scripts/manifest-sync.ps1
# Pulls the active dev-data manifest from GuitarAlchemist dev servers and boots/aligns project context.

param(
    [string]$Url = "https://demos.guitaralchemist.com/dev-data/manifest",
    [string]$LocalUrl = "http://localhost:5176/dev-data/manifest",
    [string]$OutputPath = ".claude/manifest-bootstrap.md"
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

# 1. Fetch JSON manifest
$manifest = $null
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
if ($quality -and $quality.psobject.Properties["domains"]) {
    foreach ($domainName in $quality.domains.psobject.properties.name) {
        $domain = $quality.domains.$domainName
        $source = Get-SafeProperty $domain "source"
        $data = Get-SafeProperty $domain "data"
        $status = Get-SafeProperty $data "oracle_status"
        
        $statusColor = if ($status -eq 'ok') { "🟢 OK" } else { "🔴 DEGRADED" }
        
        $md += "### $domainName ($statusColor)`n"
        $md += "- **Source:** $source`n"
        
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

# Add Regressions
if ($regressions.Count -gt 0) {
    $md += "### ⚠️ ACTIVE REGRESSIONS`n"
    foreach ($reg in $regressions) {
        $md += "- $reg`n"
    }
    $md += "`n"
} else {
    $md += "### 🟢 Regressions: None detected`n`n"
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
