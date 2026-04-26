param(
    [Parameter(Mandatory = $true)]
    [string]$SnapshotsDir,

    [string]$OutDir = "state/quality",

    [double]$RegressionThresholdPct = 5.0
)

$date = Get-Date -Format "yyyy-MM-dd"
$reportPath = Join-Path $OutDir "quality-trend-$date.md"
$artifactPath = Join-Path $OutDir "quality-health-$date.json"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

cargo run -p ix-quality-trend --bin ix-quality-trend -- `
    --snapshots-dir $SnapshotsDir `
    --out $reportPath `
    --out-json $artifactPath `
    --regression-threshold-pct $RegressionThresholdPct

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Markdown report: $reportPath"
Write-Host "Health artifact: $artifactPath"
