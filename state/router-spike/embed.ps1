# Embed TRAIN/DEV/TEST prompts via Ollama nomic-embed-text, using the router's
# exact normalization (lowercase + trim) so vectors transfer to production.
$ErrorActionPreference = "Stop"
$spike = "C:\Users\spare\source\repos\ix\state\router-spike"
$outDir = Join-Path $spike "embeddings"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Embed([string]$text) {
  $norm = $text.Trim().ToLowerInvariant()
  $body = @{ model = "nomic-embed-text"; prompt = $norm } | ConvertTo-Json -Compress
  $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/embeddings" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
  return ,$resp.embedding
}

# --- TRAIN ---
$train = Get-Content -Raw (Join-Path $spike "train-set.json") | ConvertFrom-Json
$trainRows = New-Object System.Collections.Generic.List[object]
foreach ($g in $train.inScope) {
  foreach ($p in $g.prompts) { $trainRows.Add(@{ label=$g.intentId; prompt=$p; vec=(Embed $p) }) }
}
foreach ($p in $train.outOfScope) { $trainRows.Add(@{ label="__none__"; prompt=$p; vec=(Embed $p) }) }
$trainRows | ConvertTo-Json -Depth 5 -Compress | Out-File -Encoding utf8 (Join-Path $outDir "train.json")
Write-Host "TRAIN embedded: $($trainRows.Count)"

# --- DEV (the existing 86-prompt corpus from ga) ---
$devPath = "C:\Users\spare\source\repos\ga\Tests\Common\GA.Business.ML.Tests\Data\routing-eval-prompts.json"
$dev = Get-Content -Raw $devPath | ConvertFrom-Json
$devRows = New-Object System.Collections.Generic.List[object]
foreach ($p in $dev.prompts) { $devRows.Add(@{ label=$p.expectedIntentId; prompt=$p.prompt; vec=(Embed $p.prompt) }) }
$devRows | ConvertTo-Json -Depth 5 -Compress | Out-File -Encoding utf8 (Join-Path $outDir "dev.json")
Write-Host "DEV embedded: $($devRows.Count)"

# --- TEST (held-out) ---
$test = Get-Content -Raw (Join-Path $spike "heldout-test.json") | ConvertFrom-Json
$testRows = New-Object System.Collections.Generic.List[object]
foreach ($p in $test.prompts) { $testRows.Add(@{ label=$p.expectedIntentId; prompt=$p.prompt; vec=(Embed $p.prompt) }) }
$testRows | ConvertTo-Json -Depth 5 -Compress | Out-File -Encoding utf8 (Join-Path $outDir "test.json")
Write-Host "TEST embedded: $($testRows.Count)"
Write-Host "DONE"
