# Jev arm, Stage 1 — the ONLY networked step. Sends jev/requests.jsonl verbatim
# (the exact bytes hashed in plan.json) to TypeSafe, one request per prompt,
# and writes a receipt for `jev-router score`. Guards (pre-registered in RESULTS.md):
#   - key from TYPESAFE_API_KEY (process, else User scope); never printed
#   - JEV_ROUTER_APPROVED=YES required on top of the key
#   - exactly one attempt per request, no retries, no redirects followed
#   - stop after any response once cumulative REPORTED input tokens x rate card
#     exceeds the budget (a post-response stop, not a billing cap)
# Usage (from the ix root): pwsh state/router-spike/jev/run-live.ps1
param(
  [string]$Out = "state/router-spike/jev/live.receipt.jsonl",
  [double]$BudgetUsd = 0.05
)
$ErrorActionPreference = "Stop"
$Api = "https://api.typesafe.ai/v1/systemone"
$Model = "jev-1.13.0"
$PricePerMillion = 0.042

$key = $env:TYPESAFE_API_KEY
if (-not $key) { $key = [Environment]::GetEnvironmentVariable("TYPESAFE_API_KEY", "User") }
if (-not $key) { throw "TYPESAFE_API_KEY is not set" }
if ($env:JEV_ROUTER_APPROVED -ne "YES") { throw "set JEV_ROUTER_APPROVED=YES after operator approval" }

$plan = Get-Content state/router-spike/jev/plan.json -Raw | ConvertFrom-Json -AsHashtable
$lines = Get-Content state/router-spike/jev/requests.jsonl -Encoding utf8
if ($lines.Count -ne $plan.calls) { throw "requests.jsonl has $($lines.Count) lines, plan says $($plan.calls) — re-run plan" }
if (Test-Path $Out) { throw "$Out exists; refusing to append to an old receipt" }

$sha = [System.Security.Cryptography.SHA256]::Create()
$utf8 = [System.Text.UTF8Encoding]::new($false)
$inputTokens = 0
$sent = 0
foreach ($line in $lines) {
  # Re-extract the request text exactly as `plan` wrote it: {"id":..,"request":<body>}
  $id = ($line | ConvertFrom-Json).id
  $body = $line.Substring($line.IndexOf('"request":') + 10, $line.Length - $line.IndexOf('"request":') - 11)
  $bytes = $utf8.GetBytes($body)
  $digest = -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") })
  if ($digest -ne $plan.request_sha256[$id]) { throw "digest mismatch for $id — requests.jsonl is not the planned body" }

  $t0 = [Diagnostics.Stopwatch]::StartNew()
  $status = $null; $resp = $null; $err = $null
  try {
    $r = Invoke-WebRequest -Uri $Api -Method Post -Body $bytes -ContentType "application/json; charset=utf-8" `
      -Headers @{ Authorization = "Bearer $key" } -MaximumRedirection 0 -TimeoutSec 30 -SkipHttpErrorCheck
    $status = [int]$r.StatusCode
    if ($status -eq 200) { $resp = $r.Content | ConvertFrom-Json -AsHashtable } else { $err = "http $status" }
  } catch { $err = $_.Exception.GetType().Name }
  $sent++
  $receipt = [ordered]@{ id = $id; request_sha256 = $digest; http_status = $status; latency_ms = $t0.ElapsedMilliseconds; error = $err; response = $resp }
  Add-Content -Path $Out -Value ($receipt | ConvertTo-Json -Depth 10 -Compress) -Encoding utf8

  $tok = if ($resp -and $resp.usage) { $resp.usage.input_tokens } else { $null }
  if ($tok -is [long] -or $tok -is [int]) { $inputTokens += [long]$tok }
  elseif ($status -eq 200) { Write-Warning "no reported usage for $id — stopping (cost unknown)"; break }
  $spent = $inputTokens / 1e6 * $PricePerMillion
  if ($spent -gt $BudgetUsd) { Write-Warning "budget stop after $sent calls: `$$spent > `$$BudgetUsd"; break }
  if ($status -in 401, 403, 429) { Write-Warning "stopping on http $status after $sent calls"; break }
}
"sent $sent / $($plan.calls) calls; reported input tokens $inputTokens; rate-card USD {0:N6}" -f ($inputTokens / 1e6 * $PricePerMillion)
