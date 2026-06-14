# run-adversarial-panel.ps1 — scheduled runner for the ix adversarial LLM panel.
#
# WHY local, not hosted CI: the panel must query the LIVE GA chatbot (Ollama +
# OPTIC-K + GaChatbot.Api on :5252). A hosted GitHub runner has none of those,
# so a cron CI job would degrade every single time (green-but-dead). This
# wrapper runs where the backend actually lives — this machine, or a
# self-hosted runner with the stack warm.
#
# Register it on a schedule (pick one):
#   - Claude Code:   /schedule  (cron a remote agent that runs this)
#   - Windows:       schtasks /Create /SC DAILY /TN "ix-adversarial-panel" /TR "pwsh -File <abs path>"
#   - Manual:        pwsh Scripts/run-adversarial-panel.ps1
#
# It (1) preflights the chatbot, (2) runs the /ix-adversarial-llm-panel dynamic
# workflow headless via `claude -p`, (3) refreshes the cross-repo fleet feed so
# the unified dashboard updates. Honest-degrades (no fabricated grade) if the
# backend is down.

[CmdletBinding()]
param(
    [string]$ChatbotUrl = "http://localhost:5252",
    [int]$Limit = 0   # 0 = full LLM-deferred tier; >0 caps for a quick run
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
$parent   = Split-Path $repoRoot -Parent
$gaRoot   = Join-Path $parent "ga"

Write-Host "── ix adversarial LLM panel — scheduled run ──" -ForegroundColor Cyan

# 1. Backend preflight — refuse to run (don't fabricate) if the chatbot is down.
$healthy = $false
try {
    $r = Invoke-WebRequest -Uri "$ChatbotUrl/api/chatbot/status" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    $healthy = ($r.StatusCode -eq 200)
} catch { $healthy = $false }

if (-not $healthy) {
    Write-Host "Chatbot unreachable at $ChatbotUrl — skipping panel (no fabricated grade)." -ForegroundColor Yellow
    Write-Host "Bring it up (ga/docs/runbooks/chatbot-deploy.md) then re-run." -ForegroundColor Yellow
    # Still refresh the fleet feed so the dashboard shows the current (stale) state honestly.
    if (Test-Path (Join-Path $gaRoot "Scripts/afk-fleet-aggregate.ps1")) {
        pwsh -NoProfile -File (Join-Path $gaRoot "Scripts/afk-fleet-aggregate.ps1")
    }
    exit 2   # 2 = environment-degraded, snapshot not written (matches the ecosystem convention)
}

# 2. Run the dynamic workflow headless. Requires the `claude` CLI on PATH and
#    dynamic workflows enabled (/config). args cap the prompt count if -Limit>0.
$argsJson = if ($Limit -gt 0) { "{ ""limit"": $Limit, ""host"": ""$ChatbotUrl"" }" } else { "{ ""host"": ""$ChatbotUrl"" }" }
$prompt = "Run the saved workflow /ix-adversarial-llm-panel with args $argsJson. It grades the expected_check=llm corpus tier against the live chatbot and writes state/adversarial/llm-panel-<date>.json."
Write-Host "Running panel via claude -p ..." -ForegroundColor White
Push-Location $repoRoot
try {
    claude -p $prompt --allowedTools "Bash,Read,Write,Glob,Grep"
    $code = $LASTEXITCODE
} finally { Pop-Location }
if ($code -ne 0) { Write-Host "claude -p exited $code" -ForegroundColor Yellow }

# 3. Refresh the cross-repo fleet feed so the unified dashboard reflects this run.
if (Test-Path (Join-Path $gaRoot "Scripts/afk-fleet-aggregate.ps1")) {
    pwsh -NoProfile -File (Join-Path $gaRoot "Scripts/afk-fleet-aggregate.ps1")
}
Write-Host "Done. Latest snapshot: ix/state/adversarial/llm-panel-*.json" -ForegroundColor Green
