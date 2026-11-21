#requires -Version 7.4
# The updater task's runner. Install-IxForgeUpdater.ps1 runs it as SYSTEM every minute, once the owner installed it.
#
# Each run writes a heartbeat (updater-seen.json) that the Desktop shows to its admin devices. When an admin device has
# asked for a release from the app (update-request.json, written by the service), the runner:
#   1. takes the request (moved away first, so it is never run twice: an uncertain outcome is reported, not retried);
#   2. checks the release folder again (name, manifest, hash), copies ixctl.exe to the staging folder, where only SYSTEM
#      and administrators write, and checks the copy's hash against the one the device was shown;
#   3. calls the reviewed Update-IxForge.ps1 with -Apply and the PC's current installed hash, spec hash and service
#      command (it backs up, replaces, restarts, checks health and the neighbours, and restores on failure);
#   4. writes update-result.json with the receipt's state.
# The browser never supplies a command or a path: only a version and a hash, both checked here.
[CmdletBinding()]
param(
 [string]$Requests = 'C:\ProgramData\ixctl\workshop\updates',
 [string]$Releases = (Join-Path $env:ProgramData 'IxForge\releases'),
 [string]$Staging = (Join-Path $env:ProgramData 'IxForge\staging'),
 [string]$Updater = (Join-Path $PSScriptRoot 'Update-IxForge.ps1'),
 [string]$InstalledExe = 'C:\Program Files\ixctl\ixctl.exe',
 [string]$SpecPath = 'C:\ProgramData\ixctl\spec.json',
 [string]$ReceiptsDir = 'C:\ProgramData\ixctl\workshop',
 [string]$ServiceName = 'ixctl',
 [string]$ExpectedImagePath
)
$ErrorActionPreference = 'Stop'

function Write-Atomic([string]$Path, $Value) {
  $tmp = "$Path.tmp"
  $Value | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $tmp -Encoding utf8NoBOM
  Move-Item -LiteralPath $tmp -Destination $Path -Force
}
function Now { (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }
function Get-Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant() }

if (!(Test-Path -LiteralPath $Requests -PathType Container)) { return } # the Desktop has not started with updates yet
Write-Atomic (Join-Path $Requests 'updater-seen.json') @{seen_at = Now}

$request = Join-Path $Requests 'update-request.json'
if (!(Test-Path -LiteralPath $request -PathType Leaf)) { return }
New-Item -ItemType Directory -Force -Path $Staging | Out-Null
$run = 'run-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$work = Join-Path $Staging $run
New-Item -ItemType Directory -Path $work | Out-Null
$taken = Join-Path $work 'request.json'
Move-Item -LiteralPath $request -Destination $taken

$version = ''
$result = [ordered]@{version = ''; state = 'failed_before_replacement'; receipt = ''; finished_at = ''; error = ''}
$started = Get-Date
try {
  $asked = Get-Content -LiteralPath $taken -Raw | ConvertFrom-Json
  $version = [string]$asked.version; $result.version = $version
  $hash = ([string]$asked.sha256).ToUpperInvariant()
  if ($version -notmatch '^\d+\.\d+\.\d+-ixforge\.\d{1,6}$' -or $hash -notmatch '^[0-9A-F]{64}$') { throw 'request_invalid: version or hash malformed' }
  $folder = Join-Path $Releases $version
  $manifestPath = Join-Path $folder 'release.json'
  if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "release_missing: no release $version" }
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  if ($manifest.version -ne $version -or ([string]$manifest.sha256).ToUpperInvariant() -ne $hash) { throw 'hash_mismatch: the manifest no longer names this release and hash' }
  $candidate = Join-Path $work 'ixctl.exe'
  Copy-Item -LiteralPath (Join-Path $folder 'ixctl.exe') -Destination $candidate
  if ((Get-Sha $candidate) -ne $hash) { throw 'hash_mismatch: the release file changed after it was offered' }

  if (!$ExpectedImagePath) { $ExpectedImagePath = (Get-CimInstance Win32_Service -Filter "Name='$ServiceName'").PathName }
  if (!$ExpectedImagePath) { throw "service_missing: no service $ServiceName" }
  $reviewed = @{
    Candidate               = $candidate
    CandidateSHA256         = $hash
    ExpectedInstalledSHA256 = Get-Sha $InstalledExe
    ExpectedSpecSHA256      = Get-Sha $SpecPath
    ExpectedImagePath       = $ExpectedImagePath
    ExpectedClientVersion   = [string]$manifest.client_version
    InstalledExe            = $InstalledExe
  }
  $result.state = 'failed'
  & $Updater @reviewed -Apply -Confirm:$false *>&1 | Out-File -LiteralPath (Join-Path $work 'update.log') -Encoding utf8NoBOM
  $result.state = 'unknown'
} catch {
  $result.error = $_.Exception.Message
} finally {
  # The receipt written during this run says what happened, whatever the updater returned.
  $receipt = Get-ChildItem -LiteralPath $ReceiptsDir -Directory -Filter 'update-*' -ErrorAction SilentlyContinue |
    ForEach-Object { Get-Item -LiteralPath (Join-Path $_.FullName 'receipt.json') -ErrorAction SilentlyContinue } |
    Where-Object { $_.LastWriteTime -ge $started } | Sort-Object LastWriteTime | Select-Object -Last 1
  if ($receipt) {
    $result.receipt = $receipt.Directory.Name
    $state = (Get-Content -LiteralPath $receipt.FullName -Raw | ConvertFrom-Json).State
    if ($state) { $result.state = [string]$state }
  }
  $result.finished_at = Now
  Write-Atomic (Join-Path $Requests 'update-result.json') $result
  Write-Atomic (Join-Path $Requests 'updater-seen.json') @{seen_at = Now}
  Remove-Item -LiteralPath (Join-Path $work 'ixctl.exe') -Force -ErrorAction SilentlyContinue
}
