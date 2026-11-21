#requires -Version 7.4
# Puts a release of ixctl built on this PC in the releases folder, where the Desktop offers it to its admin devices
# (Devices drawer, "Mise à jour du Desktop"). Run under the owner's account; no elevation.
#
#   pwsh deploy/workshop/Publish-IxForgeRelease.ps1 -Version 0.1.0-ixforge.17 -Notes "What changes"
#       builds go/ixctl from this checkout (it must be committed: the manifest names its commit)
#   pwsh deploy/workshop/Publish-IxForgeRelease.ps1 -Version 0.1.0-ixforge.17 -Exe <built ixctl.exe> -Commit <sha>
#       publishes a build made elsewhere
#
# The release folder is <Releases>\<Version>\{ixctl.exe, release.json}. A published version is never overwritten.
[CmdletBinding()]
param(
 [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+-ixforge\.\d{1,6}$')][string]$Version,
 [string]$Notes = '',
 [string]$Exe,
 [ValidatePattern('^[0-9a-f]{7,40}$')][string]$Commit,
 [string]$Source = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
 [string]$Releases = (Join-Path $env:ProgramData 'IxForge\releases')
)
$ErrorActionPreference = 'Stop'; $PSNativeCommandUseErrorActionPreference = $true

$final = Join-Path $Releases $Version
if (Test-Path -LiteralPath $final) { throw "Version $Version is already published in $final; a published version is never overwritten." }
New-Item -ItemType Directory -Force -Path $Releases | Out-Null
$work = Join-Path $Releases ('.publishing-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
try {
  $built = Join-Path $work 'ixctl.exe'
  if ($Exe) {
    if (!$Commit) { throw 'A build made elsewhere needs -Commit, the commit it was built from.' }
    Copy-Item -LiteralPath $Exe -Destination $built
  } else {
    if (git -C $Source status --porcelain) { throw "The checkout $Source has uncommitted changes: commit them, so the release names what it is built from." }
    $Commit = git -C $Source rev-parse HEAD
    $env:CGO_ENABLED = '0'
    go -C $Source build -trimpath -ldflags "-X main.version=$Version" -o $built .
  }
  $says = (& $built version) -join ' '
  if ($says -notmatch [regex]::Escape($Version) + '(\s|$)') { throw "The build says its version is '$says', not $Version." }
  $branding = Get-Content -LiteralPath (Join-Path $Source 'internal\workshop\web\branding.json') -Raw | ConvertFrom-Json
  $manifest = [ordered]@{
    version        = $Version
    sha256         = (Get-FileHash -LiteralPath $built -Algorithm SHA256).Hash.ToUpperInvariant()
    commit         = $Commit
    client_version = [string]$branding.version
    built_at       = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    notes          = $Notes
  }
  $manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $work 'release.json') -Encoding utf8NoBOM
  Move-Item -LiteralPath $work -Destination $final
  "Published $Version ($($manifest.sha256)) from $Commit in $final"
} finally {
  if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force }
}
