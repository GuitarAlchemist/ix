#requires -Version 7.4
# Installs, once, the updater task that lets an admin device update the Desktop from the app.
#
# Without -Apply it prints its plan as JSON and changes nothing. With -Apply (elevated, on the owner's explicit order):
#   - copies Run-IxForgeUpdate.ps1, Update-IxForge.ps1 and Update-IxForge.Core.ps1 from this checkout to UpdaterDir
#     (under Program Files: only administrators write there) and checks each copy's hash against the plan;
#   - creates %ProgramData%\IxForge with an explicit ACL: releases written by the owner (and administrators), read by the
#     service (NT SERVICE\ixctl) and SYSTEM; staging for SYSTEM and administrators only;
#   - registers the task "IxForge Updater": pwsh from Program Files, as SYSTEM, at startup and every minute, one
#     instance at a time.
# -Uninstall removes the task and UpdaterDir; the releases stay.
[CmdletBinding()]
param(
 [Parameter(Mandatory)][ValidatePattern('^S-1-5-21-[0-9-]+$')][string]$OwnerSid,
 [string]$UpdaterDir = (Join-Path $env:ProgramFiles 'ixctl\updater'),
 [string]$Data = (Join-Path $env:ProgramData 'IxForge'),
 [string]$ServiceAccount = 'NT SERVICE\ixctl',
 [switch]$Apply,
 [switch]$Uninstall
)
$ErrorActionPreference = 'Stop'
$task = 'IxForge Updater'
$names = 'Run-IxForgeUpdate.ps1', 'Update-IxForge.ps1', 'Update-IxForge.Core.ps1'
$files = [ordered]@{}
foreach ($n in $names) { $files[$n] = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $n) -Algorithm SHA256).Hash.ToUpperInvariant() }
$pwsh = (Get-Command pwsh -CommandType Application | Select-Object -First 1).Source
$releases = Join-Path $Data 'releases'; $staging = Join-Path $Data 'staging'
$plan = [ordered]@{
  Task = $task; Principal = 'SYSTEM'; Every = 'PT1M'; Pwsh = $pwsh; UpdaterDir = $UpdaterDir
  Releases = $releases; Staging = $staging
  ReleaseWriters = "$OwnerSid, S-1-5-32-544 (Administrators); readers: $ServiceAccount, S-1-5-18 (SYSTEM)"
  Files = $files; Apply = [string]([bool]$Apply).ToString().ToLowerInvariant(); Uninstall = [string]([bool]$Uninstall).ToString().ToLowerInvariant()
}
if (!$Apply) { $plan | ConvertTo-Json -Compress; return }

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Installing the updater requires Windows administrator elevation.' }
if ($Uninstall) {
  if (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $task -Confirm:$false }
  if (Test-Path -LiteralPath $UpdaterDir) { Remove-Item -LiteralPath $UpdaterDir -Recurse -Force }
  "UNINSTALLED: task and $UpdaterDir removed; $releases kept."
  return
}
if (!$pwsh -or !$pwsh.StartsWith($env:ProgramFiles, [StringComparison]::OrdinalIgnoreCase)) { throw "pwsh must come from Program Files (found '$pwsh'): SYSTEM would run it." }
if (!$UpdaterDir.StartsWith($env:ProgramFiles, [StringComparison]::OrdinalIgnoreCase)) { throw 'UpdaterDir must be under Program Files, where only administrators write.' }

# The scripts SYSTEM runs: copied, then checked against the hashes reviewed in the plan.
New-Item -ItemType Directory -Force -Path $UpdaterDir | Out-Null
foreach ($n in $names) {
  $copy = Join-Path $UpdaterDir $n
  Copy-Item -LiteralPath (Join-Path $PSScriptRoot $n) -Destination $copy -Force
  if ((Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash -ne $files[$n]) { throw "$n changed while it was copied." }
}

# %ProgramData%\IxForge: explicit ACLs, no inheritance from ProgramData (where every user may create folders).
function Set-ExplicitAcl([string]$Path, [object[]]$Rules) {
  New-Item -ItemType Directory -Force -Path $Path | Out-Null
  $acl = New-Object Security.AccessControl.DirectorySecurity
  $acl.SetAccessRuleProtection($true, $false)
  $acl.SetOwner([Security.Principal.SecurityIdentifier]'S-1-5-32-544')
  foreach ($r in $Rules) {
    $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($r[0], $r[1], 'ContainerInherit,ObjectInherit', 'None', 'Allow')))
  }
  Set-Acl -LiteralPath $Path -AclObject $acl
}
$system = [Security.Principal.SecurityIdentifier]'S-1-5-18'
$admins = [Security.Principal.SecurityIdentifier]'S-1-5-32-544'
$owner = [Security.Principal.SecurityIdentifier]$OwnerSid
$service = ([Security.Principal.NTAccount]$ServiceAccount).Translate([Security.Principal.SecurityIdentifier])
Set-ExplicitAcl $Data @(@($system, 'FullControl'), @($admins, 'FullControl'), @($owner, 'ReadAndExecute'), @($service, 'ReadAndExecute'))
Set-ExplicitAcl $releases @(@($system, 'FullControl'), @($admins, 'FullControl'), @($owner, 'Modify'), @($service, 'ReadAndExecute'))
Set-ExplicitAcl $staging @(@($system, 'FullControl'), @($admins, 'FullControl'))

$action = New-ScheduledTaskAction -Execute $pwsh -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + (Join-Path $UpdaterDir 'Run-IxForgeUpdate.ps1') + '"')
$every = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1)
$boot = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$runAs = New-ScheduledTaskPrincipal -UserId 'S-1-5-18' -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName $task -Action $action -Trigger @($every, $boot) -Settings $settings -Principal $runAs -Description 'IxForge: installs a release an admin device asked for from the app (Run-IxForgeUpdate.ps1).' -Force | Out-Null
"INSTALLED: task '$task' as SYSTEM every minute; scripts in $UpdaterDir; releases in $releases."
