#requires -Version 7.4
[CmdletBinding(SupportsShouldProcess,ConfirmImpact='High')]
param(
 [Parameter(Mandatory)][string]$BackupDir,
 [Parameter(Mandatory)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedCurrentSha256,
 [string]$InstalledExe='C:\Program Files\ixctl\ixctl.exe',
 [switch]$Apply
)
$ErrorActionPreference='Stop';$PSNativeCommandUseErrorActionPreference=$true
if(!$Apply){[pscustomobject]@{Service='ixctl';Backup=$BackupDir;InstalledExe=$InstalledExe;PreserveWorkshopData=$true;TailscaleChanged=$false;Apply=$false};return}
if(!$PSCmdlet.ShouldProcess('ixctl / '+$InstalledExe,'Restore the reviewed previous binary and service command; retain Workshop data and credentials; route removal is separate')){return}
$original=(Get-Content -LiteralPath (Join-Path $BackupDir 'original-image-path.txt') -Raw).Trim()
$expected=(Get-Content -LiteralPath (Join-Path $BackupDir 'new-image-path.txt') -Raw).Trim()
$service=Get-CimInstance Win32_Service -Filter "Name='ixctl'"
if($service.PathName -cne $expected -or $service.StartName -ine 'NT SERVICE\ixctl'){throw 'Service changed after the reviewed release; do not overwrite another owner.'}
if((Get-FileHash -LiteralPath $InstalledExe -Algorithm SHA256).Hash -ine $ExpectedCurrentSha256){throw 'Current binary differs from the reviewed candidate.'}
$dataDir=Split-Path $BackupDir -Parent;$configPath=Join-Path $dataDir 'config.json'
$originalHash=(Get-Content -LiteralPath (Join-Path $BackupDir 'original-sha256.txt') -Raw).Trim()
if($originalHash -notmatch '^[0-9a-fA-F]{64}$' -or (Get-FileHash -LiteralPath (Join-Path $BackupDir 'ixctl.exe')).Hash -ine $originalHash){throw 'Backup binary integrity is not proven; service remains unchanged.'}
$previousConfig=Get-Content -LiteralPath (Join-Path $BackupDir 'config.json') -Raw | ConvertFrom-Json
if($previousConfig.version -ne 1 -or $previousConfig.enabled){throw 'Backup configuration is not the disabled initial-release configuration.'}
Stop-Service ixctl -NoWait;(Get-Service ixctl).WaitForStatus('Stopped',[TimeSpan]::FromSeconds(60))
Copy-Item -LiteralPath (Join-Path $BackupDir 'ixctl.exe') -Destination $InstalledExe -Force
Copy-Item -LiteralPath (Join-Path $BackupDir 'config.json') -Destination $configPath -Force
& sc.exe config ixctl binPath= $original
Start-Service ixctl;(Get-Service ixctl).WaitForStatus('Running',[TimeSpan]::FromSeconds(60))
$shortcutFile=Join-Path $BackupDir 'shortcut-path.txt'
if(Test-Path -LiteralPath $shortcutFile){$shortcut=(Get-Content -LiteralPath $shortcutFile -Raw).Trim();if((Split-Path $shortcut -Leaf) -ne 'Administrer IxForge.lnk'){throw 'Unrecognized shortcut; retained without changing it.'};$hash=(Get-Content -LiteralPath (Join-Path $BackupDir 'shortcut-sha256.txt') -Raw).Trim();if((Test-Path -LiteralPath $shortcut) -and (Get-FileHash -LiteralPath $shortcut).Hash -ieq $hash){Move-Item -LiteralPath $shortcut -Destination (Join-Path $BackupDir 'Administrer IxForge.lnk')}}
$launcherFile=Join-Path $BackupDir 'launcher-path.txt'
if(Test-Path -LiteralPath $launcherFile){$launcher=(Get-Content -LiteralPath $launcherFile -Raw).Trim();if($launcher -cne (Join-Path (Split-Path $InstalledExe -Parent) 'Open-IxForgeOwner.ps1')){throw 'Unrecognized launcher retained.'};$hash=(Get-Content -LiteralPath (Join-Path $BackupDir 'launcher-sha256.txt') -Raw).Trim();if((Test-Path -LiteralPath $launcher) -and (Get-FileHash -LiteralPath $launcher).Hash -ieq $hash){Move-Item -LiteralPath $launcher -Destination (Join-Path $BackupDir 'Open-IxForgeOwner.ps1')}}
[pscustomobject]@{Restored=$true;WorkshopDatabaseRetained=$true;CredentialsRetained=$true;TailscaleChanged=$false}
