#requires -Version 7.4
[CmdletBinding(SupportsShouldProcess,ConfirmImpact='High')]
param(
 [Parameter(Mandatory)][string]$Candidate,
 [Parameter(Mandatory)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$CandidateSha256,
 [Parameter(Mandatory)][string]$StateDir,
 [Parameter(Mandatory)][ValidatePattern('^[a-z0-9][a-z0-9-]{0,62}$',Options='None')][string]$InstanceName,
 [string]$InstalledExe='C:\Program Files\ixctl\ixctl.exe',
 [Parameter(Mandatory)][string]$ExpectedInstalledSha256,
 [string]$ExpectedImagePath,
 [string]$OwnerSid,
 [Parameter(Mandatory)][string]$OwnerDesktop,
 [switch]$Apply
)
$ErrorActionPreference='Stop';$PSNativeCommandUseErrorActionPreference=$true
$dataDir=Join-Path ([IO.Path]::GetFullPath($StateDir)) 'workshop'
$configPath=Join-Path $dataDir 'config.json'
if(!$Apply){[pscustomobject]@{Service='ixctl';Account='NT SERVICE\ixctl';Candidate=$Candidate;CandidateSha256=$CandidateSha256;InstalledExe=$InstalledExe;ExpectedInstalledSha256=$ExpectedInstalledSha256;ExpectedImagePath=$ExpectedImagePath;Config=$configPath;RouteChanged=$false;Apply=$false};return}
if(!$ExpectedImagePath -or $ExpectedImagePath -match '(?i)workshop-config|bearer|token|password'){throw 'Require the reviewed original service command, without a Workshop flag or credential.'}
if($OwnerSid -notmatch '^S-1-[0-9-]+$'){throw 'Require the verified human owner SID for credential ACL validation.'}
$shortcutPath=Join-Path $OwnerDesktop 'Administrer IxForge.lnk'
$launcher=Join-Path (Split-Path $InstalledExe -Parent) 'Open-IxForgeOwner.ps1'
if(Test-Path -LiteralPath $launcher){throw 'An owner launcher already exists; do not overwrite it during an initial release.'}
if(!(Test-Path -LiteralPath $OwnerDesktop -PathType Container) -or (Test-Path -LiteralPath $shortcutPath)){throw 'Require the reviewed owner Desktop and an unused IxForge shortcut path.'}
if(!$PSCmdlet.ShouldProcess('ixctl / '+$InstalledExe,'Back up the old binary, stop this service, install the reviewed candidate, enable IxForge and restart; no Tailscale change')){return}
$service=Get-CimInstance Win32_Service -Filter "Name='ixctl'"
if($service.PathName -cne $ExpectedImagePath -or $service.StartName -ine 'NT SERVICE\ixctl'){throw 'Service identity or command differs from the reviewed snapshot.'}
if($service.State -ne 'Running'){throw 'The reviewed initial release expects the current service to be Running.'}
if((Get-FileHash -LiteralPath $Candidate -Algorithm SHA256).Hash -ine $CandidateSha256){throw 'Candidate binary hash mismatch.'}
if((Get-FileHash -LiteralPath $InstalledExe -Algorithm SHA256).Hash -ine $ExpectedInstalledSha256){throw 'Installed binary changed since review.'}
if(!(Test-Path -LiteralPath (Join-Path $dataDir ($InstanceName+'.workshop-owner.key')))){throw 'Explicit restricted credential provisioning must happen first.'}
$ownerKeyPath=Join-Path $dataDir ($InstanceName+'.workshop-owner.key')
$serviceSid=([Security.Principal.NTAccount]'NT SERVICE\ixctl').Translate([Security.Principal.SecurityIdentifier]).Value
$allowedSids=@($OwnerSid,$serviceSid,'S-1-5-18','S-1-5-32-544')
foreach($path in @($dataDir,$ownerKeyPath)){
 if((Get-Item -LiteralPath $path).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Workshop data and credential must be local regular paths.'}
 $acl=Get-Acl -LiteralPath $path
 foreach($rule in $acl.Access){if($rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -notin $allowedSids){throw 'Workshop credential/data ACL grants an unapproved identity.'}}
 if($acl.Access.Count -eq 0){throw 'Empty/unproven credential ACL.'}
}
$config=Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if($config.enabled -or $config.version -ne 1){throw 'Initial release expects a reviewed disabled version-1 configuration.'}
$backup=Join-Path $dataDir ('release-'+(Get-Date -Format 'yyyyMMdd-HHmmss-ffff'))
New-Item -ItemType Directory -Path $backup | Out-Null
$ExpectedImagePath | Set-Content -LiteralPath (Join-Path $backup 'original-image-path.txt') -Encoding utf8
Copy-Item -LiteralPath $configPath -Destination (Join-Path $backup 'config.json')
$changed=$false;$launcherTouched=$false;$shortcutTouched=$false
try{
 Stop-Service -Name ixctl -NoWait;(Get-Service ixctl).WaitForStatus('Stopped',[TimeSpan]::FromSeconds(60))
 Copy-Item -LiteralPath $InstalledExe -Destination (Join-Path $backup 'ixctl.exe')
 $ExpectedInstalledSha256|Set-Content -LiteralPath (Join-Path $backup 'original-sha256.txt') -Encoding utf8
 if((Get-FileHash -LiteralPath (Join-Path $backup 'ixctl.exe')).Hash -ine $ExpectedInstalledSha256){throw 'Binary backup does not match the reviewed original.'}
 $database=Join-Path $dataDir ($InstanceName+'.workshop.sqlite')
 if(Test-Path -LiteralPath ($database+'-wal')){if((Get-Item -LiteralPath ($database+'-wal')).Length -gt 0){throw 'Workshop WAL is not drained; do not copy a live database.'}}
 if(Test-Path -LiteralPath $database){Copy-Item -LiteralPath $database -Destination (Join-Path $backup ($InstanceName+'.workshop.sqlite'))}
 $changed=$true;Copy-Item -LiteralPath $Candidate -Destination $InstalledExe -Force
 $config.enabled=$true;$config|ConvertTo-Json|Set-Content -LiteralPath $configPath -Encoding utf8
 $image=$ExpectedImagePath+' --workshop-config "'+$configPath+'"'
 $image|Set-Content -LiteralPath (Join-Path $backup 'new-image-path.txt') -Encoding utf8
 & sc.exe config ixctl binPath= $image
 Start-Service ixctl;(Get-Service ixctl).WaitForStatus('Running',[TimeSpan]::FromSeconds(60))
 if(!(Test-Path -LiteralPath (Join-Path $dataDir ($InstanceName+'.workshop.json')))){throw 'No IxForge ready-state file after service start.'}
 $launcherStream=[IO.File]::Open($launcher,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None);$launcherTouched=$true
 try{$source=[IO.File]::OpenRead((Join-Path $PSScriptRoot 'Open-IxForgeOwner.ps1'));try{$source.CopyTo($launcherStream);$launcherStream.Flush($true)}finally{$source.Dispose()}}finally{$launcherStream.Dispose()}
 $launcher|Set-Content -LiteralPath (Join-Path $backup 'launcher-path.txt') -Encoding utf8
 (Get-FileHash -LiteralPath $launcher).Hash|Set-Content -LiteralPath (Join-Path $backup 'launcher-sha256.txt') -Encoding utf8
 $temporaryShortcut=Join-Path $backup 'owner-shortcut-staged.lnk'
 $shell=New-Object -ComObject WScript.Shell;$shortcut=$shell.CreateShortcut($temporaryShortcut)
 $shortcut.TargetPath=(Get-Command pwsh -ErrorAction Stop).Source
 $shortcut.Arguments='-NoProfile -WindowStyle Hidden -File "'+$launcher+'" -StateDir "'+$StateDir+'" -InstanceName "'+$InstanceName+'"'
 $shortcut.Description='Administrer IxForge sur ce Desktop';$shortcut.Save();[IO.File]::Move($temporaryShortcut,$shortcutPath,$false);$shortcutTouched=$true
 $shortcutPath|Set-Content -LiteralPath (Join-Path $backup 'shortcut-path.txt') -Encoding utf8
 (Get-FileHash -LiteralPath $shortcutPath).Hash|Set-Content -LiteralPath (Join-Path $backup 'shortcut-sha256.txt') -Encoding utf8
 [pscustomobject]@{Service='ixctl';Backup=$backup;LocalReady=$true;TailscaleChanged=$false;OwnerLauncher=Join-Path (Split-Path $InstalledExe -Parent) 'Open-IxForgeOwner.ps1'}
}catch{
 try{if($shortcutTouched -and (Test-Path -LiteralPath $shortcutPath)){Move-Item -LiteralPath $shortcutPath -Destination (Join-Path $backup 'owner-shortcut-failed.lnk')}}catch{Write-Warning 'New shortcut cleanup failed; continue restoring the service.'}
 try{if($launcherTouched -and (Test-Path -LiteralPath $launcher)){Move-Item -LiteralPath $launcher -Destination (Join-Path $backup 'owner-launcher-failed.ps1')}}catch{Write-Warning 'New launcher cleanup failed; continue restoring the service.'}
 if($changed){
  Stop-Service ixctl -NoWait -ErrorAction SilentlyContinue;(Get-Service ixctl).WaitForStatus('Stopped',[TimeSpan]::FromSeconds(60))
  Copy-Item -LiteralPath (Join-Path $backup 'ixctl.exe') -Destination $InstalledExe -Force
  Copy-Item -LiteralPath (Join-Path $backup 'config.json') -Destination $configPath -Force
  & sc.exe config ixctl binPath= $ExpectedImagePath
  Start-Service ixctl
 }else{Start-Service ixctl -ErrorAction SilentlyContinue}
 throw
}
