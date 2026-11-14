#requires -Version 7.4
[CmdletBinding(SupportsShouldProcess,ConfirmImpact='High')]
param(
 [string]$Candidate,[string]$CandidateSHA256,[string]$ExpectedInstalledSHA256,[string]$ExpectedImagePath,
 [string]$StateDir='C:\ProgramData\ixctl',[string]$InstanceName='ixctl-trial',[string]$ExpectedClientVersion='0.1.1',
 [string]$InstalledExe='C:\Program Files\ixctl\ixctl.exe',[string]$ExpectedSpecSHA256,
 [switch]$Apply,[switch]$Rollback,[string]$BackupDirectory
)
$ErrorActionPreference='Stop';$PSNativeCommandUseErrorActionPreference=$true
. (Join-Path $PSScriptRoot 'Update-IxForge.Core.ps1')
if($Apply -and $Rollback){throw 'Choose update or rollback.'}
if($Rollback){
 if(!$BackupDirectory){throw 'Require the reviewed update backup.'}
 if($PSCmdlet.ShouldProcess('ixctl','Restore the reviewed previous executable, retain data/configuration/access and restart the same service')){
  Undo-IxForgeUpdate $BackupDirectory $CandidateSHA256 $ExpectedInstalledSHA256
 };return
}
$plan=New-IxForgeUpdatePlan -Candidate $Candidate -CandidateSHA256 $CandidateSHA256 -ExpectedInstalledSHA256 $ExpectedInstalledSHA256 -ExpectedImagePath $ExpectedImagePath -StateDir $StateDir -InstanceName $InstanceName -ExpectedClientVersion $ExpectedClientVersion -InstalledExe $InstalledExe -ExpectedSpecSHA256 $ExpectedSpecSHA256
if(!$Apply){$plan;return}
if($PSCmdlet.ShouldProcess('ixctl','Back up the exact current executable, atomically install the reviewed candidate, restart and verify; preserve data, keys, routes and configuration')){Invoke-IxForgeUpdate $plan}
