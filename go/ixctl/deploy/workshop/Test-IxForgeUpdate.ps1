#requires -Version 7.4
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'Update-IxForge.Core.ps1')
$root=Join-Path ([IO.Path]::GetTempPath()) ('ixforge-update-fixture-'+[Guid]::NewGuid().ToString('N'))
$root=[IO.Path]::GetFullPath($root);$allowed=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
if((Split-Path $root -Parent) -cne $allowed -or !(Split-Path $root -Leaf).StartsWith('ixforge-update-fixture-')){throw 'Unsafe fixture path.'}
$results=[Collections.Generic.List[string]]::new()
function Assert($condition,[string]$why){if(!$condition){throw $why}}
function Expect-Refusal([scriptblock]$action,[string]$why){$failed=$false;try{& $action|Out-Null}catch{$failed=$true};Assert $failed $why}
# These service/network adapters exist only inside this test process. No SCM or HTTP mutation is called.
function Test-IxForgeUpdateAdministrator {$true}
function Get-IxForgeUpdateService {[pscustomobject]@{State=$script:serviceState;PathName=$script:command;StartName='NT SERVICE\ixctl';ProcessId=$script:fixturePID}}
function Stop-IxForgeUpdateService {$script:stops++;$script:serviceState='Stopped'}
function Start-IxForgeUpdateService {$script:starts++;if($script:failCandidateStart -and (Get-FileHash -LiteralPath $script:installed).Hash -ceq $script:fixturePlanArgs.CandidateSHA256){throw 'Synthetic candidate startup failure'};$script:serviceState='Running';$script:fixturePID++}
function Get-IxForgeUpdateBinaryVersion([string]$Path){if($Path -ceq $script:candidate){'0.1.0-fixture.3'}else{'0.1.0-fixture.2'}}
function Get-IxForgeUpdatePublic([string]$URL,[string]$HostValue=''){if($URL -eq 'http://127.0.0.1:59999/workshop/branding.json'){return @{Status=200;Body='{"version":"0.1.0"}';SHA256='fixture-brand'}};Assert ($URL -eq 'https://workshop.invalid/') 'Unexpected fixture network URL';@{Status=200;Body='Scribe fixture';SHA256='fixture-scribe-unchanged'}}
function Test-IxForgeUpdatedApplication($Plan){$script:healthChecks++;return !($script:failHealth -and (Get-FileHash -LiteralPath $script:installed).Hash -ceq $Plan.CandidateSHA256)}
function Reset-Fixture {
 $script:caseRoot=Join-Path $root ([Guid]::NewGuid().ToString('N'));$state=Join-Path $script:caseRoot 'state';$data=Join-Path $state 'workshop'
 New-Item -ItemType Directory -Path $data|Out-Null
 $script:installed=Join-Path $script:caseRoot 'ixctl.exe';$script:candidate=Join-Path $script:caseRoot 'candidate.exe'
 [IO.File]::WriteAllText($script:installed,'old fixture binary');[IO.File]::WriteAllText($script:candidate,'new fixture binary')
 [IO.File]::WriteAllText((Join-Path $state 'spec.json'),'fixture spec')
 @{version=1;enabled=$true;origin='https://workshop.invalid';listen='127.0.0.1:59999'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $data 'config.json') -Encoding utf8
 $script:fixturePID=123;$script:serviceState='Running';$script:command='"'+$script:installed+'" service run "fixture-spec" --workshop-config "fixture-config"'
 @{pid=$script:fixturePID;addr='127.0.0.1:59999';schema_version=1}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $data 'fixture.workshop.json') -Encoding utf8
 foreach($file in @('fixture.workshop.sqlite','fixture.workshop-owner.key')){[IO.File]::WriteAllText((Join-Path $data $file),'synthetic retained data')}
 $script:stops=0;$script:starts=0;$script:healthChecks=0;$script:failHealth=$false;$script:failCandidateStart=$false
 $script:fixturePlanArgs=@{Candidate=$script:candidate;CandidateSHA256=(Get-FileHash $script:candidate).Hash;ExpectedInstalledSHA256=(Get-FileHash $script:installed).Hash;ExpectedImagePath=$script:command;StateDir=$state;InstanceName='fixture';ExpectedClientVersion='0.1.1';InstalledExe=$script:installed;ExpectedSpecSHA256=(Get-FileHash (Join-Path $state 'spec.json')).Hash}
}
function Assert-Retained($Plan){foreach($file in @('fixture.workshop.sqlite','fixture.workshop-owner.key')){Assert ([IO.File]::ReadAllText((Join-Path $Plan.DataDir $file)) -ceq 'synthetic retained data') ('Retained data changed: '+$file)};Assert ((Get-FileHash $Plan.ConfigPath).Hash -ceq $Plan.ConfigSHA256) 'Config changed';Assert ($script:command -ceq $Plan.ExpectedImagePath) 'Service command changed'}
try{
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs
 Assert ($plan.State -eq 'available' -and $plan.IntegrityVerified -and !$plan.SignatureVerified -and !$plan.Automatic) 'Wrong plan capability claims'
 Assert ($script:stops -eq 0 -and $script:starts -eq 0 -and !$plan.BackupDirectory) 'Plan caused mutation';$results.Add('PASS active/available versions and hash integrity; plan is inert, no signature/auto-update claim')
 $bad=$script:fixturePlanArgs.Clone();$bad.CandidateSHA256='0'*64;Expect-Refusal {New-IxForgeUpdatePlan @bad} 'Bad artifact accepted';Assert ($script:stops -eq 0) 'Stopped for bad artifact';$results.Add('PASS tampered candidate refused before service stop')
 $same=$script:fixturePlanArgs.Clone();$same.CandidateSHA256=$same.ExpectedInstalledSHA256;Expect-Refusal {New-IxForgeUpdatePlan @same} 'Same-version restart admitted';Assert ($script:stops -eq 0) 'Restarted the same artifact';$results.Add('PASS already-installed artifact refused without restart')
 $plan=Invoke-IxForgeUpdate $plan
 Assert ($plan.State -eq 'completed' -and $script:stops -eq 1 -and $script:starts -eq 1 -and (Get-FileHash $script:installed).Hash -ceq $plan.CandidateSHA256) 'Apply failed'
 Assert ((Get-FileHash (Join-Path $plan.BackupDirectory 'previous.exe')).Hash -ceq $plan.PreviousSHA256) 'No verified backup';Assert-Retained $plan
 $disk=Get-Content -LiteralPath (Join-Path $plan.BackupDirectory 'receipt.json') -Raw|ConvertFrom-Json;Assert ($disk.State -eq 'completed') 'No durable completion receipt';$results.Add('PASS backup, atomic replacement, restart, health and durable receipt; data/key/config/command retained')
 $plan=Undo-IxForgeUpdate $plan.BackupDirectory $plan.CandidateSHA256 $plan.PreviousSHA256
 Assert ($plan.State -eq 'rolled_back' -and (Get-FileHash $script:installed).Hash -ceq $plan.PreviousSHA256) 'Rollback failed';Assert-Retained $plan
 $stops=$script:stops;Undo-IxForgeUpdate $plan.BackupDirectory $plan.CandidateSHA256 $plan.PreviousSHA256|Out-Null;Assert ($stops -eq $script:stops) 'Rollback replay restarted twice';$results.Add('PASS explicit rollback and idempotent already-restored replay')
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:failHealth=$true;Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Failed health accepted'
 Assert ((Get-FileHash $script:installed).Hash -ceq $plan.PreviousSHA256 -and $script:serviceState -eq 'Running') 'Automatic failure restoration failed';Assert-Retained $plan
 $disk=Get-Content -LiteralPath (Join-Path $plan.BackupDirectory 'receipt.json') -Raw|ConvertFrom-Json;Assert ($disk.State -eq 'rolled_back') 'Failure rollback not recorded';$results.Add('PASS failed health restores prior executable without rewinding data')
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:failCandidateStart=$true;Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Failed startup accepted';Assert ((Get-FileHash $script:installed).Hash -ceq $plan.PreviousSHA256 -and $script:serviceState -eq 'Running') 'Startup failure did not restore old service';Assert-Retained $plan;$results.Add('PASS failed candidate startup restores the prior running version')
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:command+=' --foreign-change';Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Foreign owner command overwritten';Assert ($script:stops -eq 0) 'Stopped after owner changed';$results.Add('PASS changed owner/service command refused before mutation')
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$lock=[IO.File]::Open((Join-Path $plan.DataDir '.workshop.update.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
 try{Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Concurrent update admitted';Assert ($script:stops -eq 0) 'Concurrent stop performed'}finally{$lock.Dispose()};$results.Add('PASS concurrent updater excluded before stop')
 Reset-Fixture;$plan=Invoke-IxForgeUpdate (New-IxForgeUpdatePlan @script:fixturePlanArgs);[IO.File]::WriteAllText((Join-Path $plan.BackupDirectory 'previous.exe'),'tampered backup');$stops=$script:stops
 Expect-Refusal {Undo-IxForgeUpdate $plan.BackupDirectory $plan.CandidateSHA256 $plan.PreviousSHA256} 'Tampered backup restored';Assert ($stops -eq $script:stops) 'Stopped for corrupt backup';$results.Add('PASS corrupt rollback backup refused before stop')
}finally{
 if(Test-Path -LiteralPath $root){Remove-Item -LiteralPath $root -Recurse -Force}
 Assert (!(Test-Path -LiteralPath $root)) 'Fixture cleanup incomplete'
}
$results;$results.Count.ToString()+' synthetic update tests passed; fixture cleaned; no production service/network/data touched'
