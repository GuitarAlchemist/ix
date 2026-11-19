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
function Stop-IxForgeUpdateService {$script:stops++;$script:serviceState='Stopped';if($script:onStop){& $script:onStop}}
function Start-IxForgeUpdateService {if($script:beforeStart){& $script:beforeStart};$script:starts++;$script:startedHashes.Add($(if(Test-Path -LiteralPath $script:installed){(Get-FileHash -LiteralPath $script:installed).Hash}else{'missing'}));if($script:failCandidateStart -and (Get-FileHash -LiteralPath $script:installed).Hash -ceq $script:fixturePlanArgs.CandidateSHA256){throw 'Synthetic candidate startup failure'};$script:serviceState='Running';$script:fixturePID++}
function Get-IxForgeUpdateBinaryVersion([string]$Path){if($Path -ceq $script:candidate){'0.1.0-fixture.3'}else{'0.1.0-fixture.2'}}
function Get-IxForgeUpdatePublic([string]$URL,[string]$HostValue=''){if($URL -eq 'http://127.0.0.1:59999/workshop/branding.json'){return @{Status=200;Body='{"version":"0.1.0"}';SHA256='fixture-brand'}};Assert ($URL -eq 'https://workshop.invalid/') 'Unexpected fixture network URL';@{Status=200;Body='Scribe fixture';SHA256='fixture-scribe-unchanged'}}
function Test-IxForgeUpdatedApplication($Plan){$script:healthChecks++;$h=(Get-FileHash -LiteralPath $script:installed).Hash;if($h -ceq $Plan.PreviousSHA256){$script:previousHealthVersions.Add([string]$Plan.ExpectedClientVersion)};return !($script:failHealth -and $h -ceq $Plan.CandidateSHA256) -and !($script:failPreviousHealth -and $h -ceq $Plan.PreviousSHA256)}
# Another process acting on the installed executable after the stop or after the backup (fixture files only).
$script:saveReceipt=${function:Save-IxForgeUpdateReceipt}
function Save-IxForgeUpdateReceipt($Plan){& $script:saveReceipt $Plan;if($Plan.State -eq 'backed_up' -and $script:onBackedUp){& $script:onBackedUp}}
$script:holders=[Collections.Generic.List[IDisposable]]::new()
function Hold-Installed {$script:holders.Add([IO.File]::Open($script:installed,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read))}
function Release-Installed {foreach($h in $script:holders){$h.Dispose()};$script:holders.Clear()}
function Get-DiskState($Plan){(Get-ChildItem -LiteralPath $Plan.DataDir -Directory -Filter 'update-*'|ForEach-Object{Get-Content -LiteralPath (Join-Path $_.FullName 'receipt.json') -Raw|ConvertFrom-Json}).State}
# Neighbour adapters: one fixed observed neighbour, so neither the plan nor the pre-stop baseline reads this machine.
function Get-IxForgeServeStatusJson {if($script:neighboursObservable){'{"TCP":{"443":{"HTTPS":true}},"Web":{"workshop.invalid:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:59998"},"/workshop":{"Proxy":"http://127.0.0.1:59999/workshop"}}}}}'}else{$null}}
function Get-IxForgeNeighbourOwner([int]$Port){Assert ($Port -eq 59998) ('Unexpected neighbour port '+$Port);[pscustomobject]@{Listening=$true;PID=77;Image='C:\fixture\neighbour.exe'}}
function Get-IxForgeNeighbourStatus([string]$URL){Assert ($URL -eq 'http://127.0.0.1:59998/') ('Unexpected neighbour probe '+$URL);200}
function Reset-Fixture {
 Release-Installed;$script:neighboursObservable=$true;$script:onStop=$null;$script:onBackedUp=$null;$script:beforeStart=$null;$script:startedHashes=[Collections.Generic.List[string]]::new()
 $script:caseRoot=Join-Path $root ([Guid]::NewGuid().ToString('N'));$state=Join-Path $script:caseRoot 'state';$data=Join-Path $state 'workshop'
 New-Item -ItemType Directory -Path $data|Out-Null
 $script:installed=Join-Path $script:caseRoot 'ixctl.exe';$script:candidate=Join-Path $script:caseRoot 'candidate.exe'
 [IO.File]::WriteAllText($script:installed,'old fixture binary');[IO.File]::WriteAllText($script:candidate,'new fixture binary')
 [IO.File]::WriteAllText((Join-Path $state 'spec.json'),'fixture spec')
 @{version=1;enabled=$true;origin='https://workshop.invalid';listen='127.0.0.1:59999'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $data 'config.json') -Encoding utf8
 $script:fixturePID=123;$script:serviceState='Running';$script:command='"'+$script:installed+'" service run "fixture-spec" --workshop-config "fixture-config"'
 @{pid=$script:fixturePID;addr='127.0.0.1:59999';schema_version=1}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $data 'fixture.workshop.json') -Encoding utf8
 foreach($file in @('fixture.workshop.sqlite','fixture.workshop-owner.key')){[IO.File]::WriteAllText((Join-Path $data $file),'synthetic retained data')}
 $script:stops=0;$script:starts=0;$script:healthChecks=0;$script:failHealth=$false;$script:failCandidateStart=$false;$script:failPreviousHealth=$false;$script:previousHealthVersions=[Collections.Generic.List[string]]::new()
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
 Reset-Fixture;$script:neighboursObservable=$false;Expect-Refusal {New-IxForgeUpdatePlan @script:fixturePlanArgs} 'Unobservable neighbours admitted by the plan';$results.Add('PASS unobservable neighbours refuse the plan')
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:neighboursObservable=$false;Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Unobservable neighbours admitted at stop';Assert ($script:stops -eq 0) 'Stopped with unknown neighbours';$results.Add('PASS unobservable neighbours refuse the update before any stop')
 # A failed replacement during a normal update: the service may only restart on the verified previous executable.
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:onStop={[IO.File]::WriteAllText($script:installed,'foreign binary')}
 Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Foreign executable accepted';Release-Installed
 Assert ($script:starts -eq 0 -and $script:serviceState -eq 'Stopped') ('Unknown executable restarted: '+($script:startedHashes -join ','));Assert ((Get-DiskState $plan) -ceq 'reconciliation_required') 'Unknown executable not recorded for reconciliation'
 Assert ([IO.File]::ReadAllText($script:installed) -ceq 'foreign binary') 'Foreign executable overwritten';$results.Add('PASS executable replaced by another process after the stop: never restarted, reconciliation_required')
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:onBackedUp={Hold-Installed}
 Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Held executable replaced';Release-Installed
 Assert ($script:starts -eq 1 -and $script:startedHashes[0] -ceq $plan.PreviousSHA256 -and $script:serviceState -eq 'Running') 'Verified previous executable not restarted';Assert ((Get-DiskState $plan) -ceq 'failed_before_replacement') 'Held replacement mis-recorded';Assert-Retained $plan
 Assert ($script:previousHealthVersions.Count -eq 1 -and $script:previousHealthVersions[0] -ceq $plan.PreviousClientVersion) ('Restarted previous executable not health-checked as the previous version: '+($script:previousHealthVersions -join ','));$results.Add('PASS replacement blocked by a holder: restarts only the verified previous executable, then checks its health')
 # The previous executable restarts but is not healthy: no failed_before_replacement verdict, reconciliation instead.
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:onBackedUp={Hold-Installed};$script:failPreviousHealth=$true
 Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Held executable replaced';Release-Installed
 Assert ($script:startedHashes.Count -eq 1 -and $script:startedHashes[0] -ceq $plan.PreviousSHA256) 'Not the verified previous executable';Assert ($script:previousHealthVersions.Count -ge 1) 'Previous executable never health-checked'
 Assert ((Get-DiskState $plan) -ceq 'reconciliation_required') ('Unhealthy previous executable recorded as '+(Get-DiskState $plan));$results.Add('PASS unhealthy previous executable after a failed replacement: reconciliation_required, no reprise verdict')
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:onBackedUp={[IO.File]::Delete($script:installed)}
 Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Missing executable accepted'
 Assert ((Test-Path -LiteralPath $script:installed) -and (Get-FileHash $script:installed).Hash -ceq $plan.PreviousSHA256) 'Verified previous executable not put back';Assert ($script:starts -eq 1 -and $script:startedHashes[0] -ceq $plan.PreviousSHA256) ('Started: '+($script:startedHashes -join ','));Assert ((Get-DiskState $plan) -ceq 'failed_before_replacement') 'Removed executable mis-recorded';$results.Add('PASS executable removed before replacement: verified backup put back, then restarted')
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:onBackedUp={[IO.File]::WriteAllText($script:installed,'foreign binary');Hold-Installed}
 Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Foreign held executable accepted';Release-Installed
 Assert ($script:starts -eq 0 -and $script:serviceState -eq 'Stopped') ('Unknown executable restarted: '+($script:startedHashes -join ','));Assert ((Get-DiskState $plan) -ceq 'reconciliation_required') 'Unknown executable not recorded for reconciliation';$results.Add('PASS unknown executable after backup: never restarted, reconciliation_required')
 # Race between the SHA check and the start: another process swaps the executable after the check. The start cannot be
 # prevented by the updater; the verdict must not claim the reviewed version, and the receipt must say reconciliation.
 $swap={param($from)if((Get-FileHash -LiteralPath $script:installed).Hash -ceq $from){Release-Installed;[IO.File]::WriteAllText($script:installed,'swapped binary')}}
 $race=[Collections.Generic.List[string]]::new()
 try{Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:onBackedUp={Hold-Installed};$script:beforeStart={& $swap $plan.PreviousSHA256}.GetNewClosure()
 Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Swapped executable accepted after a failed replacement'
 Assert ((Get-DiskState $plan) -ceq 'reconciliation_required') ('Swap before the restart of the previous executable recorded as '+(Get-DiskState $plan));$results.Add('PASS swap between SHA check and restart after a failed update: reconciliation_required')}catch{$race.Add([string]$_)}
 try{
 Reset-Fixture;$plan=New-IxForgeUpdatePlan @script:fixturePlanArgs;$script:beforeStart={& $swap $plan.CandidateSHA256}.GetNewClosure()
 Expect-Refusal {Invoke-IxForgeUpdate $plan} 'Swapped candidate accepted'
 Assert ((Get-DiskState $plan) -ceq 'reconciliation_required') ('Swap before the candidate start recorded as '+(Get-DiskState $plan));$results.Add('PASS swap between replacement and candidate start: reconciliation_required')}catch{$race.Add([string]$_)}
 try{
 Reset-Fixture;$plan=Invoke-IxForgeUpdate (New-IxForgeUpdatePlan @script:fixturePlanArgs);$script:beforeStart={& $swap $plan.PreviousSHA256}.GetNewClosure()
 Expect-Refusal {Undo-IxForgeUpdate $plan.BackupDirectory $plan.CandidateSHA256 $plan.PreviousSHA256} 'Swapped executable accepted by rollback'
 Assert ((Get-DiskState $plan) -ceq 'reconciliation_required') ('Swap before the rollback restart recorded as '+(Get-DiskState $plan));$results.Add('PASS swap between rollback replacement and restart: reconciliation_required')}catch{$race.Add([string]$_)}
 Assert ($race.Count -eq 0) ('Race cases failed: '+($race -join ' | '))
}finally{
 Release-Installed
 if(Test-Path -LiteralPath $root){Remove-Item -LiteralPath $root -Recurse -Force}
 Assert (!(Test-Path -LiteralPath $root)) 'Fixture cleanup incomplete'
}
$results;$results.Count.ToString()+' synthetic update tests passed; fixture cleaned; no production service/network/data touched'
