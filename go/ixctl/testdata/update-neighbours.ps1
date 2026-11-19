#requires -Version 7.4
# Harness for workshop_update_neighbours_test.go. It runs the REAL neighbour checks, health check and rollback of
# deploy/workshop/Update-IxForge.Core.ps1. Only two observation adapters (Tailscale Serve JSON, port owner) read the
# fixture of the current stage, "before" until the service is stopped and "after" from then on; the HTTP probe is real
# and only reaches the Go test's loopback servers. Every file lives inside -Dir, which must carry the fixture marker.
param(
 [Parameter(Mandatory)][ValidateSet('verdict','rollback')][string]$Mode,
 [Parameter(Mandatory)][string]$Origin,
 [Parameter(Mandatory)][string]$Listen,
 [Parameter(Mandatory)][string]$Dir,
 # rollback only: receipt variant (plain, old, stale, truncated, empty, schema2, bogus-state, partial:<Field>), which
 # binary is installed when the rollback (re)starts, and whether the service is running then (interrupted rollback).
 [string]$Receipt='plain',
 [ValidateSet('candidate','previous')][string]$Binary='candidate',
 [ValidateSet('Running','Stopped')][string]$ServiceState='Running',
 # rollback only: what another process does to the installed executable once the service is stopped. 'hold' keeps it
 # open without delete sharing (an antivirus or a stray process), 'remove' deletes it; either makes the replacement fail.
 [ValidateSet('none','hold','remove')][string]$OnStop='none'
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '..\deploy\workshop\Update-IxForge.Core.ps1')
$Dir=[IO.Path]::GetFullPath($Dir)
if(!(Test-Path -LiteralPath (Join-Path $Dir 'ixforge-neighbours-fixture'))){throw 'Refusing a directory that is not a test fixture.'}
function Set-Stage([string]$Name){$script:stage=$Name;[IO.File]::WriteAllText((Join-Path $Dir 'stage'),$Name)}
Set-Stage 'before'
function Get-IxForgeServeStatusJson {$f=Join-Path $Dir ($script:stage+'-serve.json');if(Test-Path -LiteralPath $f){[IO.File]::ReadAllText($f)}else{$null}}
function Get-IxForgeNeighbourOwner([int]$Port){
 $f=Join-Path $Dir ($script:stage+'-owners.json')
 $map=if(Test-Path -LiteralPath $f){Get-Content -LiteralPath $f -Raw|ConvertFrom-Json -AsHashtable}else{@{}}
 $o=$map[[string]$Port];if(!$o){return [pscustomobject]@{Listening=$false;PID=$null;Image=$null}}
 [pscustomobject]@{Listening=[bool]$o['listening'];PID=$o['pid'];Image=$o['image']}
}
$installed=Join-Path $Dir 'ixctl.exe';$command='"'+$installed+'" service run "fixture-spec"'
$script:fixturePID=4242;$script:state='Running';$script:stops=0;$script:starts=0
function Test-IxForgeUpdateAdministrator {$true}
function Get-IxForgeUpdateService {[pscustomobject]@{State=$script:state;PathName=$command;StartName='NT SERVICE\ixctl';ProcessId=$script:fixturePID}}
function Stop-IxForgeUpdateService {$script:stops++;$script:state='Stopped';Set-Stage 'after'
 if($OnStop -eq 'hold'){$script:holder=[IO.File]::Open($installed,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)}
 if($OnStop -eq 'remove'){[IO.File]::Delete($installed)}
}
function Start-IxForgeUpdateService {$script:starts++;$script:state='Running'}

if($Mode -eq 'verdict'){
 $baseline=Get-IxForgeNeighbourBaseline $Origin
 Set-Stage 'after'
 $v=Test-IxForgeNeighbours $baseline $Origin
 [pscustomobject]@{baseline_known=[bool]$baseline.Known;baseline_reason=[string]$baseline.Reason;state=$v.State;reasons=@($v.Reasons|ForEach-Object{[string]$_})
  baseline_routes=@($baseline.Routes|ForEach-Object{[string]($_.Key+' -> '+$_.Target+' @ '+$_.OwnerImage)})}|ConvertTo-Json -Compress -Depth 4
 return
}

$data=Join-Path $Dir 'workshop';New-Item -ItemType Directory -Force -Path $data|Out-Null
[IO.File]::WriteAllText((Join-Path $Dir 'spec.json'),'fixture spec')
[IO.File]::WriteAllText((Join-Path $data 'config.json'),'fixture config')
@{pid=$script:fixturePID;addr=$Listen;schema_version=1}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $data 'fixture.workshop.json') -Encoding utf8
[IO.File]::WriteAllText($installed,'candidate binary')
$runID=[Guid]::NewGuid().ToString('N');$backup=Join-Path $data ('update-'+$runID)
New-Item -ItemType Directory -Path $backup|Out-Null
[IO.File]::WriteAllText((Join-Path $backup 'previous.exe'),'previous binary')
$plan=[pscustomobject]@{Schema=1;RunID=$runID;Service='ixctl';InstalledExe=$installed;CandidateSHA256=(Get-FileHash $installed).Hash
 PreviousSHA256=(Get-FileHash (Join-Path $backup 'previous.exe')).Hash;PreviousClientVersion='0.1.0';ExpectedImagePath=$command
 StateDir=$Dir;DataDir=$data;InstanceName='fixture';ConfigPath=(Join-Path $data 'config.json');ConfigSHA256=(Get-FileHash (Join-Path $data 'config.json')).Hash
 SpecPath=(Join-Path $Dir 'spec.json');SpecSHA256=(Get-FileHash (Join-Path $Dir 'spec.json')).Hash;Origin=$Origin;Listen=$Listen
 ExpectedClientVersion='0.1.1';State='completed';BackupDirectory=$backup;UpdatedUTC=''}
$candidateSHA=$plan.CandidateSHA256;$previousSHA=$plan.PreviousSHA256
switch -Regex($Receipt){
 '^old$'{$plan|Add-Member -NotePropertyName ScribeSHA256 -NotePropertyValue ('AB'*32)} # receipt written before HEALTH-R4
 # a new receipt whose recorded baseline would fail every route: a rollback must not compare with it
 '^stale$'{$plan|Add-Member -NotePropertyName Neighbours -NotePropertyValue ([pscustomobject]@{Known=$true;Reason='';Routes=@([pscustomobject]@{Key='Web/x:443/Handlers/gone';Value='Proxy=http://127.0.0.1:1';Target='http://127.0.0.1:1';Listening=$true;OwnerPID=1;OwnerImage='C:\gone.exe';Status=200})})}
 '^schema2$'{$plan.Schema=2}
 '^bogus-state$'{$plan.State='bogus'}
 '^partial:(.+)$'{$plan.PSObject.Properties.Remove($Matches[1])}
}
Save-IxForgeUpdateReceipt $plan
$receiptPath=Join-Path $backup 'receipt.json'
if($Receipt -eq 'truncated'){$t=[IO.File]::ReadAllText($receiptPath);[IO.File]::WriteAllText($receiptPath,$t.Substring(0,[int]($t.Length/2)))}
if($Receipt -eq 'empty'){[IO.File]::WriteAllText($receiptPath,'')}
if($Binary -eq 'previous'){[IO.File]::WriteAllText($installed,'previous binary')}
$script:state=$ServiceState
$failure=''
try{Undo-IxForgeUpdate $backup $candidateSHA $previousSHA|Out-Null}catch{$failure=$_.Exception.Message}
if($script:holder){$script:holder.Dispose()}
$diskState='';try{$diskState=[string](Get-Content -LiteralPath $receiptPath -Raw|ConvertFrom-Json).State}catch{$diskState='unreadable'}
[pscustomobject]@{error=$failure;receipt_state=$diskState;installed_is_previous=((Test-Path -LiteralPath $installed) -and (Get-FileHash $installed).Hash -ceq $previousSHA);stops=$script:stops
 starts=$script:starts;service=$script:state;verdict=[string]$script:IxForgeNeighbourVerdict.State}|ConvertTo-Json -Compress
