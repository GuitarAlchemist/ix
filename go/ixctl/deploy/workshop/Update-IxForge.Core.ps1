#requires -Version 7.4
# Native, one-shot maintenance. No HTTP endpoint accepts a command or artifact path.
function Get-IxForgeUpdateService { Get-CimInstance Win32_Service -Filter "Name='ixctl'" }
function Test-IxForgeUpdateAdministrator {
 $principal=[Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
 $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Stop-IxForgeUpdateService { $service=Get-Service ixctl;if($service.Status -eq 'Stopped'){return};Stop-Service ixctl -NoWait;(Get-Service ixctl).WaitForStatus('Stopped',[TimeSpan]::FromSeconds(60)) }
function Start-IxForgeUpdateService { Start-Service ixctl;(Get-Service ixctl).WaitForStatus('Running',[TimeSpan]::FromSeconds(60)) }
function Get-IxForgeUpdateBinaryVersion([string]$Path) {
 $value=(& $Path version | Out-String).Trim()
 if($LASTEXITCODE -ne 0 -or $value -notmatch '^ixctl [0-9][A-Za-z0-9.+-]{0,80}$'){throw 'Executable version probe failed.'}
 $value.Substring(6)
}
function Assert-IxForgeUpdateRegularPath([string]$Path) {
 $item=Get-Item -LiteralPath $Path -ErrorAction Stop
 while($item){if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Update paths must not traverse a reparse point.'};$item=if($item.PSIsContainer){$item.Parent}else{$item.Directory}}
}
function Get-IxForgeUpdatePublic([string]$URL,[string]$HostValue='') {
 $handler=[Net.Http.HttpClientHandler]::new();$handler.UseProxy=$false;$handler.AllowAutoRedirect=$false
 $client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(10)
 $request=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get,$URL)
 if($HostValue){$request.Headers.Host=$HostValue}
 try{$response=$client.Send($request);try{$body=$response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult();return @{Status=[int]$response.StatusCode;Body=[Text.Encoding]::UTF8.GetString($body);SHA256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($body))}}finally{$response.Dispose()}}finally{$request.Dispose();$client.Dispose()}
}
# Neighbours are everything Tailscale Serve publishes besides this Workshop's own route. IxForge does not own them and
# cannot tell what they are from their content, so it never claims an identity (not even Scribe's). A maintenance must
# leave them as it found them: the same routes, each loopback target still owned by the same executable image (a new PID
# is fine; a recycled PID running another image is not), and still answering below 500, without following redirects.
# Whatever cannot be observed is unknown, and unknown is never healthy.
function Get-IxForgeServeStatusJson { try{$out=& tailscale serve status --json 2>$null;if($LASTEXITCODE -ne 0){return $null};($out|Out-String)}catch{$null} }
function Get-IxForgeNeighbourOwner([int]$Port) {
 $listeners=@(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
 if($listeners.Count -eq 0){return [pscustomobject]@{Listening=$false;PID=$null;Image=$null}}
 $ids=@($listeners.OwningProcess|Sort-Object -Unique)
 if($ids.Count -ne 1){return [pscustomobject]@{Listening=$true;PID=$null;Image=$null}}
 $image=try{(Get-Process -Id $ids[0] -ErrorAction Stop).Path}catch{$null}
 [pscustomobject]@{Listening=$true;PID=[int]$ids[0];Image=$image}
}
function Get-IxForgeNeighbourStatus([string]$URL) { try{(Get-IxForgeUpdatePublic $URL).Status}catch{0} }
function ConvertFrom-IxForgeServeStatus([string]$Json,[string]$Origin) {
 if([string]::IsNullOrWhiteSpace($Json)){return $null}
 try{$serve=$Json|ConvertFrom-Json -AsHashtable -ErrorAction Stop}catch{return $null}
 if($serve -isnot [Collections.IDictionary]){return $null}
 $originUri=[Uri]$Origin;$own=$originUri.Host+':'+$originUri.Port
 $flat={param($value) if($value -is [Collections.IDictionary]){($value.Keys|Sort-Object|ForEach-Object{$_+'='+($value[$_]|ConvertTo-Json -Compress -Depth 4)}) -join ';'}else{$value|ConvertTo-Json -Compress -Depth 4}}
 $routes=[Collections.Generic.List[object]]::new()
 foreach($section in @($serve.Keys|Sort-Object)){
  $entries=$serve[$section]
  if($section -eq 'Web' -and $entries -is [Collections.IDictionary]){
   foreach($hostKey in @($entries.Keys)){
    $handlers=$entries[$hostKey]['Handlers']
    if($handlers -isnot [Collections.IDictionary]){$routes.Add([pscustomobject]@{Key="web $hostKey";Value=(& $flat $entries[$hostKey]);Target=$null});continue}
    foreach($path in @($handlers.Keys)){
     if($hostKey -ceq $own -and ($path -ceq '/workshop' -or $path.StartsWith('/workshop/'))){continue}
     $h=$handlers[$path];$target=if($h -is [Collections.IDictionary] -and $h['Proxy']){[string]$h['Proxy']}else{$null}
     $routes.Add([pscustomobject]@{Key="web $hostKey$path";Value=(& $flat $h);Target=$target})
    }
   }
  }elseif($section -eq 'TCP' -and $entries -is [Collections.IDictionary]){
   foreach($port in @($entries.Keys)){
    $t=$entries[$port];$target=if($t -is [Collections.IDictionary] -and $t['TCPForward']){'tcp://'+[string]$t['TCPForward']}else{$null}
    $routes.Add([pscustomobject]@{Key="tcp $port";Value=(& $flat $t);Target=$target})
   }
  }else{$routes.Add([pscustomobject]@{Key="$section";Value=(& $flat $entries);Target=$null})}
 }
 ,@($routes|Sort-Object Key)
}
function Get-IxForgeNeighbourObservation([string]$Origin) {
 $routes=ConvertFrom-IxForgeServeStatus (Get-IxForgeServeStatusJson) $Origin
 if($null -eq $routes){return [pscustomobject]@{Known=$false;Reason='Tailscale Serve configuration unavailable or unreadable';Routes=@()}}
 $unknown=[Collections.Generic.List[string]]::new()
 foreach($r in $routes){
  $r|Add-Member -Force -NotePropertyName Listening -NotePropertyValue $null;$r|Add-Member -Force -NotePropertyName OwnerPID -NotePropertyValue $null
  $r|Add-Member -Force -NotePropertyName OwnerImage -NotePropertyValue $null;$r|Add-Member -Force -NotePropertyName Status -NotePropertyValue $null
  if(!$r.Target){continue}
  $u=$null
  if(![Uri]::TryCreate($r.Target,[UriKind]::Absolute,[ref]$u) -or $u.Scheme -notin @('http','tcp') -or !$u.IsLoopback -or $u.Port -le 0 -or $u.UserInfo){$unknown.Add($r.Key+': target is not a plain loopback http/tcp address');continue}
  $owner=Get-IxForgeNeighbourOwner $u.Port
  $r.Listening=[bool]$owner.Listening;$r.OwnerPID=$owner.PID;$r.OwnerImage=$owner.Image
  if($owner.Listening -and !$owner.Image){$unknown.Add($r.Key+': owner executable not observable')}
  if($u.Scheme -eq 'http' -and $owner.Listening){$r.Status=[int](Get-IxForgeNeighbourStatus $u.AbsoluteUri)}
 }
 [pscustomobject]@{Known=($unknown.Count -eq 0);Reason=($unknown -join '; ');Routes=$routes}
}
function Get-IxForgeNeighbourBaseline([string]$Origin) {
 $now=Get-IxForgeNeighbourObservation $Origin
 if($now.Known){foreach($r in $now.Routes|Where-Object Target){if(!$r.Listening -or ($r.Target.StartsWith('http') -and ($r.Status -lt 100 -or $r.Status -ge 500))){$now.Known=$false;$now.Reason=$r.Key+': not available before maintenance';break}}}
 $now
}
function Test-IxForgeNeighbours($Baseline,[string]$Origin) {
 if($null -eq $Baseline -or !$Baseline.Known){return [pscustomobject]@{State='unknown';Reasons=@('no observed neighbour baseline')}}
 $now=Get-IxForgeNeighbourObservation $Origin
 if($null -eq $now.Routes -or ($now.Routes.Count -eq 0 -and !$now.Known)){return [pscustomobject]@{State='unknown';Reasons=@($now.Reason)}}
 $bad=[Collections.Generic.List[string]]::new();$unknown=[Collections.Generic.List[string]]::new()
 if(!$now.Known){$unknown.Add($now.Reason)}
 $before=@{};foreach($r in $Baseline.Routes){$before[$r.Key]=$r};$after=@{};foreach($r in $now.Routes){$after[$r.Key]=$r}
 foreach($k in $before.Keys){if(!$after.ContainsKey($k)){$bad.Add($k+': route removed')}elseif($after[$k].Value -cne $before[$k].Value){$bad.Add($k+': route changed')}}
 foreach($k in $after.Keys){if(!$before.ContainsKey($k)){$bad.Add($k+': route added')}}
 foreach($k in $before.Keys){
  $b=$before[$k];$a=$after[$k]
  if(!$b.Target -or !$a -or $a.Value -cne $b.Value){continue}
  if(!$a.Listening){$bad.Add($k+': target no longer listening');continue}
  if(!$a.OwnerImage){continue}
  if($a.OwnerImage -ine $b.OwnerImage){$bad.Add($k+': target now owned by another executable');continue}
  if($a.Target.StartsWith('http') -and ($a.Status -lt 100 -or $a.Status -ge 500)){$bad.Add($k+': target unavailable (status '+$a.Status+')')}
 }
 $state=if($bad.Count){'unhealthy'}elseif($unknown.Count){'unknown'}else{'healthy'}
 [pscustomobject]@{State=$state;Reasons=@($bad)+@($unknown)}
}
function New-IxForgeUpdatePlan {
 param([string]$Candidate,[string]$CandidateSHA256,[string]$ExpectedInstalledSHA256,[string]$ExpectedImagePath,[string]$StateDir,[string]$InstanceName,[string]$ExpectedClientVersion,[string]$InstalledExe='C:\Program Files\ixctl\ixctl.exe',[string]$ExpectedSpecSHA256)
 if($CandidateSHA256 -notmatch '^[A-Fa-f0-9]{64}$' -or $ExpectedInstalledSHA256 -notmatch '^[A-Fa-f0-9]{64}$' -or $ExpectedSpecSHA256 -notmatch '^[A-Fa-f0-9]{64}$'){throw 'Require reviewed candidate, installed and spec hashes.'}
 if($CandidateSHA256 -ieq $ExpectedInstalledSHA256){throw 'The reviewed candidate is already installed; no restart is needed.'}
 if($InstanceName -cnotmatch '^[a-z0-9][a-z0-9-]{0,62}$' -or $ExpectedClientVersion -notmatch '^[0-9][A-Za-z0-9.+-]{0,80}$'){throw 'Invalid instance or client version.'}
 $Candidate=[IO.Path]::GetFullPath($Candidate);$InstalledExe=[IO.Path]::GetFullPath($InstalledExe);$StateDir=[IO.Path]::GetFullPath($StateDir)
 if($StateDir.StartsWith('\\') -or $StateDir -match '(?i)(^|[\\/])OneDrive([^\\/]*)([\\/]|$)' -or $Candidate -ieq $InstalledExe){throw 'Require separate local candidate and non-synchronized state.'}
 if(!$ExpectedImagePath.StartsWith('"'+$InstalledExe+'" service run ',[StringComparison]::OrdinalIgnoreCase) -or $ExpectedImagePath -match '(?i)bearer|token|password'){throw 'Reviewed command must identify ixctl and contain no credential.'}
 $dataDir=Join-Path $StateDir 'workshop';$configPath=Join-Path $dataDir 'config.json';$specPath=Join-Path $StateDir 'spec.json'
 foreach($path in @($Candidate,$InstalledExe,$dataDir,$configPath,$specPath)){Assert-IxForgeUpdateRegularPath $path}
 if((Get-FileHash -LiteralPath $Candidate).Hash -ine $CandidateSHA256 -or (Get-FileHash -LiteralPath $InstalledExe).Hash -ine $ExpectedInstalledSHA256 -or (Get-FileHash -LiteralPath $specPath).Hash -ine $ExpectedSpecSHA256){throw 'Artifact, installed binary or spec changed since review.'}
 $service=Get-IxForgeUpdateService
 if($service.State -ne 'Running' -or $service.StartName -ine 'NT SERVICE\ixctl' -or $service.PathName -cne $ExpectedImagePath){throw 'Service changed since review.'}
 $config=Get-Content -LiteralPath $configPath -Raw|ConvertFrom-Json
 if($config.version -ne 1 -or !$config.enabled -or $config.listen -notmatch '^127\.0\.0\.1:([1-9][0-9]{0,4})$'){throw 'Require existing enabled version-1 Workshop configuration.'}
 $port=[int]$Matches[1];$origin=[Uri]$config.origin
 if($port -gt 65535 -or $port -in 8737,8738,8739,8740 -or $origin.Scheme -ne 'https' -or $origin.AbsolutePath -ne '/' -or $origin.Query -or $origin.Fragment -or $origin.UserInfo){throw 'Existing Workshop endpoint is outside the update scope.'}
 $readyPath=Join-Path $dataDir ($InstanceName+'.workshop.json');Assert-IxForgeUpdateRegularPath $readyPath
 $ready=Get-Content -LiteralPath $readyPath -Raw|ConvertFrom-Json
 if($ready.pid -ne $service.ProcessId -or $ready.schema_version -ne 1 -or $ready.addr -cne $config.listen){throw 'Current Workshop ready-state does not match the service.'}
 $neighbours=Get-IxForgeNeighbourBaseline $config.origin
 if(!$neighbours.Known){throw ('Neighbour preservation cannot be observed before maintenance: '+$neighbours.Reason)}
 $activeBrand=Get-IxForgeUpdatePublic ('http://'+$config.listen+'/workshop/branding.json') $origin.Authority
 if($activeBrand.Status -ne 200){throw 'Active client version is unavailable.'}
 $previousClientVersion=($activeBrand.Body|ConvertFrom-Json).version
 if($previousClientVersion -notmatch '^[0-9][A-Za-z0-9.+-]{0,80}$'){throw 'Invalid active client version.'}
 [pscustomobject]@{Schema=1;RunID=[Guid]::NewGuid().ToString('N');Service='ixctl';Candidate=$Candidate;CandidateSHA256=$CandidateSHA256.ToUpperInvariant();CandidateVersion=Get-IxForgeUpdateBinaryVersion $Candidate;InstalledExe=$InstalledExe;PreviousSHA256=$ExpectedInstalledSHA256.ToUpperInvariant();PreviousVersion=Get-IxForgeUpdateBinaryVersion $InstalledExe;PreviousClientVersion=$previousClientVersion;ExpectedImagePath=$ExpectedImagePath;StateDir=$StateDir;DataDir=$dataDir;InstanceName=$InstanceName;ConfigPath=$configPath;ConfigSHA256=(Get-FileHash -LiteralPath $configPath).Hash;SpecPath=$specPath;SpecSHA256=$ExpectedSpecSHA256.ToUpperInvariant();Origin=$config.origin;Listen=$config.listen;ExpectedClientVersion=$ExpectedClientVersion;Neighbours=$neighbours;IntegrityVerified=$true;SignatureVerified=$false;Automatic=$false;DatabaseAction='retain; no copy, migration or rewind';State='available';BackupDirectory=$null;UpdatedUTC=[DateTime]::UtcNow.ToString('o')}
}
function Save-IxForgeUpdateReceipt($Plan) {
 $Plan.UpdatedUTC=[DateTime]::UtcNow.ToString('o');$path=Join-Path $Plan.BackupDirectory 'receipt.json'
 [IO.File]::WriteAllText($path+'.tmp',($Plan|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false));[IO.File]::Move($path+'.tmp',$path,$true)
}
function Assert-IxForgeUpdateHostUnchanged($Plan) {
 $service=Get-IxForgeUpdateService
 if($service.PathName -cne $Plan.ExpectedImagePath -or $service.StartName -ine 'NT SERVICE\ixctl' -or (Get-FileHash -LiteralPath $Plan.ConfigPath).Hash -ine $Plan.ConfigSHA256 -or (Get-FileHash -LiteralPath $Plan.SpecPath).Hash -ine $Plan.SpecSHA256){throw 'Host command, account or configuration changed; do not overwrite another owner.'}
}
function Test-IxForgeUpdatedApplication($Plan) {
 try{
  for($i=0;$i -lt 50;$i++){
   $service=Get-IxForgeUpdateService;$readyPath=Join-Path $Plan.DataDir ($Plan.InstanceName+'.workshop.json')
   if($service.State -eq 'Running' -and (Test-Path -LiteralPath $readyPath)){
    $ready=Get-Content -LiteralPath $readyPath -Raw|ConvertFrom-Json
    if($ready.pid -eq $service.ProcessId -and $ready.schema_version -eq 1 -and $ready.addr -ceq $Plan.Listen){break}
   };Start-Sleep -Milliseconds 200
  }
  if($service.State -ne 'Running' -or $ready.pid -ne $service.ProcessId -or $ready.schema_version -ne 1 -or $ready.addr -cne $Plan.Listen){return $false}
  $brand=Get-IxForgeUpdatePublic ('http://'+$Plan.Listen+'/workshop/branding.json') ([Uri]$Plan.Origin).Authority
  $private=Get-IxForgeUpdatePublic ($Plan.Origin+'/workshop/')
  $auth=Get-IxForgeUpdatePublic ($Plan.Origin+'/workshop/api/me')
  $script:IxForgeNeighbourVerdict=Test-IxForgeNeighbours $Plan.Neighbours $Plan.Origin
  return $brand.Status -eq 200 -and ($brand.Body|ConvertFrom-Json).version -ceq $Plan.ExpectedClientVersion -and $private.Status -eq 200 -and $private.Body -match '<title>IxForge</title>' -and $auth.Status -eq 401 -and $script:IxForgeNeighbourVerdict.State -ceq 'healthy'
 }catch{return $false}
}
function Restore-IxForgeUpdateBinary($Plan) {
 Assert-IxForgeUpdateHostUnchanged $Plan
 $backup=Join-Path $Plan.BackupDirectory 'previous.exe'
 if((Get-FileHash -LiteralPath $backup).Hash -ine $Plan.PreviousSHA256){throw 'Rollback backup integrity failed; no stop performed.'}
 $current=(Get-FileHash -LiteralPath $Plan.InstalledExe).Hash
 if($current -ine $Plan.PreviousSHA256 -and $current -ine $Plan.CandidateSHA256){throw 'Installed executable is neither reviewed version; stop for reconciliation.'}
 if(!$Plan.PSObject.Properties['Neighbours'] -or !$Plan.Neighbours -or !$Plan.Neighbours.Known){
  $Plan|Add-Member -Force -NotePropertyName Neighbours -NotePropertyValue (Get-IxForgeNeighbourBaseline $Plan.Origin)
 }
 if($current -ieq $Plan.PreviousSHA256){
  # A rollback interrupted after the replacement: the previous executable is back, but nothing was verified yet.
  # Restarting the verified previous executable is the safe action; "rolled_back" still requires the full check.
  if((Get-IxForgeUpdateService).State -ne 'Running'){Start-IxForgeUpdateService}
  if(!$Plan.Neighbours.Known){throw ('Previous executable restored, but neighbour preservation cannot be observed; not recorded as rolled back: '+$Plan.Neighbours.Reason)}
  Confirm-IxForgeRollback $Plan;return
 }
 if(!$Plan.Neighbours.Known){throw ('Neighbour preservation cannot be observed; no stop performed: '+$Plan.Neighbours.Reason)}
 Stop-IxForgeUpdateService
 $staged=$Plan.InstalledExe+'.rollback-'+$Plan.RunID
 $replaceFailure=''
 try{
  [IO.File]::Copy($backup,$staged,$false)
  if((Get-FileHash -LiteralPath $staged).Hash -ine $Plan.PreviousSHA256){throw 'Staged rollback integrity failed.'};[IO.File]::Replace($staged,$Plan.InstalledExe,[NullString]::Value)
 }catch{$replaceFailure=$_.Exception.Message}finally{if(Test-Path -LiteralPath $staged){Remove-Item -LiteralPath $staged}}
 if($replaceFailure){
  # The service is stopped and the replacement failed. A missing executable (removed, or lost by ReplaceFile) is put
  # back from the verified backup without overwriting anything. The service is restarted only on the verified previous
  # executable of an unchanged host; anything else stays stopped and is recorded, never guessed.
  try{if(!(Test-Path -LiteralPath $Plan.InstalledExe)){[IO.File]::Copy($backup,$Plan.InstalledExe,$false)}}catch{}
  $installedHash=try{(Get-FileHash -LiteralPath $Plan.InstalledExe -ErrorAction Stop).Hash}catch{''}
  if($installedHash -ine $Plan.PreviousSHA256){
   $Plan.State='reconciliation_required';Save-IxForgeUpdateReceipt $Plan
   throw ('Rollback replacement failed: '+$replaceFailure+'; the verified previous executable is not in place, so the service was left stopped for reconciliation.')
  }
  Assert-IxForgeUpdateHostUnchanged $Plan
 }
 Start-IxForgeUpdateService
 Confirm-IxForgeRollback $Plan
}
function Test-IxForgePreviousApplication($Plan) {
 # The full health check (service, readiness, branding, private page, auth, neighbours) against the previous version.
 $nextVersion=$Plan.ExpectedClientVersion;$Plan.ExpectedClientVersion=$Plan.PreviousClientVersion
 try{return [bool](Test-IxForgeUpdatedApplication $Plan)}finally{$Plan.ExpectedClientVersion=$nextVersion}
}
function Test-IxForgeRunningPrevious($Plan) {
 # Checked AFTER the start: the hash checked before it cannot prove what the service manager actually launched.
 $hash=try{(Get-FileHash -LiteralPath $Plan.InstalledExe -ErrorAction Stop).Hash}catch{''}
 return (Get-IxForgeUpdateService).State -eq 'Running' -and $hash -ieq $Plan.PreviousSHA256
}
function Confirm-IxForgeRollback($Plan) {
 $failure=''
 if(!(Test-IxForgeRunningPrevious $Plan)){$failure='Rollback restart was not verified.'}
 elseif(!(Test-IxForgePreviousApplication $Plan)){$failure='Rollback application health was not verified.'}
 elseif(!(Test-IxForgeRunningPrevious $Plan)){$failure='Rollback restart was not verified.'}
 if($failure){$Plan.State='reconciliation_required';Save-IxForgeUpdateReceipt $Plan;throw $failure}
 $Plan.State='rolled_back';Save-IxForgeUpdateReceipt $Plan
}
function Invoke-IxForgeUpdate($Plan) {
 if(!(Test-IxForgeUpdateAdministrator)){throw 'Approved maintenance requires Windows administrator elevation.'}
 $fresh=New-IxForgeUpdatePlan -Candidate $Plan.Candidate -CandidateSHA256 $Plan.CandidateSHA256 -ExpectedInstalledSHA256 $Plan.PreviousSHA256 -ExpectedImagePath $Plan.ExpectedImagePath -StateDir $Plan.StateDir -InstanceName $Plan.InstanceName -ExpectedClientVersion $Plan.ExpectedClientVersion -InstalledExe $Plan.InstalledExe -ExpectedSpecSHA256 $Plan.SpecSHA256
 $lock=[IO.File]::Open((Join-Path $Plan.DataDir '.workshop.update.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
 $staged=$Plan.InstalledExe+'.update-'+$Plan.RunID;$changed=$false;$stopped=$false
 try{
  Assert-IxForgeUpdateHostUnchanged $Plan
  if((Get-FileHash -LiteralPath $Plan.InstalledExe).Hash -ine $Plan.PreviousSHA256){throw 'Installed binary changed after preflight.'}
  $Plan.BackupDirectory=Join-Path $Plan.DataDir ('update-'+$Plan.RunID);New-Item -ItemType Directory -Path $Plan.BackupDirectory|Out-Null
  Save-IxForgeUpdateReceipt $Plan
  [IO.File]::Copy($Plan.Candidate,$staged,$false)
  if((Get-FileHash -LiteralPath $staged).Hash -ine $Plan.CandidateSHA256){throw 'Staged candidate integrity failed.'}
  $Plan.Neighbours=Get-IxForgeNeighbourBaseline $Plan.Origin
  if(!$Plan.Neighbours.Known){throw ('Neighbour preservation cannot be observed; no stop performed: '+$Plan.Neighbours.Reason)}
  Stop-IxForgeUpdateService;$stopped=$true
  Assert-IxForgeUpdateHostUnchanged $Plan
  [IO.File]::Copy($Plan.InstalledExe,(Join-Path $Plan.BackupDirectory 'previous.exe'),$false)
  if((Get-FileHash -LiteralPath (Join-Path $Plan.BackupDirectory 'previous.exe')).Hash -ine $Plan.PreviousSHA256){throw 'Rollback backup integrity failed.'}
  $Plan.State='backed_up';Save-IxForgeUpdateReceipt $Plan
  [IO.File]::Replace($staged,$Plan.InstalledExe,[NullString]::Value);$changed=$true
  $Plan.State='installed';Save-IxForgeUpdateReceipt $Plan
  Start-IxForgeUpdateService
  if((Get-FileHash -LiteralPath $Plan.InstalledExe).Hash -ine $Plan.CandidateSHA256 -or !(Test-IxForgeUpdatedApplication $Plan) -or (Get-FileHash -LiteralPath $Plan.InstalledExe).Hash -ine $Plan.CandidateSHA256){throw 'Updated application health/integrity check failed.'}
  Assert-IxForgeUpdateHostUnchanged $Plan
  $Plan.State='completed';Save-IxForgeUpdateReceipt $Plan
  return $Plan
 }catch{
  $failure=$_.Exception.Message
  if($changed){try{Restore-IxForgeUpdateBinary $Plan}catch{$Plan.State='reconciliation_required';Save-IxForgeUpdateReceipt $Plan;throw ('Update failed: '+$failure+'; rollback unresolved: '+$_.Exception.Message)}}
  elseif($stopped){
   # Restart only the verified previous executable. A missing one is put back from the verified backup without
   # overwriting; anything else stays stopped and is recorded, never guessed.
   $backup=Join-Path $Plan.BackupDirectory 'previous.exe'
   try{if(!(Test-Path -LiteralPath $Plan.InstalledExe) -and (Get-FileHash -LiteralPath $backup -ErrorAction Stop).Hash -ieq $Plan.PreviousSHA256){[IO.File]::Copy($backup,$Plan.InstalledExe,$false)}}catch{}
   $installedHash=try{(Get-FileHash -LiteralPath $Plan.InstalledExe -ErrorAction Stop).Hash}catch{''}
   if($installedHash -ine $Plan.PreviousSHA256){
    $Plan.State='reconciliation_required';Save-IxForgeUpdateReceipt $Plan
    throw ('Update failed: '+$failure+'; the verified previous executable is not in place, so the service was left stopped for reconciliation.')
   }
   Assert-IxForgeUpdateHostUnchanged $Plan;Start-IxForgeUpdateService
   if(!(Test-IxForgeRunningPrevious $Plan) -or !(Test-IxForgePreviousApplication $Plan) -or !(Test-IxForgeRunningPrevious $Plan)){
    $Plan.State='reconciliation_required';Save-IxForgeUpdateReceipt $Plan
    throw ('Update failed: '+$failure+'; the verified previous executable was restarted but its health was not verified; reconciliation required.')
   }
  }
  if($Plan.BackupDirectory -and $Plan.State -ne 'rolled_back'){$Plan.State='failed_before_replacement';Save-IxForgeUpdateReceipt $Plan}
  throw ('Update failed: '+$failure+'; final state: '+$Plan.State)
 }finally{if(Test-Path -LiteralPath $staged){Remove-Item -LiteralPath $staged};$lock.Dispose()}
}
function Undo-IxForgeUpdate([string]$BackupDirectory,[string]$ExpectedCurrentSHA256,[string]$ExpectedPreviousSHA256) {
 if(!(Test-IxForgeUpdateAdministrator)){throw 'Rollback requires approved administrator maintenance.'}
 Assert-IxForgeUpdateRegularPath $BackupDirectory
 $plan=Get-Content -LiteralPath (Join-Path $BackupDirectory 'receipt.json') -Raw|ConvertFrom-Json
 if($plan.Schema -ne 1 -or $plan.RunID -notmatch '^[a-f0-9]{32}$' -or [IO.Path]::GetFullPath($BackupDirectory) -cne (Join-Path $plan.DataDir ('update-'+$plan.RunID)) -or $plan.BackupDirectory -cne $BackupDirectory -or $plan.CandidateSHA256 -ine $ExpectedCurrentSHA256 -or $plan.PreviousSHA256 -ine $ExpectedPreviousSHA256){throw 'Rollback receipt does not match the reviewed update.'}
 # Every reference the rollback and its verification read must be present before the lock, so that an incomplete or
 # unknown receipt is refused before any stop. Receipts from before HEALTH-R4 (ScribeSHA256) carry all of them.
 foreach($field in @('Service','InstalledExe','ExpectedImagePath','StateDir','DataDir','InstanceName','ConfigPath','ConfigSHA256','SpecPath','SpecSHA256','Origin','Listen','PreviousClientVersion','ExpectedClientVersion','State')){
  if(!$plan.PSObject.Properties[$field] -or [string]::IsNullOrWhiteSpace([string]$plan.$field)){throw ('Rollback receipt is incomplete ('+$field+'); no stop performed.')}
 }
 if($plan.State -cnotin @('available','backed_up','installed','completed','failed_before_replacement','reconciliation_required','rolled_back')){throw ('Rollback receipt state is unknown ('+$plan.State+'); no stop performed.')}
 $lock=[IO.File]::Open((Join-Path $plan.DataDir '.workshop.update.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
 $plan|Add-Member -Force -NotePropertyName Neighbours -NotePropertyValue $null # a standalone rollback compares neighbours with their state just before it
 try{Restore-IxForgeUpdateBinary $plan;return $plan}finally{$lock.Dispose()}
}
