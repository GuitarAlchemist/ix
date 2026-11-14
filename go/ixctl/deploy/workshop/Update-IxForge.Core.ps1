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
 $scribe=Get-IxForgeUpdatePublic ($config.origin+'/')
 if($scribe.Status -ne 200){throw 'Scribe must be healthy before maintenance.'}
 $activeBrand=Get-IxForgeUpdatePublic ('http://'+$config.listen+'/workshop/branding.json') $origin.Authority
 if($activeBrand.Status -ne 200){throw 'Active client version is unavailable.'}
 $previousClientVersion=($activeBrand.Body|ConvertFrom-Json).version
 if($previousClientVersion -notmatch '^[0-9][A-Za-z0-9.+-]{0,80}$'){throw 'Invalid active client version.'}
 [pscustomobject]@{Schema=1;RunID=[Guid]::NewGuid().ToString('N');Service='ixctl';Candidate=$Candidate;CandidateSHA256=$CandidateSHA256.ToUpperInvariant();CandidateVersion=Get-IxForgeUpdateBinaryVersion $Candidate;InstalledExe=$InstalledExe;PreviousSHA256=$ExpectedInstalledSHA256.ToUpperInvariant();PreviousVersion=Get-IxForgeUpdateBinaryVersion $InstalledExe;PreviousClientVersion=$previousClientVersion;ExpectedImagePath=$ExpectedImagePath;StateDir=$StateDir;DataDir=$dataDir;InstanceName=$InstanceName;ConfigPath=$configPath;ConfigSHA256=(Get-FileHash -LiteralPath $configPath).Hash;SpecPath=$specPath;SpecSHA256=$ExpectedSpecSHA256.ToUpperInvariant();Origin=$config.origin;Listen=$config.listen;ExpectedClientVersion=$ExpectedClientVersion;ScribeSHA256=$scribe.SHA256;IntegrityVerified=$true;SignatureVerified=$false;Automatic=$false;DatabaseAction='retain; no copy, migration or rewind';State='available';BackupDirectory=$null;UpdatedUTC=[DateTime]::UtcNow.ToString('o')}
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
  $private=Get-IxForgeUpdatePublic ($Plan.Origin+'/workshop/');$scribe=Get-IxForgeUpdatePublic ($Plan.Origin+'/')
  $auth=Get-IxForgeUpdatePublic ($Plan.Origin+'/workshop/api/me')
  return $brand.Status -eq 200 -and ($brand.Body|ConvertFrom-Json).version -ceq $Plan.ExpectedClientVersion -and $private.Status -eq 200 -and $private.Body -match '<title>IxForge</title>' -and $auth.Status -eq 401 -and $scribe.Status -eq 200 -and $scribe.SHA256 -ceq $Plan.ScribeSHA256
 }catch{return $false}
}
function Restore-IxForgeUpdateBinary($Plan) {
 Assert-IxForgeUpdateHostUnchanged $Plan
 $backup=Join-Path $Plan.BackupDirectory 'previous.exe'
 if((Get-FileHash -LiteralPath $backup).Hash -ine $Plan.PreviousSHA256){throw 'Rollback backup integrity failed; no stop performed.'}
 $current=(Get-FileHash -LiteralPath $Plan.InstalledExe).Hash
 if($current -ieq $Plan.PreviousSHA256){if((Get-IxForgeUpdateService).State -ne 'Running'){Start-IxForgeUpdateService};$Plan.State='rolled_back';Save-IxForgeUpdateReceipt $Plan;return}
 if($current -ine $Plan.CandidateSHA256){throw 'Installed executable is neither reviewed version; stop for reconciliation.'}
 Stop-IxForgeUpdateService
 $staged=$Plan.InstalledExe+'.rollback-'+$Plan.RunID
 [IO.File]::Copy($backup,$staged,$false)
 try{if((Get-FileHash -LiteralPath $staged).Hash -ine $Plan.PreviousSHA256){throw 'Staged rollback integrity failed.'};[IO.File]::Replace($staged,$Plan.InstalledExe,[NullString]::Value)}finally{if(Test-Path -LiteralPath $staged){Remove-Item -LiteralPath $staged}}
 Start-IxForgeUpdateService
 if((Get-IxForgeUpdateService).State -ne 'Running' -or (Get-FileHash -LiteralPath $Plan.InstalledExe).Hash -ine $Plan.PreviousSHA256){throw 'Rollback restart was not verified.'}
 $nextVersion=$Plan.ExpectedClientVersion;$Plan.ExpectedClientVersion=$Plan.PreviousClientVersion
 try{if(!(Test-IxForgeUpdatedApplication $Plan)){throw 'Rollback application health was not verified.'}}finally{$Plan.ExpectedClientVersion=$nextVersion}
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
  Stop-IxForgeUpdateService;$stopped=$true
  Assert-IxForgeUpdateHostUnchanged $Plan
  [IO.File]::Copy($Plan.InstalledExe,(Join-Path $Plan.BackupDirectory 'previous.exe'),$false)
  if((Get-FileHash -LiteralPath (Join-Path $Plan.BackupDirectory 'previous.exe')).Hash -ine $Plan.PreviousSHA256){throw 'Rollback backup integrity failed.'}
  $Plan.State='backed_up';Save-IxForgeUpdateReceipt $Plan
  [IO.File]::Replace($staged,$Plan.InstalledExe,[NullString]::Value);$changed=$true
  $Plan.State='installed';Save-IxForgeUpdateReceipt $Plan
  Start-IxForgeUpdateService
  if((Get-FileHash -LiteralPath $Plan.InstalledExe).Hash -ine $Plan.CandidateSHA256 -or !(Test-IxForgeUpdatedApplication $Plan)){throw 'Updated application health/integrity check failed.'}
  Assert-IxForgeUpdateHostUnchanged $Plan
  $Plan.State='completed';Save-IxForgeUpdateReceipt $Plan
  return $Plan
 }catch{
  $failure=$_.Exception.Message
  if($changed){try{Restore-IxForgeUpdateBinary $Plan}catch{$Plan.State='reconciliation_required';Save-IxForgeUpdateReceipt $Plan;throw ('Update failed: '+$failure+'; rollback unresolved: '+$_.Exception.Message)}}
  elseif($stopped){Assert-IxForgeUpdateHostUnchanged $Plan;Start-IxForgeUpdateService}
  if($Plan.BackupDirectory -and $Plan.State -ne 'rolled_back'){$Plan.State='failed_before_replacement';Save-IxForgeUpdateReceipt $Plan}
  throw ('Update failed: '+$failure+'; final state: '+$Plan.State)
 }finally{if(Test-Path -LiteralPath $staged){Remove-Item -LiteralPath $staged};$lock.Dispose()}
}
function Undo-IxForgeUpdate([string]$BackupDirectory,[string]$ExpectedCurrentSHA256,[string]$ExpectedPreviousSHA256) {
 if(!(Test-IxForgeUpdateAdministrator)){throw 'Rollback requires approved administrator maintenance.'}
 Assert-IxForgeUpdateRegularPath $BackupDirectory
 $plan=Get-Content -LiteralPath (Join-Path $BackupDirectory 'receipt.json') -Raw|ConvertFrom-Json
 if($plan.Schema -ne 1 -or $plan.RunID -notmatch '^[a-f0-9]{32}$' -or [IO.Path]::GetFullPath($BackupDirectory) -cne (Join-Path $plan.DataDir ('update-'+$plan.RunID)) -or $plan.BackupDirectory -cne $BackupDirectory -or $plan.CandidateSHA256 -ine $ExpectedCurrentSHA256 -or $plan.PreviousSHA256 -ine $ExpectedPreviousSHA256){throw 'Rollback receipt does not match the reviewed update.'}
 $lock=[IO.File]::Open((Join-Path $plan.DataDir '.workshop.update.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
 try{Restore-IxForgeUpdateBinary $plan;return $plan}finally{$lock.Dispose()}
}
