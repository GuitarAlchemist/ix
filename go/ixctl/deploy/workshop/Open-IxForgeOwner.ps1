#requires -Version 7.4
[CmdletBinding()]
param(
 [Parameter(Mandatory)][string]$StateDir,
 [Parameter(Mandatory)][ValidatePattern('^[a-z0-9][a-z0-9-]{0,62}$',Options='None')][string]$InstanceName,
 [switch]$DryRun,
 [switch]$VerifyOnly
)
$ErrorActionPreference='Stop'
$dataDir=Join-Path ([IO.Path]::GetFullPath($StateDir)) 'workshop'
$config=Get-Content -LiteralPath (Join-Path $dataDir 'config.json') -Raw | ConvertFrom-Json
if($config.version -ne 1 -or !$config.enabled -or $config.listen -notmatch '^127\.0\.0\.1:([1-9][0-9]{0,4})$'){throw 'IxForge is not configured with an enabled fixed loopback endpoint.'}
$port=[int]$Matches[1]
if($port -gt 65535 -or $port -in 8737,8738,8739,8740){throw 'Invalid or reserved endpoint.'}
$keyPath=Join-Path $dataDir ($InstanceName+'.workshop-owner.key')
if((Get-Item -LiteralPath $keyPath).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Owner key must be a regular local file.'}
$key=(Get-Content -LiteralPath $keyPath -Raw).Trim()
if($key -notmatch '^[0-9a-fA-F]{64}$'){throw 'Owner access has not been provisioned correctly.'}
$origin='http://127.0.0.1:'+$port
if($DryRun){[pscustomobject]@{OwnerOrigin=$origin;DataDir=$dataDir;Launch=$false};return}
function Sign-OwnerMessage([string]$Purpose,[string]$Nonce,[string]$Epoch,[long]$Expires,[string[]]$Extra=@()){
 $parts=@('ixforge-owner/1',$Purpose,$origin,$Nonce,$Epoch,$Expires.ToString([Globalization.CultureInfo]::InvariantCulture))+$Extra
 $mac=[Security.Cryptography.HMACSHA256]::new([Convert]::FromHexString($key))
 try{return [Convert]::ToHexString($mac.ComputeHash([Text.Encoding]::UTF8.GetBytes(($parts -join "`n")))).ToLowerInvariant()}finally{$mac.Dispose()}
}
function Assert-Proof([string]$Expected,[string]$Received){
 if(!$Received -or $Received -notmatch '^[a-fA-F0-9]{64}$'){throw 'The process on the local port did not prove its IxForge identity.'}
 $one=[Convert]::FromHexString($Expected);$two=[Convert]::FromHexString($Received);$difference=0
 for($i=0;$i -lt 32;$i++){$difference=$difference -bor ([int]$one[$i] -bxor [int]$two[$i])}
 if($difference -ne 0){throw 'The process on the local port did not prove its IxForge identity.'}
}
$random=[byte[]]::new(32);$rng=[Security.Cryptography.RandomNumberGenerator]::Create();try{$rng.GetBytes($random)}finally{$rng.Dispose()};$nonce=[Convert]::ToHexString($random).ToLowerInvariant()
$challenge=Invoke-RestMethod -Uri ($origin+'/workshop-owner/challenge?nonce='+$nonce) -NoProxy -MaximumRedirection 0 -TimeoutSec 5
if($challenge.epoch -notmatch '^[a-f0-9]{48}$' -or [long]$challenge.expires -le [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()){throw 'Owner challenge missing, stale or invalid.'}
Assert-Proof (Sign-OwnerMessage 'challenge' $nonce $challenge.epoch $challenge.expires) $challenge.proof
$request=@{nonce=$nonce;epoch=$challenge.epoch;expires=[long]$challenge.expires;proof=Sign-OwnerMessage 'session' $nonce $challenge.epoch $challenge.expires}|ConvertTo-Json
$session=Invoke-RestMethod -Uri ($origin+'/workshop-owner/session') -Method Post -ContentType 'application/json' -Headers @{Origin=$origin} -Body $request -NoProxy -MaximumRedirection 0 -TimeoutSec 5
if($session.session -notmatch '^[a-f0-9]{48}$'){throw 'No valid temporary owner session.'}
Assert-Proof (Sign-OwnerMessage 'receipt' $nonce $challenge.epoch $challenge.expires @($session.session,([long]$session.expires).ToString([Globalization.CultureInfo]::InvariantCulture))) $session.proof
if($VerifyOnly){[pscustomobject]@{OwnerOrigin=$origin;Verified=$true;SessionExpires=$session.expires;Launch=$false};return}
# Only this short-lived, in-memory, runtime-bound session reaches the browser. The persistent key never leaves this process.
Start-Process -FilePath ($origin+'/workshop-owner/#session='+$session.session)
