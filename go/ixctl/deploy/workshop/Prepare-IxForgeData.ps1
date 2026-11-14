#requires -Version 7.4
[CmdletBinding(SupportsShouldProcess,ConfirmImpact='High')]
param(
 [Parameter(Mandatory)][string]$StateDir,
 [Parameter(Mandatory)][ValidatePattern('^[a-z0-9][a-z0-9-]{0,62}$',Options='None')][string]$InstanceName,
 [Parameter(Mandatory)][ValidatePattern('^S-1-[0-9-]+$')][string]$OwnerSid,
 [Parameter(Mandatory)][string]$Origin,
 [string]$Listen='127.0.0.1:8761',
 [switch]$Provision
)
$ErrorActionPreference='Stop'
$stateRoot=[IO.Path]::GetFullPath($StateDir)
if($stateRoot.StartsWith('\\') -or $stateRoot -match '(?i)(^|[\\/])OneDrive([^\\/]*)([\\/]|$)'){throw 'Select the approved local, non-OneDrive IX state directory.'}
$originUri=[Uri]$Origin
if($originUri.Scheme -ne 'https' -or $originUri.AbsolutePath -ne '/' -or $originUri.Query -or $originUri.Fragment -or $originUri.UserInfo -or $Origin.EndsWith('/')){throw 'Use an exact HTTPS origin without a trailing slash.'}
if($Listen -notmatch '^127\.0\.0\.1:([1-9][0-9]{0,4})$'){throw 'Use a fixed IPv4 loopback endpoint.'}
$port=[int]$Matches[1];if($port -gt 65535 -or $port -in 8737,8738,8739,8740){throw 'Port invalid or reserved for Scribe.'}
$dataDir=Join-Path $stateRoot 'workshop'
$configPath=Join-Path $dataDir 'config.json'
$plan=[pscustomobject]@{DataDir=$dataDir;Database=Join-Path $dataDir ($InstanceName+'.workshop.sqlite');OwnerKey=Join-Path $dataDir ($InstanceName+'.workshop-owner.key');OwnerSid=$OwnerSid;ServiceAccount='NT SERVICE\ixctl';Origin=$Origin;Listen=$Listen;Enabled=$false;Provision=$Provision.IsPresent}
if(!$Provision){$plan;return}
if(!$PSCmdlet.ShouldProcess($dataDir,'Create the separate IxForge folder, restricted ACL, owner credential and disabled configuration')){return}
if(Test-Path -LiteralPath $configPath){throw 'Configuration already exists; inspect and preserve it before any provisioning.'}
if(Test-Path -LiteralPath $plan.OwnerKey){throw 'An owner credential already exists; retain it for explicit ACL/content review instead of reusing or rotating it automatically.'}
if(!(Test-Path -LiteralPath $stateRoot -PathType Container)){throw 'The approved existing IX state directory must already exist.'}
$ancestor=Get-Item -LiteralPath $stateRoot
while($ancestor){if($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse-point ancestors require separate path review.'};$ancestor=$ancestor.Parent}
$serviceSid=([Security.Principal.NTAccount]'NT SERVICE\ixctl').Translate([Security.Principal.SecurityIdentifier]).Value
$allowed=@($OwnerSid,$serviceSid,'S-1-5-18','S-1-5-32-544')
if(Test-Path -LiteralPath $dataDir){
 if((Get-Item -LiteralPath $dataDir).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Workshop directory must not be a junction or symlink.'}
 $acl=Get-Acl -LiteralPath $dataDir
 if(!$acl.AreAccessRulesProtected){throw 'Existing Workshop folder ACL is not protected; explicit security review is needed.'}
 foreach($rule in $acl.Access){$sid=$rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value;if($sid -notin $allowed){throw 'Existing Workshop folder grants an unapproved identity; no automatic ACL replacement.'}}
}else{
 New-Item -ItemType Directory -Path $dataDir | Out-Null
 $acl=[Security.AccessControl.DirectorySecurity]::new();$acl.SetAccessRuleProtection($true,$false)
 $inherit=[Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit'
 foreach($entry in @(@($OwnerSid,'ReadAndExecute'),@($serviceSid,'Modify'),@('S-1-5-18','FullControl'),@('S-1-5-32-544','FullControl'))){$rule=[Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new($entry[0]),[Security.AccessControl.FileSystemRights]$entry[1],$inherit,[Security.AccessControl.PropagationFlags]::None,[Security.AccessControl.AccessControlType]::Allow);$acl.AddAccessRule($rule)}
 Set-Acl -LiteralPath $dataDir -AclObject $acl
}
if(!(Test-Path -LiteralPath $plan.OwnerKey)){
 $random=[byte[]]::new(32);$rng=[Security.Cryptography.RandomNumberGenerator]::Create();try{$rng.GetBytes($random)}finally{$rng.Dispose()}
 $hex=[Convert]::ToHexString($random).ToLowerInvariant()
 $file=[IO.File]::Open($plan.OwnerKey,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
 try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes($hex+"`n");$file.Write($bytes,0,$bytes.Length);$file.Flush($true)}finally{$file.Dispose()}
}
@{version=1;enabled=$false;origin=$Origin;listen=$Listen}|ConvertTo-Json|Set-Content -LiteralPath $configPath -Encoding utf8
$plan
