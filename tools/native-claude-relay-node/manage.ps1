[CmdletBinding()]
param(
    [ValidateSet('Plan','Install','Start','Stop','Status','Uninstall')][string]$Action='Plan',
    [switch]$ApprovedPersistentActivation,
    [switch]$StartNow
)
$ErrorActionPreference='Stop'
$taskName='Gaia.NativeClaudeRelay'
$taskAccount=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskNode=(Get-Command node.exe -CommandType Application -ErrorAction Stop).Source
$taskInstall=Join-Path $env:LOCALAPPDATA 'Gaia\NativeClaudeRelay'
$taskState=Join-Path $taskInstall 'state'
$taskAgent=Join-Path $taskInstall 'agent.mjs'
$taskArguments='"'+$taskAgent+'" run --root "'+$taskState+'"'
$taskPlan=[ordered]@{taskName=$taskName;account=$taskAccount;executable=$taskNode;arguments=$taskArguments;workingDirectory=$taskInstall;trigger=('AtLogOn '+$taskAccount);logonType='Interactive';runLevel='Limited';multipleInstances='IgnoreNew';restartMinutes=1;restartCount=3;activationApplied=$false}
if($Action -eq 'Plan'){$taskPlan|ConvertTo-Json;return}
if($Action -eq 'Install'){
    if(-not $ApprovedPersistentActivation){throw 'EXPLICIT_PERSISTENT_ACTIVATION_APPROVAL_REQUIRED'}
    if([Security.Principal.WindowsIdentity]::GetCurrent().Name -ine $taskAccount){throw 'USER_CONTEXT_MISMATCH'}
    if((Test-Path -LiteralPath $taskInstall) -or (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)){throw 'EXISTING_INSTALLATION_OR_TASK_REFUSED'}
    $taskManifest=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw|ConvertFrom-Json
    foreach($taskFile in $taskManifest.files){
        $taskSource=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot $taskFile.path))
        if(-not $taskSource.StartsWith([IO.Path]::GetFullPath($PSScriptRoot)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PACKAGE_PATH_REFUSED'}
        if((Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash -ine $taskFile.sha256){throw 'PACKAGE_HASH_MISMATCH'}
    }
    New-Item -ItemType Directory -Path $taskInstall|Out-Null
    foreach($taskFile in $taskManifest.files){$taskDestination=Join-Path $taskInstall $taskFile.path;New-Item -ItemType Directory -Path (Split-Path -Parent $taskDestination) -Force|Out-Null;Copy-Item -LiteralPath (Join-Path $PSScriptRoot $taskFile.path) -Destination $taskDestination}
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Destination (Join-Path $taskInstall 'manifest.json')
    New-Item -ItemType Directory -Path $taskState|Out-Null
    $taskAction=New-ScheduledTaskAction -Execute $taskNode -Argument $taskArguments -WorkingDirectory $taskInstall
    $taskPrincipal=New-ScheduledTaskPrincipal -UserId $taskAccount -LogonType Interactive -RunLevel Limited
    $taskTrigger=New-ScheduledTaskTrigger -AtLogOn -User $taskAccount
    $taskSettings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $taskAction -Principal $taskPrincipal -Trigger $taskTrigger -Settings $taskSettings|Out-Null
    if($StartNow){Start-ScheduledTask -TaskName $taskName}
    [pscustomobject]@{state='INSTALLED';started=[bool]$StartNow;task=$taskName}|ConvertTo-Json
    return
}
if($Action -eq 'Status'){
    if(Test-Path -LiteralPath $taskAgent){& $taskNode $taskAgent status --root $taskState}else{[pscustomobject]@{state='NOT_INSTALLED'}|ConvertTo-Json}
    return
}
$taskExisting=Get-ScheduledTask -TaskName $taskName
$taskAccountSid=([Security.Principal.NTAccount]::new($taskAccount)).Translate([Security.Principal.SecurityIdentifier]).Value
function Resolve-TaskOwnerSid([string]$taskUserId){
    if($taskUserId -match '^S-1-') {return ([Security.Principal.SecurityIdentifier]::new($taskUserId)).Value}
    return ([Security.Principal.NTAccount]::new($taskUserId)).Translate([Security.Principal.SecurityIdentifier]).Value
}
$taskActualSid=Resolve-TaskOwnerSid $taskExisting.Principal.UserId
if($taskExisting.Actions.Count -ne 1 -or $taskExisting.Actions[0].Execute -ine $taskNode -or $taskExisting.Actions[0].Arguments -cne $taskArguments -or $taskActualSid -ne $taskAccountSid){throw 'TASK_OWNERSHIP_MISMATCH'}
if($Action -eq 'Start'){
    $taskDisabled=Join-Path $taskState 'disabled.flag'
    if(Test-Path -LiteralPath $taskDisabled){Remove-Item -LiteralPath $taskDisabled}
    Start-ScheduledTask -TaskName $taskName
    [pscustomobject]@{state='START_REQUESTED'}|ConvertTo-Json
    return
}
& $taskNode $taskAgent stop --root $taskState
$taskDeadline=[DateTime]::UtcNow.AddSeconds(12)
do{
    $taskStatusPath=Join-Path $taskState 'status.json'
    $taskStatus=if(Test-Path -LiteralPath $taskStatusPath){Get-Content -LiteralPath $taskStatusPath -Raw|ConvertFrom-Json}else{$null}
    if((Get-ScheduledTask -TaskName $taskName).State -ne 'Running' -and (-not $taskStatus -or -not $taskStatus.activeRequest)){break}
    Start-Sleep -Milliseconds 250
}while([DateTime]::UtcNow -lt $taskDeadline)
if((Get-ScheduledTask -TaskName $taskName).State -eq 'Running'){throw 'STOP_NOT_CONFIRMED_NO_UNINSTALL'}
if($taskStatus -and $taskStatus.activeRequest){throw 'OWNED_REQUEST_CLEANUP_UNCERTAIN_NO_UNINSTALL'}
if($Action -eq 'Stop'){[pscustomobject]@{state='STOPPED'}|ConvertTo-Json;return}
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
$taskManifest=Get-Content -LiteralPath (Join-Path $taskInstall 'manifest.json') -Raw|ConvertFrom-Json
foreach($taskFile in $taskManifest.files){
    $taskRemoval=[IO.Path]::GetFullPath((Join-Path $taskInstall $taskFile.path))
    if(-not $taskRemoval.StartsWith([IO.Path]::GetFullPath($taskInstall)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'UNINSTALL_PATH_REFUSED'}
    if(Test-Path -LiteralPath $taskRemoval){Remove-Item -LiteralPath $taskRemoval}
}
Remove-Item -LiteralPath (Join-Path $taskInstall 'manifest.json')
[pscustomobject]@{state='UNINSTALLED_STATE_RETAINED';evidenceDirectory=$taskState}|ConvertTo-Json
