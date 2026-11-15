#requires -Version 7.4
# Harness for TestRollbackReceiptContract. It puts one fictitious receipt from testdata/receipts into a marked fixture
# directory and asks the REAL Undo-IxForgeUpdate whether it admits it. Restore-IxForgeUpdateBinary is replaced by a
# recorder, so nothing is stopped, replaced or probed: this checks the receipt contract only, not the rollback itself.
param(
 [Parameter(Mandatory)][string]$Receipt,
 [Parameter(Mandatory)][string]$Dir
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '..\deploy\workshop\Update-IxForge.Core.ps1')
$Dir=[IO.Path]::GetFullPath($Dir)
if(!(Test-Path -LiteralPath (Join-Path $Dir 'ixforge-receipt-fixture'))){throw 'Refusing a directory that is not a test fixture.'}
function Test-IxForgeUpdateAdministrator {$true}
$script:admitted=$null
function Restore-IxForgeUpdateBinary($Plan){$script:admitted=$Plan}
$runID='0123456789abcdef0123456789abcdef'
$data=Join-Path $Dir 'workshop';$backup=Join-Path $data ('update-'+$runID)
New-Item -ItemType Directory -Force -Path $backup|Out-Null
$json={param($s)$s.Replace('\','\\')}
$text=[IO.File]::ReadAllText($Receipt).Replace('{{BACKUP}}',(& $json $backup)).Replace('{{DATA}}',(& $json $data)).Replace('{{STATE}}',(& $json $Dir))
[IO.File]::WriteAllText((Join-Path $backup 'receipt.json'),$text)
$failure=''
try{Undo-IxForgeUpdate $backup ('AA'*32) ('BB'*32)|Out-Null}catch{$failure=$_.Exception.Message}
[pscustomobject]@{admitted=($null -ne $script:admitted);error=$failure
 neighbours_reset=($null -ne $script:admitted -and $null -eq $script:admitted.Neighbours)}|ConvertTo-Json -Compress
