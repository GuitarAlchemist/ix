---
title: "A printed PowerShell plan for a Windows service goes on after a failed step: deletes files while the service still runs, reports a start that failed"
category: workflow-patterns
date: 2026-10-30
tags: [powershell, windows-service, scm, sc.exe, ixctl, go/ixctl, windows, fail-open]
symptom: "`sc.exe stop` then `Remove-Item` on the state directory fails on open files; `sc.exe start` exits 0 although the service stops with code 1 a moment later; a WaitForStatus timeout prints an exception and the next line (`sc.exe delete`) still runs"
root_cause: "sc.exe start/stop only ask the SCM (StartService returns at START_PENDING); a native non-zero exit and a .NET method exception do not stop a PowerShell block unless $ErrorActionPreference = 'Stop' and $PSNativeCommandUseErrorActionPreference = $true, the second mainstream only from PowerShell 7.4; Stop-Service waits with no deadline on a wedged STOP_PENDING"
---

## Problem

`ixctl service plan` prints the install, start, status, stop and uninstall commands an
administrator pastes into PowerShell. Three Codex rounds on the first cut found the plan failing
open in three different ways, each a behaviour of PowerShell or the SCM rather than of ixctl:

- `sc.exe stop X` followed by `sc.exe delete X` and `Remove-Item` of the state directory: the
  stop is only a request, so the removal races the service's own stop (up to its 30 s bound) and
  fails half-way on open log and journal files.
- `(Get-Service X).WaitForStatus('Stopped', [TimeSpan]::FromSeconds(60))` added after it: on
  timeout it throws a `MethodInvocationException`, which in a default session is not terminating,
  so the following delete still runs.
- `sc.exe start X`: `StartService` returns once the service's main thread exists, at
  START_PENDING. A service whose setup fails (missing spec) exits with code 1 after `sc.exe start`
  already printed success.
- `Stop-Service X` instead: correct for an already stopped service, but it polls every two seconds
  with no overall deadline while the service stays STOP_PENDING.

## Root cause

PowerShell keeps going after a native command's non-zero exit and after a statement-terminating
error unless the session says otherwise. `$ErrorActionPreference = 'Stop'` promotes the
exceptions; `$PSNativeCommandUseErrorActionPreference = $true` makes a native non-zero exit an
error. The latter is experimental (off) in 7.3 and mainstream from 7.4. The SCM's start and stop
are asynchronous by design.

## Fix

Commits b06af23e, 11a52cc8 and ba16cd22 on `feat/ix-windows-service` (`go/ixctl/service.go`,
`installPlan`): the plan names PowerShell 7.4 or later; every section opens with
`$ErrorActionPreference = 'Stop'; $PSNativeCommandUseErrorActionPreference = $true`; a start is
`sc.exe start X` then `WaitForStatus('Running', 30 s)`; a stop is `Stop-Service X -NoWait` then
`WaitForStatus('Stopped', 60 s)`, before any delete. `TestTheInstallPlanIsExplicitAndStoresNoSecret`
checks the text; nothing of the plan has run against an installed service.

## How to find it again

In pwsh 7.4+, with no service touched:

```powershell
pwsh -NoProfile -Command { cmd.exe /c exit 5; 'AFTER' }   # prints AFTER: fails open
pwsh -NoProfile -Command { $ErrorActionPreference = 'Stop'; $PSNativeCommandUseErrorActionPreference = $true; cmd.exe /c exit 5; 'AFTER' }   # stops
pwsh -NoProfile -Command { $ErrorActionPreference = 'Stop'; (Get-Service Spooler).WaitForStatus('Paused', [TimeSpan]::FromMilliseconds(300)); 'AFTER' }   # stops on the timeout
```
