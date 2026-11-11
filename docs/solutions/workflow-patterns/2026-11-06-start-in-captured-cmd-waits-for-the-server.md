---
title: "A server started with `start` inside a captured `cmd /c` keeps the caller waiting until the server exits"
category: workflow-patterns
date: 2026-11-06
tags: [windows, cmd, start, pipe, handle-inheritance, powershell, runbook, server]
symptom: "A runbook script that runs `start \"name\" server.exe serve ...` and then a few curl calls, launched from PowerShell as `cmd.exe /c run.cmd 2>&1 | Out-String`, finishes its own lines but the PowerShell call never returns (tool timeout after 120 s), while the server is up and answering"
root_cause: "`start` gives the new process a copy of the caller's inheritable handles, including the stdout/stderr pipe that PowerShell reads. The pipe reaches end-of-file only when every writer has closed it, and the server, still running, holds a writer, so `Out-String` waits for the server to exit"
---

# `start` in a captured `cmd /c`: the caller waits for the server

## Problem

Checking the `go/ixmux` runbook on 2026-11-06, Branche ran the cmd.exe steps from PowerShell:

```powershell
$o = cmd.exe /c "C:\...\run.cmd" 2>&1 | Out-String
```

`run.cmd` started `ixmux.exe serve` with `start "ixmux-verify" /min ...`, waited for its token file, called
`/v1/status` and wrote the answer to a file. Every line of the script ran: the server was up, the status file was
written. The PowerShell call still hung until the tool's 120 s timeout moved it to the background.

## Why

`start` launches the server with the caller's inheritable handles. Among them is the write end of the pipe that
PowerShell reads `cmd`'s output from. A pipe reports end-of-file only when every write handle is closed; `cmd` closes
its own when it exits, but the server keeps its copy for as long as it runs. `Out-String` waits for end-of-file, so
it waits for the server.

## Fix

Do not capture the output of a command that starts a long-lived process. Either of these returns as soon as the
script ends:

- redirect the script's output to a file inside the `cmd` line, and read the file afterwards:
  `Start-Process cmd.exe -ArgumentList '/c', "C:\...\run.cmd > C:\...\run.out 2>&1" -NoNewWindow -Wait`;
- start the server from PowerShell itself with `Start-Process -WindowStyle Minimized`, which does not hand it the
  pipe, and keep `cmd` only for the short client calls.

Typing the runbook in an interactive cmd.exe window has no such problem: there is no pipe to inherit.

## Prevention

A runbook that starts a server says so next to the `start` line (the `go/ixmux` README EN/FR do). An agent that
checks a runbook from a tool call redirects to a file rather than capturing.
