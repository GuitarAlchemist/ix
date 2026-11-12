---
title: "The ixctl Windows service is Running and healthy, yet it will not come back after a reboot"
category: integration-issues
date: 2026-11-12
tags: [ixctl, windows-service, scm, start-type, reboot, resume-check, windows]
symptom: "services.msc and sc.exe query show ixctl RUNNING, /health answers ok:true, the workshop answers 200; after the next reboot nothing listens until someone starts the service by hand"
root_cause: "The install plan creates the service with `sc.exe create ... start= demand` (Manual). Running is the current state, start type is what the SCM does at boot: a manual service that was started by hand stays up until the reboot, then stays down."
---

# A Running service that will not survive a reboot

## Problem

The installed ixctl service was checked the usual way and looked healthy on every surface:

- `Get-CimInstance Win32_Service -Filter "Name='ixctl'"`: `State Running`;
- the control plane's `GET /health` (the only route without a token): `200 {"ok":true,...}`;
- the IxForge workshop's `GET /workshop/branding.json` (with its configured `Host`):
  `200 {"name":"IxForge","short_name":"IxForge","version":"0.1.1"}`, locally and over the tailnet.

None of these says what happens at boot. The resume check did:

```text
resume check: 2 components, 1 not ready
1  plane     degraded  service ixctl is running, start type manual: it will not start after a reboot
                       remedy (run by a human): elevated: sc.exe config ixctl start= delayed-auto
2  workshop  ok        a listener holds 127.0.0.1:8761
```

## Root cause

`ixctl service install`'s plan (`go/ixctl/service.go`) creates the service with `start= demand`,
which the SCM calls Manual. A manual service started once stays Running until the machine
restarts, then nothing starts it. State and start type are two different fields of the SCM, and
every health probe only sees the first.

Changing it needs `SERVICE_CHANGE_CONFIG`. The installed service's DACL
(`sc.exe sdshow ixctl`) grants that to SYSTEM and elevated Administrators only, so a
medium-integrity session (Administrators "deny only" in `whoami /groups`) cannot do it, and should
not try to work around it.

## Fix

A human runs, in an elevated terminal:

```powershell
sc.exe config ixctl start= delayed-auto
```

It changes only the boot behaviour: the service is not restarted, so nothing it serves goes down.
Then `ixctl resume check <inventory.json>` reports `plane ok ... start type automatic (delayed)`
and exits 0. Whether the install plan itself should say `delayed-auto` is a product choice the
plan does not document yet.

## How to find it again

```powershell
(Get-CimInstance Win32_Service -Filter "Name='ixctl'").StartMode   # Manual = will not come back
sc.exe qc ixctl                                                    # START_TYPE : 3 DEMAND_START
ixctl resume check inventory.json                                  # a component with "service": "ixctl"
```

The resume check reads the start type with query rights only; a running service whose start type
is not automatic is `degraded`, never `ok`.
