# IxForge host deployment — prepared, not executed

The candidate mounts IxForge in the existing ixctl process, behind an explicit `service run --workshop-config FILE` flag. The default remains disabled. One additional IPv4 loopback listener serves the application and the independently authenticated local owner UI; this is not another daemon. `/v1` continues on its original listener and guard, and is absent from the application face.

## Deployment status

This source candidate does not validate or modify any installed service. Before deployment, the host owner must establish the current executable hash, command, instance/state directory, permissions, listeners and approved HTTPS origin. Supply those values explicitly; no machine-specific defaults are included. Preserve unrelated applications and routes. Deployment remains a separate human-approved action.

## Concrete proposal to review

| Item | Proposed target / change |
|---|---|
| Process/account | Existing `ixctl` / `NT SERVICE\ixctl`; no account, startup-mode or second-service change |
| Binary | Review the new candidate commit and its built binary SHA256, then replace only `C:\Program Files\ixctl\ixctl.exe` |
| Service command | Preserve the exact approved existing command; append only `--workshop-config "<stateDir>\workshop\config.json"` |
| Local application port | `127.0.0.1:8761`, only if the approved inspection confirms availability; no bind/probe of that production port was attempted |
| Application origin | Proposed `https://your-approved-host.example`; same origin for PWA and API |
| Tablet path | `/workshop/`, with its existing scope, IndexedDB identity and cache prefix unchanged |
| Data | `<existing IX stateDir>\workshop\<instance>.workshop.sqlite`; if the existing directory is confirmed as the installer default, `C:\ProgramData\ixctl\workshop\<instance>.workshop.sqlite` |
| Owner secret | `<stateDir>\workshop\<instance>.workshop-owner.key`; separate, explicitly provisioned, never auto-created/rotated by service startup |
| Owner page | `http://127.0.0.1:8761/workshop-owner/`, opened by an `Administrer IxForge` Desktop shortcut; runtime-bound temporary session passed in fragment and removed from history; persistent key never transmitted or put in the shortcut |
| Owner access | Service SID: Modify; verified human owner SID: ReadAndExecute; SYSTEM/Administrators: FullControl; inherited broad access removed on the new Workshop folder only |
| Network | A private HTTPS Serve mount for `/workshop` only; no Funnel, firewall, ACL, CA or manual certificate changes |

The Desktop shortcut may live in the owner's OneDrive-backed Bureau. The active database and owner secret must not. The data-preparation script rejects UNC/OneDrive paths and reparse-point ancestors and existing Workshop junctions. Actual rights are still unknown; the listed rights are the minimum required proposal, not a claim that they are already granted.

## Ordered release and confirmation boundaries

1. **Read-only inspection approval first.** Obtain the current service command/account and non-secret instance/state-path fields; the owner's SID and target-folder ACL; `tailscale version` and `tailscale serve status --json`; and availability of only `127.0.0.1:8761`. Do not fetch global/Scribe tokens, full session files or manuscript data. If permission is refused again, stop the corresponding inspection and keep its fields unknown.
2. **Build/review locally.** `go generate ./internal/workshop`, applicable tests, `go vet ./...`, `go build`. The new module is off unless a version-1 enabled configuration is supplied. Unknown/corrupt database versions are refused before existing-schema DDL; fresh initialization and metadata/revision tables commit atomically. No production migration runs during preparation.
3. **Confirm data/security provisioning.** Review `Prepare-IxForgeData.ps1` in its default inert mode with the approved paths/SIDs. Only a subsequent explicit `-Provision` may create the dedicated restricted folder, new owner credential and a **disabled** configuration. It never reads a Scribe/global key, never replaces an existing configuration and never silently rotates a key.
4. **Confirm the service release/maintenance window.** `Release-IxForge.ps1` is inert without `-Apply`. It requires the candidate hash and exact reviewed old service command/hash. It rechecks the service identity, stops only ixctl, waits for stop, backs up the reviewed executable and config/service command, and copies only its dedicated SQLite file if it already exists and is drained. It never copies the global state/token files, the whole IX state directory or Scribe data. It arms restoration before the first installed-binary mutation, enables the reviewed config, appends the flag and restarts ixctl. A new local owner shortcut opens the UI without a terminal. No Tailscale route is changed by this script.
5. **Confirm the application route separately.** After approved inspection proves `/workshop` is unused and the local release is ready, the proposed current-CLI command is:

   `tailscale serve --bg --https=443 --set-path=/workshop http://127.0.0.1:8761/workshop`

   The repeated path in the target is deliberate: current Tailscale source strips the mount prefix before joining the proxy target path and preserves the incoming Host. This proposal is based on [official Serve CLI documentation](https://tailscale.com/docs/reference/tailscale-cli/serve) and [the current proxy implementation](https://github.com/tailscale/tailscale/blob/main/ipn/ipnlocal/serve.go); it has not been proven against the installed version or live route configuration. Check the approved installed version before execution. Never run `serve reset`, replace `/`, or publish the owner path or `/v1`. The previous health/version/clock authorization does not authorize this writable application mount.
6. **Confirm real pairing at the action.** Open the owner shortcut, press **Créer un QR de pairage**, scan on the real tablet in the same tailnet, request pairing, then explicitly approve the named pending device on Desktop. Creation of the real device credential is a new persistent access action. Install through the tablet browser. Verify offline/reconnect, a real client update and that device's supported microphone path. Draft creation never starts an IXLET.

## Rollback

If the newly approved `/workshop` mount was added, remove only that mount using the installed CLI's verified equivalent of:

`tailscale serve --bg --https=443 --set-path=/workshop off`

This is a separately confirmed network action. Do not reset Serve or alter Scribe's root mount. Then review `Rollback-IxForge.ps1` in its inert mode. Its `-Apply` checks the current candidate hash and expected service command and validates the complete binary/config backup before stopping ixctl. It restores the old binary/command and disabled config, restarts the same service, and moves only its recorded owner shortcut into its backup.

Keep Workshop SQLite, its operation receipts and its owner/device credentials in place. Do not automatically rewind the database, delete browser storage/outbox, copy SQLite/WAL live, or restore a whole state directory. The old binary ignores Workshop; an approved later retry can resume the same store identity. Credential revocation/removal or restoring older project data requires a distinct, concrete decision. The server executable does not auto-update; only the client PWA has the tested automatic update cycle.

## Script validation performed

All scripts were parsed with PowerShell's AST parser. The preparation/release/rollback default modes are plans only. The owner launcher was exercised with synthetic key/config using DryRun and VerifyOnly against the real fixture host and a counterfeit local endpoint. HMAC challenge/session/receipt verification passed; the fake endpoint never received the persistent key. Only a 15-minute session from the current host epoch reaches the browser. No browser was launched by the PowerShell verification test; a separate fresh-profile browser tested the full owner UI. No provisioning, service mutation, installed-binary copy, real shortcut, Tailscale command, certificate or production credential was executed/created by this tranche. Apply/rollback behavior remains subject to the concrete review/confirmation gates above.
