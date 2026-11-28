# IxForge controlled updates

The reference IX Control Plane inspected by the parent is commit `4d758c1`, app0.5.14: authenticated version/deployment identity, explicit guarded reload, opt-in auto-reload. Its live correspondence and issue identifier are unknown. It has no inspected service-worker, binary installer, signature verification or service rollback. Do not describe it as an existing Windows updater.

IxForge already has its own scoped service-worker update negotiation. It preserves IndexedDB drafts and exact pending operations, waits for typing/dictation and coordinates tabs. Its scope, start URL and storage identity remain stable. Client0.1.1 adds functional local project creation and the canonical QR navigation fix. Replacing the PC executable is a separate, explicitly approved operation.

`Update-IxForge.ps1` provides a native one-shot maintenance path in the existing host ecosystem. It opens no server, creates no credential and accepts no browser-supplied command. Default invocation returns an inert plan: active/available binary and client versions, exact SHA256, unchanged host/spec/configuration checks, and `Automatic=false`. Hash verification is integrity against the reviewed value; `SignatureVerified=false` is explicit. There is no remote artifact discovery, publisher signature or unattended service updating.

Only `-Apply` after a concrete owner approval and Windows elevation may execute it. The updater rechecks the current binary, command, account, spec and configuration, holds an exclusive maintenance lock, stages and hashes the candidate, drains/stops ixctl, backs up the exact old executable, atomically replaces it, and restarts the same service with the same arguments. It checks the following:
- the ready-state PID and schema;
- the new local client version (`version` in `branding.json`);
- the private HTML with `<title>IxForge</title>`;
- that the API refuses unauthenticated requests;
- that the Tailscale Serve neighbours are unchanged: routes, the executable that owns each port, and availability. The updater re-observes them just before every stop and never identifies them by content.

Unknown is never treated as healthy. Keys, SQLite, browser replicas/outbox, network routes and configuration are retained. No live SQLite/WAL copying or schema migration occurs. From the counter-IXLET release on, the first start creates the additive `workshop_xlets` table, and from the checklist release on `workshop_xlet_items` (schema version stays 1); a restored older binary still opens the file and ignores those tables, so a rollback keeps the instances for a later re-update.

Each attempt has `workshop/update-<run-id>/receipt.json` and a verified `previous.exe`. States include available, backed_up, installed, completed, rolled_back, failed_before_replacement and reconciliation_required. An uncertain outcome is inspected through that receipt and current hashes before any retry. A failed startup or health check restores the reviewed prior executable. An unrelated host or configuration change, or a corrupt backup, stops for reconciliation rather than overwriting another owner.

After a failed replacement, the service restarts only on the verified previous executable, and only after the full health check. Otherwise it stays stopped as `reconciliation_required`.

[RECEIPTS.md](RECEIPTS.md) and [receipt.schema.json](receipt.schema.json) are the consumer contract. They cover the receipt fields, the states, the `ScribeSHA256` → `Neighbours` migration and the `branding.json` fields.

Explicit rollback uses `-Rollback -BackupDirectory <recorded-directory> -CandidateSHA256 <updated-hash> -ExpectedInstalledSHA256 <previous-hash>`. It preserves project data and access, verifies the backup and current host, and does not stop an already restored version again. It does not remove the application route or return to a pre-IxForge configuration. The older initial-release rollback is a separate fallback.

## Updating from the app

From R30 on, the owner can start the same reviewed update from the app, without a terminal or a UAC prompt each time. The browser never supplies a command or a path, only a version and a hash.

1. `Publish-IxForgeRelease.ps1 -Version 0.1.0-ixforge.N -Notes "…"` builds `go/ixctl` from a committed checkout. It writes `%ProgramData%\IxForge\releases\<version>\{ixctl.exe, release.json}`, where `release.json` holds the version, SHA-256, commit, client version, build time and notes. It runs under the owner's account and never overwrites a published version.
2. The Desktop shows admin devices only the releases that are newer than the one it runs and whose folder, manifest and file hash agree. It shows them in the Devices drawer under « Mise à jour du Desktop ».
3. « Installer » writes `update-request.json` (version and hash) to `<state>\workshop\updates`. The Desktop refuses with:
   - `updater_absent` when the updater has not been seen in the last five minutes;
   - `update_pending` while another request waits;
   - 400 for a release that is not offered or whose hash differs.
4. The task « IxForge Updater » runs `Run-IxForgeUpdate.ps1` as SYSTEM every minute. On each run it writes a heartbeat. When a request is there, it:
   - takes the request, so it is never run twice;
   - checks the release again;
   - copies `ixctl.exe` to `%ProgramData%\IxForge\staging` (SYSTEM and administrators only) and checks the copy's hash;
   - runs `Update-IxForge.ps1 -Apply` with the PC's current installed hash, spec hash and service command;
   - writes the receipt's state to `update-result.json`, which the drawer shows.

   An uncertain or failed outcome is reported and never retried.
5. `Install-IxForgeUpdater.ps1 -OwnerSid <sid>` prints its plan and changes nothing. With `-Apply`, it needs elevation and the owner's explicit order, and installs once:
   - the three scripts in `C:\Program Files\ixctl\updater`, hash-checked against the plan;
   - the ACLs of `%ProgramData%\IxForge`;
   - the task.

   `-Uninstall -Apply` removes the task and the scripts.

Installing the updater is the owner's one approval for later updates. Each update is still the reviewed `Update-IxForge.ps1`, with its backup, health checks, neighbour checks and restore on failure.

`Test-IxForgeUpdate.ps1` exercises seventeen synthetic cases:
- the inert version and integrity plan;
- a changed artifact;
- a replay of the same artifact;
- a successful backup, replacement, restart and receipt;
- an explicit, idempotent rollback;
- a failed health check;
- a failed candidate startup;
- a concurrent maintainer;
- a changed owner command;
- a corrupt backup;
- unobservable neighbours;
- a failed replacement: unknown, held, removed or unhealthy executable.

`workshop_update_neighbours_test.go` and `workshop_receipt_contract_test.go` cover the neighbours, the receipts and the contract. The SCM/network adapters exist only in that test process; fixture data and keys remain byte-identical and the fixture is cleaned. No real service mutation is part of those tests.

The actual PC browser is not exposed by the available UI adapter. Installing the PWA or proving its standalone launch is still unverified, even if its HTTPS manifest and assets are valid. No PWA installation, pairing, session widget or xlet runtime is implied by a successful backend update. The proposed session widget (mandate/activity/blocker/result, bounded authorized actions) remains a later product slice.
