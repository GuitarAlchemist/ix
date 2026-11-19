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
