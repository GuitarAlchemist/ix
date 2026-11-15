# IxForge updater contract for consumers

Contract version: **receipt schema 1, documentation revision R9 (2026-11-11)**. It describes the code of `Update-IxForge.Core.ps1` at this revision. `workshop_receipt_contract_test.go` keeps the two in step: if the code drifts, that test fails, and the document must be updated with it.

## 1. `branding.json`, the IxForge name and version

`GET <origin>/workshop/branding.json` serves the embedded file `internal/workshop/web/branding.json` as it is, with no runtime field added. At this revision the file contains exactly:

```json
{"name":"IxForge","short_name":"IxForge","version":"0.1.1"}
```

| Field | Meaning | Who reads it |
|---|---|---|
| `name` | Product name shown by the PWA (title, header) | PWA (`app.js`), the manifest handler (`http.go` copies `name` and `short_name`). **The updater does not read it.** |
| `short_name` | Short PWA name | Manifest handler |
| `version` | **IxForge client (PWA) version**, pattern `^[0-9][A-Za-z0-9.+-]{0,80}$` | Updater: the plan records it as `PreviousClientVersion`, and the health check compares it with `ExpectedClientVersion`. The PWA also shows it as `name · version`. |

To check the name **and** the version, a consumer such as IX Service reads `name` and `version` from this one document. These are the exact field names; no other field carries the version.

What this field is not:
- `version` is not the binary version. The `ixctl` binary has its own version, from `ixctl version`, recorded in receipts as `CandidateVersion` and `PreviousVersion`. It is not a hash either.
- No field carries a commit or an executable hash. Those are only in the receipt.

The updater itself checks the name differently: it looks for `<title>IxForge</title>` on `GET <origin>/workshop/`, not for `name` in `branding.json`.

## 2. Update receipt

**Location:**
- receipt: `<DataDir>\update-<RunID>\receipt.json` (UTF-8 JSON, atomic write);
- backup next to it: `previous.exe`.

**Schema:** [`receipt.schema.json`](receipt.schema.json) (JSON Schema 2020-12). The schema is open: unknown fields are tolerated and ignored.

**Admission rules for a rollback** (`Undo-IxForgeUpdate`). A receipt that breaks any of them is **refused before the lock and before any service stop**:
1. The JSON is readable. A truncated or empty file is refused.
2. `Schema == 1`, and `RunID` is 32 lowercase hex characters.
3. `BackupDirectory` equals both `<DataDir>\update-<RunID>` and the directory passed to the rollback.
4. `CandidateSHA256` and `PreviousSHA256` equal the hashes passed to the rollback.
5. Each of the following fields is present and not empty: `Service`, `InstalledExe`, `ExpectedImagePath`, `StateDir`, `DataDir`, `InstanceName`, `ConfigPath`, `ConfigSHA256`, `SpecPath`, `SpecSHA256`, `Origin`, `Listen`, `PreviousClientVersion`, `ExpectedClientVersion`, `State`.
6. `State` is one of `available`, `backed_up`, `installed`, `completed`, `failed_before_replacement`, `reconciliation_required`, `rolled_back`.

**States:**

| State | Meaning for a consumer |
|---|---|
| `available` | Plan recorded, nothing stopped yet |
| `backed_up` | Service stopped, verified `previous.exe` written |
| `installed` | Candidate in place, health not yet verified |
| `completed` | Candidate running, full health check passed (the neighbours included) |
| `rolled_back` | Verified previous executable running, full health check passed against the previous version |
| `failed_before_replacement` | The update failed before the replacement. The verified previous executable was restarted, and its full health check passed. |
| `reconciliation_required` | Identity or health not proven: unknown executable, failed replacement, or a restarted previous version that is not healthy. **The service may be stopped. Never treat this state as healthy**: a human or the owner must reconcile. |

What a state does not tell:
- No state other than `completed` and `rolled_back` means that IxForge is healthy.
- A consumer must not infer the current state of the machine from the receipt alone. It must compare the receipt with the current executable hash and service state.

## 3. Migration from `ScribeSHA256` to `Neighbours`

| Receipt | Written by | Field | Rollback |
|---|---|---|---|
| Before HEALTH-R4 | `e998e3e` (ixforge.4) | `ScribeSHA256` (hash of the root page) | Admitted. `ScribeSHA256` is **ignored**. |
| Since HEALTH-R4 | this candidate | `Neighbours` (`Known`, `Reason`, `Routes[]`), or null before the stop | Admitted. `Neighbours` is **ignored and replaced**. |

There is no file migration step: old receipts stay as they are.

A rollback never trusts a recorded baseline. Just before it stops the service, it observes again every Tailscale Serve route except `/workshop`:
- the exact route;
- the owner executable of a loopback target;
- the HTTP status, without following redirects.

If that observation is not possible, the rollback is refused before the stop. After the restart, the verdict must be `healthy`.

**Neighbours carry no identity.** IxForge does not claim to know what a neighbour is. On 2026-11-10 the root route was observed serving WMUX, and Scribe was not published on any route. That fact is dated, not a rule.

### Fixtures

The fixtures in `testdata/receipts/` are fictitious. Their paths are placeholders (`{{STATE}}`, `{{DATA}}`, `{{BACKUP}}`), and their hashes are `AA…`, `BB…` and so on.

- **Accepted:**
  - `accepted-old-scribe-completed`
  - `accepted-new-neighbours-completed`
  - `accepted-new-reconciliation-required`
  - `accepted-new-neighbours-null-installed`
- **Refused:**
  - `refused-truncated`
  - `refused-unknown-schema`
  - `refused-unknown-state`
  - `refused-old-missing-listen`
  - `refused-new-empty-previous-client-version`
  - `refused-other-candidate`
  - `refused-runid-not-backup-directory`

## 4. Guarantees and non-guarantees for consumers

**Guaranteed** by the code at this revision and its tests:
- No service stop without a known baseline of the neighbours.
- No `completed`, `rolled_back` or `failed_before_replacement` without a full health check.
- No restart of an unknown executable.

**Not guaranteed:**
- Signature: `SignatureVerified` is always false.
- Behaviour on the real machine: it has not been observed.
- A choice between stopping and restarting when the candidate stays installed after a failed rollback. That policy is still open with the owner. The code leaves the service stopped and records `reconciliation_required`.
