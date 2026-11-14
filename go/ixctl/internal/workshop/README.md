# IxForge — local integration candidate

IxForge creates versioned Project/Workspace drafts from an editable French or English transcript. It is an installable tablet PWA backed by a dedicated Desktop SQLite database. It does not generate arbitrary programs or execute IXLETs. The counter is a draft template, with a clearly labelled preview. The IXLET runtime is not connected.

The integration is implemented in this isolated ixctl host through an optional service-run configuration, disabled by default. No installed service, Tailscale route, production credential or production data was changed. It uses an additional loopback application face in the same ixctl process, not a new daemon.

## Integration boundary

The intended host is the existing IX process. In an owned checkout of its current source, copy this package and reconcile the module dependency changes. The isolated repository's two snapshot commits are provenance, not commits to cherry-pick into IX.

The host entry point is service run --workshop-config FILE. Without that flag (or with enabled=false), the module opens no file or listener. See deploy/workshop/README.md for the prepared configuration and release boundaries.


Choose `existingHostStateDir` from the existing service's local state directory and its already granted rights. The proposed default, if the existing directory is confirmed, is `C:\ProgramData\ixctl\workshop\<instance>.workshop.sqlite`. No real file or directory was created there and no ACL was changed. Do not use the OneDrive-backed Bureau, Scribe's database, a shared SQLite/WAL file, or a tablet copy of SQLite.

`Handler` requires one exact HTTPS origin and refuses foreign Host/Origin requests. PWA and API use the same origin, under `/workshop/`. The optional host integration mounts this separate face; actual activation/origin remain subject to the IX host owner's release decision. Preserve all existing `/v1` loopback, local Host/Origin and global bearer checks. Never send that global bearer to this package's client or turn Workshop into an ixctl proxy.

The local owner UI is implemented under /workshop-owner/ with independent Host/Origin checks and temporary owner sessions. It displays QR codes and pending devices and supports explicit approval/refusal/revocation without a terminal. A Desktop shortcut proves the running endpoint through HMAC and obtains an in-memory, runtime-bound session; the persistent owner key never traverses HTTP or reaches the browser. Sessions expire after 15 minutes and do not survive host restart. The tablet cannot access this authority or /v1. See deploy/workshop/README.md.

## Data and pairing

- SQLite is authoritative. A save transaction writes the current document, append-only revision/event and device-scoped operation receipt together. The identical `op_id` and payload return the original receipt; changed payloads are refused. Base-revision mismatches are visible conflicts.
- The tablet keeps a replica, transcript, drafts and exact outbox operations in IndexedDB. Local transforms are transactional. A receipt is removed from the outbox only after local persistence succeeds. Authentication errors, explicit refusal/revocation, ambiguous 401, outages and HTTP body timeouts retain local data.
- Conflict resolution is explicit and operates on a whole project. The local and received Desktop documents remain available until the user chooses. Automatic field-level merging and revision-history browsing are outside this slice.
- QR invites last two minutes, are single-use, and are invalidated after five wrong guesses or a process restart. The secret is in the URL fragment and is removed from browser history before pairing. Device credentials are independent of IX and Scribe; SQLite stores their hashes. Access remains pending until owner approval.
- After explicit refusal/revocation, the user can request fresh pairing without deleting drafts/outbox. The expected store identity is checked before consuming the new invite. A different database does not silently acquire the previous database's outbox.

## Client updates

The stable technical identity remains `/workshop/`, `ix-workshop-replica` and the `ix-workshop-shell-` cache prefix. The display name/version come from `web/branding.json`. The build generates manifest/HTML names and a service-worker cache digest covering every bundled client asset. The Go release test fails if assets changed without rebuilding the worker.

The client checks on startup, reconnect and every minute. A new worker installs the complete shell and waits. It asks every open Workshop tab to finish editing/dictation and local writes/synchronization, briefly locks ready tabs, then activates and reloads them safely. A busy/missing/new tab defers activation; a bounded lock lease recovers from a lost negotiation. Each browser profile coordinates its own tabs independently. Outbox receipts are never cleared by a client update. HTTP 502 and network failures can reopen the cached shell; API responses, tokens and SQLite are never put in the service-worker cache.

This is client updating only. Replacing the Desktop executable or restarting the Windows service remains a separately approved host release.

## Voice and transcript

Try `Créer un workspace "Maison" avec un compteur "Lectures"` or switch to English and use `Create a workspace "Home" with a counter "Readings"`. Interpretation shows an intention first; creation requires the explicit next action. The bounded parser is a deterministic template, not a connected language model.

Dictation starts only on request and only if the browser reports an already available on-device language pack (`fr-FR` or `en-US`). There is no remote recognizer fallback or automatic pack installation. Unsupported and denied states leave the editable transcript usable. Browser tests inject a synthetic recognizer; no microphone/audio, paid model or provider API was used or verified.

## Build and verify

From this directory, pinned development dependencies are TypeScript 5.9.3 and jsQR 1.4.0. Runtime i18next 25.6.0 is vendored locally.

```text
npm run build
npm run check-build
npm test
npm run lint
npm run typecheck
```

From `go/ixctl`:

```text
go generate ./internal/workshop
go vet ./...
go build ./...
go test -mod=readonly ./...
```

The JS type check covers the replica, synchronizer, voice adapter and updater. The DOM application and service worker additionally have syntax and real-browser verification. No claim of TypeScript coverage for every UI line is made.

For the full isolated browser/QR suite, set `WORKSHOP_BROWSER=1`, `WORKSHOP_QR_DECODER` to jsQR's `dist/jsQR.js` when not installed locally, and optionally `WORKSHOP_ARTIFACT_DIR` to an evidence folder, then run `go test -mod=readonly -v -timeout 240s ./internal/workshop`.

The harness creates a new SQLite fixture and ephemeral loopback TLS listener, refuses Scribe's production ports, uses `https://workshop.invalid:<ephemeral-port>`, and pins only the test certificate in each freshly created Chrome profile. DNS mapping exists only in that disposable browser; no system trust, Tailscale route or certificate is installed. Browser identity is checked against the spawned PID and its own DevToolsActivePort. Test-created browsers/profiles and temporary databases are cleaned up. The fixture owner endpoint exists only in `_test.go`.

The test certificate is the Go httptest fixture certificate. This is not a proposal to use a manual certificate in production. Real installation requires Tailscale HTTPS and an explicitly approved app route, as described in [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve). PWA service workers require a secure context; see [MDN's service-worker guide](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API/Using_Service_Workers).

## Remaining steps for a real tablet

1. Reconcile the integrated candidate with the IX owner's current checkout; its lifecycle and owner UI are implemented in this isolated copy.
2. Confirm the exact application-facing mux/origin and existing writable local state directory, preserving `/v1` and the Scribe route.
3. Review the concrete host release and private Tailscale HTTPS path before any binary replacement, service restart, route change or real database creation.
4. With confirmation at the action, create a real temporary invite, pair and approve the real tablet. Install from its browser and verify offline/reconnect and an actual client update.
5. Separately verify that tablet's on-device speech support and microphone permission. Keep transcript entry available when unsupported.

No source was extracted from the private Claude artifact and none was republished. The runtime reactor work provides no execution binding to this candidate.
