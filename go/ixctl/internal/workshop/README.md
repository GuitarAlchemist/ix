# IxForge — local integration candidate

IxForge creates versioned Project/Workspace drafts from an editable French or English transcript. It is an installable tablet PWA backed by a dedicated Desktop SQLite database. It does not generate arbitrary programs. The templates create a counter or a checklist: a widget and an IXLET definition of the same kind and title. Once the project is saved, a device can activate that IXLET and the Desktop then runs it.

Commands: `Créer un workspace "Maison" avec un compteur "Lectures"` and `Créer un workspace "Courses" avec une liste "Épicerie"` (also `liste à cocher`); in English, `Create a workspace "Home" with a counter "Readings"` and `Create a workspace "Home" with a checklist "Groceries"`.

## IXLET runtime (counter and checklist)

- **Activation is explicit.** A definition stays a draft until a device presses **Activer sur le Desktop**, and only for a project already committed on the Desktop (`ixlet_not_saved` otherwise). Activating again never resets a running instance.
- **The Desktop is the authority.** One instance per activated definition lives in `workshop_xlets` (kind, value, version) in the same Workshop SQLite file, with checklist entries in `workshop_xlet_items` (id, position, text, done). It survives a restart and is shared by every approved device. A counter starts from the presenting widget's `initial_value`; a checklist starts empty and holds at most 200 items of 1 to 240 characters.
- **Widget pairing.** A widget presents the IXLET of the same workspace with the same kind and title (the template's pairing). No document field was added. The document kinds are closed to `counter` and `checklist`; a checklist widget has no initial value.
- **Actions** go to `POST /workshop/api/xlets/actions` with an `op_id`: `increment`, `decrement`, `reset` for a counter; `add` (with `text`), `check`, `uncheck`, `remove` (with `item_id`) for a checklist. A field that does not belong to the action, or an action of the other kind, is refused; a removed item is `ixlet_item_missing`. The same device, `op_id` and payload return the first result without running again, and another payload under that `op_id` is refused. `GET /workshop/api/xlets` lists every instance with its items; each sync rereads it, the higher version winning per instance.
- **No uncertain retry.** A lost, 5xx or malformed reply is shown as an uncertain result; nothing is resent automatically and the next sync rereads the Desktop's state. Controls are disabled while the Desktop is not connected, and the last known state is labelled as such. Text being typed for a new item survives the periodic re-render.
- **Local run before activation.** While the Desktop holds no instance of a definition, the device runs the IXLET itself under the same rules (`applyLocal` in `web/sync.js`): the state lives in that browser's IndexedDB replica only (`localXlets`), survives closing the app, works unpaired and offline, and the tool says "Sur cet appareil seulement". Nothing is sent before pairing.
- **Automatic sync and activation.** There is no save or activate button. Each sync (every 5 s, and after each change) queues every changed project without a conflict, then activates on the Desktop every IXLET of a saved, settled project that has no instance yet. The activation carries the device's local state as its `seed` (`{"value":n}` or `{"items":[{"text","done"}]}`, validated like actions, at most 200 items); the Desktop applies a seed only when it creates the instance (`seeded: true`), never over a running one. Each activation is sent once: an unclear reply ends the turn, and the next reread of the Desktop's instances decides; a local copy equal to the hosted instance is dropped. If another device activated first, the local copy stays on the device and the tool says so. A definite refusal (400) is recorded and not retried.
- **Administration from the app.** The PC the owner linked from the owner page ("Relier ce PC"), or a device the owner granted from there (`POST /workshop-owner/api/devices/{id}/admin`, approved devices only), administers the others from the Devices drawer: `GET /workshop/api/devices`, `POST /workshop/api/devices/invite` (an ordinary invite, approval still required), `POST /workshop/api/devices/{id}/{approved|refused|revoked}`. Other devices get 403, anonymous requests 401, and the administering device cannot revoke itself (400). Its authority is its device token, weaker than the owner page's local key; the owner accepted this on 2026-11-14.

## Client (one screen, Lit widgets)

- **Screen.** A top bar of icons (sync pill, surface, devices, language, appearance), a rail of tools with the new-tool field, and the open tool taking the rest of the screen; no page scroll. Phones get the rail as a strip.
- **Widgets** are custom elements built with Lit 3.3.1 (`web/widgets.js`, vendored `web/vendor/lit-all.min.js`, see `THIRD_PARTY.md`): `<ix-widget>` picks `<ix-checklist>` or `<ix-counter>`. They render into the light DOM, so the stylesheet and the CSP (no inline styles, no inline scripts) apply, and they never talk to the network: the shell hands them the state, whether they can act, and how.
- **New tool.** The field takes a name (Liste or Compteur, Enter makes a list), the documented sentence ("Créer un workspace … avec une liste …"), or a pasted widget. Each tool is its own project.
- **Detach.** A tool opens alone in its own window (`/workshop/#widget=<key>`), to put on another screen. The windows share the IndexedDB replica and talk over a `BroadcastChannel` ("changed", a "detached" heartbeat, "gone", "reattach"); the main window shows where the tool went and brings it back.
- **Surface.** `/workshop/#surface` shows the pinned tools side by side and nothing else, in a window of its own.
- **Export and import (Claude artifacts).** A tool exports as a standalone HTML page that works offline, with its state in `<script type="application/json" id="ixforge-widget">`: `{"format":"ixforge-widget/1","kind":"checklist","title":"…","items":[{"text":"…","done":false}]}` or `{"format":"ixforge-widget/1","kind":"counter","title":"…","value":0}`. The page can be published as a Claude artifact; Claude can also write a page carrying such a block. Its "Copier pour IxForge" button copies the JSON. Pasting the JSON or the whole page in the new-tool field, or importing the file, makes a new local tool with that state, shared once the Desktop has it.
- **Access** follows device approval: anonymous, pending, refused or revoked devices can neither read nor drive an instance.
- **Schema.** Both tables are additive to schema version 1 (created if absent, version unchanged), so restoring an earlier binary still opens the file; it simply ignores the instances. That earlier binary refuses to save a document containing a checklist (`invalid_document`); such a project stays on the device.

## Terminal (the PC only)

Authorized by the owner on 2026-11-14 (« autorisé terminal ») in its "PC only" form (`terminal.go`):

- **Who.** The PC the owner links from the owner page ("Relier ce PC") has the terminal. Any other device only once the owner allows it from the owner page, device by device ("Autoriser le terminal", with a confirmation), and withdraws it the same way. Administering devices does not include it. A revoked device loses it with its access. The standalone site has none.
- **How.** `POST /workshop/api/terminal {"command": …}` runs the command with Windows PowerShell (`-NoProfile -NonInteractive -EncodedCommand`, UTF-8 output) as the service's own account (`NT SERVICE\ixctl`, not the owner's Windows account), in the folder `terminal` beside the database. One command at a time, stopped after 30 s (`timed_out`, exit code −1), output capped at 256 KiB (`truncated`), command at most 4096 bytes. The reply carries the output and the exit code; an uncertain reply is shown as such and never resent.
- **Journal.** Each run is written to `workshop_terminal_log` (time, device, command, exit code, timeout, duration, output size; the last 1000), never its output. The app keeps the runs in the page's memory only.
- **Tables.** `workshop_terminal` and `workshop_terminal_log` are additive to schema version 1, like the others.

## Standalone site and optional AI

- **Standalone site.** `go run ./cmd/ixforge-site -out <new folder>` writes IxForge as static files for any static host (`WriteSite`): the client alone under `workshop/`, `branding.json` with `"standalone": true`, a service worker with its own cache name, `_headers` (the client's CSP, `nosniff`, `no-referrer`, `no-cache`) and `_redirects` (`/` → `/workshop/`). No server code, owner page or database. Opened in a browser, it is installable (PWA), runs on that device only (data in IndexedDB), works offline once loaded, offers no pairing and never calls a Desktop API. Publishing it is a separate, explicit step.
- **Any folder, any static host.** The client's paths are relative to its own folder, so `go run ./cmd/ixforge-site -app -out <new folder>` (`WriteApp`) writes the app alone for a folder of a host that sets no headers, such as GitHub Pages (`spareilleux.github.io/apps/ixforge/`); its page then carries the client's policy in a `<meta http-equiv="Content-Security-Policy">` (without `frame-ancestors`, which a meta element cannot carry). Pages of the same origin share its IndexedDB, caches and service-worker namespace; the app only touches its own (`ix-workshop-replica`, `ix-workshop-shell-*`, its folder's scope).
- **Optional AI.** In Réglages (standalone) or Appareils (Desktop), a person may save their own Claude API key; it stays in the device's IndexedDB, and only its last four characters are shown. With a key, "Créer avec l'IA" sends the sentence in the new-tool field from the browser straight to `https://api.anthropic.com/v1/messages` (`web/ai.js`: model `claude-opus-5-5`, effort `low`, structured output against a JSON schema, header `anthropic-dangerous-direct-browser-access`). The cost is billed to that key. The answer is validated as an `ixforge-widget/1` widget before a local tool is created; a refused key, a declined request, a rate limit or a malformed answer is explained and creates nothing. "Oublier la clé" removes it. Without a key, nothing about AI runs. The client's CSP allows `connect-src 'self' https://api.anthropic.com` and nothing else.

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
- **Linking the Desktop's own PC.** The IxForge window on the PC that hosts the Desktop is a browser client like any other and needs access too. On the owner page, "Relier ce PC" (`POST /workshop-owner/api/link`, owner session required) issues an invite whose claim is approved at once: the owner's own action is the approval. Same secret, two-minute lifetime, single use and guess limit as any invite. Its link (`#pair=…&link=1`) makes the client claim it without a code, under the name "PC" unless one was typed. Ordinary pairing is unchanged and stays pending until approved.
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
