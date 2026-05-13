---
date: 2026-05-13
purpose: Grounded analysis of GA issue #204 (Decide support status for legacy chatbot REST/SSE routes)
reversibility: two-way door — analysis only, no code/state change; surfaces a recommendation the GA owner can accept or reject
revisit-trigger: ChatWidget.tsx migrates to AG-UI, OR a SignalR client lands for /hubs/chatbot, OR docs/plans/2026-05-12-chatbot-cleanup-tracker.md appears in ga/
status: complete — finding: cannot deprecate yet; sequence migration first
---

# GA issue #204 — analysis and chat-surfaces.md draft

## TL;DR

The issue says "AG-UI and SignalR are the active public paths" and offers a binary keep-or-deprecate choice. **The premise is wrong as of 2026-05-13.** Legacy REST `/api/chatbot/chat` is the *only* chat surface with a canonical UI consumer. AG-UI and SignalR are server-side scaffolding without visible canonical clients. Deprecating now would break the canonical UI.

**Recommendation: keep legacy REST as the active surface, sequence migration of `ChatWidget.tsx` to AG-UI first, then deprecate.**

## Source of authority

- Issue body references `docs/plans/2026-05-12-chatbot-cleanup-tracker.md` — **does not exist** in ga repo (`ls` returns 404).
- Acceptance criterion in issue body: *"chat-surfaces.md lists explicit support status and caller inventory for each route."* That doc also doesn't exist yet — draft below.
- A github-actions bot tried to action this issue 2026-05-13T02:09:18Z, took 41 seconds, posted "I was unable to process your request" at 02:09:59. Another agent-fails-on-real-task data point — same shape as the green-but-dead Seldon Plan finding from `docs/plans/2026-05-12-self-improvement-loops.md`.

## Surface inventory

### Surface 1 — Legacy REST (`Apps/ga-server/GaApi/Controllers/ChatbotController.cs`)

`[Route("api/[controller]")]` → `/api/chatbot/*`

| Method | Path | Line | Description |
|---|---|---|---|
| `POST` | `/api/chatbot/chat/stream` | 43 | SSE streaming |
| `POST` | `/api/chatbot/chat` | 152 | sync chat |
| `GET` | `/api/chatbot/status` | 199 | health |
| `GET` | `/api/chatbot/examples` | 217 | sample prompts |
| `GET` | `/api/chatbot/demo` | 248 | demo page |

### Surface 2 — AG-UI REST (`Apps/ga-server/GaApi/Controllers/AgUiChatController.cs`)

`[Route("api/chatbot")]` (literal, not `[controller]`)

| Method | Path | Line | Description |
|---|---|---|---|
| `GET` | `/api/chatbot/skills` | 41 | skill enumeration |
| `POST` | `/api/chatbot/agui/json` | 51 | sync AG-UI |
| `POST` | `/api/chatbot/agui/stream` | 109 | streaming AG-UI |

### Surface 3 — SignalR (`Apps/ga-server/GaApi/Hubs/ChatbotHub.cs`)

Mapped at `/hubs/chatbot` (Program.cs:500). Doc comment: *"Anonymous: this hub backs the public demo at demos.guitaralchemist.com/chatbot/"*.

| Method | Signature |
|---|---|
| `SendMessage` | `(string message, bool useSemanticSearch = true)` |
| `ClearHistory` | `()` |
| `GetHistory` | `()` → `List<ChatMessage>` |
| `SearchKnowledge` | `(string query, int limit = 10)` → `List<SemanticSearchResult>` |

## Caller inventory

| Surface / route | Canonical UI (`ga-react-components`, port 5176) | Deprecated UI (`ga-client`, port 5173) |
|---|---|---|
| Legacy `/api/chatbot/chat` | **YES** — `ChatWidget.tsx:917` | YES — `chatApi.ts:179` |
| Legacy `/api/chatbot/chat/stream` | no | YES — `chatApi.ts:85`, `chatService.ts:94` |
| Legacy `/api/chatbot/status` | no | YES — `chatApi.ts:47`, `chatService.ts:170` |
| Legacy `/api/chatbot/examples` | no | YES — `chatApi.ts:217`, `chatService.ts:185` |
| Legacy `/api/chatbot/demo` | no | YES — `chatService.ts:218` |
| AG-UI any route | no | no |
| SignalR `/hubs/chatbot` | no | no — possibly used by external `demos.guitaralchemist.com/chatbot/` (out of repo) |

**Verification:** `grep -rn 'chatbotHub\|/chatbot-hub\|HubConnection.*[Cc]hat' ../ga --include="*.ts" --include="*.tsx" --include="*.cs"` returns empty for non-server code. No SignalR client in the workspace.

**Verification:** `grep -n 'agui/stream\|agui/json' ../ga --include="*.ts" --include="*.tsx"` (would need to run — left as exercise; the absence in caller inventory above is from the initial `api/chatbot` grep).

## Decision

The issue framed this as binary (keep vs deprecate). The honest answer is **neither — sequence the migration**:

1. **Phase A (this PR):** publish `chat-surfaces.md` documenting current state. Mark legacy REST as **active**, AG-UI as **available but unused**, SignalR as **available (external client only)**.
2. **Phase B (next PR):** migrate `ChatWidget.tsx:917` from `/api/chatbot/chat` → `/api/chatbot/agui/stream`. Verify SSE semantics match.
3. **Phase C (PR after that):** add `[Obsolete("Use /api/chatbot/agui/stream", error: false)]` to `ChatbotController.cs` chat actions with sunset date 90 days out.
4. **Phase D (90 days later):** delete `ChatbotController` chat/chat-stream actions. Keep `status/examples/demo` if still needed by external consumers.

The deprecated `ga-client` frontend's dependence on legacy REST does **not** block this — that frontend is documented as "usually off" per `reference_ga_two_react_apps.md` and can either migrate alongside the canonical UI or be retired.

## Drafted `chat-surfaces.md` content

Copy-paste-ready for `ga/docs/chatbot/chat-surfaces.md`:

````markdown
# Chatbot surfaces — support status and caller inventory

Last reviewed: 2026-05-13. Issue: [#204](https://github.com/GuitarAlchemist/ga/issues/204).

GA exposes **three** chatbot wire surfaces. Only one has a canonical UI consumer today.

## Surface matrix

| Surface | Routes / hub | Server controller / hub | Canonical UI uses it? | Deprecated UI uses it? | Status |
|---|---|---|---|---|---|
| Legacy REST | `POST /api/chatbot/chat`, `/chat/stream`, `GET /api/chatbot/status`, `/examples`, `/demo` | `ChatbotController.cs` | **YES** — `ChatWidget.tsx:917` calls `/api/chatbot/chat` | yes — `ga-client/services/{chatApi,chatService}.ts` | **active** (only canonical-UI surface) |
| AG-UI | `GET /api/chatbot/skills`, `POST /api/chatbot/agui/json`, `/agui/stream` | `AgUiChatController.cs` | no | no | **available, no in-repo client** |
| SignalR | hub at `/hubs/chatbot`: `SendMessage`, `ClearHistory`, `GetHistory`, `SearchKnowledge` | `ChatbotHub.cs` | no | no | **available, external client only** (`demos.guitaralchemist.com/chatbot/`) |

## Per-route caller inventory

### Legacy REST (`ChatbotController`)

- `POST /api/chatbot/chat`
  - canonical: `ReactComponents/ga-react-components/src/components/PrimeRadiant/ChatWidget.tsx:917`
  - deprecated: `Apps/ga-client/src/services/chatApi.ts:179`
- `POST /api/chatbot/chat/stream`
  - canonical: none
  - deprecated: `Apps/ga-client/src/services/chatApi.ts:85`, `chatService.ts:94`
- `GET /api/chatbot/status`
  - canonical: none
  - deprecated: `Apps/ga-client/src/services/chatApi.ts:47`, `chatService.ts:170`
- `GET /api/chatbot/examples`
  - canonical: none
  - deprecated: `Apps/ga-client/src/services/chatApi.ts:217`, `chatService.ts:185`
- `GET /api/chatbot/demo`
  - canonical: none
  - deprecated: `Apps/ga-client/src/services/chatService.ts:218`

### AG-UI REST (`AgUiChatController`)

- `GET /api/chatbot/skills` — no in-repo callers
- `POST /api/chatbot/agui/json` — no in-repo callers
- `POST /api/chatbot/agui/stream` — no in-repo callers

### SignalR hub (`ChatbotHub` at `/hubs/chatbot`)

- `SendMessage(message, useSemanticSearch)` — no in-repo callers (external demo page only)
- `ClearHistory()` — no in-repo callers
- `GetHistory()` — no in-repo callers
- `SearchKnowledge(query, limit)` — no in-repo callers

## Support decision (2026-05-13)

**Legacy REST is the only canonical-UI surface today.** Deprecation cannot proceed until the canonical UI migrates. Sequence:

1. **Phase A (now):** this doc — publish current state.
2. **Phase B:** migrate `ChatWidget.tsx:917` to `POST /api/chatbot/agui/stream`. Confirm SSE chunk semantics match.
3. **Phase C:** add `[Obsolete("Use /api/chatbot/agui/stream", error: false)]` to `ChatbotController` chat/chat-stream actions with 90-day sunset.
4. **Phase D (90 days after C):** delete `chat`/`chat/stream`. Re-evaluate `status`/`examples`/`demo` based on remaining usage.

The deprecated `ga-client` frontend (port 5173, "usually off" per `reference_ga_two_react_apps.md`) is not a blocker — migrate it alongside the canonical UI or retire.

## Verification commands

```bash
# Caller inventory
grep -rn 'api/chatbot' Apps/ ReactComponents/ --include='*.ts' --include='*.tsx' --include='*.cs'

# SignalR consumers (should be empty in-repo)
grep -rn 'chatbotHub\|HubConnection.*[Cc]hat' Apps/ ReactComponents/ --include='*.ts' --include='*.tsx'
```
````

## Next actions on the issue

When the GA owner reviews this:

1. Drop the `chat-surfaces.md` block above into `ga/docs/chatbot/chat-surfaces.md`.
2. Edit issue #204 body to replace "AG-UI and SignalR are the active public paths" with "AG-UI is the migration target; SignalR backs the external demo."
3. Close #204 once `chat-surfaces.md` lands (Phase A complete).
4. Open a follow-up issue for Phase B (`ChatWidget` AG-UI migration).
