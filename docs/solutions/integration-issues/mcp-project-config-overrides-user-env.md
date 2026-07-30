---
title: "Project-scoped .mcp.json silently overrides the user-global server def and strips its env"
category: integration-issues
date: 2026-07-24
tags: [mcp, mcp-config, environment-variables, notebooklm, claude-code, config-precedence]
symptom: "An MCP server ignores an env var set in ~/.claude.json (e.g. NOTEBOOKLM_ACCOUNT=main); the server keeps launching with the wrong config no matter how many times you /mcp reconnect or /reload-plugins or even restart"
root_cause: "A project-scoped .mcp.json defines the same server WITHOUT the env block; project scope wins wholesale over the user-global definition, so the env never reaches the spawned process"
---

# Project .mcp.json overrides the user-global MCP server and drops its env

## Problem

The `notebooklm` MCP server kept reporting `authenticated: false` and could not be
queried (`ask_question` → *"Could not find NotebookLM chat input"*). The user-global
config (`~/.claude.json`) declared it correctly:

```json
"notebooklm": {
  "command": "npx", "args": ["notebooklm-mcp@latest"],
  "env": { "NOTEBOOKLM_ACCOUNT": "main", "NOTEBOOKLM_BROWSER_CHANNEL": "chromium" }
}
```

…yet the running server showed the *default* profile's library, not `main`'s. A
Google login (real Chrome / patchright) had written valid cookies + `state.json`
into the `main` profile, but the server never read that profile.

Things that did **not** fix it (all reuse the running process / its original env):
- `/mcp` → reconnect
- `/reload-plugins`
- A full quit-and-relaunch of Claude Code (!) — because the override persisted

## Root cause

The repo's **project-scoped `.mcp.json`** also declared `notebooklm`, but with **no
`env` block**:

```json
"notebooklm": { "command": "npx", "args": ["notebooklm-mcp@latest"] }
```

Project scope wins **wholesale** over the user-global definition — it does not merge
per-key. So in this repo the server always spawned with **no `NOTEBOOKLM_ACCOUNT`**,
landing on the default profile regardless of what `~/.claude.json` said. The auth
flag gates on a `state.json` storage-state file in whatever profile the process
actually reads — which was never the authenticated one.

## Solution

Diagnose by comparing **both** config scopes for the same server name:

```bash
grep -A8 -i "notebooklm" <repo>/.mcp.json        # project scope (wins)
grep -A20 -i "notebooklm" ~/.claude.json          # user-global (shadowed)
```

Then either add the missing `env` to the project entry, or remove the project
entry so the user-global one applies. (Here ownership moved to another repo, so
the entry was removed from this repo's `.mcp.json` entirely.)

**Env changes only take effect on a fresh server process** — a `/mcp` reconnect and
`/reload-plugins` both reuse the running process and its original environment. After
editing config, fully restart so the server re-spawns with the new env.

## Prevention

- When an MCP env var "isn't taking", **first check for a project `.mcp.json` that
  redeclares the same server** and shadows the user-global def — don't assume the
  user-global entry is the one in force.
- Keep a server defined in **one** scope, or keep the two definitions' `env` blocks
  in sync. Project scope replaces, it does not merge.
- Related gotcha for browser-driven MCP servers (NotebookLM): the working login used
  **real Chrome via patchright** (Google flags the bundled Chrome-for-Testing as "not
  secure"), so `NOTEBOOKLM_BROWSER_CHANNEL=chromium` may not match the profile the
  login wrote cookies to — the channel must match the login browser.
