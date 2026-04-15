# Friday Brief — NotebookLM Vendoring Runbook

**Status:** DRAFT — phase 2, not production.

This runbook is a placeholder for the vendoring steps needed to bring the
NotebookLM MCPs into the Friday Brief pipeline. **None of the steps below
have been executed.** Do not copy-paste commands without reviewing them
first — the SHAs are placeholders and the firewall rules depend on your
host OS.

## 1. Create the sacrificial Workspace identity

1. In Google Admin (or ask your Workspace admin), create a new user:
   `friday-brief-bot@customer.com`.
2. Restrict scopes: **no Drive, no Gmail, no admin SDK**. The account
   should only be able to sign in to NotebookLM.
3. Enable 2FA with a hardware key bound to the sandbox host.
4. Record the recovery email in the team password manager.

## 2. Clone and pin the NotebookLM MCPs

Clone into `./vendor/` at the repo root:

```bash
# Reader MCP — drives a headless Chrome to query NotebookLM
git clone https://github.com/PleasePrompto/notebooklm-mcp vendor/notebooklm-mcp
cd vendor/notebooklm-mcp
git checkout REPLACE_WITH_SHA   # pin before building
npm ci && npm run build
cd ../..

# Writer MCP — uploads arbitrary sources into a NotebookLM notebook
git clone https://github.com/qiaomu/anything-to-notebooklm vendor/anything-to-notebooklm
cd vendor/anything-to-notebooklm
git checkout REPLACE_WITH_SHA   # pin before installing
python -m venv .venv && source .venv/bin/activate && pip install -e .
cd ../..
```

Record the two SHAs in `.mcp.friday-brief.example.json` and in
`state/knowledge/{date}-friday-brief-vendor.knowledge.json`. **Never pin
to a tag or a branch** — git tags can be rewritten, branches move.

## 3. Sandbox profile directory

The two MCPs share a single Chrome user-data-dir so they operate on the
same NotebookLM session. That directory must be isolated from the
developer's main Chrome profile.

Conceptual setup on Windows:

1. Enable Windows Sandbox (`Turn Windows features on or off`).
2. Mount an encrypted VHD (BitLocker, per-user key) as `S:\nlm-sandbox`.
3. Set `SANDBOX_DIR=S:\nlm-sandbox` in the MCP launcher environment.
4. Inside the sandbox, launch Chrome with
   `--user-data-dir=%SANDBOX_DIR%\chrome-nlm`.

On Linux/macOS, substitute a rootless container (podman / toolbx) with
the encrypted volume mounted at `$SANDBOX_DIR`.

## 4. Egress firewall allowlist

Configure the sandbox's egress rules to allow **only** `*.google.com`.
Block everything else. On Windows Sandbox, this is easiest via a
WFP filter in the host policy:

```
netsh advfirewall firewall add rule ^
  name="friday-brief-allow-google" ^
  dir=out program="<path-to-sandbox-chrome.exe>" ^
  remoteip=any action=allow ^
  description="NotebookLM only"
```

(The real rule should use a FQDN filter; the exact syntax depends on
your firewall product. Document which one you used.)

## 5. Validation

Before running the Friday Brief against the real NotebookLM:

1. Launch Chrome inside the sandbox and confirm the profile path shows
   `S:\nlm-sandbox\chrome-nlm` (or the equivalent). It must **not**
   resolve to the developer's `%LOCALAPPDATA%\Google\Chrome\User Data`.
2. Browse to `chrome://version` inside the sandbox and confirm the
   profile path matches step 1.
3. Browse to any non-Google domain; it must fail to load.
4. Sign in to NotebookLM as `friday-brief-bot@customer.com`; confirm
   the account badge shows the Workspace tier (not `@gmail.com`).
5. Run `cargo run -p ix-friday-brief -- run` from the host and verify
   the MVP still produces a local brief — the sandbox is additive, not
   a replacement for the current local path.

Only after all five validations pass should you merge the contents of
`.mcp.friday-brief.example.json` into `.mcp.json` and rerun the
pipeline with the `tier_gate`/`upload`/`audio`/`scrape` nodes swapped
from stubs to real MCP calls.
