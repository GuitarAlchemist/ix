#!/usr/bin/env bash
# Run one Pi probe against a throwaway checkout of this repository.
#
# Pi is a second, independent agent harness: a different runtime, a different
# provider, and no governance layer. That independence is the whole point -- its
# answer is only evidence about the repository if it was produced without the
# governed lane's prompts, skills or tools in the loop.
#
# Two properties this script is responsible for:
#
#   1. Pi cannot damage anything. Its `bash` and `write` tools can mutate
#      whatever they can reach and Pi ships no path restriction, so the probe
#      runs in an ephemeral `git worktree`. The real working tree is never the
#      cwd. Afterwards the worktree is diffed, so "it did not write" is a
#      measurement rather than a hope.
#   2. The raw event stream is kept verbatim. Normalisation is a separate step
#      (`to-contract.mjs`) so a parsing mistake can never destroy the evidence.
#
# usage: run-probe.sh <prompt-file> [model]
set -euo pipefail

PROMPT_FILE_ARG="${1:?usage: run-probe.sh <prompt-file> [model]}"
# Absolute, and read before any `cd`: the prompt used to be interpolated inside
# the subshell that had already chdir'd into the worktree, so a relative path
# silently resolved to nothing and pi ran with an empty message -- exiting 0.
PROMPT_FILE="$(cd "$(dirname "$PROMPT_FILE_ARG")" && pwd)/$(basename "$PROMPT_FILE_ARG")"
MODEL="${2:-gpt-5.2}"
PROVIDER="${PI_PROBE_PROVIDER:-openai}"
# `read` alone cannot search -- Pi has no grep or glob tool -- so a search probe
# needs `bash`. Containment is the worktree, not the allowlist.
TOOLS="${PI_PROBE_TOOLS:-read,bash}"

# bash here understands MSYS paths like /c/Users/...; the native Windows python
# and node binaries do not. Convert at every boundary where a path leaves bash.
winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi; }

ROOT="$(git rev-parse --show-toplevel)"
IX_PI="$ROOT/IX-pi"
PI_CLI="$IX_PI/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
[[ -f "$PI_CLI" ]] || { echo "pi is not installed; run: (cd IX-pi && npm install --ignore-scripts)" >&2; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { echo "no such prompt file: $PROMPT_FILE_ARG" >&2; exit 1; }
PROMPT_TEXT="$(cat "$PROMPT_FILE")"
[[ -n "${PROMPT_TEXT//[[:space:]]/}" ]] || { echo "prompt file is empty: $PROMPT_FILE" >&2; exit 1; }

SLUG="$(basename "$PROMPT_FILE" .md)"
STAMP="$(date -u +%Y-%m-%d)"
RUN_ID="${STAMP}-${SLUG}"
WORKTREE="$IX_PI/worktree/$SLUG"
RAW="$IX_PI/state/raw/${RUN_ID}.events.jsonl"
META="$IX_PI/state/raw/${RUN_ID}.meta.json"

# The probe must describe a known commit, not a dirty tree, or a disagreement
# cannot be attributed to anything.
BASE_SHA="$(git rev-parse HEAD)"

rm -rf "$WORKTREE"
mkdir -p "$(dirname "$WORKTREE")" "$IX_PI/state/raw"
git worktree prune
git worktree add --detach --quiet "$WORKTREE" "$BASE_SHA"
trap 'cd "$ROOT" && git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true' EXIT

STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
( cd "$WORKTREE" && node "$PI_CLI" \
    --print --mode json \
    --provider "$PROVIDER" --model "$MODEL" \
    --tools "$TOOLS" \
    --session-dir "$IX_PI/sessions" \
    -- "$PROMPT_TEXT" ) > "$RAW" 2> "${RAW%.jsonl}.stderr"
PI_EXIT=$?
set -e
ENDED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# What did the agent actually touch? Measured, not assumed.
DIRTY="$(cd "$WORKTREE" && git status --porcelain | head -50)"
DIRTY_COUNT="$(printf '%s' "$DIRTY" | grep -c . || true)"

python - "$(winpath "$META")" <<PY
import hashlib, json, os, subprocess, sys
prompt = open(r"""$(winpath "$PROMPT_FILE")""", "rb").read()
json.dump({
    "run_id": "$RUN_ID",
    "prompt_file": os.path.relpath(r"""$(winpath "$PROMPT_FILE")""", r"""$(winpath "$ROOT")""")
        # Forward slashes regardless of host: the artifact is meant to be
        # diffable across machines, and Windows relpath yields os.sep.
        .replace(os.sep, "/"),
    "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
    "base_sha": "$BASE_SHA",
    "harness": {
        "name": "pi",
        "version": subprocess.run(
            ["node", r"""$(winpath "$PI_CLI")""", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "provider": "$PROVIDER",
        "model": "$MODEL",
        "tool_allowlist": "$TOOLS".split(","),
    },
    "run": {"started_at": "$STARTED", "ended_at": "$ENDED", "exit_code": $PI_EXIT},
    "worktree_dirty_paths": $DIRTY_COUNT,
}, open(sys.argv[1], "w"), indent=2)
PY

# A run that produced only a session header asserted nothing about this
# repository, and reporting it as a success is the same green-but-dead shape the
# probe itself is looking for. Fail closed instead.
EVENTS="$(grep -c . "$RAW" || true)"
if [[ "$EVENTS" -le 1 ]]; then
  echo "PROBE FAILED: pi emitted $EVENTS event(s) -- a session header and nothing else." >&2
  echo "The agent never ran, so this run is not evidence. stderr:" >&2
  sed 's/^/  /' "${RAW%.jsonl}.stderr" >&2
  exit 1
fi

echo "raw events : $RAW"
echo "meta       : $META"
echo "pi exit    : $PI_EXIT"
echo "worktree   : $DIRTY_COUNT path(s) modified by the agent"
[[ "$DIRTY_COUNT" -gt 0 ]] && printf '%s\n' "$DIRTY"
exit 0
