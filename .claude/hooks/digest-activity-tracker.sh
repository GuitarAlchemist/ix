#!/usr/bin/env bash
# PostToolUse hook (matcher: Edit|Write|Bash) — increments mutation counter.

set -e
repoRoot="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$repoRoot" ] && exit 0

digestDir="$repoRoot/state/digests"
counterPath="$digestDir/.activity-counter"
mkdir -p "$digestDir"

count=0
if [ -f "$counterPath" ]; then
  raw="$(cat "$counterPath" 2>/dev/null | tr -d '[:space:]')"
  if echo "$raw" | grep -qE '^[0-9]+$'; then
    count="$raw"
  fi
fi
count=$((count + 1))
echo "$count" > "$counterPath"
exit 0
