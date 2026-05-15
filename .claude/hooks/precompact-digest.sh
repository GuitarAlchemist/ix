#!/usr/bin/env bash
# PreCompact hook — archives state/digests/latest.md and writes a metadata-only
# fallback if /digest wasn't invoked before compaction.

set -e
repoRoot="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$repoRoot" ] && exit 0

digestDir="$repoRoot/state/digests"
archDir="$digestDir/archive"
latest="$digestDir/latest.md"
mkdir -p "$digestDir" "$archDir"

sessionId="unknown"
if [ -t 0 ]; then
  :
else
  payload="$(cat || true)"
  if [ -n "$payload" ] && command -v jq >/dev/null 2>&1; then
    val="$(echo "$payload" | jq -r '.session_id // empty' 2>/dev/null || true)"
    [ -n "$val" ] && sessionId="$val"
  fi
fi

ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
tsFile="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"

if [ -f "$latest" ]; then
  cp -f "$latest" "$archDir/$tsFile-$sessionId.md"
  if [ -n "$(find "$latest" -mmin -30 2>/dev/null)" ]; then
    exit 0
  fi
fi

branch="$(git -C "$repoRoot" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
headSha="$(git -C "$repoRoot" rev-parse --short HEAD 2>/dev/null || echo unknown)"
headSubj="$(git -C "$repoRoot" log -1 --format='%s' 2>/dev/null || echo unknown)"

openPr=""
if command -v gh >/dev/null 2>&1; then
  prNum="$(gh pr view --json number -q .number 2>/dev/null || true)"
  [ -n "$prNum" ] && openPr="#$prNum"
fi
prLine=""
[ -n "$openPr" ] && prLine="**Open PR:** $openPr"$'\n'

cat > "$latest" <<EOF
---
schema_version: 1
session_id: $sessionId
written_at: $ts
trigger: precompact-hook-fallback
branch: $branch
head_sha: $headSha
head_subject: $headSubj
open_pr: $openPr
---

# Session digest (fallback — /digest was not invoked before compaction)

**Branch:** $branch @ $headSha — $headSubj
$prLine
## Model-driven sections

_No \`/digest\` invocation was captured before this compaction. Re-orient from
\`git log\` and the open PR. Invoke \`/digest\` mid-session to populate the
**Next action**, **In-flight**, **Live hypotheses**, **Open questions**, and
**Do NOT carry forward** sections before the next compaction event._
EOF
exit 0
