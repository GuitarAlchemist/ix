#!/usr/bin/env bash
# PostToolUse hook (matcher: Edit|Write|Bash) — increments mutation counter.
# Enhancement 1 (Cherny periodic mid-session digest): when activity threshold
# OR time threshold is hit, write a mid-session metadata digest so we survive
# crashes/network drops without waiting for Stop or PreCompact.

set -e
repoRoot="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$repoRoot" ] && exit 0

digestDir="$repoRoot/state/digests"
archDir="$digestDir/archive"
counterPath="$digestDir/.activity-counter"
midCounterPath="$digestDir/.activity-count"
latest="$digestDir/latest.md"
mkdir -p "$digestDir" "$archDir"

# Existing mutation counter (for staleness nudge)
count=0
if [ -f "$counterPath" ]; then
  raw="$(cat "$counterPath" 2>/dev/null | tr -d '[:space:]')"
  if echo "$raw" | grep -qE '^[0-9]+$'; then
    count="$raw"
  fi
fi
count=$((count + 1))
echo "$count" > "$counterPath"

# Enhancement 1: independent counter for mid-session digest gating
midCount=0
if [ -f "$midCounterPath" ]; then
  raw="$(cat "$midCounterPath" 2>/dev/null | tr -d '[:space:]')"
  if echo "$raw" | grep -qE '^[0-9]+$'; then
    midCount="$raw"
  fi
fi
midCount=$((midCount + 1))
echo "$midCount" > "$midCounterPath"

# Thresholds: N=20 mutations OR M=30 minutes since last digest
THRESHOLD_COUNT="${IX_DIGEST_MID_COUNT:-20}"
THRESHOLD_MIN="${IX_DIGEST_MID_MIN:-30}"

get_mtime_sec() {
  stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}

ageMin=99999
if [ -f "$latest" ]; then
  mtime="$(get_mtime_sec "$latest")"
  if [ -n "$mtime" ] && [ "$mtime" != "0" ]; then
    ageMin=$(( ($(date +%s) - mtime) / 60 ))
  fi
fi

shouldWrite=false
if [ "$midCount" -ge "$THRESHOLD_COUNT" ]; then
  shouldWrite=true
elif [ "$ageMin" -ge "$THRESHOLD_MIN" ] && [ "$midCount" -ge 3 ]; then
  shouldWrite=true
fi

[ "$shouldWrite" = false ] && exit 0

# Reset counter
echo "0" > "$midCounterPath"

tsIso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
tsFile="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"

branch="$(git -C "$repoRoot" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
headSha="$(git -C "$repoRoot" rev-parse --short HEAD 2>/dev/null || echo unknown)"
headSubj="$(git -C "$repoRoot" log -1 --format='%s' 2>/dev/null || echo unknown)"

# Rotate latest -> archive before overwriting
[ -f "$latest" ] && cp -f "$latest" "$archDir/$tsFile-pre-mid.md"

midPath="$digestDir/mid-$tsFile.md"
cat > "$midPath" <<EOF
---
schema_version: 1
session_id: mid-session-auto
written_at: $tsIso
trigger: activity-tracker-mid-session
branch: $branch
head_sha: $headSha
head_subject: $headSubj
mutations_since_last: $midCount
---

# Session digest (mid-session auto — activity threshold reached)

**Branch:** $branch @ $headSha — $headSubj

## Model-driven sections

_Auto-written by digest-activity-tracker after $midCount mutations / ${ageMin}m since last digest.
Invoke \`/digest\` at your next natural breakpoint to populate **Next action**,
**In-flight**, **Live hypotheses**, **Open questions**, **Do NOT carry forward**,
and **Success criteria**._
EOF

cp -f "$midPath" "$latest"
exit 0
