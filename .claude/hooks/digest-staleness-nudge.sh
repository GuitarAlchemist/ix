#!/usr/bin/env bash
# UserPromptSubmit hook — emits additionalContext nudge when state has drifted.
# Karpathy R1+R4. Silent when digest is fresh.

set -e

# Cross-platform mtime — closes macOS/BSD portability gap from code review.
get_mtime_sec() {
  stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}
get_age_min() {
  local mtime
  mtime="$(get_mtime_sec "$1")"
  if [ -z "$mtime" ] || [ "$mtime" = "0" ]; then echo 0; return; fi
  echo "$(( ($(date +%s) - mtime) / 60 ))"
}

repoRoot="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$repoRoot" ] && exit 0

latest="$repoRoot/state/digests/latest.md"
counter="$repoRoot/state/digests/.activity-counter"

digestAgeMin=""
if [ -f "$latest" ]; then
  digestAgeMin="$(get_age_min "$latest")"
fi

mutationCount=0
if [ -f "$counter" ]; then
  raw="$(cat "$counter" 2>/dev/null | tr -d '[:space:]')"
  if echo "$raw" | grep -qE '^[0-9]+$'; then
    mutationCount="$raw"
  fi
fi

shouldNudge=false
reason=""
if [ -z "$digestAgeMin" ]; then
  shouldNudge=true
  reason="No session digest exists yet. Invoke /digest at your next natural breakpoint."
elif [ "$digestAgeMin" -gt 30 ] && [ "$mutationCount" -gt 10 ]; then
  shouldNudge=true
  reason="Last digest $digestAgeMin min ago; $mutationCount mutations since. Karpathy R4: task complete != goal achieved — consider /digest to mark success criteria status."
elif [ "$digestAgeMin" -gt 90 ]; then
  shouldNudge=true
  reason="Last digest $digestAgeMin min ago. Session drifting — invoke /digest before the next compaction."
fi

[ "$shouldNudge" = false ] && exit 0

if command -v jq >/dev/null 2>&1; then
  jq -n --arg ctx "[digest-nudge] $reason" '{additionalContext: $ctx}'
else
  printf '{"additionalContext":"[digest-nudge] %s"}\n' "$reason"
fi
exit 0
