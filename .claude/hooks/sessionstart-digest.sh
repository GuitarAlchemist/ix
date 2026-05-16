#!/usr/bin/env bash
# SessionStart hook — emits state/digests/latest.md to stdout for additionalContext.
# Skips silently if missing or >24h stale.

set -e

# Cross-platform mtime (replaces GNU-only `find -printf '%T@'`).
# Closes the macOS/BSD portability gap from the 2026-05-15 code review.
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
[ ! -f "$latest" ] && exit 0

if [ -z "$(find "$latest" -mmin -1440 2>/dev/null)" ]; then
  echo ""
  echo "Session digest exists but is >24h old — skipping injection. Run /digest to refresh."
  echo ""
  exit 0
fi

ageMin="$(get_age_min "$latest")"

if [ "$ageMin" -lt 60 ]; then
  ageStr="$ageMin min"
else
  ageStr="$((ageMin / 60))h $((ageMin % 60))m"
fi

echo ""
echo "=== Session digest (last written $ageStr ago) ==="
cat "$latest"
echo "=== End digest ==="
echo ""
exit 0
