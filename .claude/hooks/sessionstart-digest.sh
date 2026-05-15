#!/usr/bin/env bash
# SessionStart hook — emits state/digests/latest.md to stdout for additionalContext.
# Skips silently if missing or >24h stale.

set -e
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

ageMin="$(find "$latest" -printf '%T@\n' 2>/dev/null | awk -v now="$(date +%s)" '{ printf "%d", (now - $1) / 60 }')"
[ -z "$ageMin" ] && ageMin=0

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
