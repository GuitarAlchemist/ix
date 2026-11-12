#!/usr/bin/env bash
# SessionStart hook — emits state/digests/latest.md to stdout for additionalContext.
# Skips silently if missing; says so in one line if >24h stale or an automatic stub.
# The ecosystem manifest is injected on every source but compact.
# Fixture: .claude/hooks/test-sessionstart-digest.sh

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
bootstrap="$repoRoot/.claude/manifest-bootstrap.md"

# What started the session: startup, resume, clear or compact (empty when not said).
payload="$(cat 2>/dev/null || true)"
event="$(printf '%s' "$payload" | sed -n 's/.*"source"[[:space:]]*:[[:space:]]*"\([a-z]*\)".*/\1/p')"

# After a compaction the summary carries the session, and the manifest (another repo's
# snapshot, already injected at startup) would only take room from it.
if [ "$event" != "compact" ]; then
  # Sync manifest dynamically via pwsh if available
  if command -v pwsh &> /dev/null; then
    pwsh -File "$repoRoot/scripts/manifest-sync.ps1" &> /dev/null || true
  elif command -v powershell &> /dev/null; then
    powershell -File "$repoRoot/scripts/manifest-sync.ps1" &> /dev/null || true
  fi

  # Inject local/live ecosystem manifest into context
  if [ -f "$bootstrap" ]; then
    echo ""
    echo "=== Ecosystem Manifest (Vite Dev Data) ==="
    cat "$bootstrap"
    echo "=== End Manifest ==="
    echo ""
  fi
fi

[ ! -f "$latest" ] && exit 0

if [ -z "$(find "$latest" -mmin -1440 2>/dev/null)" ]; then
  echo ""
  echo "Session digest exists but is >24h old — skipping injection. Run /digest to refresh."
  echo ""
  exit 0
fi

# The tracker's and the pre-compaction stubs hold the branch and HEAD only; nothing in them
# was written by a model, so they are named, not injected.
trigger="$(sed -n 's/^trigger:[[:space:]]*//p' "$latest" | head -n 1)"
case "$trigger" in
  activity-tracker-mid-session|precompact-hook-fallback)
    echo ""
    echo "state/digests/latest.md is an automatic stub ($trigger): nothing model-written to carry forward."
    echo ""
    exit 0
    ;;
esac

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
