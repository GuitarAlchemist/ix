#!/usr/bin/env bash
# PreCompact hook — archives state/digests/latest.md and writes a metadata-only
# fallback if /digest wasn't invoked before compaction.

set -e

# Sanitizers — applied to ALL untrusted strings (stdin payload, branch name,
# commit subject, PR number) before interpolation into paths or YAML.
# Closes path-traversal + YAML-injection findings from the 2026-05-15
# octo security review.
safe_id() {
  local v="${1:-}" f="${2:-unknown}" m="${3:-64}"
  [ -z "$v" ] && { printf '%s' "$f"; return; }
  local c
  c="$(printf '%s' "$v" | tr -d '\r\n\t' | head -c "$m")"
  if printf '%s' "$c" | grep -qE '^[A-Za-z0-9._-]+$'; then
    printf '%s' "$c"
  else
    printf '%s' "$f"
  fi
}
safe_yaml() {
  local v="${1:-}" m="${2:-200}"
  if [ -z "$v" ]; then printf 'null'; return; fi
  local c
  c="$(printf '%s' "$v" | tr '\r\n' '  ' | head -c "$m")"
  c="$(printf '%s' "$c" | sed "s/'/''/g")"
  printf "'%s'" "$c"
}

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

safeSession="$(safe_id "$sessionId")"
if [ -f "$latest" ]; then
  cp -f "$latest" "$archDir/$tsFile-$safeSession.md"
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

sessionIdYaml="$(safe_yaml "$sessionId")"
branchYaml="$(safe_yaml "$branch")"
headShaYaml="$(safe_yaml "$headSha")"
headSubjYaml="$(safe_yaml "$headSubj")"
openPrYaml="$(safe_yaml "$openPr")"
cat > "$latest" <<EOF
---
schema_version: 1
session_id: $sessionIdYaml
written_at: $ts
trigger: precompact-hook-fallback
branch: $branchYaml
head_sha: $headShaYaml
head_subject: $headSubjYaml
open_pr: $openPrYaml
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
