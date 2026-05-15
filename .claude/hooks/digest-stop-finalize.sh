#!/usr/bin/env bash
# Stop hook — writes a finalize digest if /digest hasn't run in last 10 min.

set -e
repoRoot="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$repoRoot" ] && exit 0

digestDir="$repoRoot/state/digests"
archDir="$digestDir/archive"
latest="$digestDir/latest.md"

if [ -f "$latest" ] && [ -n "$(find "$latest" -mmin -10 2>/dev/null)" ]; then
  exit 0
fi

mkdir -p "$digestDir" "$archDir"

tsFile="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
tsIso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

[ -f "$latest" ] && cp -f "$latest" "$archDir/$tsFile-stop.md"

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
session_id: stop-finalize
written_at: $tsIso
trigger: stop-hook-finalize
branch: $branch
head_sha: $headSha
head_subject: $headSubj
open_pr: $openPr
---

# Session digest (Stop-hook finalize — /digest not invoked in last 10 min)

**Branch:** $branch @ $headSha — $headSubj
$prLine
## Model-driven sections

_Session ended without a recent \`/digest\`. Next session: re-orient from
\`git log\` + open PR. Prior digests (if any) are in \`state/digests/archive/\`._
EOF

bash "$repoRoot/.claude/hooks/digest-validate.sh" "$latest" 2>/dev/null || true
exit 0
