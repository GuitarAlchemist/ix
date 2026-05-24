#!/usr/bin/env bash
# PostToolUse(matcher=Bash) hook — Enhancement 3 (Cherny PR rationale capture).
# When a Bash invocation runs `gh pr create`, snapshot the title + body + diff
# stats to state/digests/pr-<num>-<slug>.md so the rationale survives later edits.

set -e

repoRoot="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$repoRoot" ] && exit 0

if [ -t 0 ]; then exit 0; fi
payload="$(cat || true)"
[ -z "$payload" ] && exit 0
command -v jq >/dev/null 2>&1 || exit 0

cmd="$(echo "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
output="$(echo "$payload" | jq -r '.tool_response.output // .tool_response // empty' 2>/dev/null || true)"

# Only react to gh pr create
echo "$cmd" | grep -qE 'gh[[:space:]]+pr[[:space:]]+create' || exit 0

# Extract title and body from command (best-effort — quoted strings vary)
title="$(printf '%s' "$cmd" | grep -oE -- '--title[[:space:]]+("[^"]*"|'\''[^'\'']*'\'')' | head -1 | sed -E 's/^--title[[:space:]]+(.)(.*)\1$/\2/' || true)"
body="$(printf '%s' "$cmd" | grep -oE -- '--body[[:space:]]+("[^"]*"|'\''[^'\'']*'\'')' | head -1 | sed -E 's/^--body[[:space:]]+(.)(.*)\1$/\2/' || true)"

# Extract PR number from gh CLI output (URL like https://github.com/x/y/pull/123)
prNum="$(printf '%s' "$output" | grep -oE 'pull/[0-9]+' | head -1 | sed 's|pull/||')"
[ -z "$prNum" ] && prNum="unknown"

# Slug from title (lowercased, alphanum + dash, max 40 chars)
slug="$(printf '%s' "$title" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | tr -s '-' | sed 's/^-//;s/-$//' | head -c 40)"
[ -z "$slug" ] && slug="untitled"

digestDir="$repoRoot/state/digests"
mkdir -p "$digestDir"
outPath="$digestDir/pr-$prNum-$slug.md"

tsIso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
branch="$(git -C "$repoRoot" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
shortStat="$(git -C "$repoRoot" diff --shortstat HEAD~1 2>/dev/null || echo "")"

cat > "$outPath" <<EOF
---
schema_version: 1
trigger: pr-rationale-capture
captured_at: $tsIso
branch: $branch
pr_number: $prNum
diff_shortstat: $shortStat
---

# PR #$prNum — $title

**Branch:** $branch
**Captured:** $tsIso
**Diff:** $shortStat

## Title

$title

## Body

$body
EOF

exit 0
