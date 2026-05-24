#!/usr/bin/env bash
# UserPromptSubmit hook — Enhancement 2 (Cherny auto /correct trigger).
# If the user's prompt matches correction language, nudge Claude to invoke
# /correct so the rule lands in CLAUDE.md. Throttled: fires once per N=5 prompts.

set -e

repoRoot="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$repoRoot" ] && exit 0

digestDir="$repoRoot/state/digests"
mkdir -p "$digestDir"
counterPath="$digestDir/.correction-counter"

prompt=""
if [ ! -t 0 ]; then
  payload="$(cat || true)"
  if [ -n "$payload" ] && command -v jq >/dev/null 2>&1; then
    prompt="$(echo "$payload" | jq -r '.prompt // .user_prompt // empty' 2>/dev/null || true)"
  fi
  # Fallback: raw stdin if not JSON
  if [ -z "$prompt" ]; then
    prompt="$payload"
  fi
fi

[ -z "$prompt" ] && exit 0

# Match correction language (case-insensitive, anchored at start, word boundary)
firstWord="$(printf '%s' "$prompt" | head -c 200 | tr '[:upper:]' '[:lower:]')"
if ! printf '%s' "$firstWord" | grep -qE "^(no|don't|dont|stop|wait|actually|that's wrong|thats wrong|incorrect)\b"; then
  exit 0
fi

# Throttle: increment counter, fire only every N=5
THROTTLE="${IX_CORRECTION_THROTTLE:-5}"
count=0
if [ -f "$counterPath" ]; then
  raw="$(cat "$counterPath" 2>/dev/null | tr -d '[:space:]')"
  if echo "$raw" | grep -qE '^[0-9]+$'; then
    count="$raw"
  fi
fi
count=$((count + 1))
echo "$count" > "$counterPath"

# Only fire on count == 1 (first correction) or every Nth thereafter
if [ "$count" -ne 1 ] && [ $((count % THROTTLE)) -ne 0 ]; then
  exit 0
fi

msg="Detected correction language. Consider invoking /correct to formalize this into CLAUDE.md so the rule persists across sessions (Cherny self-improvement loop)."

if command -v jq >/dev/null 2>&1; then
  jq -n --arg ctx "[correction-nudge] $msg" '{additionalContext: $ctx}'
else
  printf '{"additionalContext":"[correction-nudge] %s"}\n' "$msg"
fi
exit 0
