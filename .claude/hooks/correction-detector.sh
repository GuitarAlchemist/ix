#!/usr/bin/env bash
# UserPromptSubmit hook: when the user's message reads as a correction, tell Claude to run
# /correct in the same turn and to end with a teaching line, so the rule lands in CLAUDE.md and
# the user learns the moment to type /correct themselves (Cherny's self-improvement loop).
#
# French first: the user dictates in French ("non, pas comme ça", "plutôt", "on en a déjà
# parlé"); the English openers stay. Every correction fires. The first version matched English
# only and spoke once, then one time in five: with a French speaker it never spoke at all.
# Behavioural fixture: bash .claude/hooks/test-correction-detector.sh

set -e

prompt=""
if [ ! -t 0 ]; then
  payload="$(cat || true)"
  if [ -n "$payload" ] && command -v jq >/dev/null 2>&1; then
    prompt="$(printf '%s' "$payload" | jq -r '.prompt // .user_prompt // empty' 2>/dev/null || true)"
  fi
  # Fallback: raw stdin if not JSON
  if [ -z "$prompt" ]; then
    prompt="$payload"
  fi
fi

[ -z "$prompt" ] && exit 0

# An opener at the start of the message, after optional punctuation and a filler word. Accented
# letters are written as alternations, not bracket classes: in the C locale grep runs under, a
# bracket class holds single bytes and a two-byte letter never matches.
openers="^[[:space:][:punct:]]*((ah|euh|bon|ok|mais|hmm|hum)[[:space:],]+)?(non|no|nope|pas (comme )?(ç|c)a|arr(ê|e)te|attends?|plut(ô|o)t|c'est faux|faux|mauvais|pas du tout|stop|wait|don'?t|dont|actually|incorrect|that'?s wrong|thats wrong)\b"
# A reminder anywhere in the message.
phrases="on en a d(é|e)j(à|a) parl(é|e)|je (te l'|t.)ai d(é|e)j(à|a) dit|on avait dit|we (already )?discussed this|je te l.avais dit"

head="$(printf '%s' "$prompt" | head -c 300)"
if ! printf '%s' "$head" | grep -qiE "$openers" && ! printf '%s' "$prompt" | grep -qiE "$phrases"; then
  exit 0
fi

msg="The user's message reads as a correction. In this same turn: run /correct if the correction would apply to future work (typos and one-off nudges are not rules), and end your reply with one teaching line telling the user what you ran, why, and that they can type /correct themselves right after a correction."

if command -v jq >/dev/null 2>&1; then
  jq -n --arg ctx "[correction-nudge] $msg" '{additionalContext: $ctx}'
else
  printf '{"additionalContext":"[correction-nudge] %s"}\n' "$msg"
fi
exit 0
