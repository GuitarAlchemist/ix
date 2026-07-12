#!/bin/bash
# Behavioral fixture for governance-check.sh (ix#228 P0-1).
# Contract under test: PreToolUse hook reads JSON on stdin; BLOCK = exit 2;
# safe input = exit 0. Run: bash .claude/hooks/test-governance-check.sh

HOOK="$(dirname "$0")/governance-check.sh"
fails=0

check() {
    local desc="$1" json="$2" want="$3"
    printf '%s' "$json" | bash "$HOOK" >/dev/null 2>&1
    local got=$?
    if [[ "$got" -eq "$want" ]]; then
        echo "PASS  $desc (exit $got)"
    else
        echo "FAIL  $desc (want exit $want, got $got)"
        fails=$((fails + 1))
    fi
}

# (a) unsafe: recursive delete of root → BLOCKED (exit 2)
check "rm -rf / is blocked" \
    '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' 2

# (b) safe command → allowed (exit 0)
check "ls is allowed" \
    '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"ls -la"}}' 0

# (b2) legitimate absolute-path cleanup → allowed (warns, does not block).
#      Guards the Codex finding: unanchored 'rm -rf /' matched every abs path.
check "rm -rf /tmp/build-cache is allowed (warn only)" \
    '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"rm -rf /tmp/build-cache"}}' 0

# (b3) rm -rf /* (root wildcard) → BLOCKED (exit 2)
check "rm -rf /* is blocked" \
    '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"rm -rf /*"}}' 2

# (c) force-push to main → BLOCKED (exit 2)
check "git push --force origin main is blocked" \
    '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"git push --force origin main"}}' 2

# (d) DROP DATABASE → BLOCKED (exit 2)
check "DROP DATABASE is blocked" \
    '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"psql -c \"DROP DATABASE prod\""}}' 2

# (e) warn-only command (git reset --hard) → allowed but warns (exit 0)
check "git reset --hard warns without blocking" \
    '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"git reset --hard HEAD~1"}}' 0

# (f) non-Bash tool with sensitive path → warn only (exit 0)
check "Write to .env warns without blocking" \
    '{"hook_event_name":"PreToolUse","tool_name":"Write","tool_input":{"file_path":"/app/.env"}}' 0

# (g) garbage stdin → fail-open with warning (exit 0), never block everything
check "unparseable stdin fails open" \
    'not json at all' 0

echo
if [[ $fails -eq 0 ]]; then echo "ALL PASS"; else echo "$fails FAILURE(S)"; exit 1; fi
