#!/bin/bash
# Demerzel Governance Enforcement Hook (PreToolUse)
#
# Contract (code.claude.com/docs/en/hooks): event data arrives as JSON on
# stdin ({tool_name, tool_input:{command|file_path,...}}); BLOCK = exit 2
# (stderr is fed back to Claude and the tool call is prevented); WARN =
# stderr + exit 0 (non-blocking).
#
# References: Demerzel Constitution v2.0.0

INPUT=$(cat)

_field() {
    printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
v = d
for k in sys.argv[1:]:
    v = v.get(k, {}) if isinstance(v, dict) else {}
print(v if isinstance(v, str) else '')
" "$@" 2>/dev/null
}

TOOL_NAME=$(_field tool_name)
COMMAND=$(_field tool_input command)
FILE_PATH=$(_field tool_input file_path)

# Fail-open with a visible warning if parsing yielded nothing (never silently
# block every tool call because the harness or python3 changed under us).
if [[ -z "$TOOL_NAME" ]]; then
    echo "[demerzel] WARN: governance hook received no parseable tool_name on stdin — rules not evaluated." >&2
    exit 0
fi

if [[ "$TOOL_NAME" == "Bash" ]]; then

    # ── BLOCK: Catastrophic operations (Article 3: Reversibility) ──────────
    # Anchored to root ("/", "/*") or bare wildcard only — a plain substring
    # match on "rm -rf /" would block every absolute path (e.g. rm -rf
    # /tmp/build-cache), which the WARN rule below already covers.
    if [[ "$COMMAND" =~ rm\ -rf\ +/+([[:space:]]|$) ]] || [[ "$COMMAND" =~ rm\ -rf\ +/+\* ]] || [[ "$COMMAND" =~ rm\ -rf\ +\* ]]; then
        echo "[demerzel] BLOCKED: Article 3 (Reversibility) — recursive delete of root or wildcard is catastrophic." >&2
        exit 2
    fi
    if [[ "$COMMAND" =~ git\ push\ --force.*(main|master) ]] || [[ "$COMMAND" =~ git\ push\ -f.*(main|master) ]]; then
        echo "[demerzel] BLOCKED: Article 3 (Reversibility) — force push to main/master destroys shared history." >&2
        exit 2
    fi
    if [[ "$COMMAND" =~ DROP\ DATABASE ]] || [[ "$COMMAND" =~ drop\ database ]]; then
        echo "[demerzel] BLOCKED: Article 3 (Reversibility) — DROP DATABASE is irreversible." >&2
        exit 2
    fi
    if [[ "$COMMAND" =~ chmod\ 777\ / ]]; then
        echo "[demerzel] BLOCKED: Article 9 (Bounded Autonomy) — chmod 777 on root is unsafe." >&2
        exit 2
    fi

    # ── WARN: Risky operations ─────────────────────────────────────────────
    if [[ "$COMMAND" =~ (rm\ -rf|git\ reset\ --hard|git\ clean\ -fd) ]]; then
        echo "[demerzel] WARN: Article 3 (Reversibility) — destructive command detected. Confirm before proceeding." >&2
    fi
    if [[ "$COMMAND" =~ (git\ push\ --force|git\ push\ -f) ]]; then
        echo "[demerzel] WARN: Article 3 (Reversibility) — force push detected. Verify target branch." >&2
    fi
    if [[ "$COMMAND" =~ (chmod\ 777|sudo|chown\ root|--no-verify) ]]; then
        echo "[demerzel] WARN: Article 9 (Bounded Autonomy) — permission escalation detected." >&2
    fi
    if [[ "$COMMAND" =~ (DROP\ TABLE|TRUNCATE|DELETE\ FROM.*WHERE\ 1) ]]; then
        echo "[demerzel] WARN: Article 3 (Reversibility) — destructive SQL detected." >&2
    fi
fi

# Article 4: Proportionality — warn on sensitive file writes
if [[ "$TOOL_NAME" == "Write" ]]; then
    if [[ "$FILE_PATH" =~ (/etc/|\.env$|credentials|secret|\.pem$|\.key$) ]]; then
        echo "[demerzel] WARN: Article 4 (Proportionality) — writing to sensitive file." >&2
    fi
fi

exit 0
