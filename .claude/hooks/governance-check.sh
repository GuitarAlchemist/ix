#!/bin/bash
# Demerzel Governance Enforcement Hook
#
# BLOCK (exit 1): Catastrophic/irreversible operations that should never proceed without explicit override
# WARN (stderr): Risky operations that need human attention but aren't automatically blocked
#
# References: Demerzel Constitution v2.0.0

TOOL_NAME="${CLAUDE_TOOL_NAME:-}"
COMMAND="${CLAUDE_BASH_COMMAND:-}"

if [[ "$TOOL_NAME" == "Bash" ]]; then

    # ── BLOCK: Catastrophic operations (Article 3: Reversibility) ──────────
    if [[ "$COMMAND" =~ rm\ -rf\ / ]] || [[ "$COMMAND" =~ rm\ -rf\ \* ]]; then
        echo "[demerzel] BLOCKED: Article 3 (Reversibility) — recursive delete of root or wildcard is catastrophic." >&2
        exit 1
    fi
    if [[ "$COMMAND" =~ git\ push\ --force.*(main|master) ]] || [[ "$COMMAND" =~ git\ push\ -f.*(main|master) ]]; then
        echo "[demerzel] BLOCKED: Article 3 (Reversibility) — force push to main/master destroys shared history." >&2
        exit 1
    fi
    if [[ "$COMMAND" =~ DROP\ DATABASE ]] || [[ "$COMMAND" =~ drop\ database ]]; then
        echo "[demerzel] BLOCKED: Article 3 (Reversibility) — DROP DATABASE is irreversible." >&2
        exit 1
    fi
    if [[ "$COMMAND" =~ chmod\ 777\ / ]]; then
        echo "[demerzel] BLOCKED: Article 9 (Bounded Autonomy) — chmod 777 on root is unsafe." >&2
        exit 1
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
    FILE_PATH="${CLAUDE_FILE_PATH:-}"
    if [[ "$FILE_PATH" =~ (/etc/|\.env$|credentials|secret|\.pem$|\.key$) ]]; then
        echo "[demerzel] WARN: Article 4 (Proportionality) — writing to sensitive file." >&2
    fi
fi
