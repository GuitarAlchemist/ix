#!/bin/bash
# Post-edit Rust check — runs `cargo check` on the affected crate after .rs edits
# Provides fast feedback loop (verification-driven, harness engineering principle 6)

# Contract: hook event data arrives as JSON on stdin (tool_name, tool_input.file_path).
INPUT=$(cat)
read -r TOOL_NAME FILE_PATH < <(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(' ')
    sys.exit(0)
print(d.get('tool_name', ''), d.get('tool_input', {}).get('file_path', ''))
" 2>/dev/null)

# Only act on Write/Edit of .rs files
if [[ ! "$TOOL_NAME" =~ ^(Write|Edit)$ ]]; then
    exit 0
fi
if [[ ! "$FILE_PATH" == *.rs ]]; then
    exit 0
fi

# Determine the crate from the path: crates/<name>/...
CRATE=""
if [[ "$FILE_PATH" =~ /crates/([^/]+)/ ]]; then
    CRATE="${BASH_REMATCH[1]}"
fi

if [[ -z "$CRATE" ]]; then
    exit 0
fi

# Run cargo check -p <crate>, capture output
cd "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null || exit 0

OUTPUT=$(cargo check -p "$CRATE" --message-format=short 2>&1)
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    # Print only error lines to stderr (keep context tight)
    echo "[rust-check] cargo check -p $CRATE failed:" >&2
    echo "$OUTPUT" | grep -E "^(error|warning)" | head -10 >&2
fi

# Never block — PostToolUse should report, not block
exit 0
