#!/bin/bash
# Pipeline validation hook — runs before tool execution
# Checks if pipeline-related code has obvious issues

# Only act on Write/Edit tool calls to pipeline files.
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

if [[ "$TOOL_NAME" =~ ^(Write|Edit)$ ]] && [[ "$FILE_PATH" == *pipeline* ]]; then
    # Check for potential cycle indicators in the file content
    if [[ -f "$FILE_PATH" ]]; then
        # Warn if a node references itself as input
        if grep -qE '\.input\("[^"]+",\s*"([^"]+)"\).*\.node\("\1"' "$FILE_PATH" 2>/dev/null; then
            echo "[ix-pipeline] WARNING: Possible self-referencing node detected" >&2
        fi
    fi
fi
