#!/bin/bash
# Pipeline validation hook — runs before tool execution
# Checks if pipeline-related code has obvious issues

# Only act on Write/Edit tool calls to pipeline files
TOOL_NAME="${CLAUDE_TOOL_NAME:-}"
FILE_PATH="${CLAUDE_FILE_PATH:-}"

if [[ "$TOOL_NAME" =~ ^(Write|Edit)$ ]] && [[ "$FILE_PATH" == *pipeline* ]]; then
    # Check for potential cycle indicators in the file content
    if [[ -f "$FILE_PATH" ]]; then
        # Warn if a node references itself as input
        if grep -qE '\.input\("[^"]+",\s*"([^"]+)"\).*\.node\("\1"' "$FILE_PATH" 2>/dev/null; then
            echo "[machin-pipeline] WARNING: Possible self-referencing node detected" >&2
        fi
    fi
fi
