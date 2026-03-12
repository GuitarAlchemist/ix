#!/bin/bash
# Cache lifecycle hook — manages machin-cache across sessions
# Lightweight: only logs to stderr, actual cache management is in-process

SNAPSHOT_DIR="${HOME}/.machin/cache"

case "${CLAUDE_HOOK_EVENT:-}" in
    session_start)
        if [[ -f "$SNAPSHOT_DIR/snapshot.json" ]]; then
            echo "[machin-cache] Found cache snapshot at $SNAPSHOT_DIR/snapshot.json" >&2
        fi
        ;;
    session_end)
        mkdir -p "$SNAPSHOT_DIR" 2>/dev/null
        echo "[machin-cache] Session ended. Cache snapshot should be persisted by the application." >&2
        ;;
esac
