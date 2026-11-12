#!/bin/bash
# Behavioral fixture for sessionstart-digest.sh.
# Contract under test: the SessionStart hook reads {"source": ...} on stdin and always exits 0.
# - The ecosystem manifest is injected at startup, resume and clear, never after a compaction
#   (the compaction summary carries the session; the manifest is another repo's snapshot).
# - A digest a model wrote is injected whatever the source.
# - An automatic stub (trigger activity-tracker-mid-session or precompact-hook-fallback) carries
#   only the branch and HEAD, so it is never injected; one line says it was skipped.
# Run: bash .claude/hooks/test-sessionstart-digest.sh

HOOK="$(cd "$(dirname "$0")" && pwd)/sessionstart-digest.sh"
fails=0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

repo() { # $1 = digest trigger
    rm -rf "$tmp/r"
    mkdir -p "$tmp/r/.claude" "$tmp/r/state/digests"
    git -C "$tmp/r" init -q
    printf 'MANIFEST-MARKER\n' > "$tmp/r/.claude/manifest-bootstrap.md"
    printf -- '---\nschema_version: 1\ntrigger: %s\n---\n\nDIGEST-MARKER\n' "$1" \
        > "$tmp/r/state/digests/latest.md"
}

run() { # $1 = source ("" sends an empty payload)
    local payload='{}'
    [ -n "$1" ] && payload="{\"source\": \"$1\"}"
    (cd "$tmp/r" && printf '%s' "$payload" | bash "$HOOK" 2>/dev/null)
}

expect() { # desc trigger source want_manifest want_digest
    local desc="$1" out got
    repo "$2"
    out="$(run "$3")"
    got=$?
    local m=no d=no
    printf '%s' "$out" | grep -q MANIFEST-MARKER && m=yes
    printf '%s' "$out" | grep -q DIGEST-MARKER && d=yes
    if [ "$got" -eq 0 ] && [ "$m" = "$4" ] && [ "$d" = "$5" ]; then
        echo "PASS  $desc"
    else
        echo "FAIL  $desc (exit $got, manifest $m want $4, digest $d want $5)"
        fails=$((fails + 1))
    fi
}

expect "startup injects the manifest and a written digest" manual startup yes yes
expect "resume injects the manifest"                      manual resume  yes yes
expect "an empty payload is a startup"                    manual ""      yes yes
expect "compact injects no manifest, keeps the digest"    manual compact no  yes
expect "compact skips a tracker stub"         activity-tracker-mid-session compact no no
expect "startup skips a tracker stub"         activity-tracker-mid-session startup yes no
expect "compact skips a precompact stub"      precompact-hook-fallback     compact no no

repo activity-tracker-mid-session
if run startup | grep -q 'automatic stub'; then
    echo "PASS  a skipped stub is named in one line"
else
    echo "FAIL  a skipped stub is named in one line"
    fails=$((fails + 1))
fi

if [ "$fails" -gt 0 ]; then
    echo "$fails failure(s)"
    exit 1
fi
echo "all passed"
