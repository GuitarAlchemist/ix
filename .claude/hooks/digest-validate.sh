#!/usr/bin/env bash
# Validates state/digests/latest.md frontmatter against docs/contracts/digest-schema.json.
# Karpathy R11: every AI step declares an output schema; runtime rejects mismatches.

set -e
DIGEST_PATH="${1:-}"

repoRoot="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$repoRoot" ] && exit 0

[ -z "$DIGEST_PATH" ] && DIGEST_PATH="$repoRoot/state/digests/latest.md"
[ ! -f "$DIGEST_PATH" ] && exit 0

fm="$(awk '/^---$/{c++; next} c==1{print}' "$DIGEST_PATH" 2>/dev/null || true)"
if [ -z "$fm" ]; then
  echo "digest-validate: missing or malformed YAML frontmatter in $DIGEST_PATH" >&2
  exit 1
fi

required="schema_version session_id written_at trigger branch head_sha head_subject"
for k in $required; do
  if ! echo "$fm" | grep -qE "^${k}:[[:space:]]*[^[:space:]]"; then
    echo "digest-validate: required field '$k' missing or null" >&2
    exit 1
  fi
done

sv="$(echo "$fm" | grep -E '^schema_version:' | sed -E 's/^schema_version:[[:space:]]*//' | tr -d '[:space:]')"
if [ "$sv" != "1" ]; then
  echo "digest-validate: schema_version must be 1 (got '$sv')" >&2
  exit 1
fi

trig="$(echo "$fm" | grep -E '^trigger:' | sed -E 's/^trigger:[[:space:]]*//' | tr -d '[:space:]')"
case "$trig" in
  digest-skill|precompact-hook-fallback|stop-hook-finalize|auto-write-routine) ;;
  *)
    echo "digest-validate: trigger '$trig' not valid" >&2
    exit 1
    ;;
esac

exit 0
