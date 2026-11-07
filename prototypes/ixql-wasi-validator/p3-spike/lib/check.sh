# Assertion helpers for run.sh. Source it; it needs $WT (a wasmtime binary) and $CASES (a dir).
#
#   run_case NAME WASM [STDIN_FILE]   run WASM under $WT once; keep stdout, stderr, exit code
#   expect   NAME RC STDOUT STDERR    0 if case NAME matches, else 1 with the reason on stderr
#              RC      exact exit code expected
#              STDOUT  a file whose bytes stdout must equal, or `empty`, or `-` for no check
#              STDERR  a fixed string stderr must contain, or `-` for no check
#   check          NAME RC STDOUT STDERR   expect, counted as a PASS or a FAIL
#   check_rejects  LABEL NAME RC STDOUT STDERR
#              negative control: a deliberately wrong expectation must be rejected
#
# FAILS counts failed checks; the caller exits non-zero when it is not 0.
FAILS=0
CHECKS=0

run_case() {
  local name=$1 wasm=$2 stdin=${3:-/dev/null} rc=0
  "$WT" run "$wasm" <"$stdin" >"$CASES/$name.out" 2>"$CASES/$name.err" || rc=$?
  echo "$rc" >"$CASES/$name.rc"
}

expect() {
  local name=$1 want_rc=$2 want_out=$3 want_err=$4 got_rc
  got_rc=$(cat "$CASES/$name.rc")
  if [ "$got_rc" != "$want_rc" ]; then
    echo "  $name: exit code $got_rc, expected $want_rc" >&2; return 1
  fi
  case "$want_out" in
    -) ;;
    empty) if [ -s "$CASES/$name.out" ]; then
             echo "  $name: stdout not empty: $(head -c 120 "$CASES/$name.out")" >&2; return 1; fi ;;
    *) if ! cmp -s "$CASES/$name.out" "$want_out"; then
         echo "  $name: stdout differs from $(basename "$want_out"): $(head -c 120 "$CASES/$name.out")" >&2
         return 1; fi ;;
  esac
  if [ "$want_err" != "-" ] && ! grep -qF -- "$want_err" "$CASES/$name.err"; then
    echo "  $name: stderr lacks \"$want_err\": $(head -c 120 "$CASES/$name.err")" >&2; return 1
  fi
  return 0
}

check() {
  CHECKS=$((CHECKS + 1))
  if expect "$@"; then
    echo "PASS  $1 (exit $2)"
  else
    echo "FAIL  $1"; FAILS=$((FAILS + 1))
  fi
}

check_rejects() {
  local label=$1 why; shift
  CHECKS=$((CHECKS + 1))
  if why=$(expect "$@" 2>&1); then
    echo "FAIL  negative control \"$label\": the wrong expectation was accepted"; FAILS=$((FAILS + 1))
  else
    echo "PASS  negative control \"$label\" rejected:$(echo "$why" | sed 's/^ *[^:]*:/ /')"
  fi
}
