#!/usr/bin/env bash
# SPIKE: WASI 0.3 feasibility for the IXQL -> Go validator pipeline. Reproduces every result in
# ../README.md (WASI 0.3 spike + follow-ups). Downloads pinned official tools into $1 (default: a temp dir),
# checks their sha256, builds, and runs. Needs: go, cargo, rustup target wasm32-wasip2, curl.
# Nothing is installed globally; nothing is published.
#
# Steps 6-8 ASSERT their results (exact exit codes, stdout against ./expected/*, the adapter trap
# marker on stderr) and include negative controls; the script exits 1 if any assertion fails.
# Steps 1-5 only PRINT evidence for a human to read; they assert nothing.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROTO="$(cd "$HERE/.." && pwd)"
TOOLS="${1:-$(mktemp -d)}"
OUT="$TOOLS/out"
mkdir -p "$TOOLS" "$OUT"

WASMTIME_V=50.0.1
WASM_TOOLS_V=1.263.0
fetch() { # url dest sha256
  [ -f "$2" ] || curl -sSfL -o "$2" "$1"
  echo "$3  $2" | sha256sum -c --quiet
}
fetch "https://github.com/bytecodealliance/wasmtime/releases/download/v$WASMTIME_V/wasmtime-v$WASMTIME_V-x86_64-linux.tar.xz" \
  "$TOOLS/wasmtime.tar.xz" 685466455dd42a14774a86c36c8f2e45cbb9bd6d92a850139315df60a75498a5
fetch "https://github.com/bytecodealliance/wasm-tools/releases/download/v$WASM_TOOLS_V/wasm-tools-$WASM_TOOLS_V-x86_64-linux.tar.gz" \
  "$TOOLS/wasm-tools.tar.gz" 21c036c1740a0552124c79c6e2170cb61c5b016005671a6d5bc06cf642165431
fetch "https://github.com/bytecodealliance/wasmtime/releases/download/v$WASMTIME_V/wasi_snapshot_preview1.command.wasm" \
  "$TOOLS/wasi_snapshot_preview1.command.wasm" 0b4fe490390ecc9c2222f969f65b179e658a3327e005efd4ddc959766132a3c4
tar -C "$TOOLS" -xJf "$TOOLS/wasmtime.tar.xz"
tar -C "$TOOLS" -xzf "$TOOLS/wasm-tools.tar.gz"
WT="$TOOLS/wasmtime-v$WASMTIME_V-x86_64-linux/wasmtime"
WTL="$TOOLS/wasm-tools-$WASM_TOOLS_V-x86_64-linux/wasm-tools"
export RUST_BACKTRACE=0
"$WT" --version; "$WTL" --version; go version

step() { printf '\n=== %s\n' "$*"; }
VALID="$PROTO/fixtures/candidate-valid.json"
INVALID="$PROTO/fixtures/candidate-invalid.json"
CASES="$OUT/cases"; rm -rf "$CASES"; mkdir -p "$CASES"
# shellcheck source=lib/check.sh
. "$HERE/lib/check.sh"
TRAP="assertion failed at adapter line 2857"
WANT_REPORT="$HERE/expected/validator-invalid.stdout"
WANT_STRESS="$HERE/expected/stress.stdout"

step "1. Go fixture validator, GOOS=wasip1 (same flags as ix-wasm's Toolchain)"
mkdir -p "$OUT/go" && cp "$PROTO/validator/main.go" "$OUT/go/"
(cd "$OUT/go" && { [ -f go.mod ] || GOTOOLCHAIN=local go mod init stage >/dev/null 2>&1; } &&
  GOOS=wasip1 GOARCH=wasm CGO_ENABLED=0 GOPROXY=off GOTOOLCHAIN=local go build -trimpath -o "$OUT/validator.p1.wasm" .)
"$WT" run "$OUT/validator.p1.wasm" < "$INVALID"

step "2. Same module adapted to a component (wasi_snapshot_preview1 command adapter)"
"$WTL" component new "$OUT/validator.p1.wasm" \
  --adapt "wasi_snapshot_preview1=$TOOLS/wasi_snapshot_preview1.command.wasm" -o "$OUT/validator.p2.wasm"
"$WTL" component wit "$OUT/validator.p2.wasm" | grep -E '^\s*export'
"$WT" run "$OUT/validator.p2.wasm" < "$VALID" 2>&1 | head -1 || true

step "3. Native WASI 0.3 guests (Rust, wasm32-wasip2 target, wasip3 crate) on Wasmtime $WASMTIME_V"
for f in wasi-0-3-0 wasi-0-3-2; do
  (cd "$HERE/guest" && CARGO_TARGET_DIR="$OUT/target-$f" cargo build -q --release --target wasm32-wasip2 --features "$f")
  W="$OUT/target-$f/wasm32-wasip2/release/p3_echo_guest.wasm"
  echo "--- $f exports: $("$WTL" component wit "$W" | grep -E '^\s*export' | xargs)"
  "$WT" run "$W" < "$VALID" | head -2
  "$WT" run -S p3=n "$W" < "$VALID" 2>&1 | grep -m1 'imports instance' || true
done

step "4. A WIT type mismatch fails at link time"
for t in string u32; do
  printf '(component (import "wasi:cli/environment@0.3.0" (instance (export "get-arguments" (func (result (list %s)))))))' "$t" > "$OUT/env-$t.wat"
  "$WTL" parse "$OUT/env-$t.wat" -o "$OUT/env-$t.wasm"
  echo "--- list<$t>: $("$WT" run "$OUT/env-$t.wasm" 2>&1 | grep -m1 -E 'wrong type|no exported' | sed 's/^ *//')"
done

step "5. WASI 0.3 async: concurrent vs sequential vs blocking, guest stream, host stream + future"
(cd "$HERE/async-guest" && CARGO_TARGET_DIR="$OUT/target-async" cargo build -q --release --target wasm32-wasip2)
ASYNC="$OUT/target-async/wasm32-wasip2/release/p3_async_guest.wasm"
"$WT" run "$ASYNC" all
(printf 'a'; sleep 0.3; printf 'b'; sleep 0.2) | "$WT" run "$ASYNC" stdin
echo "--- future-intra (expected to be refused): $("$WT" run "$ASYNC" future-intra 2>&1 | grep -m1 -o 'cannot read from and write to.*')"
echo "--- canonical ABI names in the core module:"
"$WTL" print "$ASYNC" | grep -oE '"(\[async-lower\]wait-for|\[async-lift\]wasi:cli/run@0.3.0#run|\[method\]pollable.block)"' | sort -u
echo "--- with -W component-model-async=n: $("$WT" run -W component-model-async=n "$ASYNC" concurrent 2>&1 | grep -m1 -o '`stream` requires.*')"

step "6. Adapter trap, controlled (no Go): which memory-ownership pattern breaks the adapter [asserted]"
for t in c0-control c1-go-sbrk c2-honest-grow c3-heap-on-stack-page; do
  "$WTL" parse "$HERE/adapter-trap/$t.wat" -o "$OUT/$t.p1.wasm"
  "$WTL" component new "$OUT/$t.p1.wasm" --adapt "wasi_snapshot_preview1=$TOOLS/wasi_snapshot_preview1.command.wasm" -o "$OUT/$t.component.wasm"
  run_case "$t.p1" "$OUT/$t.p1.wasm"
  run_case "$t.component" "$OUT/$t.component.wasm"
done
# Every core module runs cleanly without the adapter.
for t in c0-control c1-go-sbrk c2-honest-grow c3-heap-on-stack-page; do check "$t.p1" 0 empty -; done
check c0-control.component            0   empty -        # control
check c1-go-sbrk.component            134 empty "$TRAP"  # Go-style sbrk: the exact Go trap
check c2-honest-grow.component        0   empty -        # same grow, honest ownership: fine
check c3-heap-on-stack-page.component 1   empty -        # adapter stack overwrote "heap" bytes (proc_exit != 0 -> 1)

step "7. Go runtime with the diagnostic patch (go build -overlay; GOROOT untouched) [asserted]"
GOROOT_REAL="$(go env GOROOT)"
mkdir -p "$OUT/overlay/runtime" "$OUT/stress"
cp "$GOROOT_REAL/src/runtime/os_wasm.go" "$GOROOT_REAL/src/runtime/mem_wasm.go" "$OUT/overlay/runtime/"
(cd "$OUT/overlay" && patch -s -p1 < "$HERE/adapter-trap/go1.24.7-runtime-diagnostic.diff")
printf '{"Replace":{"%s/src/runtime/os_wasm.go":"%s/overlay/runtime/os_wasm.go","%s/src/runtime/mem_wasm.go":"%s/overlay/runtime/mem_wasm.go"}}' \
  "$GOROOT_REAL" "$OUT" "$GOROOT_REAL" "$OUT" > "$OUT/overlay.json"
cp "$HERE/adapter-trap/stress/main.go" "$OUT/stress/"
(cd "$OUT/stress" && { [ -f go.mod ] || GOTOOLCHAIN=local go mod init stress >/dev/null 2>&1; })
for prog in go stress; do for v in stock patched; do
  flags=(); [ $v = patched ] && flags=(-overlay="$OUT/overlay.json")
  (cd "$OUT/$prog" && GOOS=wasip1 GOARCH=wasm CGO_ENABLED=0 GOPROXY=off GOTOOLCHAIN=local go build -trimpath "${flags[@]}" -o "$OUT/$prog.$v.p1.wasm" .)
  "$WTL" component new "$OUT/$prog.$v.p1.wasm" --adapt "wasi_snapshot_preview1=$TOOLS/wasi_snapshot_preview1.command.wasm" -o "$OUT/$prog.$v.component.wasm"
  for kind in p1 component; do run_case "$prog.$v.$kind" "$OUT/$prog.$v.$kind.wasm" "$INVALID"; done
done; done
# The patch must actually change the build, or the "patched" results prove nothing.
CHECKS=$((CHECKS + 1))
if cmp -s "$OUT/go.stock.p1.wasm" "$OUT/go.patched.p1.wasm"; then
  echo "FAIL  go.patched.p1.wasm is identical to go.stock.p1.wasm: the overlay was not applied"; FAILS=$((FAILS + 1))
else echo "PASS  patched build differs from stock build (overlay applied)"; fi
check go.stock.p1            0   "$WANT_REPORT" -        # reference output
check go.stock.component     134 empty          "$TRAP"  # stock Go traps behind the adapter
check go.patched.p1          0   "$WANT_REPORT" -        # patch leaves core p1 unchanged
check go.patched.component   0   "$WANT_REPORT" -        # patch removes the trap; same bytes as stock p1
check stress.stock.p1        0   "$WANT_STRESS" -
check stress.stock.component 134 empty          "$TRAP"
check stress.patched.p1      0   "$WANT_STRESS" -
check stress.patched.component 0 "$WANT_STRESS" -

step "8. Negative controls: wrong expectations against the same real artifacts must be rejected"
check_rejects "stock component expected to succeed"   go.stock.component 0   "$WANT_REPORT" -
check_rejects "patched component expected to trap"    go.patched.component 134 empty "$TRAP"
check_rejects "trap marker required on a clean run"   go.patched.p1 0 - "$TRAP"
check_rejects "stress output compared to the report"  stress.patched.component 0 "$WANT_REPORT" -
run_case go.stock.p1.valid "$OUT/go.stock.p1.wasm" "$VALID"
check_rejects "valid candidate's report vs invalid's" go.stock.p1.valid 0 "$WANT_REPORT" -
check_rejects "c1 expected to pass like c2"           c1-go-sbrk.component 0 empty -
check_rejects "c3 corruption expected to be absent"   c3-heap-on-stack-page.component 0 empty -

step "Summary"
echo "$((CHECKS - FAILS)) of $CHECKS assertions passed (steps 6-8; steps 1-5 are printed evidence only)"
[ "$FAILS" -eq 0 ] || { echo "FAILED: $FAILS assertion(s)"; exit 1; }
