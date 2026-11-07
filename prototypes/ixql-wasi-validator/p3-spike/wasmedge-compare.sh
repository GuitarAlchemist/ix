#!/usr/bin/env bash
# COMPARISON: the small Go FIXTURE validator (../validator/main.go — not the complete, previously
# validated Go package, which is not in this repository) as a WASI Preview 1 core module, run
# under Wasmtime and under WasmEdge with the same inputs. Stdout, stderr and exit codes are
# compared byte for byte.
#
# WasmEdge is downloaded from the official GitHub release into a temporary directory, checked
# against the release's SHA256SUM, run from there with LD_LIBRARY_PATH set for its own commands
# only, and deleted at the end. The installer script is NOT used: by default it also fetches the
# wasi_nn-ggml AI plugin. Nothing is installed globally.
#
# Usage: wasmedge-compare.sh <tools-dir>   (the same dir run.sh uses; Wasmtime is fetched if missing)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROTO="$(cd "$HERE/.." && pwd)"
TOOLS="${1:?usage: $0 <tools-dir>}"
mkdir -p "$TOOLS"
WE_TMP="$(mktemp -d "$TOOLS/wasmedge-tmp.XXXXXX")"
cleanup() { rm -rf "$WE_TMP"; }
trap cleanup EXIT

WASMTIME_V=50.0.1
WASMEDGE_V=0.18.1
WE_PKG="WasmEdge-$WASMEDGE_V-manylinux_2_28_x86_64.tar.gz"
fetch() { # url dest sha256
  [ -f "$2" ] || curl -sSfL -o "$2" "$1"
  echo "$3  $2" | sha256sum -c --quiet
}
fetch "https://github.com/bytecodealliance/wasmtime/releases/download/v$WASMTIME_V/wasmtime-v$WASMTIME_V-x86_64-linux.tar.xz" \
  "$TOOLS/wasmtime.tar.xz" 685466455dd42a14774a86c36c8f2e45cbb9bd6d92a850139315df60a75498a5
tar -C "$TOOLS" -xJf "$TOOLS/wasmtime.tar.xz"
WT="$TOOLS/wasmtime-v$WASMTIME_V-x86_64-linux/wasmtime"

# WasmEdge: the official release asset, verified against the release's own SHA256SUM, and also
# against the digest recorded here on 2026-11-03.
curl -sSfL -o "$WE_TMP/$WE_PKG" "https://github.com/WasmEdge/WasmEdge/releases/download/$WASMEDGE_V/$WE_PKG"
curl -sSfL -o "$WE_TMP/SHA256SUM" "https://github.com/WasmEdge/WasmEdge/releases/download/$WASMEDGE_V/SHA256SUM"
(cd "$WE_TMP" && grep " $WE_PKG\$" SHA256SUM | sha256sum -c -)
echo "ec97ccfc172a1fbb2116e9dd00ca12d41f9f9ad3d7b1c89a63b8477703fa67c9  $WE_TMP/$WE_PKG" | sha256sum -c --quiet
mkdir "$WE_TMP/install" && tar -C "$WE_TMP/install" -xzf "$WE_TMP/$WE_PKG"
we() { env -u WASMEDGE_PLUGIN_PATH LD_LIBRARY_PATH="$WE_TMP/install/lib64" "$WE_TMP/install/bin/wasmedge" "$@"; }

"$WT" --version
we --version
go version

# The fixture, built with ix-wasm's Toolchain flags.
OUT="$WE_TMP/out"; mkdir -p "$OUT/go"
cp "$PROTO/validator/main.go" "$OUT/go/"
(cd "$OUT/go" && GOTOOLCHAIN=local go mod init stage >/dev/null 2>&1 &&
  GOOS=wasip1 GOARCH=wasm CGO_ENABLED=0 GOPROXY=off GOTOOLCHAIN=local go build -trimpath -o "$OUT/validator.p1.wasm" .)
sha256sum "$OUT/validator.p1.wasm" "$PROTO/validator/main.go"

# name | stdin
cases=(
  "valid|$(cat "$PROTO/fixtures/candidate-valid.json")"
  "invalid|$(cat "$PROTO/fixtures/candidate-invalid.json")"
  "not-an-object|[1,2,3]"
  "failure:not-json|nope"
  "failure:empty-stdin|"
)
mismatch=0
printf '\n%-22s %-9s %-9s %-8s %-8s %s\n' case wt.exit we.exit stdout= stderr= "stdout (wasmtime)"
for c in "${cases[@]}"; do
  name="${c%%|*}"; input="${c#*|}"
  wt_rc=0; printf '%s' "$input" | "$WT" run "$OUT/validator.p1.wasm" >"$OUT/$name.wt.out" 2>"$OUT/$name.wt.err" || wt_rc=$?
  we_rc=0; printf '%s' "$input" | we "$OUT/validator.p1.wasm" >"$OUT/$name.we.out" 2>"$OUT/$name.we.err" || we_rc=$?
  so=yes; cmp -s "$OUT/$name.wt.out" "$OUT/$name.we.out" || { so=NO; mismatch=1; }
  se=yes; cmp -s "$OUT/$name.wt.err" "$OUT/$name.we.err" || { se=NO; mismatch=1; }
  [ "$wt_rc" = "$we_rc" ] || mismatch=1
  printf '%-22s %-9s %-9s %-8s %-8s %s\n' "$name" "$wt_rc" "$we_rc" "$so" "$se" \
    "$(tr -d '\n' < "$OUT/$name.wt.out" | cut -c1-90)"
  [ -s "$OUT/$name.wt.err" ] && echo "    stderr (wasmtime): $(tr -d '\n' < "$OUT/$name.wt.err")"
  [ "$se" = NO ] && echo "    stderr (wasmedge): $(tr -d '\n' < "$OUT/$name.we.err")"
done

# A directory the fixture would see only if it were preopened: neither runtime is given one.
echo
echo "preopens: none passed to either runtime (no --dir); environment: none passed."
echo "result: $([ $mismatch = 0 ] && echo 'all cases identical (stdout, stderr, exit code)' || echo 'DIFFERENCES FOUND')"

cleanup; trap - EXIT
[ ! -e "$WE_TMP" ] && echo "removed: $WE_TMP"
exit $mismatch
