#!/usr/bin/env bash
# ix showcase — end-to-end walkthrough of the 7-verb CLI.
#
# Run from the workspace root:   bash examples/showcase/demo.sh
#
# The script only calls `ix` (built binary) — it never mutates state outside
# its own temp workspace.

set -euo pipefail

# Locate the ix binary (prefer release if present, fall back to debug).
IX_BIN="./target/release/ix.exe"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/release/ix"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/debug/ix.exe"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/debug/ix"
[[ -f "$IX_BIN" ]] || { echo "error: ix binary not found — run 'cargo build -p ix-cli' first" >&2; exit 1; }

IX_BIN="$(cd "$(dirname "$IX_BIN")" && pwd)/$(basename "$IX_BIN")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Point governance-loading paths at the workspace — needed because the
# demo runs from a temp dir, and handlers resolve paths relative to IX_ROOT.
export IX_ROOT="$WORKSPACE_ROOT"
export IX_GOVERNANCE_DIR="$WORKSPACE_ROOT/governance/demerzel"

# Dedicated temp workspace so the demo never touches user state.
DEMO_DIR="$(mktemp -d -t ix-showcase-XXXXXX)"
trap 'rm -rf "$DEMO_DIR"' EXIT
cd "$DEMO_DIR"

# Copy the showcase pipeline into the temp workspace.
cp "$SCRIPT_DIR/pipeline.yaml" ./ix.yaml

# Terminal colors (disabled on non-TTY).
if [[ -t 1 ]]; then
    HEAD="\033[1;36m"  # cyan bold
    STEP="\033[1;33m"  # yellow bold
    DIM="\033[2m"
    RESET="\033[0m"
else
    HEAD="" STEP="" DIM="" RESET=""
fi

banner() { printf "\n${HEAD}━━━ %s ━━━${RESET}\n\n" "$1"; }
step()   { printf "${STEP}$ %s${RESET}\n" "$*"; }

# ───────────────────────────────────────────────────────────────────────────
banner "1. Discover: how many skills, which domains?"
step "ix list domains --format table"
"$IX_BIN" --format table list domains | head -30

# ───────────────────────────────────────────────────────────────────────────
banner "2. Introspect a single skill"
step "ix describe skill stats"
"$IX_BIN" --format table describe skill stats

# ───────────────────────────────────────────────────────────────────────────
banner "3. Run a skill ad-hoc"
step "ix run stats --input '{\"data\":[1,2,3,4,5,6,7,8,9,10]}'"
"$IX_BIN" --format table run stats --input '{"data":[1,2,3,4,5,6,7,8,9,10]}'

# ───────────────────────────────────────────────────────────────────────────
banner "4. Validate an ix.yaml pipeline"
step "ix pipeline validate"
"$IX_BIN" --format table pipeline validate

step "ix pipeline dag --format table"
"$IX_BIN" --format table pipeline dag

# ───────────────────────────────────────────────────────────────────────────
banner "5. Execute the pipeline (diamond DAG, parallel roots)"
step "ix pipeline run --json  (NDJSON event stream)"
"$IX_BIN" pipeline run --json

# ───────────────────────────────────────────────────────────────────────────
banner "6. Verify ix.lock was written for reproducibility"
step "ls -la ix.lock && head -30 ix.lock"
ls -la ix.lock
printf "${DIM}"
head -30 ix.lock
printf "${RESET}"

# ───────────────────────────────────────────────────────────────────────────
banner "7. Governance: doctor + action check"
step "ix check doctor"
"$IX_BIN" --format table check doctor || true  # may return P if optional dirs missing

step "ix check action 'update the README' --format table"
"$IX_BIN" --format table check action "update the README" || true
step "ix check action 'delete production database' --format table"
"$IX_BIN" --format table check action "delete production database" || true
echo -e "${DIM}(exit codes: 0=T ok, 1=P probable, 2=U unknown, 3=D doubtful, 4=F false, 5=C contradictory)${RESET}"

# ───────────────────────────────────────────────────────────────────────────
banner "8. Hexavalent belief state"
step "ix beliefs set deployment 'Signal pipeline passes all gates' --truth P --confidence 0.8"
"$IX_BIN" --format table beliefs set deployment "Signal pipeline passes all gates" --truth P --confidence 0.8

step "ix beliefs snapshot 'post-pipeline-run'"
"$IX_BIN" --format table beliefs snapshot "post-pipeline-run"

echo -e "${DIM}snapshot written to:${RESET}"
ls state/snapshots/

# ───────────────────────────────────────────────────────────────────────────
banner "9. List the 14 Demerzel personas"
step "ix list personas --format table"
"$IX_BIN" --format table list personas

# ───────────────────────────────────────────────────────────────────────────
banner "Done!"
cat <<EOF

${HEAD}Recap:${RESET}
  • 43 skills across 29 domains, all discoverable via the registry
  • 7-verb CLI: run | pipeline | list | describe | check | beliefs | serve
  • ix.yaml DAG executed with parallel branches + governance leaf
  • ix.lock written with sha256 content hashes
  • Hexavalent governance (T/P/U/D/F/C) with constitutional compliance
  • 14 Demerzel personas loaded from the governance submodule

${HEAD}Try next:${RESET}
  • cargo run -p ix-demo          ← visual pipeline editor
  • $IX_BIN describe persona skeptical-auditor
  • $IX_BIN list skills --domain governance --format json
  • edit the ix.yaml in $DEMO_DIR and re-run

Workspace used: ${DIM}$DEMO_DIR${RESET} (cleaned up on exit)

EOF
