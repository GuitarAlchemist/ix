#!/usr/bin/env bash
# Runs the three real-world showcase pipelines and shows per-stage output.

set -euo pipefail

IX_BIN="./target/release/ix.exe"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/release/ix"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/debug/ix.exe"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/debug/ix"
[[ -f "$IX_BIN" ]] || { echo "error: ix binary not found — run 'cargo build -p ix-cli'" >&2; exit 1; }
IX_BIN="$(cd "$(dirname "$IX_BIN")" && pwd)/$(basename "$IX_BIN")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export IX_ROOT="$WORKSPACE_ROOT"
export IX_GOVERNANCE_DIR="$WORKSPACE_ROOT/governance/demerzel"

if [[ -t 1 ]]; then
    HEAD="\033[1;35m" STEP="\033[1;33m" DIM="\033[2m" RESET="\033[0m"
else
    HEAD="" STEP="" DIM="" RESET=""
fi

run_pipeline() {
    local name="$1"
    local description="$2"
    printf "\n${HEAD}━━━ %s ━━━${RESET}\n" "$name"
    printf "${DIM}%s${RESET}\n\n" "$description"
    local dir
    dir="$(mktemp -d -t ix-adv-$name-XXXXXX)"
    cp "$SCRIPT_DIR/$name.yaml" "$dir/ix.yaml"
    (
        cd "$dir"
        printf "${STEP}$ ix pipeline dag${RESET}\n"
        "$IX_BIN" --format json pipeline dag | head -20
        printf "\n${STEP}$ ix pipeline run${RESET}\n"
        "$IX_BIN" --format json pipeline run
    )
    rm -rf "$dir"
}

run_pipeline "fraud-detection" \
  "E-commerce fraud screening — 4 parallel signals (stats, kmeans, bloom, hyperloglog) → governance gate"

run_pipeline "bandit-ab-test" \
  "Multi-armed bandit A/B test — Thompson sampling + Nash equilibrium + deployment review"

run_pipeline "sensor-anomaly" \
  "Industrial sensor monitoring — FFT + Lyapunov + stats + bloom → page on-call?"

run_pipeline "chained-spectrum" \
  "Signal chain with {from:} data flow — spectrum magnitudes feed downstream stats"

run_pipeline "music-theory" \
  "GA federation: chord/scale/progression feature extraction across ii-V-I + Autumn Leaves"

run_pipeline "code-quality" \
  "PR triage: complexity metrics (cyclomatic/cognitive/Halstead) on 3 Rust functions"

run_pipeline "graph-centrality" \
  "Service mesh routing: PageRank + Dijkstra shortest path + topological deployment order"

run_pipeline "topology-circle" \
  "Persistent homology detects the hole in a noisy circle (Betti numbers)"

run_pipeline "grammar-evolution" \
  "Grammar rule competition via replicator dynamics + Bayesian update + MCTS search"

printf "\n${HEAD}━━━ Done ━━━${RESET}\n"
cat <<EOF

Nine real-world pipelines ran end-to-end:

  • ${HEAD}fraud-detection${RESET}    — multi-signal transaction screening
  • ${HEAD}bandit-ab-test${RESET}     — exploration → exploitation → rollout
  • ${HEAD}sensor-anomaly${RESET}     — regime-change alerting
  • ${HEAD}chained-spectrum${RESET}   — {from:} reference pulls fft.magnitudes into stats
  • ${HEAD}music-theory${RESET}       — cross-repo GA music-theory federation
  • ${HEAD}code-quality${RESET}       — static analysis triage gate
  • ${HEAD}graph-centrality${RESET}   — PageRank + Dijkstra + topological sort
  • ${HEAD}topology-circle${RESET}    — persistent homology detects the hole
  • ${HEAD}grammar-evolution${RESET}  — replicator dynamics + Bayesian + MCTS

All six fuse independent signals into a governance leaf — the canonical
"parallel analysis → constitutional gate → action" pattern.

${HEAD}Try next:${RESET}
  • Open any of the .yaml files in the Pipeline tab of ix-demo
  • Modify the inputs to see how the governance leaf's verdict shifts
  • Add your own stage: $IX_BIN pipeline new my-scenario

EOF
