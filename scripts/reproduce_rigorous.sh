#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-large}"

if [[ ! -f agent/experiments/results/vitabench_authoritative_resources.csv ]]; then
  echo "[rigorous] missing VitaBench resource snapshot; generating it first"
  bash scripts/reproduce_vitabench.sh
fi

echo "[rigorous] run large-scale benchmark profile=$PROFILE"
python3 agent/experiments/rigorous_vitabench_benchmark.py --profile "$PROFILE"

echo "[rigorous] regenerate paper figures"
python3 agent/experiments/paper_figures.py

echo "[rigorous] outputs:"
echo "  agent/experiments/results/rigorous_vitabench_runs.csv"
echo "  agent/experiments/results/rigorous_vitabench_summary.csv"
echo "  agent/experiments/results/rigorous_vitabench_manifest.json"
echo "  agent/experiments/results/paper_figures/fig11_rigorous_vitabench.svg"
