#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[vitabench] setup external VitaBench package"
bash agent/integrations/setup_vitabench.sh

echo "[vitabench] run environment-derived write workload"
python3 agent/experiments/vitabench_authoritative.py "$@"

echo "[vitabench] regenerate paper figures"
python3 agent/experiments/paper_figures.py

echo "[vitabench] outputs:"
echo "  agent/experiments/results/vitabench_authoritative.csv"
echo "  agent/experiments/results/vitabench_authoritative_manifest.json"
echo "  agent/experiments/results/paper_figures/fig10_vitabench_authoritative.svg"
