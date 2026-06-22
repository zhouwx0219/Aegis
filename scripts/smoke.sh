#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[smoke] build cast_core"
bash build.sh

echo "[smoke] import cast_core"
python3 - <<'PY'
import cast_core
print(cast_core.__doc__)
PY

echo "[smoke] transaction lifecycle"
python3 agent/experiments/transaction_runtime_experiment.py

echo "[smoke] end-to-end demo"
python3 agent/experiments/demo_e2e.py

echo "[smoke] correctness boundary"
python3 agent/experiments/correctness_boundary.py

echo "[smoke] regenerate paper SVG figures from existing results"
python3 agent/experiments/paper_figures.py

echo "[smoke] OK"
