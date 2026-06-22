#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"

echo "[ccfa] mode=$MODE"
echo "[ccfa] build"
bash build.sh

echo "[ccfa] core runtime checks"
python3 agent/experiments/transaction_runtime_experiment.py
python3 agent/experiments/demo_e2e.py >/tmp/astra_demo_e2e.log
python3 agent/experiments/correctness_boundary.py >/tmp/astra_correctness_boundary.log

echo "[ccfa] rerun dependency-light data producers"
python3 agent/experiments/sweep_contention.py
PROFILE="quick"
if [[ "$MODE" == "large" ]]; then
  PROFILE="large"
fi
python3 agent/experiments/ccfa_extended_experiments.py --profile "$PROFILE"

if python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('matplotlib') else 1)"
then
  echo "[ccfa] matplotlib detected; rerun legacy measured/PNG experiments"
  python3 agent/experiments/concurrent_harness.py
  python3 agent/experiments/semantic_validation_experiment.py
  python3 agent/experiments/explore_experiment.py
  python3 agent/experiments/escrow_experiment.py
  python3 agent/experiments/cc_comparison.py
  python3 agent/experiments/hybrid_cc_adaptive.py
else
  echo "[ccfa] matplotlib not installed; skip legacy PNG experiments."
  echo "[ccfa] install with: python3 -m pip install -r requirements.txt"
fi

echo "[ccfa] paper-facing SVG figures"
python3 agent/experiments/paper_figures.py

if [[ "$MODE" == "llm-mock" ]]; then
  echo "[ccfa] run LLM pipeline in mock mode"
  python3 agent/experiments/llm_in_the_loop.py all --mock --tasks 48 --k 3 --seeds 3
fi

echo "[ccfa] outputs:"
echo "  agent/experiments/results/paper_figures/"
echo "  docs/CCFA_ARTIFACT_GUIDE.md"
