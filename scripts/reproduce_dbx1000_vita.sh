#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-quick}"
RESOURCES="${RESOURCES:-agent/experiments/results/vitabench_authoritative_resources.csv}"

if [[ ! -f "$RESOURCES" ]]; then
  echo "[dbx1000-vita] missing resource CSV: $RESOURCES" >&2
  echo "Run scripts/reproduce_vitabench.sh first, or set RESOURCES=/path/to/vitabench_authoritative_resources.csv" >&2
  exit 2
fi

echo "[dbx1000-vita] build DBx1000 ASTRA/Vita runner"
make -C third_party/dbx1000 astra_vita -j"${JOBS:-2}"

if [[ "$PROFILE" == "large" ]]; then
  TASKS="${TASKS:-10000}"
  THREADS="${THREADS:-32}"
  SEEDS="${SEEDS:-5}"
  CGEN_MS="${CGEN_MS:-2}"
else
  TASKS="${TASKS:-3000}"
  THREADS="${THREADS:-16}"
  SEEDS="${SEEDS:-3}"
  CGEN_MS="${CGEN_MS:-2}"
fi

echo "[dbx1000-vita] balanced profile"
./third_party/dbx1000/astra_vita \
  --resources "$RESOURCES" \
  --out agent/experiments/results/dbx1000_vita_balanced.csv \
  --tasks "$TASKS" \
  --threads "$THREADS" \
  --k "${K:-4}" \
  --seeds "$SEEDS" \
  --c-gen-ms "$CGEN_MS" \
  --capacity-multiplier "${CAPACITY_MULTIPLIER:-20}" \
  --hot-per-category "${HOT_PER_CATEGORY:-6}" \
  --hot-bias "${HOT_BIAS:-0.85}"

echo "[dbx1000-vita] high-contention profile"
./third_party/dbx1000/astra_vita \
  --resources "$RESOURCES" \
  --out agent/experiments/results/dbx1000_vita_contention.csv \
  --tasks "$TASKS" \
  --threads "$THREADS" \
  --k "${K:-4}" \
  --seeds "$SEEDS" \
  --c-gen-ms "$CGEN_MS" \
  --capacity-multiplier "${CONTENTION_CAPACITY_MULTIPLIER:-1}" \
  --hot-per-category "${CONTENTION_HOT_PER_CATEGORY:-3}" \
  --hot-bias "${CONTENTION_HOT_BIAS:-0.95}"

echo "[dbx1000-vita] aggregate summaries"
python3 agent/experiments/analyze_dbx1000_vita.py

echo "[dbx1000-vita] outputs:"
echo "  agent/experiments/results/dbx1000_vita_balanced.csv"
echo "  agent/experiments/results/dbx1000_vita_contention.csv"
echo "  agent/experiments/results/dbx1000_vita_summary.csv"
