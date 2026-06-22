#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESOURCES="${RESOURCES:-agent/experiments/results/vitabench_authoritative_resources.csv}"
if [[ ! -f "$RESOURCES" ]]; then
  echo "[dbx1000-vita-sensitivity] missing resource CSV: $RESOURCES" >&2
  echo "Run scripts/reproduce_vitabench.sh first, or set RESOURCES=/path/to/vitabench_authoritative_resources.csv" >&2
  exit 2
fi

echo "[dbx1000-vita-sensitivity] build DBx1000 ASTRA/Vita runner"
make -C third_party/dbx1000 astra_vita -j"${JOBS:-2}"

TASKS="${TASKS:-2000}"
THREADS="${THREADS:-16}"
SEEDS="${SEEDS:-3}"
CGEN_MS="${CGEN_MS:-2}"
THREAD_K="${THREAD_K:-4}"

INPUTS=()

for K in ${K_SWEEP:-1 2 4 8}; do
  OUT="agent/experiments/results/dbx1000_vita_sensitivity_k${K}.csv"
  echo "[dbx1000-vita-sensitivity] k sweep K=$K"
  ./third_party/dbx1000/astra_vita \
    --resources "$RESOURCES" \
    --out "$OUT" \
    --tasks "$TASKS" \
    --threads "$THREADS" \
    --k "$K" \
    --seeds "$SEEDS" \
    --c-gen-ms "$CGEN_MS" \
    --capacity-multiplier "${CAPACITY_MULTIPLIER:-1}" \
    --hot-per-category "${HOT_PER_CATEGORY:-3}" \
    --hot-bias "${HOT_BIAS:-0.95}"
  INPUTS+=(--input "k${K}=$OUT")
done

for T in ${THREAD_SWEEP:-4 8 16 32}; do
  OUT="agent/experiments/results/dbx1000_vita_sensitivity_threads${T}.csv"
  echo "[dbx1000-vita-sensitivity] thread sweep THREADS=$T"
  ./third_party/dbx1000/astra_vita \
    --resources "$RESOURCES" \
    --out "$OUT" \
    --tasks "$TASKS" \
    --threads "$T" \
    --k "$THREAD_K" \
    --seeds "$SEEDS" \
    --c-gen-ms "$CGEN_MS" \
    --capacity-multiplier "${THREAD_CAPACITY_MULTIPLIER:-1}" \
    --hot-per-category "${THREAD_HOT_PER_CATEGORY:-3}" \
    --hot-bias "${THREAD_HOT_BIAS:-0.95}"
  INPUTS+=(--input "threads${T}=$OUT")
done

echo "[dbx1000-vita-sensitivity] aggregate summaries"
python3 agent/experiments/analyze_dbx1000_vita.py \
  --out agent/experiments/results/dbx1000_vita_sensitivity_summary.csv \
  "${INPUTS[@]}"

echo "[dbx1000-vita-sensitivity] output:"
echo "  agent/experiments/results/dbx1000_vita_sensitivity_summary.csv"
