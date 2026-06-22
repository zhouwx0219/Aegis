#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESOURCES="${RESOURCES:-agent/experiments/results/vitabench_authoritative_resources.csv}"
if [[ ! -f "$RESOURCES" ]]; then
  echo "[dbx1000-vita-cgen] missing resource CSV: $RESOURCES" >&2
  echo "Run scripts/reproduce_vitabench.sh first, or set RESOURCES=/path/to/vitabench_authoritative_resources.csv" >&2
  exit 2
fi

echo "[dbx1000-vita-cgen] build DBx1000 ASTRA/Vita runner"
make -C third_party/dbx1000 astra_vita -j"${JOBS:-2}"

TASKS="${TASKS:-10000}"
THREADS="${THREADS:-64}"
SEEDS="${SEEDS:-5}"
K="${K:-4}"
CAPACITY_MULTIPLIER="${CAPACITY_MULTIPLIER:-100}"
HOT_PER_CATEGORY="${HOT_PER_CATEGORY:-4}"
HOT_BIAS="${HOT_BIAS:-0.90}"

INPUTS=()

for CGEN in ${CGEN_SWEEP:-0 2 20}; do
  tag="${CGEN//./p}"
  OUT="agent/experiments/results/dbx1000_vita_cgen_${tag}ms.csv"
  echo "[dbx1000-vita-cgen] c_gen_ms=${CGEN} tasks=${TASKS} threads=${THREADS} k=${K}"
  ./third_party/dbx1000/astra_vita \
    --resources "$RESOURCES" \
    --out "$OUT" \
    --tasks "$TASKS" \
    --threads "$THREADS" \
    --k "$K" \
    --seeds "$SEEDS" \
    --c-gen-ms "$CGEN" \
    --capacity-multiplier "$CAPACITY_MULTIPLIER" \
    --hot-per-category "$HOT_PER_CATEGORY" \
    --hot-bias "$HOT_BIAS"
  INPUTS+=(--input "cgen_${tag}ms=$OUT")
done

echo "[dbx1000-vita-cgen] aggregate summary"
python3 agent/experiments/analyze_dbx1000_vita.py \
  --out agent/experiments/results/dbx1000_vita_cgen_summary.csv \
  "${INPUTS[@]}"

echo "[dbx1000-vita-cgen] output:"
echo "  agent/experiments/results/dbx1000_vita_cgen_summary.csv"
