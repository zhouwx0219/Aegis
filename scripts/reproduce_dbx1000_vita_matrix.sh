#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESOURCES="${RESOURCES:-agent/experiments/results/vitabench_authoritative_resources.csv}"
if [[ ! -f "$RESOURCES" ]]; then
  echo "[dbx1000-vita-matrix] missing resource CSV: $RESOURCES" >&2
  echo "Run scripts/reproduce_vitabench.sh first, or set RESOURCES=/path/to/vitabench_authoritative_resources.csv" >&2
  exit 2
fi

echo "[dbx1000-vita-matrix] build DBx1000 ASTRA/Vita runner"
make -C third_party/dbx1000 astra_vita -j"${JOBS:-2}"

TASKS="${TASKS:-3000}"
SEEDS="${SEEDS:-3}"
CGEN_MS="${CGEN_MS:-2}"
CAPACITY_MULTIPLIER="${CAPACITY_MULTIPLIER:-100}"
K_VALUES="${K_VALUES:-1 4 8}"

INPUTS=()

run_case() {
  local level="$1"
  local k="$2"
  local threads="$3"
  local hot_per="$4"
  local hot_bias="$5"
  local out="agent/experiments/results/dbx1000_vita_matrix_${level}_k${k}.csv"
  echo "[dbx1000-vita-matrix] level=${level} K=${k} threads=${threads} hot_per=${hot_per} hot_bias=${hot_bias}"
  ./third_party/dbx1000/astra_vita \
    --resources "$RESOURCES" \
    --out "$out" \
    --tasks "$TASKS" \
    --threads "$threads" \
    --k "$k" \
    --seeds "$SEEDS" \
    --c-gen-ms "$CGEN_MS" \
    --capacity-multiplier "$CAPACITY_MULTIPLIER" \
    --hot-per-category "$hot_per" \
    --hot-bias "$hot_bias"
  INPUTS+=(--input "${level}_k${k}=$out")
}

for k in $K_VALUES; do
  run_case low "$k" "${LOW_THREADS:-8}" "${LOW_HOT_PER_CATEGORY:-24}" "${LOW_HOT_BIAS:-0.00}"
  run_case mid "$k" "${MID_THREADS:-64}" "${MID_HOT_PER_CATEGORY:-4}" "${MID_HOT_BIAS:-0.90}"
  run_case high "$k" "${HIGH_THREADS:-64}" "${HIGH_HOT_PER_CATEGORY:-1}" "${HIGH_HOT_BIAS:-1.00}"
done

echo "[dbx1000-vita-matrix] aggregate summary"
python3 agent/experiments/analyze_dbx1000_vita.py \
  --out agent/experiments/results/dbx1000_vita_matrix_summary.csv \
  "${INPUTS[@]}"

echo "[dbx1000-vita-matrix] output:"
echo "  agent/experiments/results/dbx1000_vita_matrix_summary.csv"
