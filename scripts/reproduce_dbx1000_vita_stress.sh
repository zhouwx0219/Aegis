#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESOURCES="${RESOURCES:-agent/experiments/results/vitabench_authoritative_resources.csv}"
if [[ ! -f "$RESOURCES" ]]; then
  echo "[dbx1000-vita-stress] missing resource CSV: $RESOURCES" >&2
  echo "Run scripts/reproduce_vitabench.sh first, or set RESOURCES=/path/to/vitabench_authoritative_resources.csv" >&2
  exit 2
fi

echo "[dbx1000-vita-stress] build DBx1000 ASTRA/Vita runner"
make -C third_party/dbx1000 astra_vita -j"${JOBS:-2}"

OUT="${OUT:-agent/experiments/results/dbx1000_vita_stress_hot64.csv}"
SUMMARY="${SUMMARY:-agent/experiments/results/dbx1000_vita_stress_summary.csv}"

echo "[dbx1000-vita-stress] hot-resource 64-thread stress"
./third_party/dbx1000/astra_vita \
  --resources "$RESOURCES" \
  --out "$OUT" \
  --tasks "${TASKS:-3000}" \
  --threads "${THREADS:-64}" \
  --k "${K:-4}" \
  --seeds "${SEEDS:-3}" \
  --c-gen-ms "${CGEN_MS:-2}" \
  --capacity-multiplier "${CAPACITY_MULTIPLIER:-100}" \
  --hot-per-category "${HOT_PER_CATEGORY:-1}" \
  --hot-bias "${HOT_BIAS:-1.0}"

echo "[dbx1000-vita-stress] aggregate summary"
python3 agent/experiments/analyze_dbx1000_vita.py \
  --out "$SUMMARY" \
  --input "hot64=$OUT"

echo "[dbx1000-vita-stress] outputs:"
echo "  $OUT"
echo "  $SUMMARY"
