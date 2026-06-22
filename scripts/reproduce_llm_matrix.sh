#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[llm-matrix] DEEPSEEK_API_KEY is required in the process environment" >&2
  exit 2
fi

bash build.sh >/tmp/cast_build_llm_matrix.log

TASKS="${TASKS:-60}"
CONC="${CONC:-8}"
THREADS="${THREADS:-8}"
SEEDS="${SEEDS:-3}"
SPEED="${SPEED:-20}"

run_case() {
  local level="$1"
  local k="$2"
  local hot_bias="$3"
  local seats_scale="$4"
  local tag="matrix_${level}_k${k}"
  echo "[llm-matrix] level=${level} k=${k} hot_bias=${hot_bias} seats_scale=${seats_scale}"
  python3 agent/experiments/llm_in_the_loop.py all \
    --tasks "$TASKS" \
    --k "$k" \
    --conc "$CONC" \
    --threads "$THREADS" \
    --seeds "$SEEDS" \
    --speed "$SPEED" \
    --hot-bias "$hot_bias" \
    --seats-scale "$seats_scale"
  cp agent/experiments/results/llm_cache.json "agent/experiments/results/llm_cache_${tag}.json"
  cp agent/experiments/results/llm_in_the_loop.json "agent/experiments/results/llm_in_the_loop_${tag}.json"
  cp agent/experiments/results/llm_in_the_loop.png "agent/experiments/results/llm_in_the_loop_${tag}.png"
}

for k in ${K_VALUES:-1 4 8}; do
  run_case low "$k" "${LOW_HOT_BIAS:-0.0}" "${LOW_SEATS_SCALE:-4.0}"
  run_case mid "$k" "${MID_HOT_BIAS:-0.6}" "${MID_SEATS_SCALE:-1.0}"
  run_case high "$k" "${HIGH_HOT_BIAS:-1.0}" "${HIGH_SEATS_SCALE:-0.5}"
done

python3 agent/experiments/analyze_llm_matrix.py
