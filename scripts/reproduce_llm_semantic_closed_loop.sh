#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required" >&2
  exit 2
fi

TASKS_PER_WORKLOAD="${TASKS_PER_WORKLOAD:-60}"
K="${K:-4}"
LLM_CONCURRENCY="${LLM_CONCURRENCY:-12}"
THREADS="${THREADS:-32}"
REPLAY_SEEDS="${REPLAY_SEEDS:-5}"
SPEED="${SPEED:-20}"
MODEL="${MODEL:-deepseek-chat}"

bash build.sh
python3 agent/experiments/llm_semantic_closed_loop.py all \
  --tasks-per-workload "$TASKS_PER_WORKLOAD" \
  --k "$K" \
  --llm-concurrency "$LLM_CONCURRENCY" \
  --threads "$THREADS" \
  --replay-seeds "$REPLAY_SEEDS" \
  --speed "$SPEED" \
  --model "$MODEL"

