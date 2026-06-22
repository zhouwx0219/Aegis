#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASKS="${TASKS:-3000}"
THREADS="${THREADS:-32}"
SEEDS="${SEEDS:-1 2 3 4 5}"
K="${K:-4}"
C_GEN="${C_GEN:-0.002}"

python3 agent/experiments/semantic_workload_benchmark.py \
  --profile quick \
  --tasks "$TASKS" \
  --threads "$THREADS" \
  --seeds $SEEDS \
  --k "$K" \
  --c-gen "$C_GEN"

