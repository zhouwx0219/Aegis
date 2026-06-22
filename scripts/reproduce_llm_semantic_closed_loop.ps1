param(
  [int]$TasksPerWorkload = 60,
  [int]$K = 4,
  [int]$LlmConcurrency = 12,
  [int]$Threads = 32,
  [int]$ReplaySeeds = 5,
  [double]$Speed = 20,
  [string]$Model = "deepseek-chat"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $env:DEEPSEEK_API_KEY) {
  throw "DEEPSEEK_API_KEY is required"
}

bash build.sh
$key = $env:DEEPSEEK_API_KEY.Replace("'", "'\''")
bash -lc "export DEEPSEEK_API_KEY='$key'; python3 agent/experiments/llm_semantic_closed_loop.py all --tasks-per-workload $TasksPerWorkload --k $K --llm-concurrency $LlmConcurrency --threads $Threads --replay-seeds $ReplaySeeds --speed $Speed --model $Model"
