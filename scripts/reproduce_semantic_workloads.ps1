param(
  [int]$Tasks = 3000,
  [int]$Threads = 32,
  [string]$Seeds = "1 2 3 4 5",
  [int]$K = 4,
  [double]$CGen = 0.002
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$seedArgs = $Seeds -split "\s+" | Where-Object { $_ -ne "" }
python agent\experiments\semantic_workload_benchmark.py `
  --profile quick `
  --tasks $Tasks `
  --threads $Threads `
  --seeds $seedArgs `
  --k $K `
  --c-gen $CGen
