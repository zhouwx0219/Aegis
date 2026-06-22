param()

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not $env:DEEPSEEK_API_KEY) {
  throw "DEEPSEEK_API_KEY is required in the process environment"
}
if (-not (Get-Command bash -ErrorAction SilentlyContinue)) {
  throw "bash/WSL is required to run the real LLM matrix"
}

bash scripts/reproduce_llm_matrix.sh
