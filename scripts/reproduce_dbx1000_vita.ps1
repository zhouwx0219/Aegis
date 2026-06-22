param(
  [string]$Profile = "quick"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Get-Command bash -ErrorAction SilentlyContinue)) {
  throw "bash/WSL is required to build and run the DBx1000 C++ runner"
}

bash scripts/reproduce_dbx1000_vita.sh $Profile
