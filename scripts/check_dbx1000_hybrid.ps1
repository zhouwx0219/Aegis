param()

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Get-Command bash -ErrorAction SilentlyContinue)) {
  throw "bash/WSL is required to build DBx1000"
}

bash scripts/check_dbx1000_hybrid.sh
