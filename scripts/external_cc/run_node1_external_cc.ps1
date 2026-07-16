param(
    [string]$Node = "node1",
    [string]$RemoteRoot = "/home/chenht/castdas_external_cc",
    [string]$Systems = "bamboo,polaris",
    [string]$Workloads = "ycsb,tpcc",
    [string]$Levels = "low,medium,high",
    [string]$ClientCounts = "8,16,24,32,40,48",
    [string]$AgentRatios = "1.0,0.8",
    [double]$Duration = 5,
    [string]$Algorithms = "",
    [string]$Output = "results\external_cc_agentic.csv",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$remoteScripts = "$RemoteRoot/castdas_scripts"
$remoteOutput = "$RemoteRoot/external_cc_agentic.csv"
$localOutput = Join-Path $repoRoot $Output

ssh $Node "mkdir -p '$remoteScripts'"
scp (Join-Path $PSScriptRoot "patch_dbx1000_agentic.py") "$Node`:$remoteScripts/patch_dbx1000_agentic.py"
scp (Join-Path $PSScriptRoot "run_external_cc_matrix.py") "$Node`:$remoteScripts/run_external_cc_matrix.py"

$smokeArg = ""
if ($Smoke) {
    $smokeArg = "--smoke"
}

$algArg = ""
if ($Algorithms.Trim().Length -gt 0) {
    $algArg = "--algorithms '$Algorithms'"
}

$cmd = @"
python3 '$remoteScripts/run_external_cc_matrix.py' \
  --root '$RemoteRoot' \
  --systems '$Systems' \
  --workloads '$Workloads' \
  --levels '$Levels' \
  --client-counts '$ClientCounts' \
  --agent-ratios '$AgentRatios' \
  --duration '$Duration' \
  --output '$remoteOutput' \
  --patch-script '$remoteScripts/patch_dbx1000_agentic.py' \
  $algArg \
  $smokeArg
"@

ssh $Node $cmd

$localDir = Split-Path -Parent $localOutput
if ($localDir -and !(Test-Path $localDir)) {
    New-Item -ItemType Directory -Path $localDir | Out-Null
}
scp "$Node`:$remoteOutput" $localOutput
Write-Host "Wrote $localOutput"

