param(
    [ValidateSet("ycsb", "tpcc", "all")]
    [string]$Workload = "all",

    [ValidateSet("low", "medium", "high", "all")]
    [string]$Profile = "all",

    [string]$Variants = "all",
    [string]$Seeds = "920104,920105,920106,920107,920108",
    [int]$TaskCount = 60,
    [int]$Workers = 24,
    [int]$AgentSlots = 4,
    [string]$TrainSeeds = "910104,910105,910106,910107,910108",
    [int]$TrainRounds = 4,
    [int]$TrainTaskCount = 0,
    [double]$TrainPolicyEpsilon = 0.05,
    [string]$ValidationSeeds = "930104,930105",
    [int]$ValidationTaskCount = 0,
    [int]$PriorityCap = 1,
    [ValidateSet("conservative", "threshold32", "naive")]
    [string]$StaticPreset = "conservative",
    [switch]$NoFreezeDynamicPolicy,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath([string]$Path) {
    $resolved = (Resolve-Path $Path).Path
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        $drive = $Matches[1].ToLower()
        $rest = $Matches[2] -replace "\\", "/"
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert path to WSL path: $resolved"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$WslRepo = Convert-ToWslPath $RepoRoot
if (-not $OutputDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = "results/atcc_ablation_$stamp"
}

$CommonArgs = @(
    "--workload $Workload",
    "--profile $Profile",
    "--variants $Variants",
    "--seeds $Seeds",
    "--task-count $TaskCount",
    "--workers $Workers",
    "--agent-slots $AgentSlots",
    "--planning-delay-ms 50",
    "--latency-distribution lognormal",
    "--latency-cv 0.8",
    "--latency-max-ms 500",
    "--max-attempts 8",
    "--background-workers 4",
    "--background-interval-ms 2",
    "--background-strategy occ",
    "--prelock-wait-budget-ms 70",
    "--prelock-wait-budget-mode object",
    "--prelock-lease-mode-ycsb yield-refresh-regenerate",
    "--prelock-lease-mode-tpcc hold",
    "--agent-execution-mode staged",
    "--snapshot-timing before-planning",
    "--train-seeds $TrainSeeds",
    "--train-rounds $TrainRounds",
    "--train-task-count $TrainTaskCount",
    "--train-policy-epsilon $TrainPolicyEpsilon",
    "--validation-seeds $ValidationSeeds",
    "--validation-task-count $ValidationTaskCount",
    "--priority-cap $PriorityCap",
    "--static-preset $StaticPreset",
    "--output-dir $OutputDir"
) -join " "
if ($NoFreezeDynamicPolicy) {
    $CommonArgs = "$CommonArgs --no-freeze-dynamic-policy"
}

$Command = "cd $WslRepo && timeout 180s python3 -m agent.evaluation.atcc_ablation_experiment $CommonArgs"
Write-Host "Running ATCC ablation -> $OutputDir"
wsl -e bash -lc $Command
