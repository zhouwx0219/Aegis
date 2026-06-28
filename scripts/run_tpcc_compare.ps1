param(
    [ValidateSet("low", "medium", "high", "all")]
    [string]$Profile = "all",

    [ValidateSet("atcc", "full")]
    [string]$StrategySet = "full",

    [string]$PolicyVariant = "default",
    [ValidateSet("hold", "yield-during-planning", "yield-refresh-regenerate", "defer-until-after-planning")]
    [string]$PrelockLeaseMode = "hold",
    [ValidateSet("legacy", "staged", "staged-local")]
    [string]$AgentExecutionMode = "staged",
    [ValidateSet("before-planning", "after-planning")]
    [string]$SnapshotTiming = "before-planning",
    [string]$PolicyArtifact = "",
    [double]$PolicyEpsilon = -1.0,
    [int]$TaskCount = 60,
    [int]$Workers = 24,
    [string]$OutputDir = "results/handoff_tpcc_compare"
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

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WslRepo = Convert-ToWslPath $RepoRoot
$PolicyArtifactArgs = @()
if ($PolicyArtifact) {
    $PolicyArtifactArgs += "--policy-artifact $(Convert-ToWslPath $PolicyArtifact)"
}
if ($PolicyEpsilon -ge 0.0) {
    $PolicyArtifactArgs += "--policy-epsilon $PolicyEpsilon"
}

$Strategies = if ($StrategySet -eq "full") {
    "occ,2pl-nowait,2pl-wait-die,mvcc-full,silo-full,tictoc-full,adaptive-op-strict,adaptive-hybrid"
} else {
    "occ,tictoc-full,adaptive-op-strict,adaptive-hybrid"
}

$Profiles = if ($Profile -eq "all") { @("low", "medium", "high") } else { @($Profile) }

$CommonArgs = @(
    "--workload tpcc",
    "--strategies $Strategies",
    "--task-count $TaskCount",
    "--seed 920104",
    "--repeats 1",
    "--workers $Workers",
    "--agent-slots 4",
    "--agent-admission-mode before-begin",
    "--planning-delay-ms 50",
    "--latency-distribution lognormal",
    "--latency-cv 0.8",
    "--latency-max-ms 500",
    "--max-attempts 8",
    "--background-workers 4",
    "--background-interval-ms 2",
    "--background-strategy occ",
    "--object-lock-scheduler bounded-priority",
    "--object-lock-priority-burst 2",
    "--prelock-wait-budget-ms 70",
    "--prelock-wait-budget-mode object",
    "--prelock-lease-mode $PrelockLeaseMode",
    "--agent-execution-mode $AgentExecutionMode",
    "--snapshot-timing $SnapshotTiming",
    "--policy-variant $PolicyVariant",
    "--transaction-mix new_order:1.0"
)
if ($PolicyArtifactArgs.Count -gt 0) {
    $CommonArgs += $PolicyArtifactArgs
}
$Common = $CommonArgs -join " "

$ProfileArgs = @{
    low = "--warehouses 8 --districts-per-warehouse 5 --customers-per-district 100 --items 500 --order-lines 5"
    medium = "--warehouses 2 --districts-per-warehouse 3 --customers-per-district 60 --items 200 --order-lines 8"
    high = "--warehouses 1 --districts-per-warehouse 2 --customers-per-district 40 --items 100 --order-lines 10"
}

foreach ($p in $Profiles) {
    $Output = "$OutputDir/tpcc-$p.json"
    $Command = "cd $WslRepo && mkdir -p $OutputDir && python3 -m agent.evaluation.atcc_retry_experiment $Common $($ProfileArgs[$p]) --output $Output"
    Write-Host "Running TPCC $p -> $Output"
    wsl -e bash -lc $Command
}

python (Join-Path $PSScriptRoot "summarize_retry_results.py") --input-dir (Join-Path $RepoRoot $OutputDir)
