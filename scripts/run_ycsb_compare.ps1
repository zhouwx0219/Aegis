param(
    [ValidateSet("low", "medium", "high", "all")]
    [string]$Profile = "all",

    [ValidateSet("atcc", "full")]
    [string]$StrategySet = "atcc",

    [string]$PolicyVariant = "ycsb-strict-tuned",
    [ValidateSet("hold", "yield-during-planning", "yield-refresh-regenerate", "defer-until-after-planning")]
    [string]$PrelockLeaseMode = "yield-refresh-regenerate",
    [ValidateSet("legacy", "staged", "staged-local")]
    [string]$AgentExecutionMode = "staged",
    [ValidateSet("before-planning", "after-planning")]
    [string]$SnapshotTiming = "before-planning",
    [string]$PolicyArtifact = "",
    [double]$PolicyEpsilon = -1.0,
    [int]$TaskCount = 60,
    [int]$Workers = 24,
    [string]$OutputDir = "results/handoff_ycsb_compare"
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
    "occ,2pl-nowait,2pl-wait-die,mvcc-full,silo-full,tictoc-full,adaptive-op-strict,transaction-atcc-strict,adaptive-hybrid"
} else {
    "occ,tictoc-full,adaptive-op-strict,transaction-atcc-strict,adaptive-hybrid"
}

$Profiles = if ($Profile -eq "all") { @("low", "medium", "high") } else { @($Profile) }

$CommonArgs = @(
    "--workload ycsb",
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
    "--policy-variant $PolicyVariant"
)
if ($PolicyArtifactArgs.Count -gt 0) {
    $CommonArgs += $PolicyArtifactArgs
}
$Common = $CommonArgs -join " "

$ProfileArgs = @{
    low = "--records 512 --fields 10 --requests-per-task 10 --candidates 3 --read-weight 0.95 --update-weight 0.05 --zipf-theta 0.0 --hotspot-fraction 0.0 --hotspot-access-probability 0.0"
    medium = "--records 128 --fields 10 --requests-per-task 10 --candidates 3 --read-weight 0.90 --update-weight 0.10 --zipf-theta 0.7 --hotspot-fraction 0.10 --hotspot-access-probability 0.50"
    high = "--records 64 --fields 10 --requests-per-task 10 --candidates 3 --read-weight 0.50 --update-weight 0.50 --zipf-theta 0.99 --hotspot-fraction 0.10 --hotspot-access-probability 0.75"
}

foreach ($p in $Profiles) {
    $Output = "$OutputDir/ycsb-$p.json"
    $Command = "cd $WslRepo && mkdir -p $OutputDir && python3 -m agent.evaluation.atcc_retry_experiment $Common $($ProfileArgs[$p]) --output $Output"
    Write-Host "Running YCSB $p -> $Output"
    wsl -e bash -lc $Command
}

python (Join-Path $PSScriptRoot "summarize_retry_results.py") --input-dir (Join-Path $RepoRoot $OutputDir)
