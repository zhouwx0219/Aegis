param(
    [ValidateSet("low", "medium", "high", "all")]
    [string]$Profile = "all",

    [ValidateSet("atcc", "full")]
    [string]$StrategySet = "atcc",

    [string]$PolicyVariant = "ycsb-strict-tuned",
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

$Strategies = if ($StrategySet -eq "full") {
    "occ,2pl-nowait,2pl-wait-die,mvcc-full,silo-full,tictoc-full,adaptive-op-strict"
} else {
    "occ,tictoc-full,adaptive-op-strict"
}

$Profiles = if ($Profile -eq "all") { @("low", "medium", "high") } else { @($Profile) }

$Common = @(
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
    "--prelock-lease-mode hold",
    "--policy-variant $PolicyVariant"
) -join " "

$ProfileArgs = @{
    low = "--records 512 --fields 10 --requests-per-task 6 --candidates 3 --read-weight 0.95 --update-weight 0.05 --zipf-theta 0.0"
    medium = "--records 128 --fields 10 --requests-per-task 6 --candidates 3 --read-weight 0.5 --update-weight 0.5 --zipf-theta 0.8"
    high = "--records 64 --fields 10 --requests-per-task 8 --candidates 3 --read-weight 0.2 --update-weight 0.8 --zipf-theta 0.99"
}

foreach ($p in $Profiles) {
    $Output = "$OutputDir/ycsb-$p.json"
    $Command = "cd $WslRepo && mkdir -p $OutputDir && python3 -m agent.evaluation.atcc_retry_experiment $Common $($ProfileArgs[$p]) --output $Output"
    Write-Host "Running YCSB $p -> $Output"
    wsl -e bash -lc $Command
}

python (Join-Path $PSScriptRoot "summarize_retry_results.py") --input-dir (Join-Path $RepoRoot $OutputDir)
