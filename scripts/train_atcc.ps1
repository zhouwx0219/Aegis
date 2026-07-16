param(
    [ValidateSet("concurrent", "mixed")]
    [string]$Benchmark = "concurrent",

    [ValidateSet("ycsb", "tpcc")]
    [string]$Workload = "tpcc",

    [ValidateSet("low", "medium", "high")]
    [string]$Level = "high",

    [ValidateSet("small", "paper")]
    [string]$WorkloadProfile = "small",
    [double]$YcsbZipfTheta = -1.0,

    [string]$Workloads = "",
    [string]$Levels = "",

    [int]$Episodes = 5,
    [int]$Tasks = 100,
    [int]$Workers = 8,
    [double]$Duration = 1.0,
    [int]$Clients = 0,
    [double]$AgentRatio = 0.80,
    [int]$Agents = 2,
    [int]$Background = 8,
    [ValidateSet("hotspot", "procedure")]
    [string]$BackgroundMode = "hotspot",
    [int]$Retries = 0,
    [switch]$RetryUntilCommit,
    [int]$MaxAttemptsPerTask = 100,
    [string]$AgentRetryBackoffMs = "500,5000",
    [string]$BackgroundRetryBackoffMs = "10,30",
    [int]$TokensPerOperation = 2703,
    [int]$Seed = 920104,
    [double]$AbortThreshold = 0.20,
    [int]$MinVisits = 5,
    [double]$ProtectCostThresholdMs = 10.0,
    [double]$LowConflictSafeAbortRate = 0.50,
    [switch]$DisableLowConflictOccGuard,
    [switch]$DisableSparseStateRiskPrior,
    [double]$CommitValue = 100.0,
    [double]$AbortPenalty = 80.0,
    [double]$ReasoningWeight = 1.0,
    [double]$LockWaitWeight = 0.5,
    [double]$LatencyWeight = 0.1,
    [double]$LockHoldWeight = 0.05,
    [double]$BackgroundAbortWeight = 2.0,
    [double]$BackgroundTpsLossWeight = 0.1,
    [double]$UcbC = 1.5,
    [string]$Actions = "auto",
    [double]$BudgetSeconds = 0,
    [ValidateSet("none", "light", "agentic", "heavy")]
    [string]$ReasoningProfile = "agentic",
    [double]$ReasoningScale = 1.0,
    [Parameter(Mandatory = $true)]
    [string]$Output
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

function Convert-ToWslOutputPath([string]$Path) {
    $parent = Split-Path $Path -Parent
    $leaf = Split-Path $Path -Leaf
    if (-not $parent) {
        $parent = "."
    }
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force $parent | Out-Null
    }
    $resolvedParent = (Resolve-Path $parent).Path
    if ($resolvedParent -match "^([A-Za-z]):\\(.*)$") {
        $drive = $Matches[1].ToLower()
        $rest = $Matches[2] -replace "\\", "/"
        return "/mnt/$drive/$rest/$leaf"
    }
    throw "Cannot convert output path to WSL path: $Path"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WslRepo = Convert-ToWslPath $RepoRoot
$WslOutput = Convert-ToWslOutputPath $Output
$BudgetArgs = ""
if ($BudgetSeconds -gt 0) {
    $BudgetArgs = "--budget-seconds $BudgetSeconds"
}
$MatrixArgs = ""
if ($Workloads) {
    $MatrixArgs = "$MatrixArgs --workloads $Workloads"
}
if ($Levels) {
    $MatrixArgs = "$MatrixArgs --levels $Levels"
}
$LowConflictGuardArgs = ""
if ($DisableLowConflictOccGuard) {
    $LowConflictGuardArgs = "--disable-low-conflict-occ-guard"
}
$SparseRiskArgs = ""
if ($DisableSparseStateRiskPrior) {
    $SparseRiskArgs = "--disable-sparse-state-risk-prior"
}
$RetryUntilCommitArgs = ""
if ($RetryUntilCommit) {
    $RetryUntilCommitArgs = "--retry-until-commit"
}
$YcsbZipfArgs = ""
if ($YcsbZipfTheta -ge 0) {
    $YcsbZipfArgs = "--ycsb-zipf-theta $YcsbZipfTheta"
}
$ArgsLine = "--benchmark $Benchmark --workload $Workload --level $Level --workload-profile $WorkloadProfile $YcsbZipfArgs $MatrixArgs --episodes $Episodes --tasks $Tasks --workers $Workers --duration $Duration --clients $Clients --agent-ratio $AgentRatio --agents $Agents --background $Background --background-mode $BackgroundMode --retries $Retries $RetryUntilCommitArgs --max-attempts-per-task $MaxAttemptsPerTask --agent-retry-backoff-ms $AgentRetryBackoffMs --background-retry-backoff-ms $BackgroundRetryBackoffMs --tokens-per-operation $TokensPerOperation --seed $Seed --abort-threshold $AbortThreshold --min-visits $MinVisits --protect-cost-threshold-ms $ProtectCostThresholdMs --low-conflict-safe-abort-rate $LowConflictSafeAbortRate $LowConflictGuardArgs $SparseRiskArgs --commit-value $CommitValue --abort-penalty $AbortPenalty --reasoning-weight $ReasoningWeight --lock-wait-weight $LockWaitWeight --latency-weight $LatencyWeight --lock-hold-weight $LockHoldWeight --background-abort-weight $BackgroundAbortWeight --background-tps-loss-weight $BackgroundTpsLossWeight --ucb-c $UcbC --actions $Actions $BudgetArgs --reasoning-profile $ReasoningProfile --reasoning-scale $ReasoningScale --output $WslOutput"
$Command = "cd $WslRepo && timeout 900s python3 -m agent.cli.train_atcc $ArgsLine"

wsl -e bash -lc $Command
