param(
    [switch]$PaperStyle,
    [string]$Workloads = "ycsb,tpcc",
    [string]$Levels = "low,medium,high",
    [string]$Seeds = "920104,920105,920106",
    [string]$ClientCounts = "",
    [ValidateSet("small", "paper")]
    [string]$WorkloadProfile = "small",
    [double]$YcsbZipfTheta = -1.0,
    [string]$Cc = "occ,dynamic-atcc",
    [double]$Duration = 3.0,
    [int]$Clients = 0,
    [double]$AgentRatio = 0.80,
    [int]$Agents = 2,
    [int]$Background = 8,
    [ValidateSet("none", "light", "agentic", "heavy")]
    [string]$ReasoningProfile = "agentic",
    [double]$ReasoningScale = 2.0,
    [int]$Retries = 0,
    [switch]$RetryUntilCommit,
    [int]$MaxAttemptsPerTask = 100,
    [string]$AgentRetryBackoffMs = "500,5000",
    [string]$BackgroundRetryBackoffMs = "10,30",
    [int]$TokensPerOperation = 2703,
    [switch]$BackgroundWait,
    [ValidateSet("hotspot", "procedure")]
    [string]$BackgroundMode = "hotspot",
    [double]$ReservationTtlS = 5.0,
    [ValidateSet("", "train", "eval", "online")]
    [string]$PolicyMode = "",
    [string]$Policy = "",
    [string]$Output = ""
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
$PolicyArgs = ""
if ($Policy) {
    $PolicyArgs = "--policy $(Convert-ToWslPath $Policy)"
}
$PolicyModeArgs = ""
if ($PolicyMode) {
    $PolicyModeArgs = "--policy-mode $PolicyMode"
}
$OutputArgs = ""
if ($Output) {
    $OutputArgs = "--output $(Convert-ToWslOutputPath $Output)"
}
$BackgroundWaitArgs = ""
if ($BackgroundWait) {
    $BackgroundWaitArgs = "--background-wait"
}
$RetryUntilCommitArgs = ""
if ($RetryUntilCommit) {
    $RetryUntilCommitArgs = "--retry-until-commit"
}
$PaperStyleArgs = ""
if ($PaperStyle) {
    $PaperStyleArgs = "--paper-style"
}
$ClientCountsArgs = ""
if ($ClientCounts) {
    $ClientCountsArgs = "--client-counts $ClientCounts"
}
$YcsbZipfArgs = ""
if ($YcsbZipfTheta -ge 0) {
    $YcsbZipfArgs = "--ycsb-zipf-theta $YcsbZipfTheta"
}

$ArgsLine = "$PaperStyleArgs --workloads $Workloads --levels $Levels --seeds $Seeds $ClientCountsArgs --workload-profile $WorkloadProfile $YcsbZipfArgs --cc $Cc --duration $Duration --clients $Clients --agent-ratio $AgentRatio --agents $Agents --background $Background --reasoning-profile $ReasoningProfile --reasoning-scale $ReasoningScale --retries $Retries $RetryUntilCommitArgs --max-attempts-per-task $MaxAttemptsPerTask --agent-retry-backoff-ms $AgentRetryBackoffMs --background-retry-backoff-ms $BackgroundRetryBackoffMs --tokens-per-operation $TokensPerOperation --background-mode $BackgroundMode --reservation-ttl-s $ReservationTtlS $BackgroundWaitArgs $PolicyArgs $PolicyModeArgs $OutputArgs"
$Command = "cd $WslRepo && timeout 900s python3 -m agent.cli.matrix $ArgsLine"

wsl -e bash -lc $Command
