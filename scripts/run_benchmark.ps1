param(
    [ValidateSet("ycsb", "tpcc", "all")]
    [string]$Workload = "all",

    [ValidateSet("low", "medium", "high")]
    [string]$Profile = "low",

    [ValidateSet("quick", "full")]
    [string]$Strategies = "quick",

    [int]$TaskCount = 10,
    [int]$Workers = 1,
    [int]$AgentSlots = 1,
    [int]$MaxAttempts = 4,
    [int]$Seed = 920104,
    [double]$PlanningDelayMs = 1.0,
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

if (-not $Output) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = "results/delivery_benchmark_$stamp.json"
}

$OutputArgs = ""
if ($Output) {
    $outputParent = Split-Path $Output -Parent
    if ($outputParent -and -not (Test-Path $outputParent)) {
        New-Item -ItemType Directory -Force $outputParent | Out-Null
    }
    $OutputArgs = "--output $(Convert-ToWslOutputPath $Output)"
}

$BenchmarkArgs = @(
    "--workload $Workload",
    "--profile $Profile",
    "--strategies $Strategies",
    "--task-count $TaskCount",
    "--workers $Workers",
    "--agent-slots $AgentSlots",
    "--max-attempts $MaxAttempts",
    "--seed $Seed",
    "--planning-delay-ms $PlanningDelayMs",
    $OutputArgs
) -join " "
$Command = "cd $WslRepo && timeout 180s python3 -m agent.cli.benchmark $BenchmarkArgs"

wsl -e bash -lc $Command
