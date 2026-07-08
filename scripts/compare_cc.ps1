param(
    [ValidateSet("ycsb", "tpcc")]
    [string]$Workload = "ycsb",

    [ValidateSet("low", "medium", "high")]
    [string]$Level = "low",

    [ValidateSet("small", "paper")]
    [string]$WorkloadProfile = "small",

    [string]$Cc = "all",
    [int]$Tasks = 10,
    [int]$Workers = 8,
    [int]$Retries = 0,
    [ValidateSet("none", "light", "agentic", "heavy")]
    [string]$ReasoningProfile = "agentic",
    [double]$ReasoningScale = 1.0,
    [int]$Seed = 920104,
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
$OutputArgs = ""
if ($Output) {
    $outputParent = Split-Path $Output -Parent
    if ($outputParent -and -not (Test-Path $outputParent)) {
        New-Item -ItemType Directory -Force $outputParent | Out-Null
    }
    $OutputArgs = "--output $(Convert-ToWslOutputPath $Output)"
}
$PolicyArgs = ""
if ($Policy) {
    $PolicyArgs = "--policy $(Convert-ToWslPath $Policy)"
}
$PolicyModeArgs = ""
if ($PolicyMode) {
    $PolicyModeArgs = "--policy-mode $PolicyMode"
}

$ArgsLine = "--workload $Workload --level $Level --workload-profile $WorkloadProfile --cc $Cc --tasks $Tasks --workers $Workers --retries $Retries --reasoning-profile $ReasoningProfile --reasoning-scale $ReasoningScale --seed $Seed $PolicyArgs $PolicyModeArgs $OutputArgs"
$Command = "cd $WslRepo && timeout 180s python3 -m agent.cli.compare $ArgsLine"

wsl -e bash -lc $Command
