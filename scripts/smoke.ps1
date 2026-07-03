param(
    [switch]$Json
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
$JsonArg = if ($Json) { "--json" } else { "" }
$Command = "cd $WslRepo && timeout 180s python3 -m agent.cli.smoke $JsonArg"

wsl -e bash -lc $Command
