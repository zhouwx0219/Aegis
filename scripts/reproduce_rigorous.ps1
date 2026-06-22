$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$wslRoot = "/mnt/" + $root.Substring(0,1).ToLower() + $root.Substring(2).Replace("\", "/")
$profile = if ($args.Count -gt 0) { $args[0] } else { "large" }

wsl bash -lc "cd '$wslRoot' && bash scripts/reproduce_rigorous.sh '$profile'"
