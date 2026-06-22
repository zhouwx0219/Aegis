$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$wslRoot = "/mnt/" + $root.Substring(0,1).ToLower() + $root.Substring(2).Replace("\", "/")
$argsText = ($args | ForEach-Object { "'$_'" }) -join " "

wsl bash -lc "cd '$wslRoot' && bash scripts/reproduce_vitabench.sh $argsText"
