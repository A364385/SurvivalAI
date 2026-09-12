# SurvivalAI — one-command role training
# Usage: .\scripts\train_role.ps1 market_research
#        .\scripts\train_role.ps1 news_research -Version v0.2 -Dataset my_v2
#        .\scripts\train_role.ps1 market_research -Smoke   (no GPU needed)

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Role,
    [string]$Version = "v0.1",
    [string]$Dataset = "v1",
    [string]$BaseModel = "",
    [switch]$Smoke,
    [switch]$Activate,
    [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($DataDir -ne "") { $env:SURVIVALAI_DATA_DIR = $DataDir }

$cliArgs = @("scripts/train_role.py", "--role", $Role, "--version", $Version, "--dataset", $Dataset)
if ($BaseModel -ne "") { $cliArgs += @("--base-model", $BaseModel) }
if ($Smoke) { $cliArgs += "--smoke" }
if ($Activate) { $cliArgs += "--activate" }

python @cliArgs
exit $LASTEXITCODE
