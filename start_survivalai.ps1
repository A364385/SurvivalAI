# SurvivalAI — Windows launcher
# Verifies environment, checks configuration, runs health checks, starts
# backend + dashboard, and displays the local URL. Does not hide errors.
#
# Usage:
#   .\start_survivalai.ps1              # test mode (offline mocks)
#   .\start_survivalai.ps1 -Mode paper  # Alpaca PAPER mode (keys required)

param(
    [ValidateSet("test", "paper")]
    [string]$Mode = "test",
    [int]$Port = 8080,
    [switch]$NoRuntime
)

$ErrorActionPreference = "Stop"

Write-Host ("=" * 62)
Write-Host " SurvivalAI - LOCAL STARTUP (PAPER TRADING ONLY)"
Write-Host ("=" * 62)

# 1. Python check
$python = "python"
try {
    $version = & $python --version 2>&1
    Write-Host "[ok] $version"
} catch {
    Write-Host "[FAIL] Python not found. Install Python 3.11+ from python.org" -ForegroundColor Red
    exit 1
}

# 2. Repo root + move to it
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $repoRoot "app"))) {
    $repoRoot = $PSScriptRoot
}
Set-Location $repoRoot
Write-Host "[ok] project root: $repoRoot"

# 3. Data directory + disk space check
$dataDir = if ($env:SURVIVALAI_DATA_DIR) { $env:SURVIVALAI_DATA_DIR } else { Join-Path $repoRoot "data" }
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$drive = Get-PSDrive -Name ($dataDir.Substring(0,1))
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Write-Host "[ok] data dir: $dataDir (${freeGB}GB free)"
if ($freeGB -lt 2) {
    Write-Host "[WARN] Less than 2GB free disk space. Model downloads and checkpoints WILL fail." -ForegroundColor Yellow
    Write-Host "       Set SURVIVALAI_DATA_DIR to a drive with space, e.g.:" -ForegroundColor Yellow
    Write-Host '       $env:SURVIVALAI_DATA_DIR = "D:\survivalai_data"' -ForegroundColor Yellow
}

# 4. Paper mode requires Alpaca paper keys
if ($Mode -eq "paper") {
    if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
        Write-Host "[FAIL] Paper mode requires ALPACA_API_KEY and ALPACA_API_SECRET" -ForegroundColor Red
        Write-Host "       (Alpaca PAPER keys only: https://app.alpaca.markets - Paper Trading)"
        exit 1
    }
    Write-Host "[ok] Alpaca PAPER credentials found (live trading is architecturally impossible)"
} else {
    Write-Host "[ok] test mode: deterministic offline mocks, no network needed"
}

# 5. Optional: local LLM provider detection
if ($env:SURVIVALAI_LLM_PROVIDER) {
    Write-Host "[ok] local LLM provider: $env:SURVIVALAI_LLM_PROVIDER"
} else {
    Write-Host "[info] No SURVIVALAI_LLM_PROVIDER set; agents use the deterministic fallback."
    Write-Host "       For real local models: set SURVIVALAI_LLM_PROVIDER=lm_studio|ollama"
    Write-Host "       (LM Studio default endpoint http://127.0.0.1:1234, Ollama http://127.0.0.1:11434)"
}

# 6. Launch
Write-Host ""
Write-Host "Starting SurvivalAI backend + dashboard..." 
$cliArgs = @("scripts/run_local.py", "--mode", $Mode, "--port", $Port)
if ($NoRuntime) { $cliArgs += "--no-runtime" }
& $python @cliArgs
exit $LASTEXITCODE
