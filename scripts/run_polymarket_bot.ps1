# PowerShell script to run Polymarket Live Trading Bot as a service
# Usage: .\run_polymarket_bot.ps1 [-Live] [-Config <path>]

param(
    [switch]$Live,
    [string]$Config = "./live_config.json"
)

$ErrorActionPreference = "Stop"

# Determine mode
$mode = if ($Live) { "live" } else { "paper" }

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Polymarket Live Trading Bot Launcher" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Mode:        $mode"
Write-Host "Config:      $Config"
Write-Host "Log Dir:     ./live_logs"
Write-Host ""

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Please install Python 3.11+"
    exit 1
}

# Install dependencies if needed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
pip install -q requests numpy pandas scikit-learn

# Run the bot
try {
    Write-Host "Starting bot..." -ForegroundColor Green
    python polymarket_live_trading_bot.py --mode $mode --config $Config
}
catch {
    Write-Error "Bot crashed: $_"
    exit 1
}

Write-Host "Bot stopped." -ForegroundColor Cyan
