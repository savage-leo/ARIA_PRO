<#
  tools\production_preflight.ps1
  Usage (PowerShell, repo root):
    .\.venv\Scripts\Activate.ps1
    powershell -ExecutionPolicy Bypass -File .\tools\production_preflight.ps1 -AutoFix:$false

  This script:
  - Creates/activates venv (assumes you already installed Python 3.11)
  - Installs test deps (no heavy security packages)
  - Runs backend lint/tests (black check, mypy, flake8, pytest)
  - Runs frontend tests (vitest/jest) if frontend exists
  - Runs simulation/fake-data scan
  - Attempts MT5 connection check (no credentials => logs alert and stops model smoke tests)
  - Runs model smoke tests only if MT5 connection available
  - Produces a single report: tools/production_preflight_report.json
#>

param(
  [switch]$AutoFix = $false
)

Write-Host "=== ARIA PRODUCTION PREFLIGHT (Windows) ===" -ForegroundColor Cyan

# 1) Activate venv (expect .venv exists). If not, create it.
if (-not (Test-Path ".venv")) {
  Write-Host "Virtualenv not found. Creating .venv..."
  python -m venv .venv
}
Write-Host "Activating venv..."
. .\.venv\Scripts\Activate.ps1

# 2) Install minimal infra deps for running diagnostics
Write-Host "Installing diagnostic dependencies..."
python -m pip install --upgrade pip
python -m pip install black mypy flake8 pytest pytest-cov pytest-asyncio requests sqlalchemy psycopg2-binary onnxruntime MetaTrader5

# Optional heavy packages (best-effort)
python -m pip install torch
if ($LASTEXITCODE -ne 0) { Write-Host "torch install failed (optional)" }
python -m pip install tensorflow
if ($LASTEXITCODE -ne 0) { Write-Host "tensorflow install failed (optional)" }

# 3) Run DAN Audit Engine
$autoFixFlag = ""
if ($AutoFix) { $autoFixFlag = "--auto-fix" }

Write-Host "Running DAN Audit Engine (backend/tools/dan_audit_engine.py)..."
python backend\tools\dan_audit_engine.py --report tools\production_preflight_report.json $autoFixFlag

if ($LASTEXITCODE -eq 0) {
  Write-Host "DAN Audit Engine completed successfully. Report: tools/production_preflight_report.json" -ForegroundColor Green
} elseif ($LASTEXITCODE -eq 2) {
  Write-Host "DAN Audit terminated early due to no MT5 connection or critical config missing. See report." -ForegroundColor Yellow
} else {
  Write-Host "DAN Audit encountered errors. See report and console output." -ForegroundColor Red
}

Write-Host "`n=== END PREFLIGHT ==="


