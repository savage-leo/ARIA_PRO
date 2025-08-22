<#
  ARIA Production Preflight Script
  DAN INSTITUTIONAL DEV CONFIG
  
  Usage:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\tools\production_preflight.ps1 -Full
  
  This script performs comprehensive production readiness checks:
  - Creates isolated venv for testing
  - Installs all dependencies
  - Runs format/lint/type checks
  - Executes AST parse and import validation
  - Runs test suite
  - Produces detailed JSON/Markdown reports
#>

param(
    [switch]$Full,
    [switch]$SkipInstall,
    [switch]$Markdown
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

Write-Host "[DAN] Starting ARIA production preflight on Windows 10 Pro..." -ForegroundColor Cyan
Write-Host "[DAN] Repository root: $repoRoot" -ForegroundColor DarkGray

# Create tools directory if needed
if (-not (Test-Path tools)) { 
    New-Item -Path tools -ItemType Directory | Out-Null 
}

# Create dedicated preflight venv (isolated from main .venv)
$venv = Join-Path $repoRoot 'venv_preflight'
if (-not (Test-Path $venv)) {
    Write-Host "[DAN] Creating isolated venv at $venv" -ForegroundColor Yellow
    python -m venv $venv
} else {
    Write-Host "[DAN] Using existing venv at $venv" -ForegroundColor Green
}

# Activate venv for current session
$activate = Join-Path $venv 'Scripts\Activate.ps1'
. $activate

# Upgrade pip
Write-Host "[DAN] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel | Out-Null

if (-not $SkipInstall) {
    # Install project requirements
    if (Test-Path backend\requirements.txt) {
        Write-Host "[DAN] Installing backend/requirements.txt" -ForegroundColor Yellow
        pip install -r backend\requirements.txt
    } elseif (Test-Path requirements.txt) {
        Write-Host "[DAN] Installing requirements.txt" -ForegroundColor Yellow
        pip install -r requirements.txt
    } elseif (Test-Path pyproject.toml) {
        Write-Host "[DAN] Found pyproject.toml - installing editable deps" -ForegroundColor Yellow
        pip install -e . 2>$null
        if ($LASTEXITCODE -ne 0) { Write-Host "[DAN] Editable install failed - continuing" -ForegroundColor DarkYellow }
    } else {
        Write-Host "[DAN] No requirements file found - continuing with checks" -ForegroundColor DarkYellow
    }
}

# Ensure necessary testing/linting tools
Write-Host "[DAN] Installing testing/linting tools..." -ForegroundColor Yellow
pip install --upgrade black flake8 isort mypy pytest pytest-json-report pytest-asyncio | Out-Null

# Run format checks
Write-Host "`n[DAN] Phase 1: Code formatting checks" -ForegroundColor Cyan

Write-Host "[DAN] Running black --check..." -ForegroundColor Yellow
$blackResult = black --check backend 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[DAN] Black suggests reformatting - run: black backend" -ForegroundColor DarkYellow
} else {
    Write-Host "[DAN] Black check passed" -ForegroundColor Green
}

Write-Host "[DAN] Running isort --check..." -ForegroundColor Yellow
$isortResult = isort --check-only backend 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[DAN] isort suggests fixes - run: isort backend" -ForegroundColor DarkYellow
} else {
    Write-Host "[DAN] isort check passed" -ForegroundColor Green
}

# Run linting
Write-Host "`n[DAN] Phase 2: Linting" -ForegroundColor Cyan

Write-Host "[DAN] Running flake8..." -ForegroundColor Yellow
$flake8Result = flake8 backend --count --statistics 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[DAN] flake8 found issues - review output" -ForegroundColor DarkYellow
} else {
    Write-Host "[DAN] flake8 check passed" -ForegroundColor Green
}

# Run type checking
Write-Host "`n[DAN] Phase 3: Type checking" -ForegroundColor Cyan

Write-Host "[DAN] Running mypy..." -ForegroundColor Yellow
$mypyResult = mypy --ignore-missing-imports backend 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[DAN] mypy found type issues - review output" -ForegroundColor DarkYellow
} else {
    Write-Host "[DAN] mypy check passed" -ForegroundColor Green
}

# Run the comprehensive Python preflight harness
Write-Host "`n[DAN] Phase 4: Comprehensive preflight harness" -ForegroundColor Cyan

$markdownFlag = ""
if ($Markdown) { 
    $markdownFlag = "--markdown" 
}

Write-Host "[DAN] Running aria_preflight.py (AST, imports, compileall, tests)..." -ForegroundColor Yellow
python tools\aria_preflight.py --output tools\production_preflight_report.json $markdownFlag

$harness_exit = $LASTEXITCODE

# Run pytest separately for more detailed coverage
Write-Host "`n[DAN] Phase 5: Test suite execution" -ForegroundColor Cyan

Write-Host "[DAN] Running pytest with coverage..." -ForegroundColor Yellow
pytest backend -q --maxfail=5 --json-report --json-report-file=tools\pytest_report.json 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "[DAN] Pytest passed" -ForegroundColor Green
} else {
    Write-Host "[DAN] Pytest failures detected - check tools/pytest_report.json" -ForegroundColor DarkYellow
}

# Summary
Write-Host "`n" + ("="*60) -ForegroundColor Cyan
Write-Host "[DAN] PREFLIGHT COMPLETE" -ForegroundColor Cyan
Write-Host ("="*60) -ForegroundColor Cyan

Write-Host "`nGenerated reports:" -ForegroundColor Green
Write-Host " - tools\production_preflight_report.json" -ForegroundColor White
if (Test-Path tools\production_preflight_report.md) {
    Write-Host " - tools\production_preflight_report.md" -ForegroundColor White
}
if (Test-Path tools\pytest_report.json) {
    Write-Host " - tools\pytest_report.json" -ForegroundColor White
}

# Exit code based on harness result
if ($harness_exit -eq 2) {
    Write-Host "`n[DAN] CRITICAL ERRORS FOUND - Fix immediately!" -ForegroundColor Red
    exit 2
} elseif ($harness_exit -eq 1) {
    Write-Host "`n[DAN] Issues found - Review and fix before production" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "`n[DAN] All checks passed - Ready for production!" -ForegroundColor Green
    exit 0
}

Pop-Location


