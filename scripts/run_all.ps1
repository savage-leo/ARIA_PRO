param(
  [ValidateSet('dev','prod')][string]$Mode = 'dev',
  [string]$BindHost = '127.0.0.1',
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5175,
  [string]$PythonPath = '.venv\Scripts\python.exe',
  [switch]$OpenBrowser,
  [switch]$SkipInstall,
  [switch]$KeepAlive
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repoRoot 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Write-Info([string]$msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Err([string]$msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Basic checks
if (-not (Test-Path $PythonPath)) { Write-Err "Python not found at $PythonPath"; exit 1 }
$npm = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npm) { Write-Err 'npm is not installed or not in PATH'; exit 1 }

# Set env that backend expects
$env:TRADE_DB = Join-Path $repoRoot 'logs\trade_memory.sqlite'
# Standard CORS env per production_config.py; keep legacy for back-compat
if (-not $env:ARIA_CORS_ORIGINS) {
  $env:ARIA_CORS_ORIGINS = "http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort"
}
if (-not $env:ARIA_ALLOWED_ORIGINS) {
  $env:ARIA_ALLOWED_ORIGINS = "$($env:ARIA_CORS_ORIGINS),http://localhost:5176,http://127.0.0.1:5176"
}

if (-not $SkipInstall) {
  # Backend deps
  Write-Info 'Installing backend dependencies...'
  & $PythonPath -m pip install -r (Join-Path $repoRoot 'backend\requirements.txt') | Out-Host
  # Frontend deps
  Write-Info 'Installing frontend dependencies...'
  Push-Location (Join-Path $repoRoot 'frontend')
  try {
    npm ci | Out-Host
  } catch {
    Write-Info 'npm ci failed, falling back to npm install'
    npm install | Out-Host
  }
  Pop-Location
}

# Backend — start if not healthy
$base = "http://$BindHost`:$BackendPort"
function Test-Health { try { Invoke-RestMethod -Uri "$base/health" -Method Get -TimeoutSec 2 } catch { $null } }
function Wait-Health([int]$TimeoutSec){ $deadline=(Get-Date).AddSeconds($TimeoutSec); while((Get-Date) -lt $deadline){ $h=Test-Health; if($h -and $h.status -eq 'ok'){ return $h }; Start-Sleep -Milliseconds 500 }; throw "Health timeout" }

$h = Test-Health
$backendStarted = $false
if (-not $h) {
  Write-Info "Starting backend on $base ..."
  $args = @('-m','uvicorn','backend.main:app','--host', $BindHost, '--port', $BackendPort.ToString())
  $proc = Start-Process -FilePath $PythonPath -ArgumentList $args -PassThru -WindowStyle Hidden
  $backendStarted = $true
  Set-Content -Path (Join-Path $logDir 'backend.pid') -Value $proc.Id
  $h = Wait-Health -TimeoutSec 30
}
Write-Info ("Backend HEALTH => " + ($h | ConvertTo-Json -Depth 5))

# Frontend
$frontendDir = Join-Path $repoRoot 'frontend'
if ($Mode -eq 'prod') {
  Write-Info 'Building frontend (vite build)...'
  Push-Location $frontendDir; npm run build | Out-Host; Pop-Location
  Write-Info "Starting vite preview on http://localhost:$FrontendPort ..."
  $fe = Start-Process -FilePath 'npm' -ArgumentList @('run','preview','--','--port',"$FrontendPort") -WorkingDirectory $frontendDir -PassThru -WindowStyle Hidden
} else {
  Write-Info "Starting frontend dev on http://localhost:$FrontendPort ..."
  $fe = Start-Process -FilePath 'npm' -ArgumentList @('run','dev','--','--port',"$FrontendPort",'--host') -WorkingDirectory $frontendDir -PassThru -WindowStyle Hidden
}
Set-Content -Path (Join-Path $logDir 'frontend.pid') -Value $fe.Id
Start-Sleep -Seconds 2

# Self-test: Trade Memory minimal flow
Write-Info 'Running Trade Memory self-test...'
$created = Invoke-RestMethod -Uri "$base/trade-memory/ideas" -Method Post -Body (@{ idea = @{ symbol='EURUSD'; bias='long'; confidence=0.82 }; meta = @{ source='orchestrator' } } | ConvertTo-Json -Depth 5) -ContentType 'application/json'
$id = $created.id
$recent = Invoke-RestMethod -Uri "$base/trade-memory/recent?limit=3" -Method Get
$item = Invoke-RestMethod -Uri "$base/trade-memory/$id" -Method Get
$outcome = Invoke-RestMethod -Uri "$base/trade-memory/$id/outcome" -Method Patch -Body (@{ outcome = @{ pnl = 1.23; status = 'closed' } } | ConvertTo-Json -Depth 5) -ContentType 'application/json'
$summary = [ordered]@{ backend=$base; frontend="http://localhost:$FrontendPort"; created=$created; recent=$recent; by_id=$item; outcome=$outcome }
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $logDir 'orchestrator_selftest.json')
Write-Info 'Self-test completed. Summary: logs\orchestrator_selftest.json'

if ($OpenBrowser) {
  Start-Process "$base/docs" | Out-Null
  Start-Process "http://localhost:$FrontendPort" | Out-Null
}

Write-Info 'All services are up.'
if (-not $KeepAlive) {
  Write-Info 'Stopping child processes (no KeepAlive)'
  try { if ($fe -and -not $fe.HasExited) { Stop-Process -Id $fe.Id -Force } } catch {}
  try { if ($backendStarted -and $proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force } } catch {}
}
