param(
  [ValidateSet('dev','prod')][string]$Mode = 'dev',
  [switch]$KeepAlive,
  [switch]$OpenBrowser
)

Write-Host "[DEPRECATED] Use scripts/run_all.ps1 instead. Forwarding..." -ForegroundColor Yellow
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$orchestrator = Join-Path $scriptDir 'run_all.ps1'
if (-not (Test-Path $orchestrator)) {
  Write-Host "[ERROR] Orchestrator not found at $orchestrator" -ForegroundColor Red
  exit 1
}

# Forward to run_all.ps1 preserving key flags
$argsList = @('-Mode', $Mode)
if ($KeepAlive) { $argsList += '-KeepAlive' }
if ($OpenBrowser) { $argsList += '-OpenBrowser' }

powershell -ExecutionPolicy Bypass -NoLogo -NoProfile -File $orchestrator @argsList
