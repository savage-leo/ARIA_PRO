# ARIA PRO — Go Live (Set flags, restart launcher, health checks)
[CmdletBinding()]
param(
  [switch]$NoNgrok,
  [string]$Root
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve project root
if([string]::IsNullOrWhiteSpace($Root)){
  $Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
Set-Location $Root

# Enable full live execution + LLM monitor/tuning
$vars = @{
  'ARIA_ENABLE_EXEC'    = '1'
  'AUTO_EXEC_ENABLED'   = '1'
  'ALLOW_LIVE'          = '1'
  'LLM_MONITOR_ENABLED' = '1'
  'LLM_TUNING_ENABLED'  = '1'
}
foreach($k in $vars.Keys){ [Environment]::SetEnvironmentVariable($k, $vars[$k]) }

# Ensure ADMIN_API_KEY exists (do not overwrite if already set)
if([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable('ADMIN_API_KEY'))){
  [Environment]::SetEnvironmentVariable('ADMIN_API_KEY','aria_pro_admin_key_2024')
}
$admin = [Environment]::GetEnvironmentVariable('ADMIN_API_KEY')

# Launch the main live runner
$launcher = Join-Path $Root 'scripts\run_live.ps1'
if(-not (Test-Path $launcher)){ throw "Launcher not found: $launcher" }

$laArgs = @()
if($NoNgrok){ $laArgs += '-NoNgrok' }

# Prefer pwsh when available
$ps = 'powershell'
try { if(Get-Command pwsh -ErrorAction SilentlyContinue){ $ps = 'pwsh' } } catch {}

& $ps -NoProfile -ExecutionPolicy Bypass -File $launcher @laArgs

# Post health checks
Start-Sleep -Seconds 5
$backendHealth = 'http://127.0.0.1:8000/health'
$proxyHealth   = 'http://127.0.0.1:8101/health'

function Check([string]$name, [scriptblock]$call) {
  try { & $call | Out-Null; Write-Host "$name OK" -ForegroundColor Green }
  catch { Write-Warning "$name failed: $($_.Exception.Message)" }
}

Check 'Backend' { Invoke-RestMethod -Uri $backendHealth -Method GET -TimeoutSec 5 }
Check 'Proxy'   { Invoke-RestMethod -Uri $proxyHealth   -Method GET -TimeoutSec 5 }

try {
  $hdr = @{ 'X-ARIA-ADMIN' = $admin }
  $lm = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/debug/llm-monitor/status' -Headers $hdr -Method GET -TimeoutSec 5
  Write-Host ("LLM Monitor: " + ($lm | ConvertTo-Json -Depth 5)) -ForegroundColor DarkGray
} catch { Write-Warning "LLM Monitor check failed: $($_.Exception.Message)" }

try {
  $at = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/monitoring/auto-trader/status' -Method GET -TimeoutSec 5
  Write-Host ("AutoTrader: " + ($at | ConvertTo-Json -Depth 5)) -ForegroundColor DarkGray
} catch { Write-Warning "AutoTrader status check failed: $($_.Exception.Message)" }

# ngrok tunnel discovery (if local API available)
try {
  $tunnels = Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -Method GET -TimeoutSec 3
  foreach($t in $tunnels.tunnels){
    $port = ($t.config.addr -split ':')[-1]
    Write-Host ("ngrok port {0} => {1}" -f $port, $t.public_url) -ForegroundColor Cyan
  }
} catch {}
