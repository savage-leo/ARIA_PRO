# ARIA PRO — Live Runner (Proxy + Backend + ngrok + Logs)
# INTENT: Launch real LLM proxy, backend with LLM monitor, ngrok tunnels, and log tailing.
# NOTE: This script relies on your existing environment variables. No secrets are stored here.
# Date: 2025-08-19
# INTENTIONAL_DEFAULT_TAG: Live MT5 and execution defaults are enabled in backend/core/config.py by design.

[CmdletBinding()]
param(
    [switch]$NoNgrok
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Require-Exe($path, [string]$name){
    if(-not (Test-Path $path)){ throw "Missing $name at $path" }
}

function Require-Env([string]$name){
    $val = [Environment]::GetEnvironmentVariable($name)
    if([string]::IsNullOrWhiteSpace($val)){
        throw "Missing required environment variable: $name"
    }
}

function Load-DotEnv([string]$path){
    if(-not (Test-Path $path)){ return }
    Get-Content -Path $path | ForEach-Object {
        $line = $_.Trim()
        if([string]::IsNullOrWhiteSpace($line)){ return }
        if($line.StartsWith('#')){ return }
        $idx = $line.IndexOf('=')
        if($idx -lt 1){ return }
        $key = $line.Substring(0,$idx).Trim()
        $val = $line.Substring($idx+1).Trim().Trim('"')
        if(-not [string]::IsNullOrWhiteSpace($key)){
            if([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($key))){
                [Environment]::SetEnvironmentVariable($key, $val)
            }
        }
    }
}

function Wait-Http200([string]$url, [int]$retries = 40){
    for($i=0; $i -lt $retries; $i++){
        try {
            Invoke-RestMethod -Uri $url -Method GET -TimeoutSec 5 | Out-Null
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    throw "Timeout waiting for $url"
}

function Start-InWindow([string]$title, [string]$command){
    $ps = 'powershell'
    try {
        if (Get-Command pwsh -ErrorAction SilentlyContinue) { $ps = 'pwsh' }
    } catch {}
    $args = @('-NoExit','-Command', $command)
    Start-Process -FilePath $ps -ArgumentList $args -WindowStyle Normal -Verb runAs -PassThru |
        ForEach-Object { try { $_.MainWindowTitle = $title } catch {} } | Out-Null
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonPath  = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogsDir     = Join-Path $ProjectRoot 'logs'
$ProxyPort   = 8101
$BackendPort = 8000
$ProxyHost   = '0.0.0.0'
$BackendHost = '127.0.0.1'

Require-Exe $PythonPath 'Python (venv)'
if(-not (Test-Path $LogsDir)){ New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null }

# Load environment from .env / production.env if present
Load-DotEnv (Join-Path $ProjectRoot '.env')
Load-DotEnv (Join-Path $ProjectRoot 'production.env')

# Proxy env (real LLM; mock must be off)
if([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable('PROXY_BASE_URL'))){
    [Environment]::SetEnvironmentVariable('PROXY_BASE_URL','https://openrouter.ai/api/v1')
    Write-Host 'Defaulted PROXY_BASE_URL to https://openrouter.ai/api/v1' -ForegroundColor DarkYellow
}

$apiKey = [Environment]::GetEnvironmentVariable('PROXY_API_KEY')
if([string]::IsNullOrWhiteSpace($apiKey) -or $apiKey.Trim() -eq 'REPLACE_ME'){
    Write-Host 'PROXY_API_KEY not found in environment. Enter it now (not persisted):' -ForegroundColor Yellow
    $sec = Read-Host -AsSecureString 'PROXY_API_KEY'
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    try {
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
    if([string]::IsNullOrWhiteSpace($plain)){
        throw 'A non-empty PROXY_API_KEY is required for real LLM (mock is disabled).'
    }
    [Environment]::SetEnvironmentVariable('PROXY_API_KEY', $plain)
}

# Optional defaults for proxy
if(-not [Environment]::GetEnvironmentVariable('DEFAULT_MODEL')){ [Environment]::SetEnvironmentVariable('DEFAULT_MODEL','gpt-4o-mini') }
[Environment]::SetEnvironmentVariable('PROXY_MOCK','0')

# Optional defaults for backend LLM monitor
if(-not [Environment]::GetEnvironmentVariable('LLM_MONITOR_ENABLED')){ [Environment]::SetEnvironmentVariable('LLM_MONITOR_ENABLED','1') }
if(-not [Environment]::GetEnvironmentVariable('LLM_TUNING_ENABLED')){ [Environment]::SetEnvironmentVariable('LLM_TUNING_ENABLED','1') }
if(-not [Environment]::GetEnvironmentVariable('LLM_MONITOR_INTERVAL_SEC')){ [Environment]::SetEnvironmentVariable('LLM_MONITOR_INTERVAL_SEC','15') }
if(-not [Environment]::GetEnvironmentVariable('LLM_TUNING_MAX_REL_DELTA')){ [Environment]::SetEnvironmentVariable('LLM_TUNING_MAX_REL_DELTA','0.10') }
if(-not [Environment]::GetEnvironmentVariable('LLM_TUNING_COOLDOWN_SEC')){ [Environment]::SetEnvironmentVariable('LLM_TUNING_COOLDOWN_SEC','300') }
if(-not [Environment]::GetEnvironmentVariable('LLM_MONITOR_DAN_URL')){ [Environment]::SetEnvironmentVariable('LLM_MONITOR_DAN_URL',"http://127.0.0.1:$ProxyPort") }

# Live guard + admin (optional overrides; backend has safe defaults per config)
if(-not [Environment]::GetEnvironmentVariable('ARIA_ENABLE_MT5')){ [Environment]::SetEnvironmentVariable('ARIA_ENABLE_MT5','1') }
if(-not [Environment]::GetEnvironmentVariable('ADMIN_API_KEY')){ [Environment]::SetEnvironmentVariable('ADMIN_API_KEY','aria_admin_2024_secure_key') }

Write-Host "Project Root: $ProjectRoot" -ForegroundColor Cyan
Write-Host "Python: $PythonPath" -ForegroundColor Cyan

# Start Proxy window
$proxyCmd = @"
Set-Location "$ProjectRoot"; 
Write-Host 'Starting DAN Proxy (real LLM)...' -ForegroundColor Yellow;
& "$PythonPath" proxy/dan_proxy.py
"@
Start-InWindow -title "ARIA PRO — DAN Proxy" -command $proxyCmd

# Wait for Proxy health on localhost
$proxyHealth = "http://127.0.0.1:$ProxyPort/health"
Write-Host "Waiting for Proxy: $proxyHealth" -ForegroundColor DarkYellow
Wait-Http200 $proxyHealth
Write-Host "Proxy is up." -ForegroundColor Green

# Start Backend window
$backendCmd = @"
Set-Location "$ProjectRoot";
Write-Host 'Starting Backend (Uvicorn)...' -ForegroundColor Yellow;
& "$PythonPath" -m uvicorn backend.main:app --host $BackendHost --port $BackendPort
"@
Start-InWindow -title "ARIA PRO — Backend" -command $backendCmd

# Wait for Backend health on localhost
$backendHealth = "http://127.0.0.1:$BackendPort/health"
Write-Host "Waiting for Backend: $backendHealth" -ForegroundColor DarkYellow
Wait-Http200 $backendHealth
Write-Host "Backend is up." -ForegroundColor Green

if(-not $NoNgrok){
    # Start ngrok tunnels for backend and proxy
    $ngrokBackendCmd = "ngrok http $BackendPort"
    $ngrokProxyCmd   = "ngrok http $ProxyPort"
    Start-InWindow -title "ngrok — Backend $BackendPort" -command $ngrokBackendCmd
    Start-Sleep -Seconds 1
    Start-InWindow -title "ngrok — Proxy $ProxyPort" -command $ngrokProxyCmd
    Start-Sleep -Seconds 2

    # Fetch public URLs from ngrok API
    function Get-NgrokPublicUrl([int]$port){
        try{
            $tunnels = Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -Method GET -TimeoutSec 5
            foreach($t in $tunnels.tunnels){
                if($t.config.addr -match ":$port$"){
                    return $t.public_url
                }
            }
        } catch {}
        return $null
    }

    $backendPublic = $null; $proxyPublic = $null
    for($i=0;$i -lt 30;$i++){
        if(-not $backendPublic){ $backendPublic = Get-NgrokPublicUrl -port $BackendPort }
        if(-not $proxyPublic){ $proxyPublic = Get-NgrokPublicUrl -port $ProxyPort }
        if($backendPublic -and $proxyPublic){ break }
        Start-Sleep -Seconds 1
    }

    if($backendPublic){ Write-Host "Backend public: $backendPublic" -ForegroundColor Green }
    else { Write-Warning 'Could not resolve backend ngrok URL. Is ngrok running?' }
    if($proxyPublic){ Write-Host "Proxy public:   $proxyPublic" -ForegroundColor Green }
    else { Write-Warning 'Could not resolve proxy ngrok URL. Is ngrok running?' }

    Write-Host "If routing backend -> proxy via public URL, set:" -ForegroundColor Yellow
    if($proxyPublic){ Write-Host "  `$env:LLM_MONITOR_DAN_URL=\"$proxyPublic\"" -ForegroundColor Yellow }
}

# Start log tailing windows
$tailBackend = @"
Get-Content -Path "$LogsDir\backend.log" -Wait
"@
$tailProd    = @"
Get-Content -Path "$LogsDir\production.log" -Wait
"@
Start-InWindow -title "Logs — backend.log" -command $tailBackend
Start-InWindow -title "Logs — production.log" -command $tailProd

Write-Host "\nAll services launched. Use these to interact:" -ForegroundColor Cyan
Write-Host "  Health (backend): $backendHealth" -ForegroundColor Cyan
Write-Host "  Health (proxy):   $proxyHealth" -ForegroundColor Cyan
Write-Host "\nAdmin header for debug routes: X-ARIA-ADMIN = $( [Environment]::GetEnvironmentVariable('ADMIN_API_KEY') )" -ForegroundColor DarkGray
Write-Host "\nTo stop: close the opened PowerShell windows (Proxy, Backend, ngrok, Logs)." -ForegroundColor DarkGray

# Quick health checks
Write-Host "\nRunning quick health checks..." -ForegroundColor Cyan
try {
    $bh = Invoke-RestMethod -Uri $backendHealth -Method GET -TimeoutSec 5
    Write-Host "Backend OK" -ForegroundColor Green
} catch { Write-Warning "Backend health check failed: $($_.Exception.Message)" }

try {
    $ph = Invoke-RestMethod -Uri $proxyHealth -Method GET -TimeoutSec 5
    Write-Host "Proxy OK" -ForegroundColor Green
} catch { Write-Warning "Proxy health check failed: $($_.Exception.Message)" }

$admin = [Environment]::GetEnvironmentVariable('ADMIN_API_KEY')
if(-not [string]::IsNullOrWhiteSpace($admin)){
    $hdr = @{ 'X-ARIA-ADMIN' = $admin }
    try {
        $lm = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/debug/llm-monitor/status" -Headers $hdr -Method GET -TimeoutSec 5
        Write-Host "LLM Monitor: $( ($lm | ConvertTo-Json -Depth 5) )" -ForegroundColor DarkGray
    } catch { Write-Warning "LLM monitor status check failed: $($_.Exception.Message)" }
}

try {
    $at = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/monitoring/auto-trader/status" -Method GET -TimeoutSec 5
    Write-Host "AutoTrader: $( ($at | ConvertTo-Json -Depth 5) )" -ForegroundColor DarkGray
} catch { Write-Warning "AutoTrader status check failed: $($_.Exception.Message)" }
