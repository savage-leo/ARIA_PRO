param(
  [string]$BindHost = '127.0.0.1',
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5175,
  [switch]$Verbose
)

$ErrorActionPreference = 'Stop'
$base = "http://$BindHost`:$BackendPort"

function Write-Status([string]$msg, [string]$status = 'INFO') {
  $color = switch($status) {
    'OK' { 'Green' }
    'WARN' { 'Yellow' }
    'ERROR' { 'Red' }
    default { 'Cyan' }
  }
  Write-Host "[$status] $msg" -ForegroundColor $color
}

function Test-Endpoint([string]$url, [string]$method = 'GET', [object]$body = $null) {
  try {
    $params = @{ Uri = $url; Method = $method; TimeoutSec = 5 }
    if ($body) {
      $params.Body = ($body | ConvertTo-Json -Depth 5)
      $params.ContentType = 'application/json'
    }
    $response = Invoke-RestMethod @params
    return @{ success = $true; data = $response }
  } catch {
    return @{ success = $false; error = $_.Exception.Message }
  }
}

Write-Status "=== ARIA PRO System Verification ===" 'INFO'

# 1. Backend Health & Core Endpoints
Write-Status "Checking backend health..." 'INFO'
$health = Test-Endpoint "$base/health"
if ($health.success) {
  Write-Status "Backend health: OK" 'OK'
} else {
  Write-Status "Backend health: FAILED - $($health.error)" 'ERROR'
  exit 1
}

# 2. Data Sources Status
Write-Status "Checking data sources..." 'INFO'
$dataSources = Test-Endpoint "$base/data-sources/status"
if ($dataSources.success) {
  Write-Status "Data sources: $($dataSources.data.Count) registered" 'OK'
  if ($Verbose) {
    $dataSources.data | ForEach-Object { Write-Status "  - $($_.name): $($_.status)" 'INFO' }
  }
} else {
  Write-Status "Data sources: FAILED - $($dataSources.error)" 'WARN'
}

# 3. Signal Generation Pipeline
Write-Status "Testing signal generation..." 'INFO'
$signalTest = Test-Endpoint "$base/signals/generate" 'POST' @{
  symbol = 'EURUSD'
  timeframe = 'M5'
  bars = 100
}
if ($signalTest.success) {
  Write-Status "Signal generation: OK" 'OK'
  if ($Verbose -and $signalTest.data.signals) {
    Write-Status "  Generated $($signalTest.data.signals.Count) signals" 'INFO'
  }
} else {
  Write-Status "Signal generation: FAILED - $($signalTest.error)" 'WARN'
}

# 4. Trade Memory CRUD
Write-Status "Testing Trade Memory CRUD..." 'INFO'
$tradeIdea = Test-Endpoint "$base/trade-memory/ideas" 'POST' @{
  idea = @{ symbol = 'EURUSD'; bias = 'long'; confidence = 0.85 }
  meta = @{ source = 'verification' }
}
if ($tradeIdea.success) {
  $id = $tradeIdea.data.id
  Write-Status "Trade Memory create: OK (ID: $id)" 'OK'
  
  # Test read
  $readTest = Test-Endpoint "$base/trade-memory/$id"
  if ($readTest.success) {
    Write-Status "Trade Memory read: OK" 'OK'
  }
  
  # Test outcome update
  $outcomeTest = Test-Endpoint "$base/trade-memory/$id/outcome" 'PATCH' @{
    outcome = @{ pnl = 15.0; status = 'closed' }
  }
  if ($outcomeTest.success) {
    Write-Status "Trade Memory outcome: OK" 'OK'
  }
} else {
  Write-Status "Trade Memory: FAILED - $($tradeIdea.error)" 'WARN'
}

# 5. Monitoring Endpoints
Write-Status "Checking monitoring endpoints..." 'INFO'
$monitoring = @(
  '/monitoring/models/status',
  '/monitoring/auto-trader/status'
)
foreach ($endpoint in $monitoring) {
  $result = Test-Endpoint "$base$endpoint"
  if ($result.success) {
    Write-Status "Monitor $endpoint`: OK" 'OK'
  } else {
    Write-Status "Monitor $endpoint`: FAILED" 'WARN'
  }
}

# 6. WebSocket Status
Write-Status "Checking WebSocket endpoint..." 'INFO'
try {
  $wsTest = Invoke-WebRequest -Uri "$base/ws" -Method GET -TimeoutSec 2
  if ($wsTest.StatusCode -eq 426) {  # Upgrade Required = WebSocket endpoint exists
    Write-Status "WebSocket endpoint: OK (426 Upgrade Required)" 'OK'
  }
} catch {
  Write-Status "WebSocket endpoint: FAILED - $($_.Exception.Message)" 'WARN'
}

# 7. Frontend Availability
Write-Status "Checking frontend availability..." 'INFO'
try {
  $frontend = Invoke-WebRequest -Uri "http://$BindHost`:$FrontendPort" -Method GET -TimeoutSec 3
  if ($frontend.StatusCode -eq 200) {
    Write-Status "Frontend: OK (HTTP $($frontend.StatusCode))" 'OK'
  }
} catch {
  Write-Status "Frontend: FAILED - $($_.Exception.Message)" 'WARN'
}

# 8. C++ Integration Status
Write-Status "Checking C++ SMC integration..." 'INFO'
$cppStatus = Test-Endpoint "$base/api/smc/cpp/status"
if ($cppStatus.success) {
  Write-Status "C++ SMC: $($cppStatus.data.status)" 'OK'
} else {
  Write-Status "C++ SMC: FAILED - $($cppStatus.error)" 'WARN'
}

Write-Status "=== Verification Complete ===" 'INFO'
Write-Status "Backend: $base" 'INFO'
Write-Status "Frontend: http://$BindHost`:$FrontendPort" 'INFO'
Write-Status "API Docs: $base/docs" 'INFO'
