#!/usr/bin/env pwsh
# ----------------------------------------------------------------------
# Start Institutional Proxy for ARIA PRO
# ----------------------------------------------------------------------

Write-Host "🚀 Starting ARIA PRO Institutional Proxy..." -ForegroundColor Green

# Set environment variables
$env:LOCAL_OLLAMA_URL = "http://127.0.0.1:11436"
$env:LOCAL_MODELS = "mistral:latest,phi:latest,llama3:latest"

# Configure remote models (edit these)
$env:GPTOS_120B_URL = "https://your-gptos-server.com:8443/api/v1"
$env:GPTOS_120B_KEY = "your_api_key_here"

# Start the proxy
Write-Host "📍 Proxy will run on http://localhost:11434" -ForegroundColor Yellow
Write-Host "🔧 Edit backend/services/institutional_proxy.py to configure models" -ForegroundColor Yellow

cd backend/services
python institutional_proxy.py
