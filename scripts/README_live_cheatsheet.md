# ARIA PRO — Live Start Cheat Sheet (Windows PowerShell)

This is a 1-page quick-reference to start ARIA live with real MT5 + LLM monitoring, run health checks, and tail logs.

## 1) One-Command Live Start
From project root `C:\savage\ARIA_PRO`:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_live.ps1
```

What it does:
- Loads `.env` + `production.env` (no secrets stored in the script)
- Starts DAN Proxy (real LLM; mock OFF)
- Starts Backend (MT5 live + LLM monitor + tuning)
- Creates two ngrok tunnels (Backend: 8000, Proxy: 8101)
- Opens two log tails: `logs/backend.log`, `logs/production.log`
- Prompts for `PROXY_API_KEY` only if not set

## 2) Health Checks
```powershell
# Backend health
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET

# Proxy health
Invoke-RestMethod -Uri "http://127.0.0.1:8101/health" -Method GET

# LLM monitor status (admin header from env)
$hdr = @{ "X-ARIA-ADMIN" = "aria_admin_2024_secure_key" }
Invoke-RestMethod -Uri "http://127.0.0.1:8000/debug/llm-monitor/status" -Headers $hdr -Method GET

# AutoTrader status
Invoke-RestMethod -Uri "http://127.0.0.1:8000/monitoring/auto-trader/status" -Method GET
```

## 3) Logs to Watch
- Proxy window: upstream LLM calls
- Backend window: MT5 connect, symbols, trade cycles
- `logs/backend.log`: signals, gating, SL/TP, fills, slippage
- `logs/production.log`: LLMMonitor batches, tuning suggestions, applied changes

## 4) Optional: Backend → Proxy via ngrok
After tunnels are up, you’ll see URLs like `https://<proxy-id>.ngrok-free.app`.
To route backend → proxy via ngrok (optional):

```powershell
$env:LLM_MONITOR_DAN_URL="https://<proxy-id>.ngrok-free.app"
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_live.ps1
```

## 5) Performance Share (after 60–120 min)
Provide the following for review:
- `/monitoring/auto-trader/status` JSON
- Last ~100 lines from `logs/production.log` and `logs/backend.log`
- Printed ngrok URLs (if used)
- Any MT5 execution or HTTP errors

## 6) Quick Tuning Notes
- Regime-aware thresholds: scale with ATR + HMM regime
- Multi-timeframe confluence: boost only when aligned; cap positive bias
- Liquidity gating: session windows + spread/slippage ceilings
- Execution-aware validation: penalize repeatedly adverse signal families
- Online calibration: EWMA win-rate/profit factor guardrails; small bounded steps

---
INTENTIONAL_DEFAULT_TAG (2025-08-19): Live MT5 and execution defaults are enabled by design in `backend/core/config.py`. Override via env when needed.
