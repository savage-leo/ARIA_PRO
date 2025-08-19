#!/usr/bin/env bash
set -e
. .venv/bin/activate

echo "Running acceptance tests (dry-run order)..."
python3 - <<'PY'
import requests, os, json, time
BASE=os.getenv('VITE_BACKEND_BASE','http://localhost:8000')
print("Backend base:", BASE)
# Health
h = requests.get(f"{BASE}/health").json()
print("Health:", h)
# dry-run order
payload = {
  "symbol": "EURUSD",
  "side": "buy",
  "sl": 1.0000,
  "tp": 1.1000,
  "risk_percent": 0.5,
  "dry_run": True,
  "comment": "acceptance-test"
}
r = requests.post(f"{BASE}/api/orders/place", json=payload)
print("Order response status:", r.status_code)
try:
  print("Order response:", r.json())
except Exception:
  print("Raw response:", r.text)
PY
