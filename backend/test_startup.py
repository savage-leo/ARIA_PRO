#!/usr/bin/env python3
"""Test backend startup and identify any issues"""

import sys
import os
sys.path.insert(0, r'c:\savage\ARIA_PRO')
os.environ['PYTHONPATH'] = r'c:\savage\ARIA_PRO'

# Test all imports
errors = []
warnings = []

try:
    from backend.main import app
    print("✓ FastAPI app imported")
except Exception as e:
    errors.append(f"Main app: {e}")

try:
    from backend.services.mt5_market_data import get_mt5_service
    print("✓ MT5 service imported")
except Exception as e:
    warnings.append(f"MT5 service: {e}")

try:
    from backend.services.auto_trader import get_auto_trader
    print("✓ Auto trader imported")
except Exception as e:
    warnings.append(f"Auto trader: {e}")

try:
    from backend.services.real_ai_signal_generator import get_signal_generator
    print("✓ Signal generator imported")
except Exception as e:
    warnings.append(f"Signal generator: {e}")

try:
    from backend.core.performance_monitor import get_performance_monitor
    print("✓ Performance monitor imported")
except Exception as e:
    warnings.append(f"Performance monitor: {e}")

try:
    from backend.services.data_source_manager import get_data_source_manager
    print("✓ Data source manager imported")
except Exception as e:
    warnings.append(f"Data source manager: {e}")

try:
    from backend.services.ws_broadcaster import get_broadcaster
    print("✓ WebSocket broadcaster imported")
except Exception as e:
    warnings.append(f"WebSocket broadcaster: {e}")

# Test routes
routes = [
    'backend.routes.account',
    'backend.routes.market',
    'backend.routes.signals',
    'backend.routes.monitoring',
    'backend.routes.websocket',
    'backend.routes.trade_memory_api',
    'backend.routes.positions'
]

for route in routes:
    try:
        __import__(route)
        print(f"✓ {route} imported")
    except Exception as e:
        warnings.append(f"{route}: {e}")

print("\n" + "="*50)
if errors:
    print("CRITICAL ERRORS:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
elif warnings:
    print("WARNINGS (non-critical):")
    for w in warnings:
        print(f"  ⚠ {w}")
    print("\nBackend can start with limited functionality")
else:
    print("ALL COMPONENTS OK!")
    
print("\nStarting backend server...")
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8000)
