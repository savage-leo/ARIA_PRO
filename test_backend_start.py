#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Set up environment
project_root = Path.cwd().absolute()
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root)

# Load environment
try:
    from dotenv import load_dotenv
    env_file = project_root / "production.env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[OK] Loaded {env_file.name}")
except ImportError:
    print("[WARN] python-dotenv not available")

# Essential defaults
os.environ.setdefault('ARIA_LOG_LEVEL', 'INFO')
os.environ.setdefault('ARIA_ENV', 'production')

print("=" * 50)
print("ARIA PRO BACKEND TEST START")
print("=" * 50)
print(f"Project root: {project_root}")
print(f"MT5 enabled: {os.environ.get('ARIA_ENABLE_MT5', 'false')}")
print(f"Auto trading: {os.environ.get('AUTO_TRADE_ENABLED', 'false')}")
print(f"Dry run: {os.environ.get('AUTO_TRADE_DRY_RUN', 'true')}")
print("=" * 50)

try:
    import uvicorn
    from backend.main import app
    
    print("[OK] Backend app imported successfully")
    print("[INFO] Starting uvicorn server on 127.0.0.1:8100")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8100,
        reload=False,
        log_level="info",
        access_log=True
    )
    
except Exception as e:
    print(f"[ERROR] Startup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
