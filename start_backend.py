#!/usr/bin/env python3
"""
ARIA PRO Backend Startup Script
Production-ready launcher with comprehensive error handling
"""

import sys
import os
from pathlib import Path

# Ensure proper Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root)

# Load environment file based on ARIA_ENV
try:
    from dotenv import load_dotenv
    # Peek current ARIA_ENV (if caller already set it, we respect it)
    pre_env = (os.environ.get("ARIA_ENV", "") or "").strip().lower()
    dev_like = pre_env in ("dev", "development", "local", "test", "testing", "ci")
    candidates = []
    if dev_like:
        candidates = [project_root / ".env"]
    else:
        # In production, prefer production.env, then .env to fill missing values
        candidates = [project_root / "production.env", project_root / ".env"]
    for path in candidates:
        if path.exists():
            # Do not override existing OS environment variables
            load_dotenv(path, override=False)
            print(f"[OK] Loaded environment from {path.name}")
except ImportError:
    print("[WARN] python-dotenv not available, using environment variables only")

# Set essential environment variables if not set (safe defaults)
defaults = {
    'LOG_LEVEL': 'INFO',
    'ARIA_ENV': 'development' if '--dev' in sys.argv else 'production',
    'ARIA_ENABLE_MT5': '0' if '--dev' in sys.argv else '1',
    'AUTO_TRADE_ENABLED': '0' if '--dev' in sys.argv else '1'
}

for key, value in defaults.items():
    if key not in os.environ:
        os.environ[key] = value
        print(f"Set {key}={value}")

try:
    import uvicorn
    from backend.main_cpu import app
    
    print("=" * 60)
    print("ARIA PRO BACKEND STARTING")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")
    print(f"Environment: {os.environ.get('ARIA_ENV', 'production')} | LOG_LEVEL={os.environ.get('LOG_LEVEL','INFO')}")
    print(f"MT5 enabled: {os.environ.get('ARIA_ENABLE_MT5', '0')} | Auto trading: {os.environ.get('AUTO_TRADE_ENABLED', '0')}")
    print("=" * 60)
    
    # Run the server with institutional-grade uvicorn configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        reload=False,  # Set to True for development
        log_level="info",
        access_log=True,
        workers=1,  # Single worker for institutional trading consistency
        use_colors=True,
        server_header=False,  # Security: hide server info
        date_header=False,    # Security: hide date header
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
        limit_concurrency=1000,
        limit_max_requests=10000,
        backlog=2048
    )
    
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  ..\\..\\..venv\\Scripts\\pip install -r backend\\requirements_complete.txt")
    sys.exit(1)
except Exception as e:
    print(f"Startup error: {e}")
    sys.exit(1)
