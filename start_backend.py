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

# Load production.env if it exists
try:
    from dotenv import load_dotenv
    env_file = project_root / "production.env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[OK] Loaded production environment from {env_file.name}")
except ImportError:
    print("[WARN] python-dotenv not available, using environment variables only")

# Set essential environment variables if not set (safe defaults)
defaults = {
    'ARIA_LOG_LEVEL': 'INFO',
    'ARIA_ENV': 'production',
}

for key, value in defaults.items():
    if key not in os.environ:
        os.environ[key] = value
        print(f"Set {key}={value}")

try:
    import uvicorn
    from backend.main import app
    
    print("=" * 60)
    print("ARIA PRO BACKEND STARTING")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")
    print(f"MT5 enabled: {os.environ.get('ARIA_ENABLE_MT5', 'false')}")
    print(f"Auto trading: {os.environ.get('AUTO_TRADE_ENABLED', 'false')}")
    print(f"Environment: {os.environ.get('ARIA_ENV', 'production')}")
    print("=" * 60)
    
    # Run the server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8100,
        reload=False,  # Set to True for development
        log_level="info",
        access_log=True
    )
    
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  ..\\..\\..venv\\Scripts\\pip install -r backend\\requirements_complete.txt")
    sys.exit(1)
except Exception as e:
    print(f"Startup error: {e}")
    sys.exit(1)
