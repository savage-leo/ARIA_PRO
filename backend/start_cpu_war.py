#!/usr/bin/env python
# backend/start_cpu_war.py
"""
Simple startup script for CPU-friendly ARIA War Machine
"""
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def check_requirements():
    # Check required modules
    required_modules = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'websockets': 'websockets'
    }
    missing = []
    
    for import_name, package_name in required_modules.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"Missing required modules: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def main():
    """Main entry point"""
    print("=" * 50)
    print("ARIA CPU WAR MACHINE - INSTITUTIONAL FX TRADING")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check MT5
    try:
        import MetaTrader5 as mt5
        print("[OK] MetaTrader5 module found")
    except ImportError:
        print("[WARNING] MetaTrader5 not installed - will run in simulation mode")
    
    # Run preflight checks
    try:
        from tools.loader_guard import enforce
        enforce()
    except SystemExit:
        print("[WARNING] Loader guard failed - continuing anyway in dev mode")
    except Exception as e:
        print(f"[WARNING] Loader guard error: {e}")
    
    # Start the server
    print("\n[STARTING WAR MACHINE]")
    print("Host: 0.0.0.0")
    print("Port: 8100")
    print("\nEndpoints:")
    print("  POST /signal         - Generate trading signal")
    print("  POST /execute        - Execute trade")
    print("  POST /auto_trade     - Auto trade with signal")
    print("  GET  /positions      - Get open positions")
    print("  GET  /pnl           - Get P&L")
    print("  POST /close_all     - Emergency close all")
    print("  GET  /health        - Health check")
    print("  WS   /ws            - WebSocket for real-time data")
    print("\n" + "=" * 50 + "\n")
    
    # Import and run
    os.chdir(Path(__file__).parent)
    import uvicorn
    uvicorn.run(
        "main_cpu:app",
        host="0.0.0.0",
        port=8100,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
