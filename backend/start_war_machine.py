#!/usr/bin/env python
"""
ARIA WAR MACHINE LAUNCHER
Start the profit engine
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_redis():
    """Check if Redis is running"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("[OK] Redis is running")
        return True
    except:
        print("[X] Redis not running - starting Redis...")
        # Try to start Redis using Docker
        try:
            subprocess.run(["docker", "run", "-d", "--name", "aria_redis", 
                          "-p", "6379:6379", "redis:7-alpine"], 
                          capture_output=True)
            time.sleep(2)
            return True
        except:
            print("Warning: Redis not available. Cache disabled.")
            return False

def check_mt5():
    """Check MT5 connection"""
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            info = mt5.account_info()
            if info:
                print(f"[OK] MT5 connected: Balance={info.balance}")
                mt5.shutdown()
                return True
        print("[X] MT5 not connected - will retry on startup")
        return False
    except:
        print("[X] MetaTrader5 not installed")
        return False

def main():
    print("""
    ========================================
          ARIA WAR MACHINE vX              
       Institutional AI Trading System     
    ========================================
    """)
    
    # Set environment
    os.environ["ARIA_HOST"] = "0.0.0.0"
    os.environ["ARIA_PORT"] = "8100"
    
    # Load .env file if exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"[OK] Loaded configuration from {env_file}")
    
    # System checks
    print("\n[SYSTEM CHECKS]")
    check_redis()
    check_mt5()
    
    print("\n[STARTING WAR MACHINE]")
    print(f"Host: {os.environ.get('ARIA_HOST', '0.0.0.0')}")
    print(f"Port: {os.environ.get('ARIA_PORT', '8100')}")
    print("\nEndpoints:")
    print("  POST /signal         - Generate trading signal")
    print("  POST /execute        - Execute trade")
    print("  POST /auto_trade     - Auto trade with signal")
    print("  GET  /positions      - Get open positions")
    print("  GET  /pnl           - Get P&L")
    print("  POST /close_all     - Emergency close all")
    print("  GET  /health        - Health check")
    print("\n" + "="*40 + "\n")
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "main_cpu:app",
            host=os.environ.get("ARIA_HOST", "0.0.0.0"),
            port=int(os.environ.get("ARIA_PORT", 8100)),
            log_level="info",
            reload=False,
            workers=1,  # Single worker for institutional trading consistency
            access_log=True,
            use_colors=True,
            server_header=False,  # Security: hide server info
            date_header=False,    # Security: hide date header
            timeout_keep_alive=30,
            timeout_graceful_shutdown=10,
            limit_concurrency=1000,
            limit_max_requests=10000,
            backlog=2048
        )
    except Exception as e:
        print(f"Failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
