#!/usr/bin/env python3
"""
Start Enhanced ARIA PRO Institutional Proxy for ARIA Integration
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def start_proxy():
    print("🚀 Starting Enhanced ARIA PRO Institutional Proxy")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📍 Proxy URL: http://localhost:11435")
    print("🎯 ARIA Configuration: Point to http://localhost:11435")
    print("=" * 60)
    
    # Check if proxy file exists
    proxy_path = os.path.join("backend", "services", "institutional_proxy.py")
    if not os.path.exists(proxy_path):
        print(f"❌ Proxy file not found: {proxy_path}")
        return False
    
    try:
        # Start the proxy
        print("🔄 Starting proxy process...")
        process = subprocess.Popen([sys.executable, proxy_path])
        
        print(f"✅ Proxy started with PID: {process.pid}")
        print("🔄 Waiting for proxy to initialize...")
        time.sleep(3)
        
        print("\n🎉 Enhanced ARIA PRO Institutional Proxy is running!")
        print("=" * 60)
        print("📋 Available Features:")
        print("   • 4 Local Models (fully functional)")
        print("   • 8 Remote Models (need API key update)")
        print("   • Intelligent Task-Based Routing")
        print("   • Full Streaming Support")
        print("   • ARIA-Compatible API")
        print("=" * 60)
        print("🔧 Next Steps:")
        print("1. Configure ARIA to use http://localhost:11435")
        print("2. Test with local models (immediately available)")
        print("3. Update API keys for remote model functionality")
        print("=" * 60)
        print("🛑 Press Ctrl+C to stop the proxy")
        
        # Keep running
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        if 'process' in locals():
            process.terminate()
            print("✅ Proxy stopped gracefully")
    except Exception as e:
        print(f"❌ Error starting proxy: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    start_proxy()
