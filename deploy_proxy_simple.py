#!/usr/bin/env python3
"""
Simple Enhanced ARIA PRO Proxy Deployment
"""

import subprocess
import time
import json
import os
from datetime import datetime

def deploy_proxy():
    print("🚀 Starting Enhanced ARIA PRO Institutional Proxy Deployment")
    print("=" * 60)
    
    # Create log file
    log_file = f"proxy_deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    logs = []
    
    def log_event(event_type, message, data=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "message": message,
            "data": data or {}
        }
        logs.append(entry)
        print(f"[{entry['timestamp']}] {event_type.upper()}: {message}")
    
    try:
        # Start proxy
        log_event("deployment", "Starting proxy process")
        proxy_path = os.path.join("backend", "services", "institutional_proxy.py")
        
        process = subprocess.Popen(
            ["python", proxy_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        log_event("deployment", f"Proxy started with PID: {process.pid}")
        
        # Wait for startup
        time.sleep(5)
        
        # Test health
        import httpx
        import asyncio
        
        async def test_health():
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Health check
                    resp = await client.get("http://localhost:11435/healthz")
                    log_event("health", f"Health endpoint: {resp.status_code}")
                    
                    # Model listing
                    resp = await client.get("http://localhost:11435/api/tags")
                    if resp.status_code == 200:
                        models = resp.json().get("models", [])
                        log_event("health", f"Model listing: {len(models)} models available")
                    
                    # Test local model
                    payload = {
                        "model": "qwen2.5-coder:1.5b-base",
                        "prompt": "Hello",
                        "stream": False
                    }
                    resp = await client.post("http://localhost:11435/api/generate", json=payload)
                    log_event("test", f"Local model test: {resp.status_code}")
                    
            except Exception as e:
                log_event("error", f"Health test failed: {str(e)}")
        
        asyncio.run(test_health())
        
        # Save logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"\n📄 Logs saved to: {log_file}")
        print("✅ Proxy deployment complete!")
        print("🚀 Ready for ARIA integration at http://localhost:11435")
        
        # Keep running
        print("\n🔄 Proxy is running. Press Ctrl+C to stop...")
        process.wait()
        
    except KeyboardInterrupt:
        log_event("deployment", "Shutdown requested")
        if 'process' in locals():
            process.terminate()
        print("🛑 Proxy stopped")
    except Exception as e:
        log_event("error", f"Deployment failed: {str(e)}")
    finally:
        # Save final logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

if __name__ == "__main__":
    deploy_proxy()
