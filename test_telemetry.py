#!/usr/bin/env python3
"""
Test Script for ARIA PRO Phase 1 Telemetry Implementation
"""

import time
import requests
import random

API_BASE_URL = "http://localhost:8100"
TELEMETRY_BASE = f"{API_BASE_URL}/telemetry"

def test_telemetry():
    print("🧪 Testing ARIA PRO Phase 1 Telemetry")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{TELEMETRY_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data['data']['status']}")
        else:
            print(f"❌ Health failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health error: {e}")
    
    # Test performance metrics
    try:
        response = requests.get(f"{TELEMETRY_BASE}/performance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            perf = data['data']
            print(f"✅ Latency P95: {perf['execution_latency']['p95_ms']:.2f}ms")
            print(f"✅ MT5 Connected: {perf['mt5_connection']['healthy']}")
        else:
            print(f"❌ Performance failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Performance error: {e}")
    
    # Test business metrics
    try:
        response = requests.get(f"{TELEMETRY_BASE}/business", timeout=5)
        if response.status_code == 200:
            data = response.json()
            business = data['data']
            print(f"✅ P&L: ${business['pnl']['real_time']:.2f}")
            print(f"✅ Trades: {business['trades']['total']}")
        else:
            print(f"❌ Business failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Business error: {e}")
    
    # Simulate trading activity
    print("\n🎯 Simulating trading activity...")
    for i in range(3):
        execution_data = {
            "start_time": time.time() - random.uniform(0.01, 0.1),
            "end_time": time.time(),
            "expected_price": random.uniform(1.1000, 1.2000),
            "actual_price": random.uniform(1.1000, 1.2000),
            "trade_data": {
                "symbol": "EURUSD",
                "action": "BUY",
                "volume": 0.1,
                "pnl": random.uniform(-10, 20),
                "confidence": 0.8
            }
        }
        
        try:
            response = requests.post(f"{TELEMETRY_BASE}/track-execution", json=execution_data, timeout=5)
            if response.status_code == 200:
                print(f"✅ Execution {i+1} tracked")
            else:
                print(f"❌ Execution {i+1} failed")
        except Exception as e:
            print(f"❌ Execution {i+1} error: {e}")
        
        time.sleep(1)
    
    print("\n✅ Phase 1 Telemetry Test Complete!")

if __name__ == "__main__":
    test_telemetry()

