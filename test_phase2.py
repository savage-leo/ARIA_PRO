#!/usr/bin/env python3
"""
Test Script for ARIA PRO Phase 2 Telemetry Implementation
Tests Prometheus integration
"""

import time
import requests
import random

API_BASE_URL = "http://localhost:8100"
TELEMETRY_BASE = f"{API_BASE_URL}/telemetry"

def test_phase2():
    print("🔍 Testing ARIA PRO Phase 2 Telemetry")
    print("=" * 50)
    
    # Test Prometheus availability
    try:
        response = requests.get(f"{TELEMETRY_BASE}/dashboard", timeout=5)
        if response.status_code == 200:
            data = response.json()
            prometheus_info = data['data']['prometheus']
            print(f"✅ Prometheus Available: {prometheus_info['available']}")
        else:
            print(f"❌ Dashboard failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    
    # Test Prometheus metrics endpoint
    try:
        response = requests.get(f"{TELEMETRY_BASE}/prometheus", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            print(f"✅ Prometheus metrics retrieved")
            print(f"   Length: {len(metrics)} characters")
            
            # Check for key metrics
            key_metrics = ['aria_execution_latency_seconds', 'aria_pnl_dollars', 'aria_errors_total']
            found = sum(1 for metric in key_metrics if metric in metrics)
            print(f"   Found {found}/{len(key_metrics)} key metrics")
            
        elif response.status_code == 503:
            print("❌ Prometheus not available (503)")
        else:
            print(f"❌ Prometheus failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prometheus error: {e}")
    
    # Test direct Prometheus server
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus server running on port 8000")
        else:
            print(f"❌ Prometheus server: {response.status_code}")
    except Exception as e:
        print(f"❌ Prometheus server error: {e}")
    
    print("\n✅ Phase 2 Test Complete!")

if __name__ == "__main__":
    test_phase2()

