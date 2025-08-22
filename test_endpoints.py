#!/usr/bin/env python3
"""Quick endpoint testing script for ARIA backend"""

import requests
import json
import sys
import time

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method, endpoint, data=None):
    """Test a single endpoint and return results"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"\n{method} {endpoint}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"\n{method} {endpoint}")
        print(f"❌ Error: {e}")
        return False

def main():
    print("=== ARIA Backend Endpoint Testing ===")
    
    # Test 1: Health check
    print("\n1. Testing Health Endpoint...")
    test_endpoint("GET", "/health")
    
    # Test 2: Auto-trading status
    print("\n2. Testing Auto-Trading Status...")
    test_endpoint("GET", "/api/institutional-ai/auto-trading/status")
    
    # Test 3: Toggle auto-trading ON
    print("\n3. Toggling Auto-Trading ON...")
    test_endpoint("POST", "/api/institutional-ai/auto-trading/toggle", {"enabled": True})
    
    time.sleep(1)  # Brief pause
    
    # Test 4: Check status after ON
    print("\n4. Checking Status After ON...")
    test_endpoint("GET", "/api/institutional-ai/auto-trading/status")
    
    # Test 5: Toggle auto-trading OFF
    print("\n5. Toggling Auto-Trading OFF...")
    test_endpoint("POST", "/api/institutional-ai/auto-trading/toggle", {"enabled": False})
    
    time.sleep(1)  # Brief pause
    
    # Test 6: Check status after OFF
    print("\n6. Checking Status After OFF...")
    test_endpoint("GET", "/api/institutional-ai/auto-trading/status")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main()
