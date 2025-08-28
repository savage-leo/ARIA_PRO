#!/usr/bin/env python3
"""
Test CPU Module API Endpoints
"""
import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://localhost:8100"

def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)
    return response.status_code == 200

def test_risk_status():
    """Test risk manager status"""
    print("Testing /api/cpu/risk/status...")
    response = requests.get(f"{BASE_URL}/api/cpu/risk/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)
    return response.status_code == 200

def test_signal_status():
    """Test signal processor status"""
    print("Testing /api/cpu/signal/status...")
    response = requests.get(f"{BASE_URL}/api/cpu/signal/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)
    return response.status_code == 200

def test_orchestrator_status():
    """Test orchestrator status"""
    print("Testing /api/cpu/orchestrator/status...")
    response = requests.get(f"{BASE_URL}/api/cpu/orchestrator/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)
    return response.status_code == 200

def test_signal_generation():
    """Test signal generation"""
    print("Testing /api/cpu/signal/generate...")
    data = {
        "symbol": "EURUSD",
        "timeframe": "M5",
        "lookback": 100
    }
    response = requests.post(
        f"{BASE_URL}/api/cpu/signal/generate",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)
    return response.status_code == 200

def test_orchestrator_signal():
    """Test orchestrator signal generation"""
    print("Testing /api/cpu/orchestrator/signal...")
    data = {
        "symbol": "EURUSD",
        "timeframe": "M5"
    }
    response = requests.post(
        f"{BASE_URL}/api/cpu/orchestrator/signal",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)
    return response.status_code == 200

def test_risk_check():
    """Test risk checking for a trade"""
    print("Testing /api/cpu/risk/check...")
    data = {
        "symbol": "EURUSD",
        "side": "buy",
        "entry_price": 1.1000,
        "stop_loss": 1.0950,
        "account_balance": 10000
    }
    response = requests.post(
        f"{BASE_URL}/api/cpu/risk/check",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 50)
    print("ARIA CPU MODULE API ENDPOINT TESTS")
    print(f"Testing at: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 50)
    print()
    
    tests = [
        ("Health Check", test_health),
        ("Risk Manager Status", test_risk_status),
        ("Signal Processor Status", test_signal_status),
        ("Orchestrator Status", test_orchestrator_status),
        ("Risk Check", test_risk_check),
        ("Signal Generation", test_signal_generation),
        ("Orchestrator Signal", test_orchestrator_signal)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append((name, False))
        print()
    
    # Print summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} - {name}")
    
    print()
    print(f"Results: {passed}/{total} passed")
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
