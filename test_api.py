#!/usr/bin/env python3
"""
Test script for ARIA_PRO API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_cpp_status():
    """Test C++ integration status"""
    print("Testing C++ Status...")
    response = requests.get(f"{BASE_URL}/api/smc/cpp/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_tick_processing():
    """Test tick processing endpoint"""
    print("Testing Tick Processing...")
    data = {
        "symbol": "EURUSD",
        "bid": 1.1000,
        "ask": 1.1001,
        "volume": 1000,
        "timestamp": int(time.time() * 1000),
    }
    response = requests.post(f"{BASE_URL}/api/smc/process/tick", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_bar_processing():
    """Test bar processing endpoint"""
    print("Testing Bar Processing...")
    data = {
        "symbol": "EURUSD",
        "open": 1.1000,
        "high": 1.1010,
        "low": 1.0990,
        "close": 1.1005,
        "volume": 1000,
        "timestamp": int(time.time() * 1000),
    }
    response = requests.post(f"{BASE_URL}/api/smc/process/bar", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_smc_signals():
    """Test SMC signals endpoint"""
    print("Testing SMC Signals...")
    response = requests.get(f"{BASE_URL}/api/smc/signals/EURUSD")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_order_blocks():
    """Test order blocks endpoint"""
    print("Testing Order Blocks...")
    response = requests.get(f"{BASE_URL}/api/smc/order-blocks/EURUSD")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_fair_value_gaps():
    """Test fair value gaps endpoint"""
    print("Testing Fair Value Gaps...")
    response = requests.get(f"{BASE_URL}/api/smc/fair-value-gaps/EURUSD")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def main():
    """Run all tests"""
    print("ARIA_PRO API Integration Tests")
    print("=" * 40)

    try:
        test_cpp_status()
        test_tick_processing()
        test_bar_processing()
        test_smc_signals()
        test_order_blocks()
        test_fair_value_gaps()

        print("✅ All tests completed successfully!")

    except requests.exceptions.ConnectionError:
        print(
            "❌ Connection error: Make sure the backend is running on http://localhost:8000"
        )
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
