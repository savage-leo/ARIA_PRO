#!/usr/bin/env python3
"""
Test script for SMC endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_smc_prepare():
    """Test SMC idea preparation"""
    print("Testing SMC Idea Preparation...")

    data = {
        "symbol": "EURUSD",
        "bar": {
            "ts": int(time.time()),
            "o": 1.1200,
            "h": 1.1220,
            "l": 1.1190,
            "c": 1.1210,
            "v": 120,
        },
    }

    try:
        response = requests.post(f"{BASE_URL}/api/smc/idea/prepare", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_smc_history():
    """Test SMC history"""
    print("\nTesting SMC History...")

    try:
        response = requests.get(f"{BASE_URL}/api/smc/history")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def test_smc_execute(payload):
    """Test SMC execution (dry run)"""
    print("\nTesting SMC Execution (Dry Run)...")

    if not payload:
        print("No payload to execute")
        return

    # Add dry_run flag
    payload["dry_run"] = True

    try:
        response = requests.post(
            f"{BASE_URL}/api/smc/idea/execute",
            json=payload,
            headers={"X-ADMIN-KEY": "changeme"},
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def test_all_endpoints():
    """Test all endpoints"""
    print("ARIA_PRO SMC Endpoint Tests")
    print("=" * 40)

    try:
        # Test basic endpoints
        test_smc_history()

        # Test idea preparation
        result = test_smc_prepare()

        # Test execution if we have a payload
        if result and result.get("ok") and result.get("prepared_payload"):
            test_smc_execute(result["prepared_payload"])

        print("\n✅ All SMC tests completed!")

    except requests.exceptions.ConnectionError:
        print(
            "❌ Connection error: Make sure the backend is running on http://localhost:8000"
        )
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_all_endpoints()
