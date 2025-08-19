#!/usr/bin/env python3
"""
Test Real SMC Functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_smc_idea_preparation():
    """Test SMC idea preparation with real data"""
    print("🔍 Testing SMC Idea Preparation with Real Data")
    print("=" * 50)

    # Test data with high volume to trigger SMC patterns
    test_data = {
        "symbol": "EURUSD",
        "bar": {
            "ts": int(time.time()),
            "o": 1.1000,
            "h": 1.1020,
            "l": 1.0990,
            "c": 1.1015,
            "v": 2000,  # High volume to trigger order block
            "symbol": "EURUSD",
        },
        "recent_ticks": [
            {"price": 1.1015, "size": 200, "side": "buy"},
            {"price": 1.1010, "size": 150, "side": "sell"},
        ],
    }

    try:
        response = requests.post(f"{BASE_URL}/api/smc/idea/prepare", json=test_data)
        result = response.json()

        print(f"Response Status: {response.status_code}")
        print(f"Response: {json.dumps(result, indent=2)}")

        if result.get("ok"):
            idea = result.get("idea")
            if idea:
                print(f"\n✅ SMC Idea Generated Successfully!")
                print(f"   Symbol: {idea.get('symbol')}")
                print(f"   Bias: {idea.get('bias')}")
                print(f"   Confidence: {idea.get('confidence', 0):.2f}")
                print(f"   Entry: {idea.get('entry')}")
                print(f"   Stop: {idea.get('stop')}")
                print(f"   Take Profit: {idea.get('takeprofit')}")

                # Check for SMC structures
                if idea.get("order_blocks"):
                    print(f"   Order Blocks: {len(idea['order_blocks'])}")
                if idea.get("fair_value_gaps"):
                    print(f"   Fair Value Gaps: {len(idea['fair_value_gaps'])}")
                if idea.get("liquidity_zones"):
                    print(f"   Liquidity Zones: {len(idea['liquidity_zones'])}")
            else:
                print("❌ No idea generated")
        else:
            print(f"❌ Error: {result.get('msg', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Exception: {e}")


def test_smc_signals():
    """Test SMC signals endpoint"""
    print("\n🔍 Testing SMC Signals")
    print("-" * 30)

    try:
        response = requests.get(f"{BASE_URL}/api/smc/signals/EURUSD")
        result = response.json()

        print(f"Response Status: {response.status_code}")
        print(f"Signals found: {len(result.get('signals', []))}")

        for i, signal in enumerate(result.get("signals", [])[:3]):
            print(
                f"   Signal {i+1}: {signal.get('type', 'unknown')} - Confidence: {signal.get('confidence', 0):.2f}"
            )

    except Exception as e:
        print(f"❌ Exception: {e}")


def test_order_blocks():
    """Test order blocks endpoint"""
    print("\n🔍 Testing Order Blocks")
    print("-" * 30)

    try:
        response = requests.get(f"{BASE_URL}/api/smc/order-blocks/EURUSD")
        result = response.json()

        print(f"Response Status: {response.status_code}")
        print(f"Order blocks found: {len(result.get('order_blocks', []))}")

        for i, block in enumerate(result.get("order_blocks", [])[:3]):
            print(
                f"   Block {i+1}: {block.get('type', 'unknown')} - Strength: {block.get('strength', 0):.2f}"
            )

    except Exception as e:
        print(f"❌ Exception: {e}")


def test_fair_value_gaps():
    """Test fair value gaps endpoint"""
    print("\n🔍 Testing Fair Value Gaps")
    print("-" * 30)

    try:
        response = requests.get(f"{BASE_URL}/api/smc/fair-value-gaps/EURUSD")
        result = response.json()

        print(f"Response Status: {response.status_code}")
        print(f"Fair value gaps found: {len(result.get('fair_value_gaps', []))}")

        for i, gap in enumerate(result.get("fair_value_gaps", [])[:3]):
            print(
                f"   Gap {i+1}: {gap.get('type', 'unknown')} - Strength: {gap.get('strength', 0):.2f}"
            )

    except Exception as e:
        print(f"❌ Exception: {e}")


def test_smc_with_multiple_bars():
    """Test SMC with multiple bars to build up analysis"""
    print("\n🔍 Testing SMC with Multiple Bars")
    print("-" * 40)

    # Send multiple bars to build up SMC analysis
    bars = [
        {
            "ts": int(time.time()) - 300,
            "o": 1.1000,
            "h": 1.1005,
            "l": 1.0995,
            "c": 1.1002,
            "v": 500,
            "symbol": "EURUSD",
        },
        {
            "ts": int(time.time()) - 240,
            "o": 1.1002,
            "h": 1.1008,
            "l": 1.1000,
            "c": 1.1006,
            "v": 600,
            "symbol": "EURUSD",
        },
        {
            "ts": int(time.time()) - 180,
            "o": 1.1006,
            "h": 1.1012,
            "l": 1.1004,
            "c": 1.1010,
            "v": 800,
            "symbol": "EURUSD",
        },
        {
            "ts": int(time.time()) - 120,
            "o": 1.1010,
            "h": 1.1015,
            "l": 1.1008,
            "c": 1.1013,
            "v": 1200,
            "symbol": "EURUSD",
        },
        {
            "ts": int(time.time()) - 60,
            "o": 1.1013,
            "h": 1.1020,
            "l": 1.1010,
            "c": 1.1015,
            "v": 1500,
            "symbol": "EURUSD",
        },
        {
            "ts": int(time.time()),
            "o": 1.1015,
            "h": 1.1025,
            "l": 1.1005,
            "c": 1.1008,
            "v": 2000,
            "symbol": "EURUSD",
        },  # High volume trap
    ]

    for i, bar in enumerate(bars):
        test_data = {
            "symbol": "EURUSD",
            "bar": bar,
            "recent_ticks": [
                {
                    "price": bar["c"],
                    "size": bar["v"] // 10,
                    "side": "buy" if bar["c"] > bar["o"] else "sell",
                }
            ],
        }

        try:
            response = requests.post(f"{BASE_URL}/api/smc/idea/prepare", json=test_data)
            result = response.json()

            if result.get("ok") and result.get("idea"):
                idea = result["idea"]
                print(
                    f"   Bar {i+1}: {idea.get('bias', 'none')} - Confidence: {idea.get('confidence', 0):.2f}"
                )
            else:
                print(f"   Bar {i+1}: No signal")

        except Exception as e:
            print(f"   Bar {i+1}: Error - {e}")


if __name__ == "__main__":
    test_smc_idea_preparation()
    test_smc_signals()
    test_order_blocks()
    test_fair_value_gaps()
    test_smc_with_multiple_bars()

    print("\n" + "=" * 50)
    print("🎉 SMC Real Test Complete!")
