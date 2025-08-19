#!/usr/bin/env python3
"""
SMC Analysis Features Test Script
Demonstrates the Smart Money Concepts analysis capabilities
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_smc_endpoints():
    """Test all SMC analysis endpoints"""

    print("🔍 ARIA_PRO SMC Analysis Features Test")
    print("=" * 50)

    # 1. Check C++ Integration Status
    print("\n1️⃣ C++ Integration Status:")
    try:
        response = requests.get(f"{BASE_URL}/api/smc/cpp/status")
        status = response.json()
        print(f"   C++ Available: {status.get('cpp_available', False)}")
        print(f"   Market Processor: {status.get('market_processor', False)}")
        print(f"   SMC Engine: {status.get('smc_engine', False)}")
    except Exception as e:
        print(f"   Error: {e}")

    # 2. Test SMC Signals
    print("\n2️⃣ SMC Trading Signals:")
    try:
        response = requests.get(f"{BASE_URL}/api/smc/signals/EURUSD")
        signals = response.json()
        print(f"   Signals found: {len(signals.get('signals', []))}")
        if signals.get("signals"):
            for signal in signals["signals"][:3]:  # Show first 3
                print(f"   - {signal}")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Test Order Blocks
    print("\n3️⃣ Order Blocks Detection:")
    try:
        response = requests.get(f"{BASE_URL}/api/smc/order-blocks/EURUSD")
        blocks = response.json()
        print(f"   Order blocks found: {len(blocks.get('order_blocks', []))}")
        if blocks.get("order_blocks"):
            for block in blocks["order_blocks"][:3]:  # Show first 3
                print(f"   - {block}")
    except Exception as e:
        print(f"   Error: {e}")

    # 4. Test Fair Value Gaps
    print("\n4️⃣ Fair Value Gaps:")
    try:
        response = requests.get(f"{BASE_URL}/api/smc/fair-value-gaps/EURUSD")
        gaps = response.json()
        print(f"   Fair value gaps found: {len(gaps.get('fair_value_gaps', []))}")
        if gaps.get("fair_value_gaps"):
            for gap in gaps["fair_value_gaps"][:3]:  # Show first 3
                print(f"   - {gap}")
    except Exception as e:
        print(f"   Error: {e}")

    # 5. Test Trade History
    print("\n5️⃣ Trade History:")
    try:
        response = requests.get(f"{BASE_URL}/api/smc/history?limit=5")
        history = response.json()
        print(f"   History entries: {len(history.get('history', []))}")
        if history.get("history"):
            for entry in history["history"][:3]:  # Show first 3
                print(
                    f"   - {entry.get('symbol', 'N/A')} {entry.get('bias', 'N/A')} conf={entry.get('confidence', 0):.2f}"
                )
    except Exception as e:
        print(f"   Error: {e}")

    # 6. Test Idea Preparation
    print("\n6️⃣ SMC Idea Preparation:")
    try:
        # Sample bar data
        bar_data = {
            "symbol": "EURUSD",
            "bar": {
                "ts": int(time.time()),
                "o": 1.1000,
                "h": 1.1010,
                "l": 1.0990,
                "c": 1.1005,
                "v": 1000,
                "symbol": "EURUSD",
            },
            "recent_ticks": [
                {"price": 1.1005, "size": 100, "side": "buy"},
                {"price": 1.1003, "size": 50, "side": "sell"},
            ],
        }

        response = requests.post(f"{BASE_URL}/api/smc/idea/prepare", json=bar_data)
        idea = response.json()
        print(f"   Idea generated: {idea.get('ok', False)}")
        if idea.get("idea"):
            idea_data = idea["idea"]
            print(f"   - Symbol: {idea_data.get('symbol')}")
            print(f"   - Bias: {idea_data.get('bias')}")
            print(f"   - Confidence: {idea_data.get('confidence', 0):.2f}")
            print(f"   - Entry: {idea_data.get('entry')}")
            print(f"   - Stop: {idea_data.get('stop')}")
            print(f"   - Take Profit: {idea_data.get('takeprofit')}")
    except Exception as e:
        print(f"   Error: {e}")

    # 7. Test Current Signal
    print("\n7️⃣ Current Signal:")
    try:
        response = requests.get(f"{BASE_URL}/api/smc/current/EURUSD")
        signal = response.json()
        print(f"   Signal available: {signal.get('ok', False)}")
        if signal.get("signal"):
            sig_data = signal["signal"]
            print(f"   - {sig_data}")
    except Exception as e:
        print(f"   Error: {e}")


def test_trap_detection():
    """Test trap detection with sample data"""
    print("\n🎯 Trap Detection Test:")
    print("-" * 30)

    # Sample historical data for trap detection
    sample_bars = [
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
        },  # Potential trap
    ]

    sample_ticks = [
        {"price": 1.1025, "size": 200, "side": "buy"},  # Stop sweep
        {"price": 1.1008, "size": 150, "side": "sell"},  # Reversal
    ]

    print("   Sample bar data:")
    for i, bar in enumerate(sample_bars):
        print(
            f"   Bar {i+1}: O={bar['o']:.4f} H={bar['h']:.4f} L={bar['l']:.4f} C={bar['c']:.4f} V={bar['v']}"
        )

    print("\n   This pattern shows a potential liquidity trap:")
    print("   - Large upper wick (1.1025 high, 1.1008 close)")
    print("   - High volume (2000 vs avg ~900)")
    print("   - Price rejection from highs")
    print("   - Potential buy-stop sweep followed by reversal")


def show_smc_features():
    """Show available SMC analysis features"""
    print("\n📊 Available SMC Analysis Features:")
    print("=" * 50)

    features = [
        {
            "name": "Liquidity Trap Detection",
            "description": "Identifies potential stop-loss sweeps and liquidity traps",
            "indicators": [
                "Wick analysis",
                "Volume surges",
                "Delta divergence",
                "Price rejection",
            ],
            "endpoint": "/api/smc/idea/prepare",
        },
        {
            "name": "Order Block Analysis",
            "description": "Detects institutional order blocks and accumulation zones",
            "indicators": [
                "Volume clusters",
                "Price consolidation",
                "Breakout patterns",
            ],
            "endpoint": "/api/smc/order-blocks/{symbol}",
        },
        {
            "name": "Fair Value Gap Detection",
            "description": "Identifies price inefficiencies and fair value gaps",
            "indicators": ["Gap analysis", "Imbalance detection", "FVG zones"],
            "endpoint": "/api/smc/fair-value-gaps/{symbol}",
        },
        {
            "name": "SMC Signal Generation",
            "description": "Generates trading signals based on SMC principles",
            "indicators": [
                "Multi-timeframe analysis",
                "Confidence scoring",
                "Risk/reward calculation",
            ],
            "endpoint": "/api/smc/signals/{symbol}",
        },
        {
            "name": "Trade Memory & History",
            "description": "Tracks and analyzes historical trade performance",
            "indicators": ["Win rate", "Profit factor", "Trade analysis"],
            "endpoint": "/api/smc/history",
        },
    ]

    for i, feature in enumerate(features, 1):
        print(f"\n{i}. {feature['name']}")
        print(f"   📝 {feature['description']}")
        print(f"   🔍 Indicators: {', '.join(feature['indicators'])}")
        print(f"   🌐 Endpoint: {feature['endpoint']}")


if __name__ == "__main__":
    test_smc_endpoints()
    test_trap_detection()
    show_smc_features()

    print("\n" + "=" * 50)
    print("🎉 SMC Analysis Test Complete!")
    print("\n💡 To use these features in the frontend:")
    print("   1. Open http://localhost:5173")
    print("   2. Navigate to the SMC tab")
    print("   3. Monitor real-time SMC signals")
    print("   4. Use the trading interface for execution")
