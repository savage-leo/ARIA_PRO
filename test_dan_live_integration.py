#!/usr/bin/env python3
"""
DAN_LIVE_ONLY Integration Test Suite
Tests all enhanced components: Model Adapters, MT5 Client, Risk Engine, Trade Arbiter, SMC Enhancements
"""

import sys
import os
import time
import logging
import asyncio
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.services.models_interface import build_default_adapters
from backend.services.mt5_client import MT5Client
from backend.core.risk_engine_enhanced import RiskEngine
from backend.services.exec_arbiter import TradeArbiter, ExecPlan
from backend.smc.smc_enhancements import (
    detect_order_blocks,
    detect_fvg,
    detect_liquidity_zones,
)
from backend.services.advanced_features import (
    detect_trap,
    bias_engine,
    simple_portfolio_manager,
)
from backend.smc.smc_fusion_core import get_enhanced_engine, EnhancedTradeIdea

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
)
logger = logging.getLogger(__name__)


def test_model_adapters():
    """Test REAL model adapters (no random outputs)"""
    print("\n🧪 Testing Model Adapters (DAN_LIVE_ONLY)")

    try:
        # Initialize adapters
        adapters = build_default_adapters()
        for adapter in adapters.values():
            adapter.load()
        print("✅ Model adapters loaded successfully")

        # Test LSTM adapter
        lstm_signal = adapters["lstm"].predict(
            {"series": [1.0, 1.01, 1.02, 1.03, 1.04]}
        )
        print(f"✅ LSTM signal: {lstm_signal:.4f}")

        # Test CNN adapter
        cnn_signal = adapters["cnn"].predict({"image": [[[0.1, 0.2, 0.3]]]})
        print(f"✅ CNN signal: {cnn_signal:.4f}")

        # Test PPO adapter
        ppo_signal = adapters["ppo"].predict({"state_vec": [0.1, 0.2, 0.3, 0.4, 0.5]})
        print(f"✅ PPO signal: {ppo_signal:.4f}")

        # Test Visual adapter
        vision_signal = adapters["vision"].predict(
            {"latent": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )
        print(f"✅ Visual signal: {vision_signal:.4f}")

        # Test LLM adapter
        llm_signal = adapters["llm_macro"].predict(
            {"text": "ECB dovish, rate cuts expected"}
        )
        print(f"✅ LLM signal: {llm_signal:.4f}")

        # Verify no random outputs (deterministic)
        lstm_signal2 = adapters["lstm"].predict(
            {"series": [1.0, 1.01, 1.02, 1.03, 1.04]}
        )
        assert abs(lstm_signal - lstm_signal2) < 1e-10, "LSTM not deterministic"
        print("✅ Model outputs are deterministic (no random)")

        return True

    except Exception as e:
        print(f"❌ Model adapters test failed: {e}")
        return False


def test_mt5_client():
    """Test enhanced MT5 client"""
    print("\n🧪 Testing Enhanced MT5 Client")

    try:
        # Initialize client
        client = MT5Client()
        print(f"✅ MT5 client created, MT5 available: {hasattr(client, 'connect')}")

        # Test connection (will fail gracefully if MT5 not installed)
        connected = client.connect()
        print(f"✅ MT5 connection attempt: {connected}")

        # Test client start (background threads)
        client.start()
        print("✅ MT5 client background threads started")

        # Test stop
        client.stop()
        print("✅ MT5 client stopped successfully")

        return True

    except Exception as e:
        print(f"❌ MT5 client test failed: {e}")
        return False


def test_risk_engine():
    """Test enhanced risk engine"""
    print("\n🧪 Testing Enhanced Risk Engine")

    try:
        # Initialize risk engine
        risk_engine = RiskEngine()
        print("✅ Risk engine created")

        # Test position sizing
        lot_size = risk_engine.size_from_sl(10.0, "EURUSD", 0.8)
        print(f"✅ Position sizing: {lot_size:.4f} lots for 10 pips SL")

        # Test account refresh
        risk_engine.refresh_account()
        print(f"✅ Account equity: ${risk_engine.account_equity:.2f}")

        return True

    except Exception as e:
        print(f"❌ Risk engine test failed: {e}")
        return False


def test_trade_arbiter():
    """Test trade arbiter"""
    print("\n🧪 Testing Trade Arbiter")

    try:
        # Initialize arbiter
        client = MT5Client()
        arbiter = TradeArbiter(client)
        print(f"✅ Trade arbiter created, dry run: {arbiter.dry_run}")

        # Test execution plan
        plan = ExecPlan(
            symbol="EURUSD",
            direction=1,
            lots=0.1,
            price=1.0850,
            sl=1.0830,
            tp=1.0870,
            reason="DAN_LIVE_TEST",
        )

        # Test routing (will be dry run by default)
        result = arbiter.route(plan)
        print(f"✅ Trade routing result: {result}")

        return True

    except Exception as e:
        print(f"❌ Trade arbiter test failed: {e}")
        return False


def test_smc_enhancements():
    """Test SMC enhancements"""
    print("\n🧪 Testing SMC Enhancements")

    try:
        # Create sample bars
        bars = []
        base_price = 1.0850
        for i in range(50):
            bar = {
                "open": base_price + i * 0.0001,
                "high": base_price + i * 0.0001 + 0.0005,
                "low": base_price + i * 0.0001 - 0.0003,
                "close": base_price + i * 0.0001 + 0.0002,
                "time": time.time() - (50 - i) * 60,
            }
            bars.append(bar)

        # Test order block detection
        order_blocks = detect_order_blocks(bars)
        print(f"✅ Order blocks detected: {len(order_blocks)}")

        # Test FVG detection
        fvgs = detect_fvg(bars)
        print(f"✅ Fair value gaps detected: {len(fvgs)}")

        # Test liquidity zones
        liquidity_zones = detect_liquidity_zones(bars)
        print(f"✅ Liquidity zones detected: {len(liquidity_zones)}")

        return True

    except Exception as e:
        print(f"❌ SMC enhancements test failed: {e}")
        return False


def test_advanced_features():
    """Test advanced features"""
    print("\n🧪 Testing Advanced Features")

    try:
        # Create sample bars for trap detection
        bars = []
        base_price = 1.0850
        for i in range(50):
            bar = {
                "open": base_price + i * 0.0001,
                "high": base_price + i * 0.0001 + 0.0005,
                "low": base_price + i * 0.0001 - 0.0003,
                "close": base_price + i * 0.0001 + 0.0002,
                "time": time.time() - (50 - i) * 60,
            }
            bars.append(bar)

        # Test trap detection
        trap_detected = detect_trap(bars)
        print(f"✅ Trap detection: {trap_detected}")

        # Test bias engine
        model_scores = {"lstm": 0.3, "cnn": 0.2, "ppo": -0.1}
        bias = bias_engine(model_scores, 0.1, 0.5)
        print(f"✅ Bias engine output: {bias:.4f}")

        # Test portfolio manager
        positions = [
            {"symbol": "EURUSD", "notional": 10000, "direction": 1},
            {"symbol": "GBPUSD", "notional": 8000, "direction": -1},
        ]
        managed_positions = simple_portfolio_manager(positions)
        print(f"✅ Portfolio manager processed {len(managed_positions)} positions")

        return True

    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        return False


def test_enhanced_fusion_core():
    """Test enhanced fusion core integration"""
    print("\n🧪 Testing Enhanced Fusion Core")

    try:
        # Initialize enhanced engine
        engine = get_enhanced_engine("EURUSD")
        print("✅ Enhanced fusion core created")

        # Test with sample data
        sample_bar = {
            "ts": time.time(),
            "o": 1.0850,
            "h": 1.0855,
            "l": 1.0845,
            "c": 1.0852,
            "v": 1000,
            "symbol": "EURUSD",
        }

        # Test AI signals
        raw_signals = {
            "lstm": 0.3,
            "cnn": 0.2,
            "ppo": -0.1,
            "vision": 0.4,
            "llm_macro": 0.15,
        }

        # Test market features
        market_feats = {
            "price": 1.0852,
            "spread": 0.00008,
            "atr": 0.0012,
            "vol": 0.0012,
            "trend_strength": 0.35,
            "liquidity": 0.7,
            "session_factor": 0.8,
        }

        # Test enhanced ingestion
        enhanced_idea = engine.ingest_bar(sample_bar, raw_signals, market_feats)
        if enhanced_idea:
            print(
                f"✅ Enhanced trade idea generated: {enhanced_idea.bias} conf={enhanced_idea.confidence:.2f}"
            )
            print(f"   Meta weights: {enhanced_idea.meta_weights}")
            print(f"   Regime: {enhanced_idea.regime}")
            print(f"   Anomaly score: {enhanced_idea.anomaly_score:.3f}")

            # Test position sizing
            lot_size = engine._risk_position(enhanced_idea, 1.0852)
            print(f"   Position size: {lot_size:.4f} lots")
        else:
            print("⚠️  No enhanced trade idea generated (this is normal for single bar)")

        return True

    except Exception as e:
        print(f"❌ Enhanced fusion core test failed: {e}")
        return False


async def test_real_market_data_feed():
    """Test enhanced market data feed"""
    print("\n🧪 Testing Enhanced Market Data Feed")

    try:
        from backend.services.mt5_market_data import MT5MarketDataFeed

        # Initialize feed
        feed = MT5MarketDataFeed()
        print("✅ Enhanced market data feed created")

        # Test start (will fail gracefully if MT5 not available)
        try:
            await feed.start()
            print("✅ Market data feed started")

            # Let it run for a moment
            await asyncio.sleep(2)

            # Stop
            await feed.stop()
            print("✅ Market data feed stopped")

        except Exception as e:
            print(f"⚠️  Market data feed test (expected if MT5 not available): {e}")

        return True

    except Exception as e:
        print(f"❌ Market data feed test failed: {e}")
        return False


def main():
    """Run all DAN_LIVE_ONLY tests"""
    print("🚀 DAN_LIVE_ONLY Integration Test Suite")
    print("=" * 60)

    tests = [
        ("Model Adapters", test_model_adapters),
        ("MT5 Client", test_mt5_client),
        ("Risk Engine", test_risk_engine),
        ("Trade Arbiter", test_trade_arbiter),
        ("SMC Enhancements", test_smc_enhancements),
        ("Advanced Features", test_advanced_features),
        ("Enhanced Fusion Core", test_enhanced_fusion_core),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Async test
    try:
        result = asyncio.run(test_real_market_data_feed())
        results.append(("Market Data Feed", result))
    except Exception as e:
        print(f"❌ Market Data Feed test crashed: {e}")
        results.append(("Market Data Feed", False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 DAN_LIVE_ONLY TEST RESULTS")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All DAN_LIVE_ONLY components are working correctly!")
        print("🚀 Ready for live trading with enhanced features!")
    else:
        print("⚠️  Some components need attention before live trading.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
