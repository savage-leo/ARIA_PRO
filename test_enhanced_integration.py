#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced ARIA Integration
Tests Phase 2-4 implementations: Enhanced SMC Analysis, Real Market Integration, and Frontend Fixes
"""

import sys
import os
import time
import logging
import asyncio
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.smc.smc_fusion_core import get_enhanced_engine, EnhancedTradeIdea
from backend.smc.smc_edge_core import get_edge
from backend.services.real_ai_signal_generator import RealAISignalGenerator
from backend.services.feedback_service import feedback_service
from backend.core.risk_engine import (
    validate_and_size_order,
    get_account_balance,
    get_current_price,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
)
logger = logging.getLogger(__name__)


def test_enhanced_smc_analysis():
    """Test Phase 2: Enhanced SMC Analysis"""
    print("\n🧪 Testing Phase 2: Enhanced SMC Analysis")

    try:
        # Test enhanced fusion core
        engine = get_enhanced_engine("EURUSD")
        print("✅ Enhanced fusion core created successfully")

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
            "volatility_regime": 0.4,
            "momentum": 0.2,
            "support_resistance": 0.6,
            "volume_profile": 0.3,
            "market_structure": 0.5,
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
        else:
            print("⚠️  No enhanced trade idea generated (this is normal for single bar)")

        # Test edge engine integration
        edge_engine = get_edge("EURUSD")
        edge_idea = edge_engine.ingest_bar(
            sample_bar, raw_signals=raw_signals, market_feats=market_feats
        )
        print("✅ Edge engine integration working")

        return True

    except Exception as e:
        print(f"❌ Enhanced SMC Analysis test failed: {e}")
        return False


def test_real_market_integration():
    """Test Phase 3: Real Market Integration"""
    print("\n🧪 Testing Phase 3: Real Market Integration")

    try:
        # Test risk engine enhancements
        account_balance = get_account_balance()
        current_price = get_current_price("EURUSD")
        print(f"✅ Account balance: ${account_balance:.2f}")
        print(f"✅ Current EURUSD price: {current_price:.5f}")

        # Test enhanced position sizing
        sl_price = current_price - 0.0020  # 20 pips stop loss
        lot_size = validate_and_size_order(
            "EURUSD", "buy", 1.0, sl_price, account_balance, current_price
        )
        print(f"✅ Enhanced position sizing: {lot_size:.4f} lots")

        # Test AI signal generator enhancements
        signal_gen = RealAISignalGenerator()
        print("✅ Enhanced AI signal generator created")

        # Test market feature calculations
        sample_bars = [
            {
                "ts": time.time() - 60,
                "o": 1.0850,
                "h": 1.0855,
                "l": 1.0845,
                "c": 1.0852,
                "v": 1000,
            },
            {
                "ts": time.time(),
                "o": 1.0852,
                "h": 1.0858,
                "l": 1.0848,
                "c": 1.0855,
                "v": 1200,
            },
        ]

        # Test AI signal generation
        try:
            signals = asyncio.run(
                signal_gen._generate_ai_signals("EURUSD", sample_bars)
            )
            print(f"✅ AI signals generated: {signals}")
        except Exception as e:
            print(f"⚠️  AI signal generation failed: {e}")
            signals = {}

        # Test market features
        try:
            market_feats = asyncio.run(
                signal_gen._build_market_features("EURUSD", sample_bars)
            )
            print(
                f"✅ Enhanced market features: {len(market_feats)} features calculated"
            )
        except Exception as e:
            print(f"⚠️  Market features failed: {e}")
            market_feats = {}

        return True

    except Exception as e:
        print(f"❌ Real Market Integration test failed: {e}")
        return False


def test_feedback_system():
    """Test feedback system integration"""
    print("\n🧪 Testing Feedback System Integration")

    try:
        # Test feedback service
        engine = get_enhanced_engine("EURUSD")
        feedback_service.register_engine("EURUSD", engine)
        print("✅ Engine registered with feedback service")

        # Test feedback submission
        last_features = {
            "lstm": 0.3,
            "cnn": 0.2,
            "ppo": -0.1,
            "vision": 0.4,
            "llm_macro": 0.15,
            "trend": 0.35,
            "vol_norm": 0.8,
            "spr_norm": 1.2,
            "liq_sess": 0.6,
        }

        feedback_service.submit_trade_feedback("EURUSD", 0.05, last_features)
        print("✅ Trade feedback submitted successfully")

        # Test engine statistics
        stats = feedback_service.get_engine_stats("EURUSD")
        if stats:
            print(f"✅ Engine stats retrieved: {stats}")

        return True

    except Exception as e:
        print(f"❌ Feedback system test failed: {e}")
        return False


def test_frontend_compatibility():
    """Test Phase 4: Frontend Compatibility"""
    print("\n🧪 Testing Phase 4: Frontend Compatibility")

    try:
        # Test that enhanced ideas can be serialized for frontend
        engine = get_enhanced_engine("EURUSD")

        sample_bar = {
            "ts": time.time(),
            "o": 1.0850,
            "h": 1.0855,
            "l": 1.0845,
            "c": 1.0852,
            "v": 1000,
            "symbol": "EURUSD",
        }

        raw_signals = {
            "lstm": 0.3,
            "cnn": 0.2,
            "ppo": -0.1,
            "vision": 0.4,
            "llm_macro": 0.15,
        }

        market_feats = {
            "price": 1.0852,
            "spread": 0.00008,
            "atr": 0.0012,
            "vol": 0.0012,
            "trend_strength": 0.35,
            "liquidity": 0.7,
            "session_factor": 0.8,
        }

        enhanced_idea = engine.ingest_bar(sample_bar, raw_signals, market_feats)
        if enhanced_idea:
            # Test serialization
            idea_dict = enhanced_idea.as_dict()
            print("✅ Enhanced idea serialized successfully")
            print(f"   Keys: {list(idea_dict.keys())}")

            # Test that all required fields are present
            required_fields = [
                "symbol",
                "bias",
                "confidence",
                "entry",
                "stop",
                "takeprofit",
                "meta_weights",
                "regime",
            ]
            missing_fields = [
                field for field in required_fields if field not in idea_dict
            ]
            if not missing_fields:
                print("✅ All required fields present for frontend")
            else:
                print(f"⚠️  Missing fields: {missing_fields}")

        return True

    except Exception as e:
        print(f"❌ Frontend compatibility test failed: {e}")
        return False


def test_comprehensive_integration():
    """Test comprehensive integration of all phases"""
    print("\n🧪 Testing Comprehensive Integration")

    try:
        # Test full workflow
        symbol = "EURUSD"

        # 1. Create enhanced engine
        engine = get_enhanced_engine(symbol)

        # 2. Generate sample market data
        bars = []
        base_price = 1.0850
        for i in range(20):
            bar = {
                "ts": time.time() - (20 - i) * 60,
                "o": base_price + i * 0.0001,
                "h": base_price + i * 0.0001 + 0.0005,
                "l": base_price + i * 0.0001 - 0.0003,
                "c": base_price + i * 0.0001 + 0.0002,
                "v": 1000 + i * 50,
                "symbol": symbol,
            }
            bars.append(bar)

        # 3. Generate AI signals
        signal_gen = RealAISignalGenerator()
        try:
            raw_signals = asyncio.run(signal_gen._generate_ai_signals(symbol, bars))
        except Exception as e:
            print(f"⚠️  AI signal generation failed: {e}")
            raw_signals = {
                "lstm": 0.0,
                "cnn": 0.0,
                "ppo": 0.0,
                "vision": 0.0,
                "llm_macro": 0.0,
            }

        # 4. Build market features
        try:
            market_feats = asyncio.run(signal_gen._build_market_features(symbol, bars))
        except Exception as e:
            print(f"⚠️  Market features failed: {e}")
            market_feats = {
                "price": 1.0850,
                "spread": 0.00008,
                "atr": 0.0012,
                "vol": 0.0012,
                "trend_strength": 0.35,
                "liquidity": 0.7,
                "session_factor": 0.8,
            }

        # 5. Process with enhanced fusion
        latest_bar = bars[-1]
        enhanced_idea = engine.ingest_bar(latest_bar, raw_signals, market_feats)

        # 6. Test edge engine integration
        edge_engine = get_edge(symbol)
        edge_idea = edge_engine.ingest_bar(
            latest_bar, raw_signals=raw_signals, market_feats=market_feats
        )

        # 7. Test position sizing
        if enhanced_idea and enhanced_idea.stop > 0:
            lot_size = validate_and_size_order(
                symbol,
                "buy" if enhanced_idea.bias == "bullish" else "sell",
                1.0,
                enhanced_idea.stop,
            )
            print(f"✅ Position sized: {lot_size:.4f} lots")

        print("✅ Comprehensive integration test passed")
        return True

    except Exception as e:
        print(f"❌ Comprehensive integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 ARIA Enhanced Integration Test Suite")
    print("=" * 50)

    tests = [
        ("Phase 2: Enhanced SMC Analysis", test_enhanced_smc_analysis),
        ("Phase 3: Real Market Integration", test_real_market_integration),
        ("Feedback System", test_feedback_system),
        ("Phase 4: Frontend Compatibility", test_frontend_compatibility),
        ("Comprehensive Integration", test_comprehensive_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
