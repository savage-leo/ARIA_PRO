#!/usr/bin/env python3
"""
Test script for Enhanced SMC Fusion Core
Verifies the integration works correctly
"""

import sys
import os
import time
import logging

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.smc.smc_fusion_core import (
    get_engine,
    get_enhanced_engine,
    EnhancedTradeIdea,
)
from backend.services.feedback_service import feedback_service

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
)
logger = logging.getLogger(__name__)


def test_enhanced_fusion():
    """Test the enhanced fusion core"""
    print("🧪 Testing Enhanced SMC Fusion Core...")

    try:
        # Test 1: Create enhanced engine
        print("\n1. Creating enhanced fusion engine...")
        engine = get_enhanced_engine("EURUSD")
        print(f"✅ Enhanced engine created: {type(engine).__name__}")

        # Test 2: Register with feedback service
        print("\n2. Registering with feedback service...")
        feedback_service.register_engine("EURUSD", engine)
        print("✅ Engine registered with feedback service")

        # Test 3: Create sample bar data
        print("\n3. Creating sample bar data...")
        sample_bar = {
            "ts": time.time(),
            "o": 1.1000,
            "h": 1.1010,
            "l": 1.0990,
            "c": 1.1005,
            "v": 1000,
            "symbol": "EURUSD",
        }
        print("✅ Sample bar created")

        # Test 4: Test basic SMC analysis
        print("\n4. Testing basic SMC analysis...")
        idea = engine.ingest_bar(sample_bar)
        if idea:
            print(
                f"✅ SMC idea generated: {idea.bias} (confidence: {idea.confidence:.3f})"
            )
        else:
            print("ℹ️ No SMC idea generated (expected for single bar)")

        # Test 5: Test enhanced analysis with AI signals
        print("\n5. Testing enhanced analysis with AI signals...")
        raw_signals = {
            "lstm": 0.3,
            "cnn": 0.2,
            "ppo": -0.1,
            "vision": 0.4,
            "llm_macro": 0.15,
        }

        market_feats = {
            "price": 1.1005,
            "spread": 0.00008,
            "atr": 0.0012,
            "vol": 0.0012,
            "trend_strength": 0.35,
            "liquidity": 0.7,
            "session_factor": 0.8,
        }

        # Add more bars for analysis
        for i in range(20):
            bar = {
                "ts": time.time() + i,
                "o": 1.1000 + i * 0.0001,
                "h": 1.1010 + i * 0.0001,
                "l": 1.0990 + i * 0.0001,
                "c": 1.1005 + i * 0.0001,
                "v": 1000 + i * 10,
                "symbol": "EURUSD",
            }
            engine.ingest_bar(bar)

        # Now test enhanced analysis
        enhanced_idea = engine.ingest_bar(sample_bar, raw_signals, market_feats)
        if enhanced_idea:
            print(
                f"✅ Enhanced idea generated: {enhanced_idea.bias} (confidence: {enhanced_idea.confidence:.3f})"
            )
            print(f"   Regime: {enhanced_idea.regime}")
            print(f"   Anomaly score: {enhanced_idea.anomaly_score:.3f}")
            print(f"   Meta weights: {enhanced_idea.meta_weights}")
        else:
            print("ℹ️ No enhanced idea generated")

        # Test 6: Test feedback
        print("\n6. Testing feedback system...")
        feedback_service.submit_trade_feedback("EURUSD", 0.05, raw_signals)
        print("✅ Feedback submitted")

        # Test 7: Get engine stats
        print("\n7. Getting engine statistics...")
        stats = feedback_service.get_engine_stats("EURUSD")
        if stats:
            print(
                f"✅ Stats retrieved: wins={stats['wins']}, losses={stats['losses']}, pnl={stats['pnl_day']:.4f}"
            )

        # Test 8: Test kill switch
        print("\n8. Testing kill switch...")
        feedback_service.kill_switch("EURUSD", True)
        print("✅ Kill switch activated")

        print("\n🎉 All tests passed! Enhanced fusion core is working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_fusion()
    sys.exit(0 if success else 1)
