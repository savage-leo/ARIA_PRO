#!/usr/bin/env python3
"""
Test script for C++ integration
Run this to verify the C++ components are working correctly
"""

import sys
import os
import time

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def test_cpp_integration():
    """Test the C++ integration service"""
    print("Testing ARIA_PRO C++ Integration...")

    try:
        from services.cpp_integration import cpp_service, CPP_AVAILABLE

        print(f"C++ Available: {CPP_AVAILABLE}")
        print(f"Market Processor: {cpp_service.market_processor is not None}")
        print(f"SMC Engine: {cpp_service.smc_engine is not None}")

        # Test tick processing
        print("\nTesting tick processing...")
        tick_result = cpp_service.process_tick_data(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1001,
            volume=1000,
            timestamp=int(time.time() * 1000),
        )
        print(f"Tick Result: {tick_result}")

        # Test bar processing
        print("\nTesting bar processing...")
        bar_result = cpp_service.process_bar_data(
            symbol="EURUSD",
            open_price=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000,
            timestamp=int(time.time() * 1000),
        )
        print(f"Bar Result: {bar_result}")

        # Test SMC signals
        print("\nTesting SMC signals...")
        signals = cpp_service.get_smc_signals("EURUSD")
        print(f"SMC Signals: {signals}")

        # Test order blocks
        print("\nTesting order blocks...")
        blocks = cpp_service.get_order_blocks("EURUSD")
        print(f"Order Blocks: {blocks}")

        print("\n✅ C++ Integration Test Complete!")
        return True

    except Exception as e:
        print(f"❌ C++ Integration Test Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cpp_integration()
    sys.exit(0 if success else 1)
