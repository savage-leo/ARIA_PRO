#!/usr/bin/env python3
"""
Test script to verify MT5 connection and integration with data source manager
"""

import os
import sys
import asyncio
import logging

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Set environment variables for MT5
os.environ["ARIA_ENABLE_MT5"] = "1"

from backend.services.mt5_market_data import mt5_market_feed
from backend.services.data_source_manager import data_source_manager


async def main():
    """Test MT5 connection and integration"""
    print("Testing MT5 connection and integration...")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Test MT5 connection
        print("Testing MT5 connection...")
        connected = mt5_market_feed.connect()
        if connected:
            print("SUCCESS: MT5 connected successfully")
        else:
            print("ERROR: Failed to connect to MT5")
            return

        # Test data source manager integration
        print("\nTesting data source manager integration...")
        # Connect MT5 feed to data source manager
        data_source_manager.mt5_feed = mt5_market_feed

        # Test getting a bar
        try:
            bar = data_source_manager.get_last_bar("EURUSD")
            print(f"SUCCESS: Got last bar for EURUSD: {bar}")
        except Exception as e:
            print(f"INFO: Expected error when no data yet: {e}")

        # Start MT5 feed
        print("\nStarting MT5 feed...")
        mt5_market_feed.start()

        # Wait a moment for connection
        await asyncio.sleep(2)

        # Check connection status
        print(f"MT5 connected status: {mt5_market_feed._connected}")

        # Try to get historical bars
        try:
            bars = mt5_market_feed.get_historical_bars("EURUSD", "M1", 10)
            print(f"SUCCESS: Got {len(bars)} historical bars for EURUSD")
            if bars:
                print(f"Latest bar: {bars[-1]}")
        except Exception as e:
            print(f"INFO: Error getting historical bars: {e}")

        # Stop MT5 feed
        print("\nStopping MT5 feed...")
        mt5_market_feed.stop()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()

    print("\nTest completed.")


if __name__ == "__main__":
    asyncio.run(main())
