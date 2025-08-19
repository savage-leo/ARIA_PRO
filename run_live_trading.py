#!/usr/bin/env python3
"""
Run live trading with MT5 execution
"""

import os
import sys
import asyncio
import logging

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Set environment variables for live trading
os.environ["ARIA_ENABLE_MT5"] = "1"
os.environ["AUTO_TRADE_ENABLED"] = "1"
os.environ["AUTO_TRADE_DRY_RUN"] = "0"  # Set to "1" for dry run mode
os.environ["ARIA_ENABLE_EXEC"] = "1"

from backend.services.mt5_market_data import mt5_market_feed
from backend.services.mt5_executor import mt5_executor
from backend.services.data_source_manager import data_source_manager
from backend.services.auto_trader import auto_trader


async def main():
    """Run live trading with MT5 execution"""
    print("Starting live trading with MT5 execution...")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Connect MT5 components
        print("Connecting to MT5 market feed...")
        connected = mt5_market_feed.connect()
        if not connected:
            print("ERROR: Failed to connect to MT5 market feed")
            return

        print("MT5 market feed connected successfully")

        # Connect MT5 executor
        print("Connecting to MT5 executor...")
        exec_connected = mt5_executor.connect()
        if not exec_connected:
            print("ERROR: Failed to connect to MT5 executor")
            return

        print("MT5 executor connected successfully")

        # Connect components to data source manager
        data_source_manager.mt5_feed = mt5_market_feed

        # Start MT5 market feed
        print("Starting MT5 market feed...")
        mt5_market_feed.start()

        # Start auto trader
        print("Starting auto trader...")
        await auto_trader.start()

        print("\nLive trading started successfully!")
        print("Press Ctrl+C to stop trading")

        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping live trading...")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean shutdown
        try:
            await auto_trader.stop()
            mt5_market_feed.stop()
            mt5_executor.disconnect()
        except Exception as e:
            print(f"Error during shutdown: {e}")

    print("\nLive trading stopped.")


if __name__ == "__main__":
    asyncio.run(main())
