#!/usr/bin/env python3
"""
Test script to run AutoTrader in a controlled environment
"""

import os
import sys
import asyncio
import logging
import argparse

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.services.auto_trader import AutoTrader


async def main():
    """Run AutoTrader dry-run harness with live MT5/API data and limited cycles."""
    parser = argparse.ArgumentParser(description="AutoTrader Dry-Run Harness")
    parser.add_argument("--cycles", type=int, default=2, help="Number of cycles to run")
    parser.add_argument(
        "--interval", type=float, default=0.5, help="Seconds to wait between cycles"
    )
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol to process")
    args = parser.parse_args()

    print("AutoTrader Live Data Harness")
    print("=" * 50)

    # Configure for live data testing with dry-run execution
    os.environ["AUTO_TRADE_ENABLED"] = "0"  # Don't actually trade
    os.environ["AUTO_TRADE_SYMBOLS"] = args.symbol
    os.environ["AUTO_TRADE_INTERVAL_SEC"] = str(max(1, int(args.interval)))
    os.environ["AUTO_TRADE_DRY_RUN"] = "1"  # Simulate orders only
    os.environ["AUTO_TRADE_COOLDOWN_SEC"] = "0"  # No cooldown for testing
    # Keep live data sources enabled - remove MT5 disable
    # os.environ["ARIA_ENABLE_MT5"] = "1"  # Allow MT5 usage for live data

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a fresh AutoTrader instance so it picks up the env overrides
    trader = AutoTrader()

    try:
        for i in range(max(1, args.cycles)):
            print(f"Cycle {i + 1}/{args.cycles} -> {args.symbol} (live data)")
            await trader._process_symbol(args.symbol)
            if i < args.cycles - 1 and args.interval > 0:
                await asyncio.sleep(args.interval)
        print("SUCCESS: Live data cycles completed")
    except Exception as e:
        print(f"ERROR during live data run: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Live data harness completed.")


if __name__ == "__main__":
    asyncio.run(main())
