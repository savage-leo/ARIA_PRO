#!/usr/bin/env python3
"""
Run a backtest using MT5 data
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

from backend.scripts.backtest_bars import fetch_historical_data, backtest_fusion


async def main():
    """Run backtest using MT5 data"""
    print("Running backtest with MT5 data...")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Fetch historical data using MT5
        print("Fetching historical data from MT5...")
        bars = fetch_historical_data("EURUSD", 7)  # 7 days of data

        if not bars:
            print("ERROR: No data fetched from MT5")
            return

        print(f"Successfully fetched {len(bars)} bars from MT5")

        # Run backtest
        print("\nRunning backtest...")
        results = backtest_fusion("EURUSD", bars)

        if not results:
            print("ERROR: Backtest failed")
            return

        # Display results
        print("\n=== ARIA Backtest Results ===")
        print(f"Symbol: {results['symbol']}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print()
        print("Regime Counts:")
        for regime, count in results["regime_counts"].items():
            print(f"  {regime}: {count}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()

    print("\nBacktest completed.")


if __name__ == "__main__":
    asyncio.run(main())
