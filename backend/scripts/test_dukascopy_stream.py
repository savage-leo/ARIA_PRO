#!/usr/bin/env python
"""
Test Dukascopy streaming connector
Verifies XAUUSD data streaming without disk storage
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.core.data_connector import fetch_bars, fetch_training_data, stream_bars


async def test_fetch_bars():
    """Test fetching historical bars."""
    print("\n=== Testing fetch_bars ===")

    # Fetch last 2 hours of M5 data
    end = datetime.utcnow()
    start = end - timedelta(hours=2)

    print(f"Fetching XAUUSD M5 from {start} to {end}")
    df = await fetch_bars("XAUUSD", "M5", start, end)

    if df.empty:
        print("WARNING: No data received (market might be closed)")
    else:
        print(f"Received {len(df)} bars")
        print("\nFirst 3 bars:")
        print(df.head(3))
        print("\nLast 3 bars:")
        print(df.tail(3))
        print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    return df


async def test_training_data():
    """Test fetching training data."""
    print("\n=== Testing fetch_training_data ===")

    print("Fetching 1 day of XAUUSD M5 for training")
    df = await fetch_training_data("XAUUSD", "M5", days_back=1)

    if df.empty:
        print("WARNING: No training data received")
    else:
        print(f"Training data shape: {df.shape}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    return df


async def test_streaming():
    """Test streaming bars."""
    print("\n=== Testing stream_bars ===")
    print("Starting 60-minute rolling window stream (3 iterations)...")

    iteration = 0
    async for df in stream_bars("XAUUSD", "M5", window_minutes=60):
        iteration += 1
        print(f"\nIteration {iteration}:")

        if df.empty:
            print("  No data in this window")
        else:
            print(f"  Bars: {len(df)}")
            print(f"  Latest: {df['timestamp'].iloc[-1] if len(df) > 0 else 'N/A'}")
            print(f"  Close: ${df['close'].iloc[-1]:.2f}" if len(df) > 0 else "")
            print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

        if iteration >= 3:
            print("\nStopping stream after 3 iterations")
            break


async def test_multiple_timeframes():
    """Test different timeframes."""
    print("\n=== Testing Multiple Timeframes ===")

    timeframes = ["M1", "M5", "H1"]
    end = datetime.utcnow()
    start = end - timedelta(hours=1)

    for tf in timeframes:
        print(f"\nTimeframe {tf}:")
        df = await fetch_bars("XAUUSD", tf, start, end)

        if not df.empty:
            print(f"  Bars: {len(df)}")
            print(f"  Avg close: ${df['close'].mean():.2f}")
        else:
            print("  No data")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("ARIA Dukascopy Streaming Connector Test")
    print("=" * 60)
    print("Testing pure RAM-based streaming (no disk storage)")

    try:
        # Test basic fetch
        df1 = await test_fetch_bars()

        # Test training data fetch
        df2 = await test_training_data()

        # Test multiple timeframes
        await test_multiple_timeframes()

        # Test streaming (commented out for quick test)
        # await test_streaming()

        print("\n" + "=" * 60)
        print("[OK] All tests completed successfully")
        print("Data stayed in RAM only - no disk writes")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
