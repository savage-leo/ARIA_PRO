#!/usr/bin/env python
"""
Simple test for ARIA Dukascopy data connector
Tests pure RAM streaming without disk storage
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.data_connector import fetch_bars


async def test_simple_fetch():
    """Test basic data fetching from Dukascopy."""
    print("ARIA Data Connector Test")
    print("-" * 40)

    # Test with a known working date range (weekday)
    # Using a date from early 2024 when markets were open
    start = datetime(2024, 1, 15, 10, 0, 0)  # Monday 10:00 UTC
    end = datetime(2024, 1, 15, 12, 0, 0)  # Monday 12:00 UTC

    print(f"Fetching XAUUSD M5 bars")
    print(f"From: {start}")
    print(f"To:   {end}")
    print()

    df = await fetch_bars("XAUUSD", "M5", start, end)

    if df.empty:
        print("No data received - trying alternative symbol EURUSD")
        df = await fetch_bars("EURUSD", "M5", start, end)

    if not df.empty:
        print(f"SUCCESS: Received {len(df)} bars")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print("\nSample data (first 5 bars):")
        print(df.head())
        print("\nPrice stats:")
        print(f"  Min:  ${df['low'].min():.4f}")
        print(f"  Max:  ${df['high'].max():.4f}")
        print(f"  Mean: ${df['close'].mean():.4f}")
    else:
        print("No data available for this time range")
        print("Note: Dukascopy may not have data for all symbols/times")

    return df


if __name__ == "__main__":
    df = asyncio.run(test_simple_fetch())
    print("\n" + "-" * 40)
    print("Test complete - data stayed in RAM only")
