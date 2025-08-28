#!/usr/bin/env python3
"""
MT5 Connection Sanity Check Script
Verifies MT5 connection and basic data retrieval functionality
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.mt5_market_data import MT5MarketFeed, FeedUnavailableError
from backend.core.config import get_settings

def main():
    """Main sanity check function"""
    print("🔍 ARIA MT5 Connection Sanity Check")
    print("=" * 50)
    
    # Check environment variables
    print("\n📋 Environment Check:")
    settings = get_settings()
    
    print(f"  MT5_LOGIN: {settings.MT5_LOGIN}")
    print(f"  MT5_SERVER: {settings.MT5_SERVER}")
    print(f"  MT5_PASSWORD: {'*' * len(settings.MT5_PASSWORD) if settings.MT5_PASSWORD else 'NOT SET'}")
    print(f"  ARIA_ENABLE_MT5: {settings.ARIA_ENABLE_MT5}")
    
    if not settings.MT5_LOGIN or not settings.MT5_PASSWORD or not settings.MT5_SERVER:
        print("❌ MT5 credentials incomplete. Please set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER")
        return False
    
    # Test MT5 connection
    print("\n🔌 Testing MT5 Connection:")
    mt5_feed = MT5MarketFeed()
    
    try:
        success = mt5_feed.connect()
        if success:
            print("✅ MT5 connected successfully")
        else:
            print("❌ MT5 connection failed")
            return False
    except Exception as e:
        print(f"❌ MT5 connection error: {e}")
        return False
    
    # Test data retrieval for each symbol
    print("\n📊 Testing Data Retrieval:")
    symbols = settings.symbols_list or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]
    
    success_count = 0
    for symbol in symbols:
        try:
            bar = mt5_feed.get_last_bar(symbol)
            print(f"  {symbol}: ✅ Time:{bar['time']} O:{bar['open']:.5f} H:{bar['high']:.5f} L:{bar['low']:.5f} C:{bar['close']:.5f}")
            success_count += 1
        except FeedUnavailableError as e:
            print(f"  {symbol}: ❌ Feed error: {e}")
        except Exception as e:
            print(f"  {symbol}: ❌ Error: {e}")
    
    # Test historical data
    print("\n📈 Testing Historical Data:")
    try:
        bars = mt5_feed.get_historical_bars("EURUSD", "M1", 5)
        if bars:
            print(f"  ✅ Retrieved {len(bars)} historical bars for EURUSD M1")
            for i, bar in enumerate(bars[-3:], 1):  # Show last 3 bars
                print(f"    Bar {i}: Time:{bar['time']} OHLC:{bar['open']:.5f}/{bar['high']:.5f}/{bar['low']:.5f}/{bar['close']:.5f}")
        else:
            print("  ❌ No historical bars retrieved")
    except Exception as e:
        print(f"  ❌ Historical data error: {e}")
    
    # Cleanup
    mt5_feed.disconnect()
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  Symbols tested: {len(symbols)}")
    print(f"  Successful retrievals: {success_count}")
    print(f"  Success rate: {success_count/len(symbols)*100:.1f}%")
    
    if success_count == len(symbols):
        print("✅ All tests passed! MT5 connection is healthy.")
        return True
    else:
        print("⚠️  Some tests failed. Check MT5 connection and symbol availability.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
