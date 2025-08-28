"""Simple AI Integration Test"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.mt5_ai_integration import mt5_ai_integration
from services.mt5_market_data import mt5_market_feed

def test_connection():
    print("Testing MT5 connection...")
    
    # Test MT5 connection
    if hasattr(mt5_market_feed, 'connect'):
        connected = mt5_market_feed.connect()
        print(f"MT5 Connected: {connected}")
    
    # Test getting historical data
    bars = mt5_market_feed.get_historical_bars("EURUSD", "M5", 10)
    print(f"Retrieved {len(bars)} bars for EURUSD")
    
    if bars:
        latest_bar = bars[-1]
        print(f"Latest bar: {latest_bar}")
        
        # Test AI processing
        import numpy as np
        prices = [bar['close'] for bar in bars]
        sma_5 = np.mean(prices[-5:])
        print(f"SMA5: {sma_5:.5f}")

if __name__ == "__main__":
    test_connection()
