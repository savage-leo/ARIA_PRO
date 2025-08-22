import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Initialize MT5
if not mt5.initialize():
    print("❌ MT5 initialization failed:", mt5.last_error())
    exit()

symbols = ["EURUSD", "GBPUSD", "USDJPY"]  # replace with your ARIA symbols
timeframes = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1]

for symbol in symbols:
    for tf in timeframes:
        # Attempt to fetch a huge number of bars
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 1000000)
        if rates is None:
            print(f"❌ {symbol} TF {tf}: No data or limit reached")
        else:
            print(f"✅ {symbol} TF {tf}: Max bars available = {len(rates)}")

mt5.shutdown()
