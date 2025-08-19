from __future__ import annotations
import MetaTrader5 as mt5
from datetime import datetime, timedelta

symbol = "XAUUSD"

print("Initializing MT5...")
if not mt5.initialize():
    print("initialize failed:", mt5.last_error())
    raise SystemExit(1)

try:
    ti = mt5.terminal_info()
    ai = mt5.account_info()
    print("terminal_info:", ti)
    print("account_info:", ai)
    print("connected:", getattr(ti, "connected", None))
    print("community_connected:", getattr(ti, "community_connected", None))
    print("last_error:", mt5.last_error())

    si = mt5.symbol_info(symbol)
    print("symbol_info:", si)
    if not si or not si.visible:
        print("Selecting symbol to Market Watch:", symbol)
        print("symbol_select result:", mt5.symbol_select(symbol, True))
        si = mt5.symbol_info(symbol)
        print("post-select symbol_info:", si)

    print("Attempt copy_rates_from_pos 1000...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)
    print(
        "rates len:",
        (len(rates) if rates is not None else None),
        "last_error:",
        mt5.last_error(),
    )

    print("Attempt copy_rates_from (dt=now, count=1000)...")
    rates2 = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, datetime.utcnow(), 1000)
    print(
        "rates2 len:",
        (len(rates2) if rates2 is not None else None),
        "last_error:",
        mt5.last_error(),
    )

    print("Attempt copy_rates_range last 2 days...")
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=2)
    rates3 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_dt, end_dt)
    print(
        "rates3 len:",
        (len(rates3) if rates3 is not None else None),
        "last_error:",
        mt5.last_error(),
    )
finally:
    mt5.shutdown()
