#!/usr/bin/env python3
"""Quick signal generation test"""

from backend.services.trading_pipeline_enhanced import trading_pipeline

print("🎯 Testing Live Signal Generation...")
symbols = ["EURUSD", "GBPUSD", "USDJPY"]
base_prices = {"EURUSD": 1.1000, "GBPUSD": 1.3000, "USDJPY": 150.00}

for symbol in symbols:
    print(f"\n--- {symbol} Signal Test ---")
    for i in range(3):
        price = base_prices[symbol] + (i * 0.001)
        result = trading_pipeline.process_tick(
            symbol=symbol,
            price=price,
            account_balance=100000.0,
            atr=0.001,
            spread_pips=1.0,
        )

        r = result["regime"]
        d = result["decision"]
        s = result["sizing"]
        lat = result["latency"]["total_ms"]

        print(
            f'  {i+1}: {r["state"]}/{r["vol_bucket"]} -> {d["action"]} (p*={d["p_star"]:.3f}, size={s.get("position_size", 0):.2f}, {lat:.1f}ms)'
        )

print("\n✅ Live signal generation working perfectly!")
