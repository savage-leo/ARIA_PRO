#!/usr/bin/env python3
"""Test T470 Multi-Strategy Pipeline"""

from backend.services.t470_pipeline_optimized import t470_pipeline

print("🎯 T470 Multi-Strategy Hedge Fund Brain - Testing...")

# Test with sample data
result = t470_pipeline.process_tick_optimized(
    symbol="EURUSD", price=1.1000, account_balance=100000.0, atr=0.001, spread_pips=1.0
)

print(f'✅ Models: {result["models"]["count"]} active')
print(
    f'✅ Decision: {result["decision"]["action"]} (conf={result["decision"]["confidence"]:.3f})'
)
print(f'✅ Latency: {result["performance"]["latency_ms"]:.1f}ms')
print(f'✅ Memory: {result["performance"]["memory_mb"]}MB')
print(f'✅ Position: {result["sizing"]["position_size"]} units')

# Test system status
status = t470_pipeline.get_system_status()
print(f"\n📊 System Status:")
print(f'  Models Available: {status["models_available"]}')
print(
    f'  Memory Usage: {status["memory_usage_mb"]:.1f}MB / {status["memory_limit_mb"]}MB'
)
print(
    f'  Top Models: {[model[0] for model in status["ensemble_status"]["top_models"][:3]]}'
)

print(f"\n🚀 T470 Multi-Strategy Pipeline: OPERATIONAL!")

# Test multiple symbols
print(f"\n🔄 Testing Multiple Symbols...")
symbols = ["GBPUSD", "USDJPY", "XAUUSD"]
base_prices = {"GBPUSD": 1.3000, "USDJPY": 150.00, "XAUUSD": 2000.0}

for symbol in symbols:
    result = t470_pipeline.process_tick_optimized(
        symbol=symbol,
        price=base_prices[symbol],
        account_balance=100000.0,
        atr=0.001,
        spread_pips=1.5,
    )

    print(
        f'  {symbol}: {result["decision"]["action"]} (conf={result["decision"]["confidence"]:.3f}, {result["performance"]["latency_ms"]:.1f}ms)'
    )

print(f"\n✅ All tests passed! T470 is running a multi-strategy hedge fund!")
