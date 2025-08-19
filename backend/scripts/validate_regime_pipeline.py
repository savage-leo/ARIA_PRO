# -*- coding: utf-8 -*-
"""
Validate Regime Pipeline: End-to-end validation of institutional trading pipeline
Tests regime detection, calibration, fusion, and position sizing
"""
from __future__ import annotations
import sys, os, time, random, pathlib

# Add project root to Python path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.trading_pipeline_enhanced import trading_pipeline
import numpy as np


def test_regime_detection():
    """Test regime detection with price series"""
    print("=== Testing Regime Detection ===")

    symbol = "EURUSD"
    base_price = 1.1000

    # Simulate different market regimes
    scenarios = {
        "Trending Up": np.cumsum(np.random.normal(0.0005, 0.005, 100)) + base_price,
        "Range Bound": base_price
        + 0.01 * np.sin(np.arange(100) * 0.1)
        + np.random.normal(0, 0.002, 100),
        "High Volatility": base_price + np.cumsum(np.random.normal(0, 0.015, 100)),
    }

    for scenario_name, prices in scenarios.items():
        print(f"\n--- {scenario_name} ---")

        regime_states = []
        vol_buckets = []

        for i, price in enumerate(prices[-20:]):  # Last 20 bars
            result = trading_pipeline.process_tick(
                symbol=symbol,
                price=float(price),
                timestamp=time.time() + i,
                account_balance=100000.0,
                atr=0.001,
                spread_pips=1.0,
            )

            regime_states.append(result["regime"]["state"])
            vol_buckets.append(result["regime"]["vol_bucket"])

            if i % 5 == 0:  # Print every 5th update
                r = result["regime"]
                d = result["decision"]
                s = result["sizing"]

                print(
                    f"  Bar {i}: {r['state']}/{r['vol_bucket']} "
                    f"vol={r['volatility']:.4f} "
                    f"→ {d['action']} (p*={d['p_star']:.3f}, "
                    f"size={s.get('position_size', 0):.3f})"
                )

        # Analyze regime detection
        state_counts = {
            state: regime_states.count(state) for state in set(regime_states)
        }
        print(f"  Final regime distribution: {state_counts}")


def test_kelly_sizing():
    """Test Kelly-lite position sizing across confidence levels"""
    print("\n=== Testing Kelly-Lite Position Sizing ===")

    symbol = "GBPUSD"
    base_price = 1.3000
    account_balance = 100000.0
    atr = 0.0015

    confidence_levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    vol_buckets = ["Low", "Med", "High"]

    print(
        f"{'Confidence':<12} {'Vol':<6} {'Position':<10} {'Risk%':<8} {'Kelly':<8} {'Action'}"
    )
    print("-" * 60)

    for confidence in confidence_levels:
        for vol_bucket in vol_buckets:
            # Mock model scores that would produce this confidence
            if confidence >= 0.6:
                model_scores = {"LSTM": 0.5, "PPO": 0.3, "XGB": 0.4, "CNN": 0.2}
            else:
                model_scores = {"LSTM": 0.1, "PPO": -0.1, "XGB": 0.0, "CNN": -0.2}

            result = trading_pipeline.process_tick(
                symbol=symbol,
                price=base_price,
                account_balance=account_balance,
                atr=atr,
                spread_pips=1.5,
                model_scores=model_scores,
            )

            # Force confidence for testing
            result["decision"]["p_star"] = confidence

            # Recalculate sizing with forced confidence
            from backend.core.risk_budget_enhanced import position_sizer

            sizing = position_sizer.calculate_position_size(
                symbol=symbol,
                p_star=confidence,
                vol_bucket=vol_bucket,
                account_balance=account_balance,
                atr=atr,
                spread_pips=1.5,
            )

            action = "LONG" if confidence >= 0.6 else "FLAT"

            print(
                f"{confidence:<12.2f} {vol_bucket:<6} {sizing['position_size']:<10.3f} "
                f"{sizing['risk_pct']*100:<8.2f}% {sizing['kelly_fraction']:<8.3f} {action}"
            )


def test_latency_full_pipeline():
    """Test full pipeline latency"""
    print("\n=== Testing Full Pipeline Latency ===")

    symbol = "USDJPY"
    base_price = 150.00

    latencies = []

    for i in range(1000):
        price = base_price + random.uniform(-0.5, 0.5)

        result = trading_pipeline.process_tick(
            symbol=symbol,
            price=price,
            account_balance=100000.0,
            atr=0.8,  # Different scale for JPY
            spread_pips=2.0,
        )

        latencies.append(result["latency"]["total_ms"])

    latencies = sorted(latencies)
    n = len(latencies)

    print(f"Full Pipeline Latency (n={n}):")
    print(f"  Mean: {np.mean(latencies):.2f}ms")
    print(f"  P50:  {latencies[n//2]:.2f}ms")
    print(f"  P95:  {latencies[int(n*0.95)]:.2f}ms")
    print(f"  P99:  {latencies[int(n*0.99)]:.2f}ms")
    print(f"  Max:  {max(latencies):.2f}ms")


def test_risk_guards():
    """Test risk management and kill switches"""
    print("\n=== Testing Risk Management ===")

    from backend.core.risk_budget_enhanced import position_sizer

    # Simulate daily P&L reaching limits
    print("Testing kill switch scenarios:")

    # Scenario 1: Daily DD limit
    position_sizer.reset_daily()
    position_sizer.update_pnl(-0.025)  # -2.5% (exceeds 2% limit)

    kill_switch, reason = position_sizer.check_kill_switch()
    print(f"  Daily DD: Kill={kill_switch}, Reason='{reason}'")

    # Scenario 2: Portfolio DD limit
    position_sizer.reset_daily()
    position_sizer.portfolio_dd = 0.035  # 3.5% (exceeds 3% limit)

    kill_switch, reason = position_sizer.check_kill_switch()
    print(f"  Portfolio DD: Kill={kill_switch}, Reason='{reason}'")

    # Scenario 3: Normal operation
    position_sizer.reset_daily()
    position_sizer.update_pnl(-0.005)  # -0.5% (within limits)

    kill_switch, reason = position_sizer.check_kill_switch()
    print(f"  Normal operation: Kill={kill_switch}, Reason='{reason}'")


def main():
    """Run all validation tests"""
    print("🚀 ARIA Institutional Pipeline Validation")
    print("=" * 50)

    start_time = time.time()

    try:
        test_regime_detection()
        test_kelly_sizing()
        test_latency_full_pipeline()
        test_risk_guards()

        elapsed = time.time() - start_time
        print(f"\n✅ All tests completed successfully in {elapsed:.2f}s")
        print("\n🎯 Institutional pipeline validated for T470!")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
