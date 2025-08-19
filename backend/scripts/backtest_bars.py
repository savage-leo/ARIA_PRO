"""
Backtest script for regime detection and signal fusion.

Usage:
  python backend/scripts/backtest_bars.py --symbol EURUSD --days 30
  python backend/scripts/backtest_bars.py --symbol EURUSD --days 30 --models smc,cnn,xgb,ppo
"""

import os
import sys
import json
import math
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Ensure project root is on sys.path if run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.regime import RegimeDetector, Regime
from backend.core.fusion import SignalFusion
from backend.core.calibration import ScoreCalibrator
from backend.services.models_interface import score_and_calibrate

logger = logging.getLogger("ARIA.Backtest")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def fetch_historical_data(symbol: str, days: int) -> List[Dict]:
    """Fetch historical OHLCV data for backtesting.

    Returns list of bars: [{"ts", "o", "h", "l", "c", "v", "symbol"}]
    """
    bars: List[Dict] = []

    # Try MT5 first
    try:
        import MetaTrader5 as mt5

        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

        # Calculate number of bars needed (assuming 1-minute data)
        timeframe = mt5.TIMEFRAME_M1
        count = days * 24 * 60  # minutes in N days

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        try:
            mt5.shutdown()
        except Exception:
            pass

        if rates is not None and len(rates) > 0:
            try:
                items = sorted(rates, key=lambda r: r["time"])
            except Exception:
                items = list(rates)

            for bar in items:
                try:
                    o = float(bar["open"])
                    h = float(bar["high"])
                    l = float(bar["low"])
                    c = float(bar["close"])

                    # tick_volume preferred; fallback to real_volume; else 0
                    v = 0.0
                    try:
                        v = float(bar["tick_volume"])
                    except Exception:
                        try:
                            v = float(bar["real_volume"])
                        except Exception:
                            v = 0.0
                except Exception:
                    try:
                        o = float(bar.open)
                        h = float(bar.high)
                        l = float(bar.low)
                        c = float(bar.close)
                        v = float(getattr(bar, "tick_volume", 0.0))
                    except Exception:
                        continue

                bars.append(
                    {
                        "ts": float(bar["time"]),
                        "o": o,
                        "h": h,
                        "l": l,
                        "c": c,
                        "v": v,
                        "symbol": symbol,
                    }
                )
    except Exception as e:
        logger.warning(f"MT5 data fetch failed: {e}")

    # Fallback to synthetic data if needed
    if not bars:
        logger.info("Generating synthetic data for backtest")
        import random

        # Generate synthetic price series
        price = 1.1000
        ts = datetime.now().timestamp() - days * 24 * 60 * 60

        for i in range(days * 24 * 60):  # 1-minute bars
            # Random walk with drift
            drift = random.uniform(-0.0001, 0.0001)
            price = price * (1.0 + drift)

            # Add some volatility
            vol = random.uniform(0.00005, 0.0002)
            high = price * (1.0 + vol)
            low = price * (1.0 - vol)

            bars.append(
                {
                    "ts": ts + i * 60,
                    "o": price,
                    "h": high,
                    "l": low,
                    "c": price,
                    "v": random.uniform(1000, 10000),
                    "symbol": symbol,
                }
            )

    logger.info(f"Fetched {len(bars)} bars for backtesting")
    return bars


def compute_returns(bars: List[Dict], period: int = 1) -> List[float]:
    """Compute log returns from bars."""
    if len(bars) < period + 1:
        return []

    closes = [float(b["c"]) for b in bars]
    returns = []

    for i in range(period, len(closes)):
        if closes[i - period] > 0 and closes[i] > 0:
            ret = math.log(closes[i] / closes[i - period])
            returns.append(ret)
        else:
            returns.append(0.0)

    return returns


def compute_sharpe(returns: List[float], annualization: float = 252 * 24 * 60) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    std_return = math.sqrt(
        sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    )

    if std_return == 0:
        return 0.0

    return (mean_return / std_return) * math.sqrt(annualization)


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """Compute maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return max_dd


def backtest_fusion(
    symbol: str, bars: List[Dict], models: Optional[List[str]] = None
) -> Dict[str, object]:
    """Run backtest using regime detection and signal fusion."""
    if not bars:
        return {}

    # Initialize adapters
    if models:
        os.environ["ACTIVE_MODELS"] = ",".join(models)

    adapters = build_default_adapters()
    if not adapters:
        logger.error("No adapters available for backtest")
        return {}

    # Load adapters
    for name, adapter in adapters.items():
        try:
            adapter.load()
        except Exception as e:
            logger.warning(f"Failed to load adapter {name}: {e}")

    # Initialize fusion
    calibrator = ScoreCalibrator.load_default()
    fusion = SignalFusion(signal_keys=list(adapters.keys()), calibrator=calibrator)

    # Backtest parameters
    window_size = 120  # Number of bars for feature computation

    # Metrics tracking
    trades = []
    equity_curve = [1.0]  # Start with $1
    regime_counts = defaultdict(int)
    regime_transitions = defaultdict(int)

    prev_regime = None

    # Process bars in rolling windows
    for i in range(window_size, len(bars)):
        # Current window of bars
        window = bars[i - window_size : i]
        current_bar = bars[i]

        # Regime detection
        try:
            regime, metrics = RegimeDetector.detect(window)
            regime_counts[regime.value] += 1

            if prev_regime and prev_regime != regime:
                regime_transitions[f"{prev_regime.value}->{regime.value}"] += 1
            prev_regime = regime
        except Exception as e:
            logger.warning(f"Regime detection failed at bar {i}: {e}")
            regime = Regime.RANGE

        # Generate signals
        raw_signals = {}
        market_feats = {}

        # Build features for this window
        closes = [float(b["c"]) for b in window]
        feats = {"series": closes}

        # Add OHLCV if available
        if window:
            ohlcv = [
                [
                    float(b["o"]),
                    float(b["h"]),
                    float(b["l"]),
                    float(b["c"]),
                    float(b["v"]),
                ]
                for b in window
            ]
            feats["ohlcv"] = ohlcv

        # Generate signals from adapters
        for name, adapter in adapters.items():
            try:
                score = float(adapter.predict(feats))
                raw_signals[name] = max(-1.0, min(1.0, score))
            except Exception as e:
                logger.debug(f"Signal generation failed for {name} at bar {i}: {e}")
                raw_signals[name] = 0.0

        # Compute market features
        if len(window) >= 15:
            # ATR computation
            trs = []
            prev_close = float(window[0]["c"])
            for bar in window[-14:]:  # Last 14 bars
                h = float(bar["h"])
                l = float(bar["l"])
                c = float(bar["c"])
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                trs.append(tr)
                prev_close = c

            if trs:
                atr = sum(trs) / len(trs)
                market_feats["atr"] = atr

            # Spread approximation (as percentage of price)
            current_price = float(current_bar["c"])
            if current_price > 0:
                spread_pct = (0.0001 / current_price) * 100  # Approx 1 pip
                market_feats["spread_pct"] = spread_pct

        # Fusion
        try:
            context = {
                "timestamp": current_bar["ts"],
                "atr_pct": (
                    market_feats.get("atr", 0.0) / current_price * 100
                    if current_price > 0
                    else 0.0
                ),
                "spread_pct": market_feats.get("spread_pct", 0.0001),
            }

            fusion_result = fusion.fuse(raw_signals, regime, context)
        except Exception as e:
            logger.warning(f"Fusion failed at bar {i}: {e}")
            fusion_result = {
                "p_long": 0.5,
                "p_short": 0.5,
                "direction": "buy",
                "margin": 0.0,
            }

        # Trading logic (simplified)
        p_long = float(fusion_result.get("p_long", 0.5))
        p_short = float(fusion_result.get("p_short", 0.5))
        direction = str(fusion_result.get("direction", "buy"))
        margin = float(fusion_result.get("margin", abs(p_long - p_short)))

        # Simple threshold-based entry
        entry_threshold = 0.55
        exit_threshold = 0.45

        # Get next bar for return calculation
        if i + 1 < len(bars):
            next_bar = bars[i + 1]
            ret = math.log(float(next_bar["c"]) / float(current_bar["c"]))

            # Simple trading logic
            position = 0  # 1 for long, -1 for short, 0 for flat

            if direction == "buy" and p_long >= entry_threshold:
                position = 1
            elif direction == "sell" and p_short >= entry_threshold:
                position = -1

            # Calculate PnL
            pnl = position * ret

            # Update equity curve
            new_equity = equity_curve[-1] * math.exp(pnl)
            equity_curve.append(new_equity)

            # Record trade
            trades.append(
                {
                    "timestamp": current_bar["ts"],
                    "regime": regime.value,
                    "direction": direction,
                    "p_long": p_long,
                    "p_short": p_short,
                    "margin": margin,
                    "position": position,
                    "return": ret,
                    "pnl": pnl,
                    "equity": new_equity,
                }
            )

    # Compute final metrics
    if len(equity_curve) < 2:
        return {}

    # Total return
    total_return = (equity_curve[-1] / equity_curve[0]) - 1.0

    # Returns for Sharpe calculation
    equity_returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            ret = math.log(equity_curve[i] / equity_curve[i - 1])
            equity_returns.append(ret)

    # Sharpe ratio
    sharpe = compute_sharpe(equity_returns)

    # Max drawdown
    max_dd = compute_max_drawdown(equity_curve)

    # Win rate
    winning_trades = [t for t in trades if t["pnl"] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0.0

    # Average win/loss
    avg_win = (
        sum(t["pnl"] for t in winning_trades) / len(winning_trades)
        if winning_trades
        else 0.0
    )

    losing_trades = [t for t in trades if t["pnl"] < 0]
    avg_loss = (
        sum(t["pnl"] for t in losing_trades) / len(losing_trades)
        if losing_trades
        else 0.0
    )

    # Profit factor
    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Number of trades
    num_trades = len(trades)

    return {
        "symbol": symbol,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "regime_counts": dict(regime_counts),
        "regime_transitions": dict(regime_transitions),
        "final_equity": equity_curve[-1],
        "equity_curve": equity_curve,
        "trades": trades,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backtest regime detection and signal fusion"
    )
    parser.add_argument("--symbol", default="EURUSD", help="FX symbol, e.g., EURUSD")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to backtest"
    )
    parser.add_argument(
        "--models", default=None, help="Comma-separated models to use (smc,cnn,xgb,ppo)"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()

    models = args.models.split(",") if args.models else None

    # Fetch data
    logger.info(f"Fetching {args.days} days of data for {args.symbol}")
    bars = fetch_historical_data(args.symbol, args.days)

    if not bars:
        logger.error("No data available for backtesting")
        return

    # Run backtest
    logger.info("Running backtest...")
    results = backtest_fusion(args.symbol, bars, models)

    if not results:
        logger.error("Backtest failed")
        return

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("=== ARIA Backtest Results ===")
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
        print()
        print("Regime Transitions:")
        for transition, count in results["regime_transitions"].items():
            print(f"  {transition}: {count}")


if __name__ == "__main__":
    main()
