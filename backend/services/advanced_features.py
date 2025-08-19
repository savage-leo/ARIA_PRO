# advanced_features.py
"""
Trap detection, bias engine, portfolio manager skeleton, correlation analysis.
Add real metrics/thresholds based on historical backtest.
"""

import numpy as np
from typing import List, Dict


def detect_trap(bars: List[Dict], lookback: int = 50) -> bool:
    """
    Detect 'trap' pattern: false breakout followed by quick reversal and volume spike.
    Simple heuristic: last candle breaks above recent high but close back inside previous range.
    """
    if len(bars) < lookback + 3:
        return False
    recent = bars[-(lookback + 3) :]
    highs = [b["high"] for b in recent[:-1]]
    last = recent[-1]
    if last["high"] > max(highs) and last["close"] < np.percentile(highs, 80):
        # trap candidate
        return True
    return False


def bias_engine(model_scores: Dict[str, float], macro: float, vol: float) -> float:
    """
    Combine model scores and macro signal into a single bias factor (-1..1).
    macro: macro sentiment [-1..1], vol: current volatility (ATR)
    """
    # conservative: weight macro when vol low; weight models when vol high
    vol_factor = np.tanh(vol * 10.0)
    weights = {
        "models": 0.7 * vol_factor + 0.3 * (1 - vol_factor),
        "macro": 0.3 * (1 - vol_factor) + 0.7 * vol_factor,
    }
    model_mean = np.mean(list(model_scores.values())) if model_scores else 0.0
    bias = weights["models"] * model_mean + weights["macro"] * macro
    return float(np.tanh(bias))


def simple_portfolio_manager(positions: List[Dict], max_exposure_pct: float = 0.2):
    """
    Positions: list of {symbol, notional, direction}
    Enforce max exposure per-correlated-cluster (placeholder)
    """
    total_notional = sum(abs(p["notional"]) for p in positions)
    if total_notional == 0:
        return positions
    for p in positions:
        if abs(p["notional"]) / total_notional > max_exposure_pct:
            # trim
            p["notional"] = p["notional"] * (
                max_exposure_pct * total_notional / abs(p["notional"])
            )
    return positions


def correlation_matrix(price_series: Dict[str, List[float]]):
    syms = list(price_series.keys())
    arr = np.vstack([np.array(price_series[s]) for s in syms])
    corr = np.corrcoef(arr)
    return {"symbols": syms, "corr": corr.tolist()}
