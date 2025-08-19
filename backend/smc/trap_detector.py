# backend/smc/trap_detector.py
"""
Trap Detector v1
Heuristics-based liquidity-trap detection:
 - looks for wick+volume anomalies, delta divergence, footprint-like spikes (if tick volume available)
 - provides a trap_score [0..1] and explanation array

This purposely does not assume DOM access; it operates on OHLCV bars + tick volume if available.
"""

from typing import List, Dict, Any
import logging
import math

logger = logging.getLogger("aria.smc.trap")


def compute_avg(items):
    return sum(items) / max(1, len(items))


def detect_trap(
    hist: List[Dict[str, Any]], recent_ticks: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    hist: list of bars (dicts) with keys ts,o,h,l,c,v,symbol
    recent_ticks: optional list of ticks/deltas for footprint-like heuristics
    Returns: {trap_score: float, direction: 'buy'|'sell'|None, explain: [str], metrics: {...}}
    """

    if not hist or len(hist) < 6:
        return {
            "trap_score": 0.0,
            "direction": None,
            "explain": ["insufficient history"],
            "metrics": {},
        }

    # Use last bar + window
    last = hist[-1]
    prev = hist[-8:-1] if len(hist) >= 9 else hist[:-1]

    # Basic metrics
    ranges = [b["h"] - b["l"] for b in prev if b["h"] - b["l"] > 0]
    avg_range = compute_avg(ranges) if ranges else 0.0
    last_range = last["h"] - last["l"] if last["h"] - last["l"] > 0 else 0.0

    wick_top = last["h"] - max(last["o"], last["c"])
    wick_bottom = min(last["o"], last["c"]) - last["l"]

    vols = [b.get("v", 0.0) for b in prev]
    avg_vol = compute_avg(vols) if vols else 0.0
    vol_surge = last.get("v", 0) > max(1.0, avg_vol * 1.8)

    score = 0.0
    explain = []
    direction = None
    metrics = {
        "avg_range": avg_range,
        "last_range": last_range,
        "wick_top": wick_top,
        "wick_bottom": wick_bottom,
        "avg_vol": avg_vol,
        "last_vol": last.get("v", 0),
    }

    # Heuristic 1: large wicked candle + vol surge = potential sweep
    if vol_surge and last_range > (1.2 * avg_range if avg_range > 0 else 0):
        if wick_top > (0.5 * last_range):
            # big upper wick + vol surge => buy stops swept (liquidity above) then drop -> trap for shorts
            score += 0.45
            direction = "buy"  # sweep likely targeted buy stops -> we expect reversal buy afterwards
            explain.append("Upper wick + vol surge -> probable buy-stop sweep")
        if wick_bottom > (0.5 * last_range):
            score += 0.45
            direction = "sell"
            explain.append("Lower wick + vol surge -> probable sell-stop sweep")

    # Heuristic 2: price rejection (close near one end) on spike
    close_dist_top = last["h"] - last["c"]
    close_dist_bottom = last["c"] - last["l"]
    if close_dist_top < (0.2 * last_range) and vol_surge:
        # close near top with huge wick bottom? bias sell-side exhaustion
        explain.append("Close near top after surge -> potential exhaustion")
        score += 0.05
    if close_dist_bottom < (0.2 * last_range) and vol_surge:
        explain.append("Close near bottom after surge -> potential exhaustion")
        score += 0.05

    # Heuristic 3: delta/divergence if recent_ticks provided (tick buys vs sells)
    if recent_ticks:
        # recent_ticks: list of {'price', 'size', 'side'} where side in ('buy','sell')
        buy_volume = sum(t["size"] for t in recent_ticks if t.get("side") == "buy")
        sell_volume = sum(t["size"] for t in recent_ticks if t.get("side") == "sell")
        metrics["tick_buy_vol"] = buy_volume
        metrics["tick_sell_vol"] = sell_volume
        if buy_volume > (sell_volume * 1.8) and (last["c"] < last["o"]):
            # buying volume but price closed down -> hidden buying (delta divergence)
            score += 0.2
            explain.append(
                "Delta divergence: buy-heavy ticks but down close -> hidden buy liquidity"
            )
            direction = "buy"
        if sell_volume > (buy_volume * 1.8) and (last["c"] > last["o"]):
            score += 0.2
            explain.append(
                "Delta divergence: sell-heavy ticks but up close -> hidden sell liquidity"
            )
            direction = "sell"

    # Heuristic 4: SMC context — check if last bar intersects SMC zones (caller should provide)
    # (We keep placeholder logic; caller should combine SMC signals with trap result.)
    # Final normalization
    trap_score = min(0.99, max(0.0, score))
    # Additional boost if both wick+vol surge and delta divergence present
    if vol_surge and recent_ticks and trap_score > 0.3:
        trap_score = min(0.99, trap_score + 0.15)
        explain.append("Combined vol spike + delta divergence -> boost")

    return {
        "trap_score": round(trap_score, 3),
        "direction": direction,
        "explain": explain,
        "metrics": metrics,
    }
