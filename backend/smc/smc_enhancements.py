# smc_enhancements.py
"""
SMC detectors: Order blocks, Fair Value Gaps (FVG), Liquidity Zones.
Input: list of bars (each bar = dict with open, high, low, close, time)
Return simple list of zones for further scoring.
"""

from typing import List, Dict, Tuple


def detect_order_blocks(
    bars: List[Dict], lookback: int = 50, thresh: float = 0.003
) -> List[Dict]:
    """
    Simple order block detection:
      - find swing highs/lows with subsequent rejection candles
      - returns blocks as dict {side, top, bottom, start, end}
    """
    if len(bars) < 5:
        return []
    blocks = []
    for i in range(2, min(len(bars) - 2, lookback)):
        # find bullish order block (bearish candle then strong bullish engulf)
        prev = bars[-(i + 2)]
        curr = bars[-(i + 1)]
        nxt = bars[-i]
        # bullish block: prev is bearish, curr bullish engulfing prev
        if (
            prev["close"] < prev["open"]
            and curr["close"] > curr["open"]
            and curr["close"] > prev["open"]
        ):
            top = max(prev["open"], curr["close"])
            bottom = min(prev["close"], curr["open"])
            blocks.append(
                {"side": "bull", "top": top, "bottom": bottom, "time": curr["time"]}
            )
        # bearish block symmetric
        if (
            prev["close"] > prev["open"]
            and curr["close"] < curr["open"]
            and curr["close"] < prev["open"]
        ):
            top = max(prev["open"], curr["close"])
            bottom = min(prev["close"], curr["open"])
            blocks.append(
                {"side": "bear", "top": top, "bottom": bottom, "time": curr["time"]}
            )
    return blocks


def detect_fvg(bars: List[Dict]) -> List[Dict]:
    """
    Fair Value Gap: three-bar gap where middle bar leaves a void between wicks.
    Simple approach: scan last N bars and identify FVGs
    """
    fvg = []
    for i in range(2, len(bars)):
        a, b, c = bars[i - 2], bars[i - 1], bars[i]
        # bullish FVG if b low > max(a.close, c.open) (simple)
        if b["low"] > max(a["close"], c["open"]):
            fvg.append(
                {
                    "side": "bull",
                    "top": b["low"],
                    "bottom": min(a["close"], c["open"]),
                    "time": b["time"],
                }
            )
        if b["high"] < min(a["close"], c["open"]):
            fvg.append(
                {
                    "side": "bear",
                    "top": max(a["close"], c["open"]),
                    "bottom": b["high"],
                    "time": b["time"],
                }
            )
    return fvg


def detect_liquidity_zones(bars: List[Dict], window: int = 100) -> List[Dict]:
    """
    Liquidity zones = clusters of wicks (highs/lows) within a sliding window.
    Returns zones with top/bottom and density score.
    """
    highs = [b["high"] for b in bars[-window:]]
    lows = [b["low"] for b in bars[-window:]]
    # pick top 3 maxima of highs and minima of lows as liquidity magnets
    import numpy as np

    h_idx = np.argsort(highs)[-3:]
    l_idx = np.argsort(lows)[:3]
    zones = []
    for idx in h_idx:
        top = highs[idx]
        bottom = top - 0.0005  # small buffer - tune per symbol
        zones.append(
            {
                "side": "sell",
                "top": top,
                "bottom": bottom,
                "density": float(np.sum(np.array(highs) > top - 1e-9)),
            }
        )
    for idx in l_idx:
        bottom = lows[idx]
        top = bottom + 0.0005
        zones.append(
            {
                "side": "buy",
                "top": top,
                "bottom": bottom,
                "density": float(np.sum(np.array(lows) < bottom + 1e-9)),
            }
        )
    return zones
