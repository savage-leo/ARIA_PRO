"""
Bias Engine — Institutional Secret Layer
- Computes a trade_bias_factor ∈ [0.5, 2.0] using regime, confluence and microstructure.
- Zero external deps. CPU-light. Live-ready.
- No mock data: requires real bars + idea + (optional) live context.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Deque, Dict, Any, Optional


# ---------- helpers ----------
def _rolling_atr(bars: Deque[Dict[str, float]], length: int = 14) -> Optional[float]:
    if len(bars) < length + 1:
        return None
    trs = []
    for i in range(-length, 0):
        h = bars[i]["h"]
        l = bars[i]["l"]
        pc = bars[i - 1]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return (sum(trs) / len(trs)) if trs else None


def _atr_slope(bars: Deque[Dict[str, float]], win: int = 10) -> float:
    if len(bars) < win + 14:
        return 0.0
    vals = []
    # compute ATR value at each step for slope
    for i in range(-win - 1, -1):
        sub = list(bars)[:i]  # growing slice
        if len(sub) < 15:  # ensure 14 + 1
            continue
        atr_val = _rolling_atr(sub, 14) or 0.0
        vals.append(atr_val)
    if len(vals) < 2:
        return 0.0
    return (vals[-1] - vals[0]) / max(len(vals) - 1, 1)


def _session_weight(now_s: Optional[float] = None) -> float:
    """
    Light time-of-day weight (UTC). Favors London/NY overlap.
    0.8 off-hours, 1.0 normal, 1.2 overlap.
    """
    ts = now_s or time.time()
    hour = int((ts // 3600) % 24)
    # rough UTC session windows (tune to broker TZ if needed)
    if 12 <= hour <= 16:  # London-NY overlap
        return 1.2
    if 7 <= hour <= 19:  # active sessions
        return 1.0
    return 0.8


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------- types ----------
@dataclass
class BiasResult:
    bias_factor: float  # multiply risk_pct by this, [0.5..2.0]
    risk_multiplier: float  # alias for readability == bias_factor
    throttle: bool  # true => skip execution (under-threshold)
    score: float  # 0..1
    reasons: Dict[str, float]  # per-feature contributions (for audit)


# ---------- engine ----------
class BiasEngine:
    """
    Combines SMC idea + bar regime + (optional) orderflow context
    to yield a bias factor and pass/skip decision.
    """

    def __init__(
        self,
        min_score: float = 0.55,
        hard_throttle_below: float = 0.50,
        max_bias: float = 2.0,
        min_bias: float = 0.5,
    ):
        self.min_score = min_score
        self.hard_throttle_below = hard_throttle_below
        self.max_bias = max_bias
        self.min_bias = min_bias

    def compute(
        self,
        idea: Any,  # TradeIdea (has bias, confidence, entry/stop/tp + smc structures)
        bars: Deque[Dict[str, float]],  # live bars (o,h,l,c,v,ts)
        market_ctx: Optional[
            Dict[str, float]
        ] = None,  # optional: {'spread':..., 'of_imbalance':..., 'slippage':...}
    ) -> BiasResult:
        market_ctx = market_ctx or {}
        reasons: Dict[str, float] = {}

        # 1) Base from idea confidence
        base = _clamp(float(idea.confidence), 0.0, 1.0)
        reasons["idea_confidence"] = base

        # 2) Regime via ATR slope (trend = good; flat = neutral)
        slope = _atr_slope(bars, 10)
        # convert slope to 0..1 by squashing small magnitudes
        slope_score = _clamp(
            0.5 + 4.0 * slope, 0.0, 1.0
        )  # tuned small; FX ATR moves are tiny
        reasons["atr_slope"] = slope_score

        # 3) Confluence: number/strength of structures aligned with idea bias
        confluence = 0.0
        count = 0
        for ob in idea.order_blocks or []:
            if ob.type == idea.bias:
                confluence += _clamp(ob.strength, 0.0, 1.0)
                count += 1
        for fvg in idea.fair_value_gaps or []:
            if fvg.direction == idea.bias:
                confluence += _clamp(fvg.strength, 0.0, 1.0)
                count += 1
        for lz in idea.liquidity_zones or []:
            if (idea.bias == "bullish" and lz.type in ("equal_low", "swing_low")) or (
                idea.bias == "bearish" and lz.type in ("equal_high", "swing_high")
            ):
                confluence += _clamp(lz.strength, 0.0, 1.0)
                count += 1
        conf_score = _clamp(confluence / max(count, 1), 0.0, 1.0)
        reasons["confluence"] = conf_score

        # 4) Microstructure penalties (if provided)
        spread = float(market_ctx.get("spread", 0.0) or 0.0)
        slippage = float(market_ctx.get("slippage", 0.0) or 0.0)
        of_imb = float(
            market_ctx.get("of_imbalance", 0.0) or 0.0
        )  # positive = buy pressure
        # Spread/slippage scale: small penalty; imbalance slight boost if aligned
        # Normalize by ATR to get dimensionless penalties
        atr = _rolling_atr(bars, 14) or 0.0
        spr_pen = (
            0.0 if atr <= 0 else _clamp(1.0 - (spread / (atr * 0.2)), 0.0, 1.0)
        )  # spread < 0.2*ATR ≈ fine
        slp_pen = (
            0.0 if atr <= 0 else _clamp(1.0 - (slippage / (atr * 0.1)), 0.0, 1.0)
        )  # slippage < 0.1*ATR ≈ fine
        reasons["spread_quality"] = spr_pen
        reasons["slippage_quality"] = slp_pen

        imb_adj = 0.5 + 0.5 * _clamp(abs(of_imb), 0.0, 1.0)
        if idea.bias == "bullish" and of_imb > 0:
            imb_score = imb_adj
        elif idea.bias == "bearish" and of_imb < 0:
            imb_score = imb_adj
        else:
            imb_score = 0.5  # neutral/misaligned
        reasons["orderflow_alignment"] = _clamp(imb_score, 0.0, 1.0)

        # 5) Session weight
        sess_w = _session_weight()
        reasons["session_weight"] = sess_w  # for audit

        # 6) Aggregate score (weights tuned conservatively)
        score = (
            0.35 * base
            + 0.20 * conf_score
            + 0.15 * slope_score
            + 0.10 * reasons["orderflow_alignment"]
            + 0.10 * spr_pen
            + 0.10 * slp_pen
        )
        # Session multiplier applied at the bias stage rather than raw score
        score = _clamp(score, 0.0, 1.0)

        # 7) Bias factor mapping (convex to reward high score)
        #    factor in [min_bias..max_bias], steep after 0.70
        if score < 0.70:
            raw_factor = self.min_bias + (score / 0.70) * (1.0 - self.min_bias)
        else:
            raw_factor = 1.0 + ((score - 0.70) / 0.30) * (self.max_bias - 1.0)

        bias_factor = _clamp(raw_factor * (sess_w / 1.0), self.min_bias, self.max_bias)
        throttle = (
            score < self.hard_throttle_below or bias_factor < 0.75
        )  # hard skip if too weak

        return BiasResult(
            bias_factor=bias_factor,
            risk_multiplier=bias_factor,
            throttle=throttle or (score < self.min_score),
            score=score,
            reasons=reasons,
        )
