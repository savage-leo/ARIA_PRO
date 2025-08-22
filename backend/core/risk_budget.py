# backend/core/risk_budget.py
"""
Unified RiskBudget engine that consolidates multiple risk signals into a single
risk_units output in [0,1], with optional throttle.

Inputs considered:
- Edge (Kelly fraction from fused probabilities)
- Cost (spread vs ATR)
- Bias engine output (bias_factor, throttle)
- Exposure constraints (boolean ok, optional ratio)
- Volatility conditioning (via RegimeMetrics)

Environment knobs (sane defaults):
- ARIA_RB_COST_K=0.75            # strength of spread/ATR penalty
- ARIA_RB_VOL_LOW=0.95           # multiplier when vol bucket = low
- ARIA_RB_VOL_MED=1.00           # multiplier when vol bucket = medium
- ARIA_RB_VOL_HIGH=0.90          # multiplier when vol bucket = high
- ARIA_RB_EDGE_WEIGHT=1.0        # exponent for Kelly edge shaping
- ARIA_RB_MIN_UNITS=0.0          # floor for non-throttled output
- ARIA_RB_MAX_UNITS=1.0          # cap for output
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os

from backend.core.regime import RegimeMetrics


@dataclass
class RiskBudgetResult:
    risk_units: float
    throttle: bool
    reasons: Dict[str, float]


class RiskBudgetEngine:
    def __init__(self) -> None:
        self.cost_k = float(os.environ.get("ARIA_RB_COST_K", "0.75"))
        self.vol_low = float(os.environ.get("ARIA_RB_VOL_LOW", "0.95"))
        self.vol_med = float(os.environ.get("ARIA_RB_VOL_MED", "1.00"))
        self.vol_high = float(os.environ.get("ARIA_RB_VOL_HIGH", "0.90"))
        self.edge_weight = float(os.environ.get("ARIA_RB_EDGE_WEIGHT", "1.0"))
        self.min_units = float(os.environ.get("ARIA_RB_MIN_UNITS", "0.0"))
        self.max_units = float(os.environ.get("ARIA_RB_MAX_UNITS", "1.0"))

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    def _vol_factor(self, m: Optional[RegimeMetrics]) -> float:
        if m is None:
            return 1.0
        b = str(getattr(m, "vol_bucket", "medium") or "medium").lower()
        if b == "low":
            return self.vol_low
        if b == "high":
            return self.vol_high
        return self.vol_med

    def compute(
        self,
        *,
        symbol: str,
        base_conf: float,
        kelly_edge: float,
        spread: float,
        atr: float,
        metrics: Optional[RegimeMetrics],
        bias_factor: float,
        bias_throttle: bool,
        exposure_ok: bool,
        exposure_ratio: Optional[float] = None,
    ) -> RiskBudgetResult:
        # Throttle from upstream signals
        if bias_throttle or not exposure_ok:
            return RiskBudgetResult(
                risk_units=0.0,
                throttle=True,
                reasons={
                    "bias_throttle": 1.0 if bias_throttle else 0.0,
                    "exposure_ok": 1.0 if exposure_ok else 0.0,
                },
            )

        # Edge shaping (0..1)
        e = self._clamp(kelly_edge, 0.0, 1.0)
        if self.edge_weight != 1.0:
            e = self._clamp(e**self.edge_weight, 0.0, 1.0)

        # Cost penalty via spread/ATR ratio
        spread = float(spread or 0.0)
        atr = float(atr or 0.0)
        if atr <= 0:
            cost_ratio = 1.0  # penalize when ATR unknown
        else:
            cost_ratio = spread / max(atr, 1e-9)
        cost_factor = self._clamp(1.0 - self.cost_k * cost_ratio, 0.0, 1.0)

        # Volatility conditioning
        vfac = self._vol_factor(metrics)

        # Bias factor (clamped)
        bfac = self._clamp(bias_factor, 0.0, 1.5)

        # Exposure soft penalty if ratio provided (0..1 => 0 low, 1 near cap)
        if exposure_ratio is None:
            exfac = 1.0
        else:
            exfac = self._clamp(
                1.0 - 0.5 * self._clamp(exposure_ratio, 0.0, 1.0), 0.0, 1.0
            )

        # Combine multiplicatively off base confidence
        base = self._clamp(base_conf, 0.0, 1.0)
        units = base * e * bfac * vfac * cost_factor * exfac
        units = self._clamp(units, self.min_units, self.max_units)

        reasons = {
            "base_conf": base,
            "edge": e,
            "bias_factor": bfac,
            "vol_factor": vfac,
            "cost_factor": cost_factor,
            "exposure_factor": exfac,
        }
        return RiskBudgetResult(risk_units=units, throttle=False, reasons=reasons)

# Compatibility alias for legacy imports
RiskBudget = RiskBudgetEngine
