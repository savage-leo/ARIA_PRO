# -*- coding: utf-8 -*-
"""
Risk Budget Enhanced: Kelly-lite sizing with p_star integration and volatility caps
CPU-optimized for institutional-grade position sizing
"""
from __future__ import annotations
import os, math, time, pathlib, threading
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class RiskLimits:
    """Risk limits configuration"""

    min_units: float = 0.01
    max_units: float = 2.0
    max_daily_dd: float = 0.02  # 2%
    max_portfolio_dd: float = 0.03  # 3%
    var_cap: float = 0.025  # 2.5% 1-day VAR
    kelly_cap: float = 0.25  # Cap Kelly at 25%
    base_risk_per_trade: float = 0.005  # 0.5% base risk


class KellyLiteCalculator:
    """Kelly-lite position sizing with edge/cost estimation"""

    def __init__(self):
        env = os.environ
        self.edge_decay = float(env.get("ARIA_KELLY_EDGE_DECAY", 0.95))
        self.cost_decay = float(env.get("ARIA_KELLY_COST_DECAY", 0.9))
        self.min_edge = float(env.get("ARIA_KELLY_MIN_EDGE", 0.001))
        self.max_cost = float(env.get("ARIA_KELLY_MAX_COST", 0.02))

    def estimate_edge(self, p_star: float, expected_rr: float = 1.2) -> float:
        """Estimate edge from confidence and expected risk/reward"""
        # Convert p_star to win probability (symmetric)
        win_prob = p_star

        # Edge = win_prob * avg_win - lose_prob * avg_loss
        # Assuming symmetric: avg_win = expected_rr, avg_loss = 1.0
        edge = win_prob * expected_rr - (1.0 - win_prob) * 1.0
        return max(self.min_edge, edge)

    def estimate_cost(self, vol_bucket: str, spread_pips: float = 1.0) -> float:
        """Estimate transaction cost from volatility and spread"""
        base_cost = {
            "Low": 0.002,  # 0.2%
            "Med": 0.005,  # 0.5%
            "High": 0.010,  # 1.0%
        }.get(vol_bucket, 0.005)

        # Add spread cost (rough approximation)
        spread_cost = spread_pips * 0.001  # 0.1% per pip

        total_cost = base_cost + spread_cost
        return min(self.max_cost, total_cost)

    def kelly_fraction(self, edge: float, cost: float, variance: float = 1.0) -> float:
        """Calculate Kelly fraction with cost adjustment"""
        if edge <= cost:
            return 0.0  # No positive expectancy

        net_edge = edge - cost

        # Kelly formula: f* = edge / variance
        # For binary outcomes with variance adjustment
        kelly_f = net_edge / max(variance, 0.1)

        # Cap at reasonable level
        return max(0.0, min(0.25, kelly_f))


class PositionSizer:
    """Main position sizing engine with multi-factor risk management"""

    def __init__(self):
        env = os.environ

        # Risk limits from environment
        self.limits = RiskLimits(
            min_units=float(env.get("ARIA_RB_MIN_UNITS", 0.01)),
            max_units=float(env.get("ARIA_RB_MAX_UNITS", 2.0)),
            max_daily_dd=float(env.get("ARIA_RB_DAILY_DD_CAP", 0.02)),
            max_portfolio_dd=float(env.get("ARIA_RB_PORTFOLIO_DD_CAP", 0.03)),
            var_cap=float(env.get("ARIA_RB_VAR_CAP", 0.025)),
            kelly_cap=float(env.get("ARIA_RB_KELLY_CAP", 0.25)),
            base_risk_per_trade=float(env.get("ARIA_RB_BASE_RISK", 0.005)),
        )

        # Components
        self.kelly_calc = KellyLiteCalculator()

        # State tracking
        self.daily_pnl = 0.0
        self.portfolio_dd = 0.0
        self.current_var = 0.0
        self.last_reset = time.time()

        self.lock = threading.Lock()

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of trading day)"""
        with self.lock:
            self.daily_pnl = 0.0
            self.last_reset = time.time()

    def update_pnl(self, pnl_change: float) -> None:
        """Update daily P&L tracking"""
        with self.lock:
            self.daily_pnl += pnl_change

            # Update portfolio drawdown (simplified)
            if self.daily_pnl < 0:
                self.portfolio_dd = max(self.portfolio_dd, abs(self.daily_pnl))

    def check_kill_switch(self) -> Tuple[bool, str]:
        """Check if kill switch should be triggered"""
        with self.lock:
            # Daily DD check
            if abs(self.daily_pnl) >= self.limits.max_daily_dd:
                return True, f"Daily DD limit breached: {self.daily_pnl:.3f}"

            # Portfolio DD check
            if self.portfolio_dd >= self.limits.max_portfolio_dd:
                return True, f"Portfolio DD limit breached: {self.portfolio_dd:.3f}"

            # VAR check
            if self.current_var >= self.limits.var_cap:
                return True, f"VAR limit breached: {self.current_var:.3f}"

            return False, ""

    def calculate_position_size(
        self,
        symbol: str,
        p_star: float,
        vol_bucket: str,
        account_balance: float,
        atr: float,
        *,
        expected_rr: Optional[float] = None,
        spread_pips: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate position size using Kelly-lite with multiple risk factors

        Args:
            symbol: Trading symbol
            p_star: Confidence from fusion model [0,1]
            vol_bucket: Volatility regime (Low/Med/High)
            account_balance: Current account balance
            atr: Average True Range for stop loss calculation
            expected_rr: Expected risk/reward ratio (optional)
            spread_pips: Current spread in pips
            meta: Additional metadata

        Returns:
            Dict with position sizing decision and risk metrics
        """
        t0 = time.perf_counter_ns()

        # Check kill switch first
        kill_switch, kill_reason = self.check_kill_switch()
        if kill_switch:
            return {
                "symbol": symbol,
                "position_size": 0.0,
                "risk_units": 0.0,
                "kill_switch": True,
                "kill_reason": kill_reason,
                "timestamp": time.time(),
                "lat_ms": (time.perf_counter_ns() - t0) / 1e6,
            }

        # Get expected RR (fallback by vol bucket)
        if expected_rr is None:
            rr_fallback = {"Low": 1.4, "Med": 1.2, "High": 1.0}
            expected_rr = rr_fallback.get(vol_bucket, 1.2)

        # Kelly-lite calculation
        edge = self.kelly_calc.estimate_edge(p_star, expected_rr)
        cost = self.kelly_calc.estimate_cost(vol_bucket, spread_pips)

        # Volatility adjustment
        vol_mult = {"Low": 0.8, "Med": 1.0, "High": 1.5}.get(vol_bucket, 1.0)
        kelly_f = self.kelly_calc.kelly_fraction(edge, cost, vol_mult)

        # Base position sizing
        base_risk = self.limits.base_risk_per_trade

        # Confidence scaling
        confidence_mult = min(2.0, p_star * 2.0)  # Scale 0.5-1.0 -> 1.0-2.0

        # Kelly adjustment
        kelly_mult = 1.0 + kelly_f  # 1.0 to 1.25 multiplier

        # Combined risk
        total_risk = base_risk * confidence_mult * kelly_mult

        # Volatility capping
        if vol_bucket == "High":
            total_risk *= 0.5  # Halve size in high vol

        # Position size calculation
        risk_amount = account_balance * total_risk

        # Use ATR for stop loss distance
        stop_distance = atr * 2.0  # 2x ATR stop

        # Calculate position size (units)
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = 0.0

        # Apply limits
        position_size = max(
            self.limits.min_units, min(self.limits.max_units, position_size)
        )

        # Final risk check
        if p_star < 0.55:  # Below minimum confidence
            position_size = 0.0

        return {
            "symbol": symbol,
            "position_size": float(position_size),
            "risk_units": float(position_size),
            "risk_amount": float(risk_amount),
            "risk_pct": float(total_risk),
            "kelly_fraction": float(kelly_f),
            "edge": float(edge),
            "cost": float(cost),
            "expected_rr": float(expected_rr),
            "stop_distance": float(stop_distance),
            "confidence_mult": float(confidence_mult),
            "vol_bucket": vol_bucket,
            "p_star": float(p_star),
            "kill_switch": False,
            "kill_reason": "",
            "timestamp": time.time(),
            "lat_ms": (time.perf_counter_ns() - t0) / 1e6,
            "meta": meta or {},
        }


# Global instance
position_sizer = PositionSizer()
