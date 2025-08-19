# -*- coding: utf-8 -*-
"""
Trading Pipeline Enhanced: Full end-to-end institutional pipeline
Regime → Models → Calibration → Fusion → Risk → Execution
"""
from __future__ import annotations
import time, logging
from typing import Dict, Any, Optional, List

from backend.core.regime_online import regime_manager
from backend.services.models_interface import score_and_calibrate
from backend.smc.smc_fusion_core_enhanced import SMCFusionCoreEnhanced
from backend.core.risk_budget_enhanced import position_sizer

logger = logging.getLogger(__name__)


class TradingPipelineEnhanced:
    """Institutional-grade trading pipeline with full probability chain"""

    def __init__(self):
        self.fusion_core = SMCFusionCoreEnhanced()
        self.last_prices = {}
        self.session_detector = SessionDetector()

    def process_tick(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[float] = None,
        account_balance: float = 100000.0,
        atr: float = 0.001,
        spread_pips: float = 1.0,
        model_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline processing for a single tick

        Returns complete institutional decision with:
        - Regime state and volatility
        - Calibrated model probabilities
        - Fused confidence and EV
        - Position sizing recommendation
        - Risk metrics and guards
        """
        t0 = time.perf_counter_ns()

        # 1. Regime Detection
        regime_info = regime_manager.update_symbol(symbol, price, timestamp)
        state = regime_info["state"]
        vol_bucket = regime_info["vol_bucket"]

        # 2. Session Detection
        session = self.session_detector.get_current_session()

        # 3. Model Scoring (mock if not provided)
        if model_scores is None:
            # Generate mock scores for demo
            import random

            model_scores = {
                "LSTM": random.uniform(-1, 1),
                "PPO": random.uniform(-1, 1),
                "XGB": random.uniform(-1, 1),
                "CNN": random.uniform(-1, 1),
            }

        # 4. Calibration
        calibrated = score_and_calibrate(symbol, state, model_scores)
        model_probs = calibrated["calibrated"]

        # 5. Fusion Decision
        spread_z = min(2.0, spread_pips / 1.5)  # Normalize spread

        decision = self.fusion_core.decide(
            symbol=symbol,
            state=state,
            vol_bucket=vol_bucket,
            session=session,
            spread_z=spread_z,
            model_probs=model_probs,
            meta={
                "spread_pips": spread_pips,
                "expected_rr": regime_info.get("expected_rr"),
                "regime_confidence": max(regime_info["state_probs"].values()),
            },
        )

        # 6. Position Sizing
        if decision["action"] != "FLAT":
            sizing = position_sizer.calculate_position_size(
                symbol=symbol,
                p_star=decision["p_star"],
                vol_bucket=vol_bucket,
                account_balance=account_balance,
                atr=atr,
                spread_pips=spread_pips,
                meta={"decision_meta": decision.get("meta", {})},
            )
        else:
            sizing = {
                "position_size": 0.0,
                "risk_units": 0.0,
                "kill_switch": False,
                "reason": "No signal (FLAT)",
            }

        # 7. Combine Results
        pipeline_result = {
            "symbol": symbol,
            "timestamp": timestamp or time.time(),
            "price": price,
            # Regime
            "regime": {
                "state": state,
                "vol_bucket": vol_bucket,
                "volatility": regime_info["volatility"],
                "volatility_annualized": regime_info["volatility_annualized"],
                "state_probs": regime_info["state_probs"],
                "dwell_count": regime_info["dwell_count"],
            },
            # Models
            "models": {
                "raw_scores": calibrated["raw"],
                "calibrated_probs": model_probs,
            },
            # Decision
            "decision": {
                "action": decision["action"],
                "p_long": decision["p_long"],
                "p_short": decision["p_short"],
                "p_star": decision["p_star"],
                "EV": decision["EV"],
                "theta": decision["theta"],
                "weights_used": decision["weights_used"],
                "guards": decision["guards"],
            },
            # Position Sizing
            "sizing": sizing,
            # Performance
            "latency": {
                "total_ms": (time.perf_counter_ns() - t0) / 1e6,
                "fusion_ms": decision.get("lat_ms", 0),
                "sizing_ms": sizing.get("lat_ms", 0),
            },
            # Context
            "context": {
                "session": session,
                "spread_pips": spread_pips,
                "atr": atr,
                "account_balance": account_balance,
            },
        }

        # Log institutional-grade decision
        logger.info(
            f"Pipeline {symbol}: {state}/{vol_bucket} → {decision['action']} "
            f"(p*={decision['p_star']:.3f}, EV={decision['EV']:.3f}, "
            f"size={sizing.get('position_size', 0):.3f}, "
            f"lat={pipeline_result['latency']['total_ms']:.1f}ms)"
        )

        return pipeline_result


class SessionDetector:
    """Simple session detection based on UTC time"""

    def get_current_session(self) -> str:
        """Get current trading session (ASIA/EU/US)"""
        import datetime

        utc_hour = datetime.datetime.utcnow().hour

        if 0 <= utc_hour < 7:
            return "ASIA"
        elif 7 <= utc_hour < 15:
            return "EU"
        else:
            return "US"


# Global instance
trading_pipeline = TradingPipelineEnhanced()
