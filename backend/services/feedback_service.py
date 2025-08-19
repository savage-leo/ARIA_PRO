"""
Feedback Service for Enhanced SMC Fusion Core
Handles trade outcomes and updates meta-model weights
"""

import logging
import time
from typing import Dict, Optional
from backend.smc.smc_fusion_core import EnhancedSMCFusionCore

logger = logging.getLogger("aria.services.feedback")


class FeedbackService:
    def __init__(self):
        self.enhanced_engines: Dict[str, EnhancedSMCFusionCore] = {}

    def register_engine(self, symbol: str, engine: EnhancedSMCFusionCore):
        """Register an enhanced fusion engine for feedback"""
        self.enhanced_engines[symbol] = engine
        logger.info(f"Registered feedback engine for {symbol}")

    def submit_trade_feedback(
        self, symbol: str, realized_return: float, last_features: Dict[str, float]
    ):
        """
        Submit trade outcome for meta-model learning

        Args:
            symbol: Trading symbol
            realized_return: PnL per unit risk (positive if trade direction was correct)
            last_features: Feature snapshot from when trade was taken
        """
        try:
            if symbol not in self.enhanced_engines:
                logger.warning(f"No enhanced engine registered for {symbol}")
                return

            engine = self.enhanced_engines[symbol]
            engine.submit_feedback(realized_return, last_features)

            logger.info(
                f"Feedback submitted for {symbol}: return={realized_return:.4f}"
            )

        except Exception as e:
            logger.exception(f"Failed to submit feedback for {symbol}: {e}")

    def get_engine_stats(self, symbol: str) -> Optional[Dict]:
        """Get engine statistics"""
        try:
            if symbol not in self.enhanced_engines:
                return None

            engine = self.enhanced_engines[symbol]
            return {
                "wins": engine._state["wins"],
                "losses": engine._state["losses"],
                "pnl_day": engine._state["pnl_day"],
                "regime": engine.context.regime,
                "meta_weights": engine.meta_model.state() if engine.meta_model else {},
            }
        except Exception as e:
            logger.exception(f"Failed to get stats for {symbol}: {e}")
            return None

    def kill_switch(self, symbol: str, enable: bool = True):
        """Activate kill switch for specific symbol"""
        try:
            if symbol in self.enhanced_engines:
                self.enhanced_engines[symbol].kill_switch(enable)
                logger.warning(
                    f"Kill switch {'activated' if enable else 'deactivated'} for {symbol}"
                )
        except Exception as e:
            logger.exception(f"Failed to set kill switch for {symbol}: {e}")


# Global feedback service instance
feedback_service = FeedbackService()
