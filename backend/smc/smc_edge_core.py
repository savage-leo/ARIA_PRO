# backend/smc/smc_edge_core.py
"""
SMC Edge Core (Secret Ingredient)
- Wraps smc_fusion + trap_detector + AVCD sizing + TMI logging
- Produces TradeIdea with "trap_confirmation" and sized payload via risk engine
- Does not execute orders automatically (dry-run). Execution exposed via route that requires ADMIN_API_KEY.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from collections import deque

from backend.smc.smc_fusion_core import (
    get_engine as get_fusion_engine,
    get_enhanced_engine,
    EnhancedTradeIdea,
)
from backend.smc.trap_detector import detect_trap
from backend.smc.bias_engine import BiasEngine
from backend.core.trade_memory import TradeMemory
from backend.core.risk_engine import validate_and_size_order
from backend.services.order_executor import order_executor

# Backward compatibility
TradeIdea = EnhancedTradeIdea

logger = logging.getLogger("aria.smc.edge")

# Feature toggles via env are read by called modules (risk_executor, etc.)


class EdgeEngine:
    def __init__(
        self, symbol: str = "EURUSD", max_bars: int = 5000, use_enhanced: bool = True
    ):
        self.symbol = symbol.upper()
        self.use_enhanced = use_enhanced

        # Use enhanced fusion core if enabled
        if self.use_enhanced:
            self.fusion = get_enhanced_engine(self.symbol)
            logger.info(f"Using enhanced SMC fusion core for {self.symbol}")
        else:
            self.fusion = get_fusion_engine(self.symbol)
            logger.info(f"Using standard SMC fusion core for {self.symbol}")

        self.history = deque(maxlen=max_bars)
        self.memory = TradeMemory()
        self.bias_engine = BiasEngine(  # default thresholds; override via env if needed
            min_score=0.55, hard_throttle_below=0.50, max_bias=2.0, min_bias=0.5
        )
        # tuning
        self.trap_threshold = 0.35  # min trap_score to consider trap confirmation
        self.min_confidence_for_execution = 0.30  # fused confidence threshold
        self.max_size_pct = float(
            0.5
        )  # max fraction of account risk allowed (passed to position sizer or risk engine)

    def ingest_bar(
        self,
        bar: Dict[str, Any],
        recent_ticks: Optional[List[Dict[str, Any]]] = None,
        raw_signals: Optional[Dict[str, float]] = None,
        market_feats: Optional[Dict[str, float]] = None,
    ) -> Optional[TradeIdea]:
        """
        bar: {'ts','o','h','l','c','v','symbol'}
        recent_ticks: optional tick list for delta analysis
        raw_signals: optional AI model signals for enhanced fusion
        market_feats: optional market features for enhanced fusion
        """
        self.history.append(bar)

        # Use enhanced ingestion with secret ingredient integration
        if self.use_enhanced and hasattr(self.fusion, "ingest_bar"):
            # Get enhanced idea with secret ingredient meta-weighting
            enhanced_idea = self.fusion.ingest_bar(bar, raw_signals, market_feats)
            fused = enhanced_idea
        else:
            fused = self.fusion.ingest_bar(bar)

        # always log fused idea even if None (for debugging)
        if fused:
            if hasattr(fused, "log_idea"):
                fused.log_idea(fused)
            else:
                # Enhanced logging for secret ingredient
                meta_info = ""
                if hasattr(fused, "meta_weights") and fused.meta_weights:
                    meta_info = f" meta_weights={fused.meta_weights}"
                if hasattr(fused, "regime") and fused.regime:
                    meta_info += f" regime={fused.regime}"
                if hasattr(fused, "anomaly_score") and fused.anomaly_score > 0:
                    meta_info += f" anomaly={fused.anomaly_score:.3f}"

                logger.info(
                    f"Enhanced Trade Idea: {fused.symbol} {fused.bias} conf={fused.confidence:.2f} entry={fused.entry:.5f} stop={fused.stop:.5f} tp={fused.takeprofit:.5f}{meta_info}"
                )

        # Trap detection on the same window
        try:
            trap = detect_trap(list(self.history), recent_ticks)
        except Exception as e:
            logger.exception("Trap detection failed: %s", e)
            trap = {"trap_score": 0.0, "direction": None, "explain": ["error"]}

        # combine fusion idea + trap confirmation
        if not fused:
            # nothing from fusion -> maybe only trap? we require fusion for bias
            logger.debug("No fused idea yet")
            return None

        # require directional alignment if trap_score >0
        trap_score = trap.get("trap_score", 0.0)
        trap_dir = trap.get("direction", None)
        fused_dir = fused.bias  # 'bullish' or 'bearish'

        # Map directions
        direction_map = {"bullish": "buy", "bearish": "sell"}
        fused_dir_simple = direction_map.get(fused_dir, None)

        # Trap must be above threshold and aligned OR model confidence must be very high
        confirmed = False
        reasons = {
            "fusion_conf": fused.confidence,
            "trap_score": trap_score,
            "trap_dir": trap_dir,
            "fused_dir": fused_dir,
        }
        if (
            trap_score >= self.trap_threshold
            and trap_dir
            and fused_dir_simple
            and trap_dir == fused_dir_simple
        ):
            confirmed = True
            reasons["confirm_reason"] = "trap_align_and_threshold"
        elif fused.confidence >= max(self.min_confidence_for_execution, 0.75):
            # very high fused confidence can override trap absence (rare)
            confirmed = True
            reasons["confirm_reason"] = "high_fused_confidence"
        else:
            reasons["confirm_reason"] = "no_confirm"

        if not confirmed:
            # still log idea with trap info for TMI
            self.memory.insert_trade_idea(
                fused.as_dict(), {"trap": trap, "confirmed": False}
            )
            return None

        # prepare sized payload (call risk_engine)
        # Note: validate_and_size_order requires sl price. Use fused.stop
        try:
            lots = validate_and_size_order(
                symbol=fused.symbol,
                side="buy" if fused.bias == "bullish" else "sell",
                risk_percent=self.max_size_pct,
                sl_price=fused.stop,
            )
        except Exception as e:
            logger.exception("Sizing failed: %s", e)
            self.memory.insert_trade_idea(
                fused.as_dict(),
                {"trap": trap, "confirmed": False, "sizing_error": str(e)},
            )
            return None

        payload = {
            "symbol": fused.symbol,
            "side": "buy" if fused.bias == "bullish" else "sell",
            "size": lots,
            "entry": fused.entry,
            "stop": fused.stop,
            "takeprofit": fused.takeprofit,
            "comment": "SMC_EDGE",
            "confidence": fused.confidence,
            "trap": trap,
            "reasons": reasons,
            "ts": time.time(),
        }

        # plan partial orders (MOST) — returns plan but doesn't execute
        plan = order_executor.plan_partial_orders(payload)
        payload["execution_plan"] = plan

        # persist idea/plan to trade memory
        self.memory.insert_trade_idea(
            fused.as_dict(), {"payload": payload, "confirmed": True}
        )

        # return TradeIdea for upper layers / UI
        return fused

    def prepare_with_bias(
        self,
        idea,  # TradeIdea from FusionEngine
        market_ctx: Optional[
            Dict[str, float]
        ] = None,  # {'spread':..., 'slippage':..., 'of_imbalance':...}
        base_risk_pct: float = 0.5,  # percent per trade (e.g., 0.5% of equity)
        equity: Optional[
            float
        ] = None,  # require real equity from account svc; do not mock
    ) -> Dict[str, Any]:
        # compute bias
        bias = self.bias_engine.compute(idea, self.history, market_ctx or {})
        # throttle? -> return plan without execution flag
        gated = bias.throttle

        # derive effective risk
        eff_risk_pct = max(0.0, base_risk_pct) * bias.risk_multiplier

        # position size (no mock: if equity not provided, do not fabricate)
        pos = None
        if equity is not None and idea.stop and idea.entry and idea.stop != 0.0:
            try:
                lots = validate_and_size_order(
                    symbol=self.symbol,
                    side="buy" if idea.bias == "bullish" else "sell",
                    risk_percent=eff_risk_pct,
                    sl_price=idea.stop,
                )
                pos = {"size": lots}
            except Exception as e:
                logger.exception("Sizing failed: %s", e)
                pos = None

        # build execution plan (using existing order executor logic)
        plan = order_executor.plan_partial_orders(
            {
                "symbol": self.symbol,
                "side": "buy" if idea.bias == "bullish" else "sell",
                "size": pos["size"] if pos else None,
                "entry": idea.entry,
                "stop": idea.stop,
                "takeprofit": idea.takeprofit,
                "comment": "SMC_EDGE_BIAS",
                "confidence": idea.confidence,
            }
        )

        payload = {
            "symbol": self.symbol,
            "bias": idea.bias,
            "confidence": idea.confidence,
            "entry": idea.entry,
            "stop": idea.stop,
            "takeprofit": idea.takeprofit,
            "risk": {
                "base_risk_pct": base_risk_pct,
                "bias_factor": bias.bias_factor,
                "effective_risk_pct": eff_risk_pct,
                "score": bias.score,
                "throttle": gated,
                "reasons": bias.reasons,
            },
            "position": pos,
            "plan": plan,
            "dry_run": False,  # LIVE TRADING - NO MOCK
        }

        # log to trade memory (idea prepared)
        self.memory.insert_trade_idea(
            idea.as_dict(), {"bias_payload": payload, "confirmed": True}
        )

        return payload


# singleton helper
_edge_instances = {}


def get_edge(symbol: str = "EURUSD") -> EdgeEngine:
    key = symbol.upper()
    if key not in _edge_instances:
        _edge_instances[key] = EdgeEngine(symbol=key)
    return _edge_instances[key]
