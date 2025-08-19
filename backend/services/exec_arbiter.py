# exec_arbiter.py
"""
Trade Arbiter:
- Only component allowed to send orders to MT5Client
- Performs pre-flight checks (kill switch, equity, exposure caps)
- Supports dry-run mode
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict

from backend.services.mt5_client import MT5Client

logger = logging.getLogger("ARB")


@dataclass
class ExecPlan:
    symbol: str
    direction: int
    lots: float
    price: float
    sl: float
    tp: float
    reason: str
    meta: Dict[str, Any] = None


class TradeArbiter:
    def __init__(self, mt5_client: MT5Client):
        self.client = mt5_client
        self.kill = False
        self.dry_run = os.environ.get("ARIA_ENABLE_EXEC", "0") != "1"
        self.max_lots = float(os.environ.get("ARIA_MAX_LOTS_PER_TRADE", "5.0"))
        self.min_equity = float(os.environ.get("ARIA_MIN_EQUITY", "100"))
        self.equity = float(os.environ.get("ARIA_EQUITY", "10000"))
        # maintain exposure map
        self.open_exposure = {}

    def engage_kill(self, enable: bool):
        self.kill = enable

    def _preflight(self, plan: ExecPlan):
        if self.kill:
            raise RuntimeError("Kill switch engaged")
        if plan.lots <= 0:
            raise RuntimeError("Zero lots")
        if plan.lots > self.max_lots:
            logger.warning("Capping lots %s -> %s", plan.lots, self.max_lots)
            plan.lots = self.max_lots
        # equity check (simple)
        if self.equity < self.min_equity:
            raise RuntimeError("Equity below min threshold")
        # TODO: add correlation/exposure checks
        return True

    def route(self, plan: ExecPlan):
        try:
            self._preflight(plan)
        except Exception as e:
            logger.error("Preflight failed: %s", e)
            return None
        if self.dry_run:
            logger.info("[DRY-RUN] Route -> %s", plan)
            return {"dry_run": True, "plan": plan}
        # build MT5 request
        # Use client.send_order()
        request = {
            "action": 0,  # mt5.TRADE_ACTION_DEAL
            "symbol": plan.symbol,
            "volume": max(plan.lots, 0.01),
            "type": 0 if plan.direction == 1 else 1,  # BUY/SELL
            "price": plan.price,
            "sl": plan.sl,
            "tp": plan.tp,
            "deviation": int(float(os.environ.get("ARIA_MAX_SLIPPAGE_PIPS", "1.5"))),
            "magic": int(os.environ.get("ARIA_MAGIC", "271828")),
            "comment": plan.reason[:50],
            "type_filling": 1,
            "type_time": 0,
        }
        try:
            r = self.client.send_order(request)
            logger.info("Order sent, resp: %s", r)
            return r
        except Exception as e:
            logger.exception("Order send failed: %s", e)
            return None


if __name__ == "__main__":
    from backend.services.mt5_client import MT5Client

    c = MT5Client()
    arb = TradeArbiter(c)
    print("Arbiter dry:", arb.dry_run)
