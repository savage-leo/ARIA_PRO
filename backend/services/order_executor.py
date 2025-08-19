# Replace order executor core to route through TradeArbiter
import os
import logging
from typing import Dict, Any
from backend.services.exec_arbiter import TradeArbiter, ExecPlan
from backend.services.mt5_client import MT5Client

logger = logging.getLogger("ORDER.EXEC")


class OrderExecutor:
    def __init__(self, mt5_client: MT5Client = None):
        self.mt5_client = mt5_client or MT5Client()
        # ensure client started by orchestrator
        self.arbiter = TradeArbiter(self.mt5_client)

    def plan_partial_orders(self, plan: Dict[str, Any]):
        """
        Convert internal plan dict -> ExecPlan and route via arbiter.
        Plan expected fields: symbol, direction, lots, price, sl, tp, reason
        """
        # simple mapping
        ep = ExecPlan(
            symbol=plan["symbol"],
            direction=int(plan["direction"]),
            lots=float(plan["lots"]),
            price=float(plan["price"]),
            sl=float(plan["sl"]),
            tp=float(plan["tp"]),
            reason=str(plan.get("reason", "arbiter_route")),
            meta=plan.get("meta", {}),
        )
        return self.arbiter.route(ep)


# Global instance
order_executor = OrderExecutor()
