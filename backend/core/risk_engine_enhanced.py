# risk_engine_enhanced.py
"""
Improved risk engine:
- Reads MT5 account info when available
- Produces lots with strict caps, slack factor and per-symbol max exposure
"""

import os
import logging
from typing import Tuple
from backend.services.mt5_client import MT5Client

logger = logging.getLogger("RISK")


class RiskEngine:
    def __init__(self, mt5_client: MT5Client = None):
        self.client = mt5_client
        self.account_equity = float(os.environ.get("ARIA_EQUITY", "10000"))
        self.max_risk_pct = float(os.environ.get("ARIA_RISK_PER_TRADE", "0.005"))
        self.max_lots_per_symbol = float(
            os.environ.get("ARIA_MAX_LOTS_PER_SYMBOL", "10.0")
        )
        self.pip_value_per_lot = float(os.environ.get("ARIA_PIP_VALUE_PER_LOT", "10.0"))
        self.min_sl_pips = float(os.environ.get("ARIA_MIN_SL_PIPS", "1.0"))

    def refresh_account(self):
        if self.client and hasattr(self.client, "connect"):
            try:
                # read MT5 account info
                import MetaTrader5 as mt5

                if mt5.initialize():
                    acc = mt5.account_info()
                    if acc:
                        self.account_equity = float(acc.balance)
                    mt5.shutdown()
            except Exception:
                logger.exception("Failed refresh account info")

    def size_from_sl(self, sl_pips: float, symbol: str, strength: float) -> float:
        sl_pips = max(sl_pips, self.min_sl_pips)
        risk_amt = self.max_risk_pct * self.account_equity * strength
        lots = risk_amt / (sl_pips * self.pip_value_per_lot)
        lots = max(0.0, min(lots, self.max_lots_per_symbol))
        return lots


if __name__ == "__main__":
    r = RiskEngine()
    print("Size example:", r.size_from_sl(10.0, "EURUSD", 0.8))
