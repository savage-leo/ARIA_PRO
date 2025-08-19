import os
import logging

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - MT5 not always installed in dev
    mt5 = None

logger = logging.getLogger("aria.mt5")

DEMO_SAFE = os.getenv("ARIA_ASSUME_DEMO", "1") == "1"
ALLOWED_LIVE = os.getenv("ARIA_ALLOW_LIVE_TRADING", "0") == "1"


def safe_initialize(login=None, server=None, password=None):
    """Initialize MT5 with safety checks.

    If DEMO_SAFE is True and live trading not allowed, we still initialize MT5 but
    emit a warning that live orders will be blocked.
    """
    if mt5 is None:
        logger.warning("MetaTrader5 package not available; running in stub mode")
        return False

    if DEMO_SAFE and not ALLOWED_LIVE:
        logger.warning(
            "MT5 init running in DEMO_SAFE mode. Live trades disabled until ARIA_ALLOW_LIVE_TRADING=1"
        )
    ok = mt5.initialize(login=login, server=server, password=password)
    if not ok:
        raise RuntimeError("MT5 initialize failed")
    return ok


def safe_order_send(request):
    """Wrapper around mt5.order_send — blocks real orders unless allowed."""
    if mt5 is None:
        logger.warning("MetaTrader5 not available; order_send stubbed")
        return {"retcode": -998, "comment": "mt5_not_available"}

    if DEMO_SAFE and not ALLOWED_LIVE:
        logger.warning(
            "Blocked order_send due to DEMO_SAFE. Request logged but not sent."
        )
        return {"retcode": -999, "comment": "blocked_by_demo_safe"}
    return mt5.order_send(request)
