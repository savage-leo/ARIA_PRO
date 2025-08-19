# backend/routes/smc_routes.py
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, Optional
import logging
import os

from backend.smc.smc_edge_core import get_edge
from backend.core.trade_memory import TradeMemory
from backend.services.mt5_executor import execute_order
from backend.services.cpp_integration import cpp_service
from backend.smc.smc_fusion_core import EnhancedTradeIdea

logger = logging.getLogger("aria.routes.smc")
router = APIRouter(prefix="/api/smc", tags=["SMC"])


# simple admin API key dependency
def require_admin(request: Request):
    key = os.getenv("ADMIN_API_KEY", "")
    if not key:
        raise HTTPException(status_code=403, detail="Admin key not configured")
    header = request.headers.get("X-ADMIN-KEY") or request.headers.get("x-admin-key")
    if header != key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return True


def _latest_idea_from_memory(symbol: str) -> Optional[EnhancedTradeIdea]:
    """Return latest EnhancedTradeIdea for symbol using TradeMemory, if available."""
    try:
        m = TradeMemory()
        rows = m.list_recent(limit=200)
        sym = symbol.upper()
        for r in rows:
            if (r.get("symbol") or "").upper() == sym:
                p = r.get("payload") or {}
                # Construct a minimal EnhancedTradeIdea from stored dict
                return EnhancedTradeIdea(
                    symbol=p.get("symbol", sym),
                    bias=p.get("bias", "neutral"),
                    confidence=float(p.get("confidence", 0.0) or 0.0),
                    entry=float(p.get("entry", 0.0) or 0.0),
                    stop=float(p.get("stop", 0.0) or 0.0),
                    takeprofit=float(p.get("takeprofit", 0.0) or 0.0),
                    order_blocks=[],
                    fair_value_gaps=[],
                    liquidity_zones=[],
                    meta_weights=p.get("meta_weights"),
                    regime=p.get("regime", "neutral"),
                    anomaly_score=float(p.get("anomaly_score", 0.0) or 0.0),
                    ts=float(p.get("ts", 0.0) or 0.0),
                )
        return None
    except Exception as e:
        logger.exception("latest_idea_from_memory failed: %s", e)
        return None


@router.get("/current/{symbol}")
def get_current_signal(symbol: str):
    eng = get_edge(symbol)
    # Try latest idea from memory; if not present, derive from last bar
    idea = _latest_idea_from_memory(symbol)
    if not idea and len(eng.history) > 0:
        latest_bar = list(eng.history)[-1]
        idea = eng.ingest_bar(latest_bar)
    if not idea:
        return {"ok": False, "msg": "no current signal"}
    return {"ok": True, "signal": idea.as_dict()}


@router.post("/idea/prepare")
def prepare_idea(payload: Dict[str, Any]):
    """
    Prepare an SMC trade idea with Bias Engine and risk-adjusted sizing.
    Accepts: {
        symbol: 'EURUSD',
        base_risk_pct: 0.5,
        market_ctx: {'spread': 0.00008, 'slippage': 0.00005, 'of_imbalance': 0.3},
        equity: 10000.0
    }
    Returns bias-adjusted payload with risk details
    """
    symbol = payload.get("symbol", "EURUSD")
    base_risk_pct = payload.get("base_risk_pct", 0.5)
    market_ctx = payload.get("market_ctx", {})
    equity = payload.get("equity")

    eng = get_edge(symbol)

    # Get the latest idea from TradeMemory or derive from last bar
    idea = _latest_idea_from_memory(symbol)
    if not idea and len(eng.history) > 0:
        latest_bar = list(eng.history)[-1]
        idea = eng.ingest_bar(latest_bar)

    if not idea:
        return {
            "ok": False,
            "msg": "No valid TradeIdea available yet. Feed real bars first.",
        }

    try:
        # Use the new bias engine to prepare the idea
        prepared_payload = eng.prepare_with_bias(
            idea=idea, market_ctx=market_ctx, base_risk_pct=base_risk_pct, equity=equity
        )

        return {"ok": True, "prepared_payload": prepared_payload}

    except Exception as e:
        logger.exception("Bias preparation failed: %s", e)
        return {"ok": False, "msg": f"Preparation failed: {str(e)}"}


@router.post("/idea/execute")
def execute_idea(payload: Dict[str, Any], admin=Depends(require_admin)):
    """
    Admin-only endpoint to execute a prepared idea payload.
    payload: prepared_payload as returned by prepare_idea (must include dry_run flag)
    """
    if not payload:
        raise HTTPException(status_code=400, detail="payload required")
    # require explicit dry_run unless AUTO_EXEC_ENABLED & ALLOW_LIVE set
    dry_run = payload.get("dry_run", False)  # LIVE TRADING - NO MOCK
    auto_ok = os.getenv("AUTO_EXEC_ENABLED", "false").lower() in ("1", "true", "yes")
    allow_live = os.getenv("ALLOW_LIVE", "0") == "1"
    if not dry_run and not (auto_ok and allow_live):
        raise HTTPException(
            status_code=403,
            detail="Live execution not allowed. Set AUTO_EXEC_ENABLED and ALLOW_LIVE and call with admin key.",
        )
    # call mt5 executor; this returns simulator if MT5 disabled
    try:
        res = execute_order(
            symbol=payload["symbol"],
            side=payload["side"],
            volume=payload["size"],
            sl=payload.get("stop"),
            tp=payload.get("takeprofit"),
            comment=payload.get("comment"),
            dry_run=dry_run,
        )
        return {"ok": True, "execution_result": res}
    except Exception as e:
        logger.exception("Execution failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def smc_history(limit: int = 50):
    m = TradeMemory()
    results = m.list_recent(limit)
    return {"ok": True, "history": results}


@router.post("/process/tick")
def process_tick_data(payload: Dict[str, Any]):
    """
    Process high-frequency tick data using C++ engine
    Accepts: { symbol: 'EURUSD', bid: 1.1000, ask: 1.1001, volume: 1000 }
    """
    try:
        symbol = payload.get("symbol")
        bid = payload.get("bid")
        ask = payload.get("ask")
        volume = payload.get("volume", 0)
        timestamp = payload.get("timestamp")

        if not all([symbol, bid, ask]):
            raise HTTPException(status_code=400, detail="symbol, bid, ask required")

        result = cpp_service.process_tick_data(symbol, bid, ask, volume, timestamp)
        return {"ok": True, "processed_tick": result}

    except Exception as e:
        logger.exception("Tick processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/bar")
def process_bar_data(payload: Dict[str, Any]):
    """
    Process bar data and detect SMC patterns using C++ engine
    Accepts: { symbol: 'EURUSD', open: 1.1000, high: 1.1010, low: 1.0990, close: 1.1005, volume: 1000 }
    """
    try:
        symbol = payload.get("symbol")
        open_price = payload.get("open")
        high = payload.get("high")
        low = payload.get("low")
        close = payload.get("close")
        volume = payload.get("volume", 0)
        timestamp = payload.get("timestamp")

        if not all([symbol, open_price, high, low, close]):
            raise HTTPException(
                status_code=400, detail="symbol, open, high, low, close required"
            )

        result = cpp_service.process_bar_data(
            symbol, open_price, high, low, close, volume, timestamp
        )
        return {"ok": True, "processed_bar": result}

    except Exception as e:
        logger.exception("Bar processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/{symbol}")
def get_smc_signals(symbol: str):
    """
    Get SMC trading signals using C++ engine
    """
    try:
        signals = cpp_service.get_smc_signals(symbol)
        return {"ok": True, "signals": signals}

    except Exception as e:
        logger.exception("Signal generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/order-blocks/{symbol}")
def get_order_blocks(symbol: str):
    """
    Get order blocks using C++ engine
    """
    try:
        blocks = cpp_service.get_order_blocks(symbol)
        return {"ok": True, "order_blocks": blocks}

    except Exception as e:
        logger.exception("Order block retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fair-value-gaps/{symbol}")
def get_fair_value_gaps(symbol: str):
    """
    Get fair value gaps using C++ engine
    """
    try:
        gaps = cpp_service.get_fair_value_gaps(symbol)
        return {"ok": True, "fair_value_gaps": gaps}

    except Exception as e:
        logger.exception("Fair value gap retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cpp/status")
def get_cpp_status():
    """
    Check C++ integration status
    """
    try:
        from backend.services.cpp_integration import CPP_AVAILABLE

        return {
            "ok": True,
            "cpp_available": CPP_AVAILABLE,
            "market_processor": cpp_service.market_processor is not None,
            "smc_engine": cpp_service.smc_engine is not None,
        }

    except Exception as e:
        logger.exception("Status check failed: %s", e)
        return {"ok": False, "error": str(e)}
