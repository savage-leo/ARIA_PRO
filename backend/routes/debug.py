"""
Debug routes for monitoring Phase 3 Orchestrator and Enhanced SMC Fusion Core
"""

from fastapi import APIRouter, HTTPException, Header
from typing import Dict, Any, Optional
import logging
import os
from backend.services.ws_broadcaster import broadcaster
from backend.monitoring.llm_monitor import llm_monitor_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug", tags=["Debug"])

# Global storage for latest ideas (in production, use Redis/database)
_latest_ideas: Dict[str, Dict[str, Any]] = {}
_orchestrator_status: Dict[str, Any] = {}

ADMIN_KEY = os.environ.get("ARIA_ADMIN_KEY") or os.environ.get("ADMIN_API_KEY", "")


def _require_admin(header_val: Optional[str]):
    """Enforce admin header if ADMIN_KEY is set. If not set, allow for local dev."""
    if ADMIN_KEY:
        if not header_val or header_val != ADMIN_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/idea/{symbol}")
async def get_latest_idea(symbol: str) -> Dict[str, Any]:
    """Get the latest EnhancedTradeIdea for a symbol"""
    idea = _latest_ideas.get(symbol.upper())
    if not idea:
        raise HTTPException(status_code=404, detail=f"No idea found for {symbol}")
    return idea


@router.get("/ideas")
async def get_all_ideas() -> Dict[str, Any]:
    """Get latest ideas for all symbols"""
    return {
        "ideas": _latest_ideas,
        "count": len(_latest_ideas),
        "symbols": list(_latest_ideas.keys()),
    }


@router.get("/orchestrator/status")
async def get_orchestrator_status() -> Dict[str, Any]:
    """Get Phase 3 Orchestrator status"""
    return _orchestrator_status


@router.get("/fusion/state/{symbol}")
async def get_fusion_state(symbol: str) -> Dict[str, Any]:
    """Get Enhanced SMC Fusion Core state for a symbol"""
    # This would integrate with the actual fusion core
    # For now, return placeholder
    return {
        "symbol": symbol.upper(),
        "bars_count": 0,
        "order_blocks_count": 0,
        "fair_value_gaps_count": 0,
        "liquidity_zones_count": 0,
        "meta_model_weights": {},
        "context": {"regime": "neutral", "vol_ewma": 0.0, "spread_ewma": 0.0},
    }


@router.post("/idea/{symbol}")
async def update_latest_idea(
    symbol: str,
    idea: Dict[str, Any],
    x_aria_admin: Optional[str] = Header(default=None, alias="X-ARIA-ADMIN"),
):
    """Update the latest idea for a symbol (called by orchestrator)"""
    _require_admin(x_aria_admin)
    _latest_ideas[symbol.upper()] = idea
    logger.info(
        f"Updated idea for {symbol}: {idea.get('bias', 'unknown')} {idea.get('confidence', 0):.3f}"
    )
    try:
        # Ensure symbol present in payload for channel routing and clients
        payload = dict(idea)
        payload.setdefault("symbol", symbol.upper())
        await broadcaster.broadcast_idea(payload)
    except Exception as e:
        logger.warning(f"Failed to broadcast idea for {symbol}: {e}")


@router.post("/orchestrator/status")
async def update_orchestrator_status(
    status: Dict[str, Any],
    x_aria_admin: Optional[str] = Header(default=None, alias="X-ARIA-ADMIN"),
):
    """Update orchestrator status (called by orchestrator)"""
    _require_admin(x_aria_admin)
    _orchestrator_status.update(status)
    logger.info(f"Updated orchestrator status: {status}")


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check including all components"""
    return {
        "status": "ok",
        "components": {
            "fastapi_backend": "running",
            "phase3_orchestrator": "running" if _orchestrator_status else "unknown",
            "enhanced_fusion_cores": len(_latest_ideas),
            "mt5_connection": _orchestrator_status.get("mt5_connected", False),
        },
        "latest_ideas": len(_latest_ideas),
        "orchestrator_status": _orchestrator_status,
    }


# ====== LLM Monitor E2E debug helpers ======
@router.get("/log/warn")
async def emit_warning(
    msg: Optional[str] = None,
    x_aria_admin: Optional[str] = Header(default=None, alias="X-ARIA-ADMIN"),
) -> Dict[str, Any]:
    """Emit a WARNING log to trigger LLM monitor capture."""
    _require_admin(x_aria_admin)
    text = msg or "E2E test warning: MT5 latency spike detected; consider increasing cooldown_sec."
    logger.warning(text)
    return {"emitted": "warning", "message": text}


@router.get("/log/error")
async def emit_error(
    msg: Optional[str] = None,
    x_aria_admin: Optional[str] = Header(default=None, alias="X-ARIA-ADMIN"),
) -> Dict[str, Any]:
    """Emit an ERROR log to trigger LLM monitor capture."""
    _require_admin(x_aria_admin)
    text = msg or "E2E test error: order rejection rate elevated; raise threshold to reduce churn."
    logger.error(text)
    return {"emitted": "error", "message": text}


@router.get("/log/exception")
async def emit_exception(
    x_aria_admin: Optional[str] = Header(default=None, alias="X-ARIA-ADMIN"),
) -> Dict[str, Any]:
    """Emit an exception with stacktrace (logged at ERROR)."""
    _require_admin(x_aria_admin)
    try:
        raise ValueError("E2E test exception: ATR calculation overflow")
    except Exception:
        logger.exception("E2E exception raised for LLM monitor test")
    return {"emitted": "exception"}


@router.get("/log/burst")
async def emit_burst(
    n: int = 6,
    x_aria_admin: Optional[str] = Header(default=None, alias="X-ARIA-ADMIN"),
) -> Dict[str, Any]:
    """Emit a burst of mixed WARNING/ERROR/CRITICAL logs to ensure batching works."""
    _require_admin(x_aria_admin)
    n = max(1, min(50, int(n)))
    messages = []
    for i in range(n):
        if i % 3 == 0:
            m = f"Burst warn {i}: MT5 feed jitter; suggest cooldown adjustment"
            logger.warning(m)
        elif i % 3 == 1:
            m = f"Burst error {i}: execution backoff triggered; consider threshold uptick"
            logger.error(m)
        else:
            m = f"Burst critical {i}: health degraded; throttle signals"
            logger.critical(m)
        messages.append(m)
    return {"emitted": n, "sample": messages[:3]}


@router.get("/llm-monitor/status")
async def llm_monitor_status(
    x_aria_admin: Optional[str] = Header(default=None, alias="X-ARIA-ADMIN"),
) -> Dict[str, Any]:
    """Report LLM monitor runtime status (admin)."""
    _require_admin(x_aria_admin)
    svc = llm_monitor_service
    # Access limited internal state for diagnostics
    return {
        "running": bool(getattr(svc, "running", False)),
        "interval_sec": getattr(svc, "_interval", None),
        "tuning_enabled": getattr(svc, "_tuning_enabled", None),
        "max_rel_delta": getattr(svc, "_max_rel", None),
        "cooldown_sec": getattr(svc, "_tune_cooldown", None),
        "proxy_url": getattr(svc, "_proxy_url_base", None),
    }
