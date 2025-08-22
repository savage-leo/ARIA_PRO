"""
Admin Hot-Swap Control Endpoints
- Secure admin-only controls to enable/disable auto hot-swap
- Status endpoint exposes watchdog availability and manager state
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any
import logging
import os

from backend.core.model_loader import hot_swap_manager
from backend.core.hot_swap_manager import WATCHDOG_AVAILABLE

logger = logging.getLogger("aria.routes.hot_swap_admin")

router = APIRouter(prefix="/admin/hot_swap", tags=["Admin", "Hot Swap"]) 


# Simple admin API key dependency (reuse pattern from smc_routes)
def require_admin(request: Request):
    key = os.getenv("ADMIN_API_KEY", "")
    if not key:
        raise HTTPException(status_code=403, detail="Admin key not configured")
    header = request.headers.get("X-ADMIN-KEY") or request.headers.get("x-admin-key")
    if header != key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return True


@router.get("/status")
async def get_hot_swap_status_admin(admin: bool = Depends(require_admin)) -> Dict[str, Any]:
    """Admin: Get hot-swap manager status and watchdog availability"""
    try:
        status = hot_swap_manager.get_swap_status()
        return {
            "status": "success",
            "data": {
                "auto_swap_enabled": bool(status.get("auto_swap_enabled", False)),
                "watchdog_available": bool(WATCHDOG_AVAILABLE),
                "manager": status,
            },
            "message": "Hot-swap status retrieved",
        }
    except Exception as e:
        logger.error(f"Failed to get hot-swap status (admin): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable")
async def enable_auto_hot_swap(admin: bool = Depends(require_admin)) -> Dict[str, Any]:
    """Admin: Enable automatic hot-swapping if watchdog is available"""
    try:
        if not WATCHDOG_AVAILABLE:
            logger.warning("Auto hot-swap enable requested but watchdog is unavailable")
            raise HTTPException(status_code=503, detail="watchdog not available on server")

        hot_swap_manager.enable_auto_swap()
        status = hot_swap_manager.get_swap_status()
        enabled = bool(status.get("auto_swap_enabled", False))
        return {
            "status": "success" if enabled else "failed",
            "message": "Auto hot-swap enabled" if enabled else "Failed to enable auto hot-swap",
            "data": {
                "auto_swap_enabled": enabled,
                "watchdog_available": bool(WATCHDOG_AVAILABLE),
                "manager": status,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable auto hot-swap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable")
async def disable_auto_hot_swap(admin: bool = Depends(require_admin)) -> Dict[str, Any]:
    """Admin: Disable automatic hot-swapping"""
    try:
        hot_swap_manager.disable_auto_swap()
        status = hot_swap_manager.get_swap_status()
        enabled = bool(status.get("auto_swap_enabled", False))
        return {
            "status": "success",
            "message": "Auto hot-swap disabled",
            "data": {
                "auto_swap_enabled": enabled,
                "watchdog_available": bool(WATCHDOG_AVAILABLE),
                "manager": status,
            },
        }
    except Exception as e:
        logger.error(f"Failed to disable auto hot-swap: {e}")
        raise HTTPException(status_code=500, detail=str(e))
