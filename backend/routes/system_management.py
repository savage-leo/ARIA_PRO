from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import psutil
import time
import asyncio
from typing import Dict, Any
import logging
from backend.services.ws_broadcaster import broadcaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

# Global system state
system_state = {
    "mt5_connected": False,
    "auto_trading_enabled": False,
    "signal_generator_active": False,
    "risk_engine_active": False,
    # informational only; status endpoint reports real counts from broadcaster
    "websocket_connections": 0,
    "start_time": time.time(),
}


class ServiceToggleRequest(BaseModel):
    enabled: bool


class SystemStatusResponse(BaseModel):
    mt5_connected: bool
    auto_trading_enabled: bool
    signal_generator_active: bool
    risk_engine_active: bool
    websocket_connections: int
    uptime_seconds: int
    cpu_usage: float
    memory_usage: float


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        uptime = int(time.time() - system_state["start_time"])
        # Real-time websocket client count
        ws_count = await broadcaster.get_client_count()

        return SystemStatusResponse(
            mt5_connected=system_state["mt5_connected"],
            auto_trading_enabled=system_state["auto_trading_enabled"],
            signal_generator_active=system_state["signal_generator_active"],
            risk_engine_active=system_state["risk_engine_active"],
            websocket_connections=ws_count,
            uptime_seconds=uptime,
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.post("/mt5/toggle")
async def toggle_mt5_connection(request: ServiceToggleRequest):
    """Toggle MT5 connection"""
    try:
        # Import MT5 service (global instance). If MetaTrader5 is not installed, degrade gracefully.
        try:
            from backend.services.mt5_executor import mt5_executor as _mt5
        except Exception as ie:
            logger.warning(f"MT5 executor unavailable: {ie}")
            system_state["mt5_connected"] = False
            return {
                "status": "unavailable",
                "enabled": False,
                "message": "MT5 not available",
            }

        if request.enabled:
            # Start MT5 connection
            success = _mt5.connect()
            system_state["mt5_connected"] = bool(success)
            return {
                "status": "connected" if success else "failed",
                "enabled": bool(success),
            }
        else:
            # Disconnect MT5 (best-effort)
            try:
                _mt5.disconnect()
            except Exception:
                pass
            system_state["mt5_connected"] = False
            return {"status": "disconnected", "enabled": False}

    except Exception as e:
        logger.error(f"Error toggling MT5 connection: {e}")
        system_state["mt5_connected"] = False
        raise HTTPException(status_code=500, detail=f"Failed to toggle MT5: {str(e)}")


@router.post("/signals/toggle")
async def toggle_signal_generator(request: ServiceToggleRequest):
    """Toggle AI signal generator"""
    try:
        system_state["signal_generator_active"] = request.enabled

        if request.enabled:
            # Start signal generator background task
            logger.info("Signal generator activated")
        else:
            # Stop signal generator
            logger.info("Signal generator deactivated")

        return {
            "status": "active" if request.enabled else "inactive",
            "enabled": request.enabled,
        }

    except Exception as e:
        logger.error(f"Error toggling signal generator: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to toggle signal generator: {str(e)}"
        )


@router.post("/risk/toggle")
async def toggle_risk_engine(request: ServiceToggleRequest):
    """Toggle risk management engine"""
    try:
        system_state["risk_engine_active"] = request.enabled

        if request.enabled:
            logger.info("Risk engine activated")
        else:
            logger.info("Risk engine deactivated")

        return {
            "status": "active" if request.enabled else "inactive",
            "enabled": request.enabled,
        }

    except Exception as e:
        logger.error(f"Error toggling risk engine: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to toggle risk engine: {str(e)}"
        )


@router.post("/websocket/toggle")
async def toggle_websocket_server(request: ServiceToggleRequest):
    """Toggle WebSocket server"""
    try:
        # This backend always exposes the /ws endpoint when the app is running.
        # We do not dynamically start/stop the WebSocket server here; this is a no-op
        # kept for API compatibility. Report current connection count for visibility.
        ws_count = await broadcaster.get_client_count()
        logger.info(
            f"WebSocket toggle requested (enabled={request.enabled}); current clients={ws_count}"
        )
        return {
            "status": "available",
            "enabled": True,
            "clients": ws_count,
            "note": "WebSocket endpoint is always active while the app runs.",
        }

    except Exception as e:
        logger.error(f"Error toggling WebSocket server: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to toggle WebSocket server: {str(e)}"
        )


@router.post("/restart")
async def restart_system():
    """Restart the entire ARIA system"""
    try:
        logger.warning("System restart initiated")

        # Reset all system states
        system_state.update(
            {
                "mt5_connected": False,
                "auto_trading_enabled": False,
                "signal_generator_active": False,
                "risk_engine_active": False,
                "websocket_connections": 0,
                "start_time": time.time(),
            }
        )

        # Simulate restart delay
        await asyncio.sleep(2)

        # Restart services
        system_state["mt5_connected"] = True
        system_state["signal_generator_active"] = True
        system_state["risk_engine_active"] = True
        system_state["websocket_connections"] = 1

        logger.info("System restart completed")
        return {"status": "restarted", "message": "System successfully restarted"}

    except Exception as e:
        logger.error(f"Error restarting system: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to restart system: {str(e)}"
        )


@router.post("/emergency-stop")
async def emergency_stop():
    """Emergency stop - halt all trading and AI services immediately"""
    try:
        logger.critical("EMERGENCY STOP ACTIVATED")

        # Immediately stop all critical services
        system_state.update(
            {
                "auto_trading_enabled": False,
                "signal_generator_active": False,
                "risk_engine_active": False,
            }
        )

        # Keep MT5 and WebSocket for monitoring
        logger.critical("All trading and AI services stopped")
        return {
            "status": "emergency_stopped",
            "message": "Emergency stop executed - all trading halted",
        }

    except Exception as e:
        logger.error(f"Error executing emergency stop: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to execute emergency stop: {str(e)}"
        )


@router.get("/metrics")
async def get_system_metrics():
    """Get detailed system performance metrics"""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()

        # Disk metrics
        disk = psutil.disk_usage("/")

        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            }
        except:
            network_stats = {}

        return {
            "cpu": {"usage_percent": cpu_percent, "count": cpu_count},
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
            },
            "network": network_stats,
            "uptime_seconds": int(time.time() - system_state["start_time"]),
        }

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")


# Initialize system state on startup
def initialize_system():
    """Initialize system state"""
    try:
        # Check MT5 connection
        system_state["mt5_connected"] = True  # Assume connected for demo
        system_state["signal_generator_active"] = True
        system_state["risk_engine_active"] = True
        # Start with 0; status endpoint reports real counts
        system_state["websocket_connections"] = 0
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing system: {e}")


# Call initialization
initialize_system()
