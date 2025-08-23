"""
Alerting System API Endpoints
Manage alerts and monitoring configuration
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from backend.core.alerting_system import get_alerting_system, get_system_monitor, AlertType, AlertSeverity

router = APIRouter(prefix="/alerts", tags=["Alerting"])
logger = logging.getLogger(__name__)

@router.get("/active")
async def get_active_alerts() -> Dict[str, Any]:
    """Get all active alerts"""
    try:
        alerting = get_alerting_system()
        active_alerts = alerting.get_active_alerts()
        
        return {
            "ok": True,
            "alerts": [alert.to_dict() for alert in active_alerts],
            "count": len(active_alerts)
        }
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_alert_history(limit: int = 100) -> Dict[str, Any]:
    """Get alert history"""
    try:
        alerting = get_alerting_system()
        history = alerting.get_alert_history(limit)
        
        return {
            "ok": True,
            "alerts": [alert.to_dict() for alert in history],
            "count": len(history),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Failed to get alert history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/resolve/{alert_id}")
async def resolve_alert(alert_id: str) -> Dict[str, Any]:
    """Resolve an active alert"""
    try:
        alerting = get_alerting_system()
        resolved = await alerting.resolve_alert(alert_id)
        
        if resolved:
            return {
                "ok": True,
                "message": f"Alert {alert_id} resolved",
                "alert_id": alert_id
            }
        else:
            return {
                "ok": False,
                "message": f"Alert {alert_id} not found or already resolved",
                "alert_id": alert_id
            }
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test")
async def send_test_alert() -> Dict[str, Any]:
    """Send a test alert"""
    try:
        alerting = get_alerting_system()
        alert = await alerting.send_alert(
            AlertType.SYSTEM_FAILURE,
            AlertSeverity.LOW,
            "Test Alert",
            "This is a test alert to verify the alerting system is working",
            "test_component",
            {"test": True, "timestamp": "now"}
        )
        
        if alert:
            return {
                "ok": True,
                "message": "Test alert sent successfully",
                "alert": alert.to_dict()
            }
        else:
            return {
                "ok": False,
                "message": "Test alert was suppressed (cooldown/rate limit)"
            }
    except Exception as e:
        logger.error(f"Failed to send test alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_alerting_status() -> Dict[str, Any]:
    """Get alerting system status"""
    try:
        alerting = get_alerting_system()
        monitor = get_system_monitor()
        
        return {
            "ok": True,
            "alerting_enabled": alerting.config.enabled,
            "email_enabled": alerting.config.email_enabled,
            "webhook_configured": bool(alerting.config.webhook_url),
            "monitoring_active": monitor.monitoring,
            "active_alerts_count": len(alerting.get_active_alerts()),
            "alert_handlers_count": len(alerting.alert_handlers),
            "max_alerts_per_hour": alerting.config.max_alerts_per_hour,
            "alert_cooldown_seconds": alerting.config.alert_cooldown
        }
    except Exception as e:
        logger.error(f"Failed to get alerting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/start")
async def start_monitoring() -> Dict[str, Any]:
    """Start system monitoring"""
    try:
        monitor = get_system_monitor()
        await monitor.start_monitoring()
        
        return {
            "ok": True,
            "message": "System monitoring started",
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/stop")
async def stop_monitoring() -> Dict[str, Any]:
    """Stop system monitoring"""
    try:
        monitor = get_system_monitor()
        await monitor.stop_monitoring()
        
        return {
            "ok": True,
            "message": "System monitoring stopped",
            "status": "inactive"
        }
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_alert_types() -> Dict[str, Any]:
    """Get available alert types and severities"""
    return {
        "ok": True,
        "alert_types": [t.value for t in AlertType],
        "severities": [s.value for s in AlertSeverity]
    }
