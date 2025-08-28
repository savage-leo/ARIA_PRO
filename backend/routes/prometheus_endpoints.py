"""
Prometheus Metrics API Endpoints
Expose metrics and monitoring data
"""

from fastapi import APIRouter, Response
from typing import Dict, Any
import logging
from backend.core.prometheus_metrics import get_prometheus_metrics as get_core_metrics, get_metrics_collector

router = APIRouter(prefix="/metrics", tags=["Metrics"])
logger = logging.getLogger(__name__)

@router.get("/prometheus")
async def prometheus_export():
    """Get metrics in Prometheus format"""
    try:
        metrics = get_core_metrics()
        metrics_text = metrics.get_metrics_text()
        
        return Response(
            content=metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        return Response(
            content=f"# Error getting metrics: {e}\n",
            media_type="text/plain"
        )

@router.get("/status")
async def get_metrics_status() -> Dict[str, Any]:
    """Get metrics collection status"""
    try:
        metrics = get_core_metrics()
        collector = get_metrics_collector()
        
        return {
            "ok": True,
            "metrics_enabled": metrics.config.enabled,
            "server_running": metrics.metrics_server_started,
            "server_port": metrics.config.port,
            "push_gateway": metrics.config.push_gateway_url,
            "collector_running": collector.running,
            "prometheus_available": hasattr(metrics, 'trade_executions')
        }
    except Exception as e:
        logger.error(f"Failed to get metrics status: {e}")
        return {"ok": False, "error": str(e)}

@router.post("/start")
async def start_metrics_collection() -> Dict[str, Any]:
    """Start metrics collection"""
    try:
        collector = get_metrics_collector()
        await collector.start()
        
        return {
            "ok": True,
            "message": "Metrics collection started",
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Failed to start metrics collection: {e}")
        return {"ok": False, "error": str(e)}

@router.post("/stop")
async def stop_metrics_collection() -> Dict[str, Any]:
    """Stop metrics collection"""
    try:
        collector = get_metrics_collector()
        await collector.stop()
        
        return {
            "ok": True,
            "message": "Metrics collection stopped",
            "status": "stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop metrics collection: {e}")
        return {"ok": False, "error": str(e)}

@router.post("/push")
async def push_metrics() -> Dict[str, Any]:
    """Manually push metrics to gateway"""
    try:
        metrics = get_core_metrics()
        await metrics.push_metrics()
        
        return {
            "ok": True,
            "message": "Metrics pushed to gateway"
        }
    except Exception as e:
        logger.error(f"Failed to push metrics: {e}")
        return {"ok": False, "error": str(e)}
