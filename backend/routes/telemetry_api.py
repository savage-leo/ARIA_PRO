"""
ARIA PRO Telemetry API
Phase 1 Implementation: Real-time telemetry endpoints for monitoring and alerting
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import time
import logging

from backend.services.telemetry_monitor import telemetry_monitor

# Import Prometheus metrics if available
try:
    from backend.services.prometheus_metrics import prometheus_metrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.get("/summary")
async def get_telemetry_summary() -> Dict[str, Any]:
    """Get complete telemetry summary including performance and business metrics"""
    try:
        summary = telemetry_monitor.get_telemetry_summary()
        return {
            "success": True,
            "data": summary,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting telemetry summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get telemetry summary")


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get real-time performance metrics"""
    try:
        perf_metrics = telemetry_monitor.performance_monitor.get_performance_metrics()
        return {
            "success": True,
            "data": {
                "execution_latency": {
                    "p50_ms": perf_metrics.execution_latency_p50,
                    "p95_ms": perf_metrics.execution_latency_p95,
                    "p99_ms": perf_metrics.execution_latency_p99
                },
                "slippage": {
                    "average": perf_metrics.slippage_average,
                    "p95": perf_metrics.slippage_p95
                },
                "throughput": {
                    "orders_per_minute": perf_metrics.throughput_orders_per_minute
                },
                "error_rate": perf_metrics.error_rate,
                "mt5_connection": {
                    "healthy": perf_metrics.mt5_connection_health
                }
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@router.get("/business")
async def get_business_metrics() -> Dict[str, Any]:
    """Get real-time business metrics"""
    try:
        business_metrics = telemetry_monitor.business_tracker.get_business_metrics()
        return {
            "success": True,
            "data": {
                "pnl": {
                    "real_time": business_metrics.real_time_pnl,
                    "daily": business_metrics.daily_pnl
                },
                "performance": {
                    "win_rate": business_metrics.win_rate,
                    "profit_factor": business_metrics.profit_factor,
                    "max_drawdown": business_metrics.max_drawdown
                },
                "trades": {
                    "total": business_metrics.total_trades,
                    "winning": business_metrics.winning_trades,
                    "losing": business_metrics.total_trades - business_metrics.winning_trades
                },
                "equity": {
                    "current": business_metrics.max_equity,
                    "max": business_metrics.max_equity
                }
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting business metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get business metrics")


@router.get("/alerts")
async def get_alerts(severity: str = None, limit: int = 10) -> Dict[str, Any]:
    """Get recent alerts, optionally filtered by severity"""
    try:
        alerts = telemetry_monitor.performance_monitor.get_alerts(severity)
        recent_alerts = list(alerts)[-limit:] if alerts else []
        
        return {
            "success": True,
            "data": {
                "alerts": [
                    {
                        "timestamp": alert.timestamp,
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "action_required": alert.action_required
                    }
                    for alert in recent_alerts
                ],
                "total_alerts": len(alerts) if alerts else 0,
                "critical_alerts": len([a for a in alerts if a.severity == "critical"]) if alerts else 0
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")


@router.get("/health")
async def get_telemetry_health() -> Dict[str, Any]:
    """Get telemetry system health status"""
    try:
        summary = telemetry_monitor.get_telemetry_summary()
        
        # Determine overall health status
        critical_alerts = [a for a in summary["alerts"] if a["severity"] == "critical"]
        warning_alerts = [a for a in summary["alerts"] if a["severity"] == "warning"]
        
        if critical_alerts:
            status = "critical"
        elif warning_alerts:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "success": True,
            "data": {
                "status": status,
                "mt5_connected": summary["performance"]["mt5_connection_health"],
                "critical_alerts": len(critical_alerts),
                "warning_alerts": len(warning_alerts),
                "last_update": summary["timestamp"]
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting telemetry health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get telemetry health")


@router.post("/track-execution")
async def track_execution(execution_data: Dict[str, Any]) -> Dict[str, Any]:
    """Track a new execution for telemetry monitoring"""
    try:
        start_time = execution_data.get("start_time")
        end_time = execution_data.get("end_time")
        expected_price = execution_data.get("expected_price")
        actual_price = execution_data.get("actual_price")
        trade_data = execution_data.get("trade_data", {})
        
        if not all([start_time, end_time, expected_price, actual_price]):
            raise HTTPException(status_code=400, detail="Missing required execution data")
        
        result = telemetry_monitor.track_execution(
            start_time=start_time,
            end_time=end_time,
            expected_price=expected_price,
            actual_price=actual_price,
            trade_data=trade_data
        )
        
        return {
            "success": True,
            "data": result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error tracking execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to track execution")


@router.post("/track-error")
async def track_error(error_data: Dict[str, Any]) -> Dict[str, Any]:
    """Track an error for telemetry monitoring"""
    try:
        error_type = error_data.get("error_type", "unknown")
        telemetry_monitor.performance_monitor.track_error(error_type)
        
        return {
            "success": True,
            "message": f"Error tracked: {error_type}",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error tracking error: {e}")
        raise HTTPException(status_code=500, detail="Failed to track error")


@router.get("/dashboard")
async def get_telemetry_dashboard() -> Dict[str, Any]:
    """Get comprehensive dashboard data for telemetry monitoring"""
    try:
        summary = telemetry_monitor.get_telemetry_summary()
        
        # Calculate additional dashboard metrics
        perf_metrics = summary["performance"]
        business_metrics = summary["business"]
        
        # Performance health score (0-100)
        perf_score = 100
        if perf_metrics["execution_latency_p95"] > 100:
            perf_score -= 30
        if perf_metrics["slippage_p95"] > 0.001:
            perf_score -= 25
        if perf_metrics["error_rate"] > 0.05:
            perf_score -= 20
        if not perf_metrics["mt5_connection_health"]:
            perf_score -= 25
        
        perf_score = max(0, perf_score)
        
        # Business health score (0-100)
        business_score = 100
        if business_metrics["max_drawdown"] > 0.05:
            business_score -= 30
        if business_metrics["win_rate"] < 0.5:
            business_score -= 20
        if business_metrics["profit_factor"] < 1.0:
            business_score -= 25
        
        business_score = max(0, business_score)
        
        return {
            "success": True,
            "data": {
                "overall_status": summary["status"],
                "health_scores": {
                    "performance": perf_score,
                    "business": business_score,
                    "overall": (perf_score + business_score) / 2
                },
                "performance": {
                    "latency_ms": {
                        "p50": perf_metrics["execution_latency_p50"],
                        "p95": perf_metrics["execution_latency_p95"],
                        "p99": perf_metrics["execution_latency_p99"],
                        "status": "good" if perf_metrics["execution_latency_p95"] <= 100 else "warning"
                    },
                    "slippage": {
                        "average": perf_metrics["slippage_average"],
                        "p95": perf_metrics["slippage_p95"],
                        "status": "good" if perf_metrics["slippage_p95"] <= 0.001 else "warning"
                    },
                    "mt5_connection": {
                        "healthy": perf_metrics["mt5_connection_health"],
                        "status": "good" if perf_metrics["mt5_connection_health"] else "critical"
                    }
                },
                "business": {
                    "pnl": {
                        "real_time": business_metrics["real_time_pnl"],
                        "daily": business_metrics["daily_pnl"],
                        "status": "good" if business_metrics["real_time_pnl"] >= 0 else "warning"
                    },
                    "performance": {
                        "win_rate": business_metrics["win_rate"],
                        "profit_factor": business_metrics["profit_factor"],
                        "max_drawdown": business_metrics["max_drawdown"],
                        "status": "good" if business_metrics["max_drawdown"] <= 0.05 else "warning"
                    }
                },
                "alerts": {
                    "total": len(summary["alerts"]),
                    "critical": len([a for a in summary["alerts"] if a["severity"] == "critical"]),
                    "warning": len([a for a in summary["alerts"] if a["severity"] == "warning"]),
                    "recent": summary["alerts"][-5:]  # Last 5 alerts
                },
                "prometheus": {
                    "available": PROMETHEUS_AVAILABLE,
                    "metrics_endpoint": "/telemetry/prometheus" if PROMETHEUS_AVAILABLE else None
                }
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting telemetry dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get telemetry dashboard")


@router.get("/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus metrics endpoint"""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prometheus metrics not available")
    
    try:
        from fastapi.responses import Response
        metrics = prometheus_metrics.get_metrics()
        return Response(
            content=metrics,
            media_type=prometheus_metrics.get_metrics_content_type()
        )
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Prometheus metrics")
