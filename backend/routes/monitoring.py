from __future__ import annotations

"""
ARIA PRO Production Monitoring Dashboard
Real-time system metrics and trading performance monitoring
"""

from fastapi import APIRouter, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, Any, List
import time
import psutil
import os
import json
from datetime import datetime, timedelta
import logging
import importlib.util as importlib_util
import pathlib
import math
import asyncio

from backend.services.auto_trader import auto_trader
from backend.core.performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Global storage for system metrics
_system_metrics = {
    "start_time": time.time(),
    "uptime": 0,
    "cpu_usage": 0,
    "memory_usage": 0,
    "disk_usage": 0,
    "network_io": {"bytes_sent": 0, "bytes_recv": 0},
    "trading_metrics": {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "daily_pnl": 0.0,
        "max_drawdown": 0.0,
    },
    "system_health": {
        "mt5_connected": False,
        "orchestrator_running": False,
        "fusion_cores_active": 0,
        "last_heartbeat": time.time(),
    },
}


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    try:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Network
        network = psutil.net_io_counters()

        # Update metrics
        _system_metrics.update(
            {
                "uptime": time.time() - _system_metrics["start_time"],
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "network_io": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                },
            }
        )

        return _system_metrics
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return _system_metrics


@router.get("/dashboard")
async def get_dashboard() -> HTMLResponse:
    """Production monitoring dashboard"""
    metrics = get_system_metrics()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ARIA PRO Production Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
            .metric-label {{ font-weight: bold; color: #666; }}
            .metric-value {{ font-weight: bold; color: #333; }}
            .status-ok {{ color: #28a745; }}
            .status-warning {{ color: #ffc107; }}
            .status-error {{ color: #dc3545; }}
            .progress-bar {{ width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
            .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s; }}
            .refresh-btn {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }}
            .refresh-btn:hover {{ background: #0056b3; }}
        </style>
        <script>
            function refreshDashboard() {{
                location.reload();
            }}
            
            // Auto-refresh every 30 seconds
            setInterval(refreshDashboard, 30000);
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 ARIA PRO Production Dashboard</h1>
                <p>Institutional Forex AI Trading System</p>
                <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>
            </div>
            
            <div class="grid">
                <!-- System Health -->
                <div class="card">
                    <h3>🏥 System Health</h3>
                    <div class="metric">
                        <span class="metric-label">Uptime:</span>
                        <span class="metric-value">{timedelta(seconds=int(metrics['uptime']))}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">MT5 Connection:</span>
                        <span class="metric-value {'status-ok' if metrics['system_health']['mt5_connected'] else 'status-error'}">
                            {'✅ Connected' if metrics['system_health']['mt5_connected'] else '❌ Disconnected'}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Orchestrator:</span>
                        <span class="metric-value {'status-ok' if metrics['system_health']['orchestrator_running'] else 'status-warning'}">
                            {'✅ Running' if metrics['system_health']['orchestrator_running'] else '⚠️ Unknown'}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Fusion Cores:</span>
                        <span class="metric-value">{metrics['system_health']['fusion_cores_active']}</span>
                    </div>
                </div>
                
                <!-- System Resources -->
                <div class="card">
                    <h3>💻 System Resources</h3>
                    <div class="metric">
                        <span class="metric-label">CPU Usage:</span>
                        <span class="metric-value">{metrics['cpu_usage']:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {metrics['cpu_usage']}%"></div>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">Memory Usage:</span>
                        <span class="metric-value">{metrics['memory_usage']:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {metrics['memory_usage']}%"></div>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">Disk Usage:</span>
                        <span class="metric-value">{metrics['disk_usage']:.1f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {metrics['disk_usage']}%"></div>
                    </div>
                </div>
                
                <!-- Trading Performance -->
                <div class="card">
                    <h3>📈 Trading Performance</h3>
                    <div class="metric">
                        <span class="metric-label">Total Trades:</span>
                        <span class="metric-value">{metrics['trading_metrics']['total_trades']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Win Rate:</span>
                        <span class="metric-value">
                            {metrics['trading_metrics']['winning_trades'] / max(metrics['trading_metrics']['total_trades'], 1) * 100:.1f}%
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total P&L:</span>
                        <span class="metric-value {'status-ok' if metrics['trading_metrics']['total_pnl'] >= 0 else 'status-error'}">
                            ${metrics['trading_metrics']['total_pnl']:.2f}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Daily P&L:</span>
                        <span class="metric-value {'status-ok' if metrics['trading_metrics']['daily_pnl'] >= 0 else 'status-error'}">
                            ${metrics['trading_metrics']['daily_pnl']:.2f}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Max Drawdown:</span>
                        <span class="metric-value status-error">
                            {metrics['trading_metrics']['max_drawdown']:.2f}%
                        </span>
                    </div>
                </div>
                
                <!-- Network & I/O -->
                <div class="card">
                    <h3>🌐 Network & I/O</h3>
                    <div class="metric">
                        <span class="metric-label">Data Sent:</span>
                        <span class="metric-value">{metrics['network_io']['bytes_sent'] / 1024 / 1024:.2f} MB</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Data Received:</span>
                        <span class="metric-value">{metrics['network_io']['bytes_recv'] / 1024 / 1024:.2f} MB</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Last Heartbeat:</span>
                        <span class="metric-value">{datetime.fromtimestamp(metrics['system_health']['last_heartbeat']).strftime('%H:%M:%S')}</span>
                    </div>
                </div>
            </div>
            
            <div class="card" style="margin-top: 20px;">
                <h3>🔗 Quick Links</h3>
                <p><a href="/health">Health Check</a> | <a href="/debug/health/detailed">Detailed Health</a> | <a href="/debug/ideas">Latest Ideas</a> | <a href="/docs">API Documentation</a></p>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics as JSON"""
    return get_system_metrics()


@router.post("/update-trading-metrics")
async def update_trading_metrics(metrics: Dict[str, Any]):
    """Update trading metrics (called by orchestrator)"""
    _system_metrics["trading_metrics"].update(metrics)
    logger.info(f"Updated trading metrics: {metrics}")


@router.post("/update-system-health")
async def update_system_health(health: Dict[str, Any]):
    """Update system health (called by orchestrator)"""
    _system_metrics["system_health"].update(health)
    _system_metrics["system_health"]["last_heartbeat"] = time.time()
    logger.info(f"Updated system health: {health}")


@router.get("/alerts")
async def get_alerts() -> List[Dict[str, Any]]:
    """Get system alerts and warnings"""
    alerts = []
    metrics = get_system_metrics()

    # CPU usage alert
    if metrics["cpu_usage"] > 80:
        alerts.append(
            {
                "level": "warning",
                "message": f"High CPU usage: {metrics['cpu_usage']:.1f}%",
                "timestamp": time.time(),
            }
        )

    # Memory usage alert
    if metrics["memory_usage"] > 85:
        alerts.append(
            {
                "level": "warning",
                "message": f"High memory usage: {metrics['memory_usage']:.1f}%",
                "timestamp": time.time(),
            }
        )

    # MT5 connection alert
    if not metrics["system_health"]["mt5_connected"]:
        alerts.append(
            {
                "level": "error",
                "message": "MT5 connection lost",
                "timestamp": time.time(),
            }
        )

    # Orchestrator status alert
    if not metrics["system_health"]["orchestrator_running"]:
        alerts.append(
            {
                "level": "error",
                "message": "Phase 3 Orchestrator not running",
                "timestamp": time.time(),
            }
        )

    return alerts


# -*- coding: utf-8 -*-
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT_ROOT / "data"))
GATING_PATH = pathlib.Path(
    os.getenv("ARIA_GATING_JSON", PROJECT_ROOT / "config" / "gating.default.json")
)


def _read_json(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"JSON not found: {p}")
    return json.loads(p.read_text())


def _latest_live_log() -> pathlib.Path | None:
    files = sorted((DATA_ROOT / "live_logs").glob("decisions_*.jsonl"))
    return files[-1] if files else None


@router.get("/status")
def status() -> Dict[str, Any]:
    gating = _read_json(GATING_PATH)
    calib_dir = DATA_ROOT / "calibration" / "current"
    manifest = {}
    if calib_dir.exists():
        for sym_dir in calib_dir.iterdir():
            if sym_dir.is_dir():
                f = sym_dir / "fusion_lr.json"
                if f.exists():
                    j = _read_json(f)
                    manifest[sym_dir.name] = {
                        "fusion_type": j.get("type", "logreg"),
                        "version_hash": j.get("version_hash", ""),
                        "features_order": j.get("features_order", []),
                    }
    return {
        "utc": datetime.utcnow().isoformat() + "Z",
        "gating_version": gating.get("version"),
        "thresholds": gating.get("default_thresholds"),
        "symbols": sorted(list(manifest.keys())),
        "calibration_manifest": manifest,
        "data_root": str(DATA_ROOT),
    }


@router.get("/gating")
def gating() -> Dict[str, Any]:
    return _read_json(GATING_PATH)


@router.get("/latency")
def latency(n: int = 5000) -> Dict[str, Any]:
    path = _latest_live_log()
    if not path:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    vals: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                vals.append(float(obj.get("lat_ms", 0.0)))
            except Exception:
                continue
    if not vals:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    vals = vals[-n:] if len(vals) > n else vals
    s = sorted(vals)

    def pct(p):
        k = max(0, min(len(s) - 1, int(math.ceil(p * len(s))) - 1))
        return s[k]

    return {"count": len(s), "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


@router.get("/decisions/last")
def decisions_last(n: int = 1000, symbol: str | None = None) -> Dict[str, Any]:
    path = _latest_live_log()
    if not path:
        return {"count": 0, "items": []}
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if symbol and obj.get("symbol") != symbol:
                    continue
                items.append(obj)
            except Exception:
                continue
    return {"count": len(items), "items": items[-n:]}


@router.get("/metrics/prom")
def metrics_prom() -> Response:
    """
    Prometheus-style text metrics (no client lib). Emits latency and action dist from latest log.
    """
    path = _latest_live_log()
    lines: List[str] = []
    lines.append(
        f'# HELP aria_info Static info\n# TYPE aria_info gauge\naria_info{{data_root="{DATA_ROOT}"}} 1'
    )
    if not path or not path.exists():
        return Response("\n".join(lines), media_type="text/plain")
    lat, act = [], {"LONG": 0, "SHORT": 0, "FLAT": 0}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                lat.append(float(obj.get("lat_ms", 0.0)))
                act[str(obj.get("action", "FLAT"))] = (
                    act.get(str(obj.get("action", "FLAT")), 0) + 1
                )
            except Exception:
                continue
    if lat:
        s = sorted(lat)

        def pct(p):
            k = max(0, min(len(s) - 1, int(math.ceil(p * len(s))) - 1))
            return s[k]

        lines += [
            "# HELP aria_latency_ms Decision latency ms",
            "# TYPE aria_latency_ms summary",
            f'aria_latency_ms{{quantile="0.01"}} {pct(0.01)}',
            f'aria_latency_ms{{quantile="0.5"}} {pct(0.5)}',
            f'aria_latency_ms{{quantile="0.95"}} {pct(0.95)}',
            f'aria_latency_ms{{quantile="0.99"}} {pct(0.99)}',
            f'aria_latency_ms{{quantile="0.999"}} {pct(0.999)}',
            f"aria_latency_ms_sum {sum(lat)}",
            f"aria_latency_ms_count {len(lat)}",
        ]
    lines += [
        "# HELP aria_actions_total Decision action counts",
        "# TYPE aria_actions_total counter",
        f'aria_actions_total{{action="LONG"}} {act.get("LONG",0)}',
        f'aria_actions_total{{action="SHORT"}} {act.get("SHORT",0)}',
        f'aria_actions_total{{action="FLAT"}} {act.get("FLAT",0)}',
    ]
    return Response("\n".join(lines), media_type="text/plain")


@router.get("/models/status")
async def get_models_status() -> Dict[str, Any]:
    """
    Light-weight model status probe.
    - Does NOT load models to avoid CPU/RAM spikes on developer machines.
    - Reports presence of artifacts and required Python modules.
    """
    def _mod_available(name: str) -> bool:
        try:
            return importlib_util.find_spec(name) is not None
        except Exception:
            return False

    required_by_key = {
        "xgb": ["onnxruntime"],
        "lstm": ["onnxruntime"],
        "cnn": ["onnxruntime"],
        "vision": ["onnxruntime"],
        "ppo": ["stable_baselines3"],
        "llm_macro": ["llama_cpp"],
    }

    model_files = {
        "xgb": PROJECT_ROOT / "backend" / "models" / "xgboost_forex.onnx",
        "lstm": PROJECT_ROOT / "backend" / "models" / "lstm_forex.onnx",
        "cnn": PROJECT_ROOT / "backend" / "models" / "cnn_patterns.onnx",
        "ppo": PROJECT_ROOT / "backend" / "models" / "ppo_trader.zip",
        "vision": PROJECT_ROOT / "backend" / "models" / "vision_model.onnx",
        "llm_macro": PROJECT_ROOT / "backend" / "models" / "llm_macro.bin",
    }

    models_status: Dict[str, Any] = {}
    for key, path in model_files.items():
        p = pathlib.Path(path)
        exists = p.exists()
        size_bytes = p.stat().st_size if exists else None
        mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat() if exists else None
        reqs = {m: _mod_available(m) for m in required_by_key.get(key, [])}

        models_status[key] = {
            "name": key,
            "model_path": str(p),
            "exists": exists,
            "size_bytes": size_bytes,
            "mtime": mtime,
            "modules_available": reqs,
            "loaded": False,  # not loading here by design
            "note": "light probe; no model load attempted",
        }

    return {"ok": True, "models": models_status}


@router.get("/auto-trader/status")
async def get_auto_trader_status() -> Dict[str, Any]:
    """Get AutoTrader runtime status and configuration"""
    try:
        enabled = os.environ.get("AUTO_TRADE_ENABLED", "0") in ("1", "true", "True")
        status = auto_trader.get_status()
        return {"ok": True, "enabled": enabled, "status": status}
    except Exception as e:
        logger.error(f"Error getting AutoTrader status: {e}")
        return {"ok": False, "error": str(e)}


# Performance Monitoring Endpoints
@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get current performance metrics for all models and system."""
    try:
        monitor = get_performance_monitor()
        return {
            "system": monitor.get_system_metrics(),
            "models": monitor.get_metrics(),
            "symbols": monitor.get_symbol_metrics(),
            "thresholds": monitor.thresholds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/models/{model_name}")
async def get_model_metrics(model_name: str):
    """Get detailed metrics for a specific model."""
    monitor = get_performance_monitor()
    metrics = monitor.get_metrics(model_name)
    if not metrics:
        return JSONResponse(
            status_code=404,
            content={"error": f"No metrics found for model: {model_name}"}
        )
    return metrics


@router.get("/performance/symbols")
async def get_symbols_metrics():
    """Get per-symbol metrics for all symbols."""
    try:
        monitor = get_performance_monitor()
        return monitor.get_symbol_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/symbols/{symbol}")
async def get_symbol_metrics(symbol: str):
    """Get per-symbol metrics for a specific symbol."""
    monitor = get_performance_monitor()
    metrics = monitor.get_symbol_metrics(symbol)
    if not metrics:
        return JSONResponse(
            status_code=404,
            content={"error": f"No metrics found for symbol: {symbol}"}
        )
    return metrics


@router.get("/performance/thresholds")
async def get_thresholds():
    """Get current monitoring thresholds (from environment with defaults)."""
    monitor = get_performance_monitor()
    return monitor.thresholds


@router.websocket("/ws/performance")
async def websocket_performance_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time performance monitoring."""
    await websocket.accept()
    monitor = get_performance_monitor()
    monitor.add_connection(websocket)
    
    try:
        # Start monitoring if not already started
        await monitor.start_monitoring()
        # Immediately send a snapshot so clients don't wait for the first interval tick
        try:
            await websocket.send_json({
                "type": "system_metrics",
                "metrics": monitor.get_system_metrics(),
            })
        except Exception as e:
            logger.debug(f"Initial system metrics snapshot send failed: {e}")
        
        while True:
            # Keep connection alive and handle incoming messages
            try:
                message = await websocket.receive_text()
                # Handle any client messages if needed
                if message == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except Exception as e:
                logger.debug(f"WebSocket message handling error: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("Performance monitoring WebSocket disconnected")
    except Exception as e:
        logger.error(f"Performance monitoring WebSocket error: {e}")
    finally:
        monitor.remove_connection(websocket)
