from fastapi import FastAPI, APIRouter, Depends, HTTPException, Response
from backend.routes import (
    auth,
    trading,
    account,
    market,
    positions,
    signals,
    monitoring,
    monitoring_enhanced,
    live_execution_api,
    smc_routes,
    websocket,
    debug,
    trade_memory_api,
    analytics,
    institutional_ai,
    system_management,
    training,
    telemetry_api,
    health,
    hedge_fund_dashboard,
)
from backend.services.data_source_manager import data_source_manager
from backend.services.auto_trader import auto_trader
from backend.services.mt5_market_data import mt5_market_feed
from backend.services.cpp_integration import cpp_service
from backend.core.performance_monitor import get_performance_monitor
from backend.core.config import get_settings
from backend.core.live_guard import enforce_live_only
from backend.core.rate_limit import get_rate_limiter, RateLimitMiddleware
from backend.core.auth import require_admin, require_trader
from backend.core.security_middleware import setup_security_middleware
from backend.core.audit import get_audit_logger, AuditEventType
from backend.core.health import get_health_checker
import asyncio
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def _configure_logging(level_name: str) -> None:
    level_name = (level_name or "INFO").upper()
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    level = level_map.get(level_name, logging.INFO)

    root = logging.getLogger()
    if not root.hasHandlers():
        # Configure basic logging with UTF-8 encoding for Windows compatibility
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler()
            ]
        )
        # Force UTF-8 encoding on console handler for Windows
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and hasattr(handler.stream, 'reconfigure'):
                try:
                    handler.stream.reconfigure(encoding='utf-8')
                except:
                    pass
    root.setLevel(level)

    # align common server loggers as well
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)

# Initialize settings with validation
try:
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)
except Exception as e:
    print(f"CRITICAL: Configuration validation failed: {e}")
    print("Please check your environment variables and ensure all required secrets are set.")
    raise SystemExit(1)


def _configure_file_logging(log_dir: str) -> None:
    """Attach rotating file handlers for info and error logs."""
    try:
        os.makedirs(log_dir, exist_ok=True)

        root = logging.getLogger()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Avoid duplicating handlers if reloaded
        existing_paths = set()
        for h in root.handlers:
            if hasattr(h, "baseFilename"):
                existing_paths.add(os.path.abspath(getattr(h, "baseFilename", "")))

        info_path = os.path.abspath(os.path.join(log_dir, "backend.log"))
        err_path = os.path.abspath(os.path.join(log_dir, "backend.err"))

        if info_path not in existing_paths:
            fh_info = RotatingFileHandler(
                info_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
            )
            fh_info.setLevel(logging.INFO)
            fh_info.setFormatter(formatter)
            root.addHandler(fh_info)

        if err_path not in existing_paths:
            fh_err = RotatingFileHandler(
                err_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
            )
            fh_err.setLevel(logging.ERROR)
            fh_err.setFormatter(formatter)
            root.addHandler(fh_err)
    except Exception:
        # Never fail app due to logging configuration issues
        pass


_configure_file_logging(settings.ARIA_LOG_DIR)
logger = logging.getLogger(__name__)

# Validate production security requirements
logger.info("Validating production security configuration...")
if not settings.JWT_SECRET_KEY or len(settings.JWT_SECRET_KEY) < 32:
    logger.critical("JWT_SECRET_KEY must be at least 32 characters for production")
    raise SystemExit(1)
if not settings.ADMIN_API_KEY or len(settings.ADMIN_API_KEY) < 16:
    logger.critical("ADMIN_API_KEY must be at least 16 characters for production")
    raise SystemExit(1)
if not settings.ARIA_WS_TOKEN or len(settings.ARIA_WS_TOKEN) < 16:
    logger.critical("ARIA_WS_TOKEN must be at least 16 characters for production")
    raise SystemExit(1)
logger.info("✅ Security configuration validated")




def _validate_runtime_config(settings) -> None:
    """Validate production runtime configuration"""
    try:
        os.makedirs(settings.ARIA_LOG_DIR, exist_ok=True)
        
        # Validate MT5 configuration if enabled
        if settings.mt5_enabled:
            if not all([settings.MT5_LOGIN, settings.MT5_PASSWORD, settings.MT5_SERVER]):
                raise RuntimeError("MT5 enabled but missing required credentials (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)")
        
        # Validate auto-trading configuration
        if settings.AUTO_TRADE_ENABLED and not settings.AUTO_TRADE_DRY_RUN:
            if not settings.ARIA_ENABLE_EXEC:
                raise RuntimeError("Live auto-trading requires ARIA_ENABLE_EXEC=1")
            if not settings.mt5_enabled:
                raise RuntimeError("Live auto-trading requires MT5 connection (ARIA_ENABLE_MT5=1)")
        
        # Validate CORS configuration
        if not settings.cors_origins_list:
            logger.warning("No CORS origins configured - frontend requests will be blocked")
            
        logger.info("✅ Runtime configuration validated")
        
    except Exception as e:
        logger.critical(f"Runtime configuration validation failed: {e}")
        raise SystemExit(1)


# Validate runtime configuration
_validate_runtime_config(settings)

# Initialize FastAPI app
app = FastAPI(
    title="ARIA Institutional Forex AI",
    version="1.2.0",
    description="Production-grade institutional Forex trading platform with AI signals",
    docs_url="/docs" if settings.JWT_ENABLED else None,  # Hide docs in production
    redoc_url="/redoc" if settings.JWT_ENABLED else None
)

# Setup comprehensive security middleware stack
setup_security_middleware(app, settings, debug=False)

# Add rate limiting middleware
if settings.RATE_LIMIT_ENABLED:
    rate_limiter = get_rate_limiter()
    app.add_middleware(RateLimitMiddleware, rate_limiter)
    logger.info(f"✅ Rate limiting enabled: {settings.RATE_LIMIT_REQUESTS_PER_MINUTE} req/min")
else:
    logger.warning("Rate limiting disabled - not recommended for production")

# API v1 Router with versioning
api_v1 = APIRouter(prefix="/api")
api_v1.include_router(auth.router, tags=["Authentication"])
api_v1.include_router(trading.router, prefix="/trading", tags=["Trading"])
api_v1.include_router(account.router, prefix="/account", tags=["Account"])
api_v1.include_router(market.router, prefix="/market", tags=["Market"])
api_v1.include_router(positions.router, prefix="/positions", tags=["Positions"])
api_v1.include_router(signals.router, prefix="/signals", tags=["Signals"])

# Include API v1 router
app.include_router(api_v1)

# Legacy routes (for backward compatibility)
app.include_router(auth.router, include_in_schema=False)
app.include_router(trading.router, prefix="/trading", tags=["Trading"], include_in_schema=False)
app.include_router(account.router, prefix="/account", tags=["Account"], include_in_schema=False)
app.include_router(market.router, prefix="/market", tags=["Market"], include_in_schema=False)
app.include_router(positions.router, prefix="/positions", tags=["Positions"], include_in_schema=False)
app.include_router(signals.router, prefix="/signals", tags=["Signals"], include_in_schema=False)
app.include_router(monitoring.router, include_in_schema=False)  # router already has prefix /monitoring
app.include_router(monitoring_enhanced.router, include_in_schema=False)  # Enhanced institutional monitoring
app.include_router(live_execution_api.router, include_in_schema=False)  # Live MT5 execution API
app.include_router(smc_routes.router, include_in_schema=False)
app.include_router(websocket.router, tags=["WebSocket"], include_in_schema=False)
app.include_router(debug.router, include_in_schema=False)
app.include_router(trade_memory_api.router, include_in_schema=False)  # router defines its own prefix/tags
app.include_router(analytics.router, include_in_schema=False)
app.include_router(institutional_ai.router, include_in_schema=False)
app.include_router(system_management.router, include_in_schema=False)
app.include_router(training.router, include_in_schema=False)
app.include_router(telemetry_api.router, include_in_schema=False)
app.include_router(health.router, include_in_schema=False)
from backend.routes.model_management import router as model_management_router
from backend.routes.cicd_management import router as cicd_management_router
from backend.routes.hot_swap_admin import router as hot_swap_admin_router
app.include_router(model_management_router, include_in_schema=False)
app.include_router(cicd_management_router, include_in_schema=False)
app.include_router(hot_swap_admin_router, include_in_schema=False)

# Provide optional /api namespace for all endpoints (backward-compatible)
api_router = APIRouter(prefix="/api")
api_router.include_router(auth.router)
api_router.include_router(trading.router, prefix="/trading", tags=["Trading"])
api_router.include_router(account.router, prefix="/account", tags=["Account"])
api_router.include_router(market.router, prefix="/market", tags=["Market"])
api_router.include_router(positions.router, prefix="/positions", tags=["Positions"])
api_router.include_router(signals.router, prefix="/signals", tags=["Signals"])
api_router.include_router(monitoring.router)
api_router.include_router(monitoring_enhanced.router)
api_router.include_router(live_execution_api.router)
api_router.include_router(smc_routes.router)
api_router.include_router(websocket.router, tags=["WebSocket"])
api_router.include_router(debug.router)
api_router.include_router(trade_memory_api.router)
api_router.include_router(analytics.router)
api_router.include_router(institutional_ai.router)
api_router.include_router(system_management.router)
api_router.include_router(training.router)
api_router.include_router(telemetry_api.router)
api_router.include_router(health.router)
api_router.include_router(model_management_router)
api_router.include_router(cicd_management_router)
api_router.include_router(hot_swap_admin_router)

# Add cache management router
from backend.routes.cache_management import router as cache_management_router
api_router.include_router(cache_management_router)

# Add Prometheus metrics router
from backend.routes.prometheus_endpoints import router as prometheus_router
api_router.include_router(prometheus_router)

# Add alerting system router
from backend.routes.alerting_endpoints import router as alerting_router
api_router.include_router(alerting_router)

# Add WebSocket management router
from backend.routes.websocket_management import router as websocket_mgmt_router
api_router.include_router(websocket_mgmt_router)

# Add WebSocket endpoint
from backend.routes.websocket_endpoint import router as websocket_router
app.include_router(websocket_router)

app.include_router(api_router)

@api_router.get("/health")
async def api_health():
    """Basic health check endpoint"""
    health_checker = get_health_checker()
    health_status = await health_checker.check_all()
    
    # Return 503 if unhealthy
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@api_router.get("/ready")
async def api_ready():
    """Readiness check for load balancer"""
    health_checker = get_health_checker()
    readiness = await health_checker.readiness_check()
    
    if not readiness["ready"]:
        raise HTTPException(status_code=503, detail=readiness)
    
    return readiness

@api_router.get("/data-sources/status")
def api_data_sources_status():
    """Get status of all data sources (namespaced alias)."""
    return data_source_manager.get_status()

@api_router.get("/audit/recent")
async def get_recent_audit_events(
    event_type: str = None,
    limit: int = 100,
    current_user: dict = Depends(require_admin)
):
    """Get recent audit events (admin only)"""
    audit_logger = get_audit_logger()
    return audit_logger.get_recent_events(
        event_type=event_type,
        limit=limit
    )

@api_router.post("/audit/verify")
async def verify_audit_integrity(
    start_id: int = 1,
    end_id: int = None,
    current_user: dict = Depends(require_admin)
):
    """Verify audit log integrity (admin only)"""
    audit_logger = get_audit_logger()
    is_valid = audit_logger.verify_integrity(start_id, end_id)
    return {"valid": is_valid, "range": {"start": start_id, "end": end_id}}

@api_router.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    metrics_collector = get_metrics_collector()
    
    # Update MT5 metrics if connected
    try:
        if S.mt5_enabled:
            import MetaTrader5 as mt5
            if mt5.initialize():
                account_info = mt5.account_info()
                if account_info:
                    metrics_collector.update_mt5_metrics(
                        connected=True,
                        balance=account_info.balance,
                        equity=account_info.equity,
                        margin=account_info.margin
                    )
                else:
                    metrics_collector.update_mt5_metrics(connected=False)
                mt5.shutdown()
            else:
                metrics_collector.update_mt5_metrics(connected=False)
    except:
        metrics_collector.update_mt5_metrics(connected=False)
    
    # Update auto-trader metrics
    if auto_trader:
        metrics_collector.update_auto_trader_status(auto_trader.running)
    
    # Generate Prometheus format metrics
    metrics_data = metrics_collector.get_metrics()
    
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4"
    )

app.include_router(api_router)

# Register data sources with live-only enforcement
try:
    enforce_live_only(settings)
    if mt5_market_feed not in data_source_manager.data_sources:
        data_source_manager.data_sources.append(mt5_market_feed)
    data_source_manager.mt5_feed = mt5_market_feed
    logger.info("✅ Registered MT5MarketFeed with DataSourceManager (live-only mode)")
except Exception as e:
    logger.critical(f"Failed to enforce live MT5 market data feed: {e}")
    raise SystemExit(1)

# Register C++ SMC integration service (runs in Python fallback if C++ disabled)
try:
    if cpp_service not in data_source_manager.data_sources:
        data_source_manager.data_sources.append(cpp_service)
        logger.info("Registered CppService with DataSourceManager")
except Exception as e:
    logger.error(f"Failed to register CppService: {e}")

# Register real AI signal generator
try:
    from backend.services.real_ai_signal_generator import real_ai_signal_generator

    if real_ai_signal_generator not in data_source_manager.data_sources:
        data_source_manager.data_sources.append(real_ai_signal_generator)
        logger.info("Registered RealAISignalGenerator with DataSourceManager")
    # Expose to manager for live-mode enforcement
    data_source_manager.ai_signal_generator = real_ai_signal_generator
except Exception as e:
    logger.error(f"Failed to register RealAISignalGenerator: {e}")



@app.on_event("startup")
async def startup_event():
    """Start data sources when the application starts with production checks"""
    logger.info("Starting ARIA PRO backend...")
    
    # Log system startup in audit trail
    audit_logger = get_audit_logger()
    audit_logger.log_event(
        event_type=AuditEventType.SYSTEM_START,
        action="System startup initiated",
        details={"version": "1.2.0", "environment": "production"}
    )
    
    # Run fail-fast health checks for critical components
    health_checker = get_health_checker()
    try:
        # Check MT5 if enabled - FAIL FAST
        if settings.mt5_enabled:
            mt5_health = await health_checker.check_mt5_connection()
            if mt5_health.status == "unhealthy":
                logger.critical(f"MT5 connection failed at startup: {mt5_health.error_message}")
                raise RuntimeError(f"Cannot start: MT5 required but unavailable - {mt5_health.error_message}")
        
        # Check required models - FAIL FAST
        if settings.AUTO_TRADE_ENABLED:
            models_health = await health_checker.check_models()
            if models_health.status == "unhealthy":
                logger.critical(f"Required models missing at startup: {models_health.error_message}")
                raise RuntimeError(f"Cannot start: {models_health.error_message}")
    except Exception as e:
        audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_STOP,
            action="Startup failed - critical component unavailable",
            details={"error": str(e)}
        )
        raise
    
    # Start rate limiter cleanup task
    if settings.RATE_LIMIT_ENABLED:
        rate_limiter = get_rate_limiter()
        rate_limiter.start_cleanup_task()
        logger.info("✅ Rate limiter cleanup task started")
    
    try:
        # Log a concise configuration snapshot (no secrets)
        cfg = {
            "LOG_LEVEL": settings.LOG_LEVEL.upper(),
            "MT5_ENABLED": str(int(settings.mt5_enabled)),
            "AUTO_TRADE_ENABLED": str(int(settings.AUTO_TRADE_ENABLED)),
            "PRIMARY_MODEL": settings.AUTO_TRADE_PRIMARY_MODEL,
            "SYMBOLS": ",".join(settings.symbols_list),
            "EXEC_ENABLED": str(int(settings.ARIA_ENABLE_EXEC)),
            "JWT_ENABLED": str(int(settings.JWT_ENABLED)),
            "RATE_LIMIT_ENABLED": str(int(settings.RATE_LIMIT_ENABLED)),
        }
        logger.info(f"✅ Runtime config: {cfg}")
    except Exception:
        # Never fail startup due to logging
        pass
    try:
        # Register LLM monitor service when enabled
        try:
            if settings.LLM_MONITOR_ENABLED:
                from backend.monitoring.llm_monitor import llm_monitor_service
                if llm_monitor_service not in data_source_manager.data_sources:
                    data_source_manager.data_sources.append(llm_monitor_service)
                    logger.info("✅ Registered LLMMonitorService with DataSourceManager")
        except Exception as e:
            logger.error(f"Failed to register LLMMonitorService: {e}")

        # Initialize Redis cache
        logger.info("Initializing Redis cache...")
        from backend.core.redis_cache import get_redis_cache
        redis_cache = get_redis_cache()
        await redis_cache.connect()
        
        # Initialize Prometheus metrics
        logger.info("Starting Prometheus metrics collection...")
        from backend.core.prometheus_metrics import get_metrics_collector
        metrics_collector = get_metrics_collector()
        await metrics_collector.start()
        
        # Initialize alerting system
        logger.info("Starting alerting system...")
        from backend.core.alerting_system import get_alerting_system
        alerting_system = get_alerting_system()
        await alerting_system.start()
        logger.info("Alerting system started")
        
        # Initialize WebSocket pool
        from backend.core.websocket_pool import get_websocket_pool
        websocket_pool = get_websocket_pool()
        await websocket_pool.start()
        logger.info("WebSocket pool started")
        
        # Start periodic WebSocket broadcasting
        from backend.routes.websocket_endpoint import start_periodic_broadcast
        await start_periodic_broadcast()
        
        # Initialize system monitor
        from backend.core.system_monitor import get_system_monitor
        system_monitor = get_system_monitor()
        await system_monitor.start_monitoring()
        
        # Initialize data sources
        logger.info("Initializing data sources...")
        await data_source_manager.start_all()
        logger.info("Data sources started successfully")

        # Start AutoTrader if enabled
        if settings.AUTO_TRADE_ENABLED:
            try:
                app.state.auto_trader_task = asyncio.create_task(auto_trader.start())
                logger.info("✅ AutoTrader started")
            except Exception as e:
                logger.error(f"Failed to start AutoTrader: {e}")
                if not settings.AUTO_TRADE_DRY_RUN:
                    # Fail fast for live trading issues
                    raise

        # Start Performance Monitor background tasks
        try:
            monitor = get_performance_monitor()
            await monitor.start_monitoring()
            # Ensure only a single logger task is running
            task = getattr(app.state, "perf_log_task", None)
            if not task or task.done():
                app.state.perf_log_task = asyncio.create_task(monitor.log_metrics())
                logger.info("PerformanceMonitor started (system monitor + periodic logger)")
        except Exception as e:
            logger.error(f"Failed to start PerformanceMonitor: {e}")
    except Exception as e:
        logger.error(f"Error starting data sources: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown of all background tasks and services"""
    logger.info("Initiating graceful shutdown of ARIA PRO backend...")
    
    # Log system shutdown in audit trail
    audit_logger = get_audit_logger()
    audit_logger.log_event(
        event_type=AuditEventType.SYSTEM_STOP,
        action="System shutdown initiated",
        details={"reason": "normal_shutdown"}
    )
    
    shutdown_tasks = []
    
    try:
        # 1. Stop WebSocket broadcasting
        logger.info("Stopping WebSocket broadcasting...")
        try:
            from backend.routes.websocket_endpoint import stop_periodic_broadcast
            await stop_periodic_broadcast()
        except Exception as e:
            logger.error(f"Error stopping WebSocket broadcasting: {e}")
        
        # 2. Stop WebSocket pool
        logger.info("Stopping WebSocket pool...")
        try:
            from backend.core.websocket_pool import get_websocket_pool
            websocket_pool = get_websocket_pool()
            await websocket_pool.stop()
            logger.info("WebSocket pool stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket pool: {e}")
        
        # 3. Stop alerting system
        logger.info("Stopping alerting system...")
        try:
            from backend.core.alerting_system import get_alerting_system
            alerting_system = get_alerting_system()
            await alerting_system.stop()
            logger.info("Alerting system stopped")
        except Exception as e:
            logger.error(f"Error stopping alerting system: {e}")
        
        # 4. Stop Prometheus metrics collection
        logger.info("Stopping Prometheus metrics collection...")
        try:
            from backend.core.prometheus_metrics import get_metrics_collector
            metrics_collector = get_metrics_collector()
            await metrics_collector.stop()
            logger.info("Prometheus metrics collection stopped")
        except Exception as e:
            logger.error(f"Error stopping Prometheus metrics: {e}")
        
        # 5. Disconnect Redis cache
        logger.info("Disconnecting Redis cache...")
        try:
            from backend.core.redis_cache import get_redis_cache
            redis_cache = get_redis_cache()
            await redis_cache.disconnect()
            logger.info("Redis cache disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting Redis cache: {e}")
        
        # 6. Stop AutoTrader if running
        try:
            if hasattr(app.state, 'auto_trader_task') and app.state.auto_trader_task:
                app.state.auto_trader_task.cancel()
                shutdown_tasks.append(app.state.auto_trader_task)
        except Exception as e:
            logger.error(f"Error stopping AutoTrader: {e}")
        
        # 7. Stop data sources
        try:
            data_source_manager = get_data_source_manager()
            await data_source_manager.stop_all()
            logger.info("Data sources stopped")
        except Exception as e:
            logger.error(f"Error stopping data sources: {e}")

        # 8. Wait for all cancelled tasks to complete
        if shutdown_tasks:
            try:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
                logger.info("All background tasks cancelled")
            except Exception as e:
                logger.error(f"Error waiting for task cancellation: {e}")

        # 9. Close MT5 connection gracefully
        try:
            import MetaTrader5 as mt5
            if mt5.terminal_info() is not None:
                mt5.shutdown()
                logger.info("MT5 connection closed")
        except Exception as e:
            logger.error(f"Error closing MT5: {e}")

        # 10. Flush all logs
        try:
            for handler in logging.getLogger().handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
            logger.info("Log buffers flushed")
        except Exception as e:
            logger.error(f"Error flushing logs: {e}")

        logger.info("Graceful shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        # Force audit log entry for failed shutdown
        try:
            audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_STOP,
                action="Shutdown failed",
                details={"error": str(e)}
            )
        except:
            pass


@app.get("/health", include_in_schema=False)
async def health():
    """Enhanced health check with MT5 connectivity and model status"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check MT5 connectivity
    try:
        if settings.mt5_enabled:
            from backend.services.mt5_executor import mt5_executor
            mt5_connected = mt5_executor.is_connected()
            health_status["checks"]["mt5"] = {
                "status": "connected" if mt5_connected else "disconnected",
                "healthy": mt5_connected
            }
            if not mt5_connected:
                health_status["status"] = "degraded"
        else:
            health_status["checks"]["mt5"] = {
                "status": "disabled",
                "healthy": True
            }
    except Exception as e:
        health_status["checks"]["mt5"] = {
            "status": "error",
            "healthy": False,
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check AutoTrader status
    try:
        auto_trader_status = auto_trader.get_status()
        health_status["checks"]["auto_trader"] = {
            "status": "running" if auto_trader_status.get("running") else "stopped",
            "healthy": True,
            "circuit_breaker": auto_trader_status.get("circuit_breaker_active", False)
        }
        if auto_trader_status.get("circuit_breaker_active"):
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["auto_trader"] = {
            "status": "error",
            "healthy": False,
            "error": str(e)
        }
    
    # Check Risk Engine kill switch
    try:
        from backend.services.risk_engine import risk_engine
        kill_switch = risk_engine.is_kill_switch_engaged()
        health_status["checks"]["risk_engine"] = {
            "status": "active" if not kill_switch else "kill_switch_engaged",
            "healthy": not kill_switch
        }
        if kill_switch:
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["risk_engine"] = {
            "status": "error",
            "healthy": False,
            "error": str(e)
        }
    
    # Check model loading status
    try:
        from backend.core.model_loader import get_model_loader
        loader = get_model_loader()
        models_loaded = len(loader._models) > 0
        health_status["checks"]["models"] = {
            "status": "loaded" if models_loaded else "not_loaded",
            "healthy": models_loaded,
            "count": len(loader._models)
        }
    except Exception as e:
        health_status["checks"]["models"] = {
            "status": "error",
            "healthy": False,
            "error": str(e)
        }
    
    # Check WebSocket connections
    try:
        from backend.services.ws_broadcaster import ws_broadcaster
        ws_count = len(ws_broadcaster.clients)
        health_status["checks"]["websocket"] = {
            "status": "active",
            "healthy": True,
            "client_count": ws_count
        }
    except Exception as e:
        health_status["checks"]["websocket"] = {
            "status": "error",
            "healthy": False,
            "error": str(e)
        }
    
    return health_status


@app.get("/data-sources/status", include_in_schema=False)
def get_data_sources_status():
    """Get status of all data sources"""
    return data_source_manager.get_status()
