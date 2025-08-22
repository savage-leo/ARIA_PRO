from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import (
    account,
    market,
    positions,
    signals,
    trading,
    monitoring,
    institutional_ai,
    system_management,
    smc_routes,
    websocket,
    debug,
    trade_memory_api,
    analytics,
    model_management,
)
from backend.routes import (
    monitoring_enhanced,
    hedge_fund_dashboard,
    live_execution_api,
    training,
    telemetry_api,
)
from backend.services.data_source_manager import data_source_manager
from backend.services.auto_trader import auto_trader
from backend.services.mt5_market_data import mt5_market_feed
from backend.services.cpp_integration import cpp_service
from backend.monitoring.llm_monitor import llm_monitor_service
from backend.core.performance_monitor import get_performance_monitor
import asyncio
import os
import logging
from logging.handlers import RotatingFileHandler
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from urllib.parse import urlparse
from backend.core.config import get_settings, Settings
from backend.core.live_guard import enforce_live_only


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
        logging.basicConfig(level=level)
    root.setLevel(level)

    # align common server loggers as well
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)

S: Settings = get_settings()
_configure_logging(S.LOG_LEVEL)


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
                info_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            fh_info.setLevel(logging.INFO)
            fh_info.setFormatter(formatter)
            root.addHandler(fh_info)

        if err_path not in existing_paths:
            fh_err = RotatingFileHandler(
                err_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            fh_err.setLevel(logging.ERROR)
            fh_err.setFormatter(formatter)
            root.addHandler(fh_err)
    except Exception:
        # Never fail app due to logging configuration issues
        pass


_configure_file_logging(S.ARIA_LOG_DIR)
logger = logging.getLogger(__name__)


def _build_csp_connect_src(settings: Settings) -> str:
    """Build a safe connect-src list for CSP from env.

    Sources include:
    - 'self'
    - Origins in ARIA_CORS_ORIGINS (http/https)
    - Derived ws/wss origins for each CORS origin
    - Additional entries from ARIA_CSP_CONNECT_SRC (space- or comma-separated)
    """
    sources = {"'self'"}

    for origin in settings.cors_origins_list:
        try:
            p = urlparse(origin)
            if p.scheme and p.netloc:
                base = f"{p.scheme}://{p.netloc}"
                sources.add(base)
                if p.scheme in ("http", "https"):
                    ws_scheme = "wss" if p.scheme == "https" else "ws"
                    sources.add(f"{ws_scheme}://{p.netloc}")
        except Exception:
            # Ignore malformed entries
            continue

    for tok in settings.csp_connect_src_extra:
        sources.add(tok)

    return " ".join(sorted(sources))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        try:
            # Core security headers
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            response.headers.setdefault(
                "Permissions-Policy",
                "geolocation=(), microphone=(), camera=()",
            )
            response.headers.setdefault("Cache-Control", "no-store")

            # HSTS for HTTPS
            if getattr(request, "url", None) and getattr(request.url, "scheme", "") == "https":
                response.headers.setdefault(
                    "Strict-Transport-Security",
                    "max-age=31536000; includeSubDomains; preload",
                )

            # Conservative CSP for API responses with configurable connect-src
            connect_src = _build_csp_connect_src(S)
            response.headers.setdefault(
                "Content-Security-Policy",
                f"default-src 'none'; frame-ancestors 'none'; base-uri 'none'; connect-src {connect_src}",
            )
        except Exception:
            # Never fail request due to header setting issues
            pass
        return response


def _validate_runtime_config(s: Settings) -> None:
    """Log warnings/errors for missing or conflicting production settings.
    Never raises; startup must not fail due to config validation.
    """
    try:
        # Ensure log directory exists
        log_dir = s.ARIA_LOG_DIR
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

        mt5_enabled = s.mt5_enabled
        if mt5_enabled:
            missing = [
                k
                for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER")
                if (
                    (k == "MT5_LOGIN" and s.MT5_LOGIN is None)
                    or (k == "MT5_PASSWORD" and not s.MT5_PASSWORD)
                    or (k == "MT5_SERVER" and not s.MT5_SERVER)
                )
            ]
            if missing:
                logger.error(
                    f"MT5 is enabled but missing credentials: {', '.join(missing)}"
                )

        auto_enabled = s.AUTO_TRADE_ENABLED
        dry_run = s.AUTO_TRADE_DRY_RUN
        exec_enabled = s.ARIA_ENABLE_EXEC
        if exec_enabled and dry_run:
            logger.warning(
                "ARIA_ENABLE_EXEC=1 but AUTO_TRADE_DRY_RUN=1; no live orders will be sent."
            )
        if auto_enabled and not dry_run and not exec_enabled:
            logger.error(
                "AutoTrader live mode requested but execution is disabled (ARIA_ENABLE_EXEC=0)."
            )

        admin_key = s.ADMIN_API_KEY or ""
        if not admin_key:
            logger.warning(
                "ADMIN_API_KEY not set; admin SMC endpoints will reject requests."
            )

        allow_live = s.ALLOW_LIVE
        auto_exec_ok = s.AUTO_EXEC_ENABLED
        if auto_enabled and not dry_run and not (auto_exec_ok and allow_live):
            logger.warning(
                "AutoTrader live trades may be blocked by AUTO_EXEC_ENABLED/ALLOW_LIVE flags."
            )

        if auto_enabled and not mt5_enabled:
            logger.error(
                "Live-only policy: AUTO_TRADE_ENABLED=1 requires MT5 (ARIA_ENABLE_MT5=1). Alpha Vantage fallback is disabled."
            )
    except Exception as e:
        logger.warning(f"Config validation encountered an issue: {e}")


# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="ARIA Institutional Forex AI")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enforce trusted hosts if configured
_allowed_hosts = S.allowed_hosts_list
if _allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts)
    logger.info(f"Trusted hosts enforced: {_allowed_hosts}")
else:
    logger.warning("Trusted hosts not configured (ARIA_ALLOWED_HOSTS). Enforcement skipped.")

# Strict CORS: require explicit ARIA_CORS_ORIGINS, no localhost defaults
_cors_origins = S.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With", "X-ADMIN-KEY"],
)
if _cors_origins:
    logger.info(f"CORS allow_origins: {_cors_origins}")
else:
    logger.warning("CORS: no origins configured (ARIA_CORS_ORIGINS). Cross-origin requests will be blocked.")

# Add strict security headers
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(trading.router, prefix="/trading", tags=["Trading"])
app.include_router(account.router, prefix="/account", tags=["Account"])
app.include_router(market.router, prefix="/market", tags=["Market"])
app.include_router(positions.router, prefix="/positions", tags=["Positions"])
app.include_router(signals.router, prefix="/signals", tags=["Signals"])
app.include_router(monitoring.router)  # router already has prefix /monitoring
app.include_router(monitoring_enhanced.router)  # Enhanced institutional monitoring
app.include_router(live_execution_api.router)  # Live MT5 execution API
app.include_router(smc_routes.router)
app.include_router(websocket.router, tags=["WebSocket"])
app.include_router(debug.router)
app.include_router(trade_memory_api.router)  # router defines its own prefix/tags
app.include_router(analytics.router)
app.include_router(institutional_ai.router)
app.include_router(system_management.router)
app.include_router(training.router)
app.include_router(telemetry_api.router)
from backend.routes.model_management import router as model_management_router
from backend.routes.cicd_management import router as cicd_management_router
from backend.routes.hot_swap_admin import router as hot_swap_admin_router
app.include_router(model_management_router)
app.include_router(cicd_management_router)
app.include_router(hot_swap_admin_router)

# Register data sources
# Enforce MT5-only live feed; no simulated fallback
try:
    enforce_live_only(S)
    if mt5_market_feed not in data_source_manager.data_sources:
        data_source_manager.data_sources.append(mt5_market_feed)
    data_source_manager.mt5_feed = mt5_market_feed
    logger.info("Registered MT5MarketFeed with DataSourceManager (live-only mode)")
except Exception as e:
    logger.error(f"Failed to enforce live MT5 market data feed: {e}")
    raise

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
    """Start data sources when the application starts"""
    logger.info("Starting ARIA PRO backend...")
    try:
        # Log a concise configuration snapshot (no secrets)
        cfg = {
            "LOG_LEVEL": S.LOG_LEVEL.upper(),
            "MT5_ENABLED": str(int(S.mt5_enabled)),
            "AUTO_TRADE_ENABLED": str(int(S.AUTO_TRADE_ENABLED)),
            "PRIMARY_MODEL": S.AUTO_TRADE_PRIMARY_MODEL,
            "SYMBOLS": ",".join(S.symbols_list),
            "EXEC_ENABLED": str(int(S.ARIA_ENABLE_EXEC)),
            "CPP_SMC": S.ARIA_ENABLE_CPP_SMC,
            "LLM_MONITOR_ENABLED": str(int(S.LLM_MONITOR_ENABLED)),
            "LLM_TUNING_ENABLED": str(int(S.LLM_TUNING_ENABLED)),
        }
        logger.info(f"Runtime config: {cfg}")
        _validate_runtime_config(S)
    except Exception:
        # Never fail startup due to logging
        pass
    try:
        # Register LLM monitor service when enabled (before start_all ensures handler attached)
        try:
            if S.LLM_MONITOR_ENABLED and llm_monitor_service not in data_source_manager.data_sources:
                data_source_manager.data_sources.append(llm_monitor_service)
                logger.info("Registered LLMMonitorService with DataSourceManager")
        except Exception as e:
            logger.error(f"Failed to register LLMMonitorService: {e}")

        # Start data sources in background
        asyncio.create_task(data_source_manager.start_all())
        logger.info("Data sources started successfully")

        # Optionally start AutoTrader if enabled via env
        if S.AUTO_TRADE_ENABLED:
            try:
                app.state.auto_trader_task = asyncio.create_task(auto_trader.start())
                logger.info("AutoTrader started")
            except Exception as e:
                logger.error(f"Failed to start AutoTrader: {e}")

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
    """Stop data sources when the application shuts down"""
    logger.info("Shutting down ARIA PRO backend...")
    try:
        # Stop AutoTrader first
        try:
            await auto_trader.stop()
            task = getattr(app.state, "auto_trader_task", None)
            if task:
                task.cancel()
        except Exception as e:
            logger.error(f"Error stopping AutoTrader: {e}")

        # Stop Performance Monitor logger task
        try:
            ptask = getattr(app.state, "perf_log_task", None)
            if ptask:
                ptask.cancel()
        except Exception as e:
            logger.error(f"Error stopping PerformanceMonitor logger: {e}")

        await data_source_manager.stop_all()
        logger.info("Data sources stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping data sources: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/data-sources/status")
def get_data_sources_status():
    """Get status of all data sources"""
    return data_source_manager.get_status()
