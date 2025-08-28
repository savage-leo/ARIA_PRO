"""
Prometheus Metrics for ARIA Pro
Institutional-grade monitoring and alerting
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import psutil
import threading

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, push_to_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Prometheus metrics configuration"""
    enabled: bool = True
    port: int = 8001
    push_gateway_url: Optional[str] = None
    job_name: str = "aria_pro"
    instance_name: str = "aria_backend"
    push_interval: int = 30  # seconds

class PrometheusMetrics:
    """Prometheus metrics collector for ARIA Pro"""
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.registry = CollectorRegistry()
        self.metrics_server_started = False
        self._lock = threading.Lock()
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - metrics disabled")
            return
        
        # Trading Metrics
        self.trade_executions = Counter(
            'aria_trade_executions_total',
            'Total number of trade executions',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )
        
        self.trade_pnl = Histogram(
            'aria_trade_pnl',
            'Trade P&L distribution',
            ['symbol', 'side'],
            buckets=[-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000],
            registry=self.registry
        )
        
        self.position_size = Histogram(
            'aria_position_size',
            'Position size distribution',
            ['symbol'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # AI Model Metrics
        self.model_predictions = Counter(
            'aria_model_predictions_total',
            'Total model predictions',
            ['model_name', 'symbol'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'aria_model_accuracy',
            'Model accuracy percentage',
            ['model_name', 'symbol'],
            registry=self.registry
        )
        
        self.model_inference_time = Histogram(
            'aria_model_inference_seconds',
            'Model inference time',
            ['model_name'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # Market Data Metrics
        self.market_data_updates = Counter(
            'aria_market_data_updates_total',
            'Market data updates received',
            ['symbol', 'timeframe'],
            registry=self.registry
        )
        
        self.market_data_latency = Histogram(
            'aria_market_data_latency_seconds',
            'Market data latency',
            ['symbol'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        self.spread = Gauge(
            'aria_spread_pips',
            'Current spread in pips',
            ['symbol'],
            registry=self.registry
        )
        
        # System Metrics
        self.cpu_usage = Gauge(
            'aria_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'aria_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'aria_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # API Metrics
        self.api_requests = Counter(
            'aria_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_response_time = Histogram(
            'aria_api_response_seconds',
            'API response time',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # Cache Metrics
        self.cache_hits = Counter(
            'aria_cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'aria_cache_misses_total',
            'Cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'aria_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )
        
        # MT5 Connection Metrics
        self.mt5_connections = Gauge(
            'aria_mt5_connections_active',
            'Active MT5 connections',
            registry=self.registry
        )
        
        self.mt5_connection_errors = Counter(
            'aria_mt5_connection_errors_total',
            'MT5 connection errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Risk Metrics
        self.account_balance = Gauge(
            'aria_account_balance',
            'Account balance',
            registry=self.registry
        )
        
        self.account_equity = Gauge(
            'aria_account_equity',
            'Account equity',
            registry=self.registry
        )
        
        self.drawdown = Gauge(
            'aria_drawdown_percent',
            'Current drawdown percentage',
            registry=self.registry
        )
        
        self.risk_exposure = Gauge(
            'aria_risk_exposure',
            'Total risk exposure',
            ['symbol'],
            registry=self.registry
        )
        
        # Application Info
        self.app_info = Info(
            'aria_application_info',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        try:
            from backend.core.config import get_settings  # local import to avoid cycles
            _settings = get_settings()
            env_name = getattr(_settings, 'environment', 'production')
        except Exception:
            env_name = 'production'
        self.app_info.info({
            'version': '1.2.0',
            'environment': env_name,
            'instance': self.config.instance_name
        })
        
        logger.info("Prometheus metrics initialized")
    
    def start_metrics_server(self) -> bool:
        """Start Prometheus metrics HTTP server"""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return False
        # In dev/test, do not start a separate HTTP server; expose via FastAPI endpoint only
        try:
            from backend.core.config import get_settings  # local import to avoid cycles
            _settings = get_settings()
            if getattr(_settings, 'is_development', False) or getattr(_settings, 'is_test', False):
                logger.info("Dev/Test mode: skipping standalone Prometheus HTTP server; metrics available via API endpoint")
                return False
        except Exception:
            # If settings load fails, fall through and attempt to start server (safe default)
            pass

        with self._lock:
            if self.metrics_server_started:
                return True
            
            try:
                start_http_server(self.config.port, registry=self.registry)
                self.metrics_server_started = True
                logger.info(f"Prometheus metrics server started on port {self.config.port}")
                return True
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
                return False
    
    async def push_metrics(self):
        """Push metrics to Prometheus Push Gateway"""
        if not PROMETHEUS_AVAILABLE or not self.config.push_gateway_url:
            return
        
        try:
            push_to_gateway(
                self.config.push_gateway_url,
                job=self.config.job_name,
                registry=self.registry,
                grouping_key={'instance': self.config.instance_name}
            )
            logger.debug("Metrics pushed to gateway")
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        if not PROMETHEUS_AVAILABLE:
            return ""
        
        return generate_latest(self.registry).decode('utf-8')
    
    # Trading Metrics Methods
    def record_trade_execution(self, symbol: str, side: str, status: str):
        """Record trade execution"""
        if PROMETHEUS_AVAILABLE:
            self.trade_executions.labels(symbol=symbol, side=side, status=status).inc()
    
    def record_trade_pnl(self, symbol: str, side: str, pnl: float):
        """Record trade P&L"""
        if PROMETHEUS_AVAILABLE:
            self.trade_pnl.labels(symbol=symbol, side=side).observe(pnl)
    
    def record_position_size(self, symbol: str, size: float):
        """Record position size"""
        if PROMETHEUS_AVAILABLE:
            self.position_size.labels(symbol=symbol).observe(size)
    
    # AI Model Metrics Methods
    def record_model_prediction(self, model_name: str, symbol: str):
        """Record model prediction"""
        if PROMETHEUS_AVAILABLE:
            self.model_predictions.labels(model_name=model_name, symbol=symbol).inc()
    
    def set_model_accuracy(self, model_name: str, symbol: str, accuracy: float):
        """Set model accuracy"""
        if PROMETHEUS_AVAILABLE:
            self.model_accuracy.labels(model_name=model_name, symbol=symbol).set(accuracy)
    
    def record_model_inference_time(self, model_name: str, inference_time: float):
        """Record model inference time"""
        if PROMETHEUS_AVAILABLE:
            self.model_inference_time.labels(model_name=model_name).observe(inference_time)
    
    # Market Data Metrics Methods
    def record_market_data_update(self, symbol: str, timeframe: str):
        """Record market data update"""
        if PROMETHEUS_AVAILABLE:
            self.market_data_updates.labels(symbol=symbol, timeframe=timeframe).inc()
    
    def record_market_data_latency(self, symbol: str, latency: float):
        """Record market data latency"""
        if PROMETHEUS_AVAILABLE:
            self.market_data_latency.labels(symbol=symbol).observe(latency)
    
    def set_spread(self, symbol: str, spread_pips: float):
        """Set current spread"""
        if PROMETHEUS_AVAILABLE:
            self.spread.labels(symbol=symbol).set(spread_pips)
    
    # System Metrics Methods
    def update_system_metrics(self):
        """Update system metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_percent)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    # API Metrics Methods
    def record_api_request(self, method: str, endpoint: str, status: int):
        """Record API request"""
        if PROMETHEUS_AVAILABLE:
            self.api_requests.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    
    def record_api_response_time(self, method: str, endpoint: str, response_time: float):
        """Record API response time"""
        if PROMETHEUS_AVAILABLE:
            self.api_response_time.labels(method=method, endpoint=endpoint).observe(response_time)
    
    # Cache Metrics Methods
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        if PROMETHEUS_AVAILABLE:
            self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        if PROMETHEUS_AVAILABLE:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def set_cache_size(self, cache_type: str, size_bytes: int):
        """Set cache size"""
        if PROMETHEUS_AVAILABLE:
            self.cache_size.labels(cache_type=cache_type).set(size_bytes)
    
    # MT5 Metrics Methods
    def set_mt5_connections(self, count: int):
        """Set active MT5 connections"""
        if PROMETHEUS_AVAILABLE:
            self.mt5_connections.set(count)
    
    def record_mt5_connection_error(self, error_type: str):
        """Record MT5 connection error"""
        if PROMETHEUS_AVAILABLE:
            self.mt5_connection_errors.labels(error_type=error_type).inc()
    
    # Risk Metrics Methods
    def set_account_balance(self, balance: float):
        """Set account balance"""
        if PROMETHEUS_AVAILABLE:
            self.account_balance.set(balance)
    
    def set_account_equity(self, equity: float):
        """Set account equity"""
        if PROMETHEUS_AVAILABLE:
            self.account_equity.set(equity)
    
    def set_drawdown(self, drawdown_percent: float):
        """Set current drawdown"""
        if PROMETHEUS_AVAILABLE:
            self.drawdown.set(drawdown_percent)
    
    def set_risk_exposure(self, symbol: str, exposure: float):
        """Set risk exposure for symbol"""
        if PROMETHEUS_AVAILABLE:
            self.risk_exposure.labels(symbol=symbol).set(exposure)

class MetricsCollector:
    """Automated metrics collection service"""
    
    def __init__(self, metrics: PrometheusMetrics):
        self.metrics = metrics
        self.running = False
        self.collection_task = None
        self.push_task = None
    
    async def start(self):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        
        # Start metrics server
        self.metrics.start_metrics_server()
        
        # Start collection tasks
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        if self.metrics.config.push_gateway_url:
            self.push_task = asyncio.create_task(self._push_loop())
        
        logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection"""
        self.running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        if self.push_task:
            self.push_task.cancel()
            try:
                await self.push_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Update system metrics
                self.metrics.update_system_metrics()
                
                # Sleep for collection interval
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    async def _push_loop(self):
        """Push metrics to gateway"""
        while self.running:
            try:
                await self.metrics.push_metrics()
                await asyncio.sleep(self.metrics.config.push_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error pushing metrics: {e}")
                await asyncio.sleep(10)

# Global metrics instance
_prometheus_metrics: Optional[PrometheusMetrics] = None
_metrics_collector: Optional[MetricsCollector] = None

def get_prometheus_metrics() -> PrometheusMetrics:
    """Get or create singleton Prometheus metrics"""
    global _prometheus_metrics
    if _prometheus_metrics is None:
        from backend.core.config import get_settings
        settings = get_settings()
        
        config = MetricConfig(
            enabled=getattr(settings, 'PROMETHEUS_ENABLED', True),
            port=getattr(settings, 'PROMETHEUS_PORT', 8001),
            push_gateway_url=getattr(settings, 'PROMETHEUS_PUSH_GATEWAY', None),
            job_name=getattr(settings, 'PROMETHEUS_JOB_NAME', 'aria_pro'),
            instance_name=getattr(settings, 'PROMETHEUS_INSTANCE', 'aria_backend')
        )
        _prometheus_metrics = PrometheusMetrics(config)
    return _prometheus_metrics

def get_metrics_collector() -> MetricsCollector:
    """Get or create singleton metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(get_prometheus_metrics())
    return _metrics_collector
