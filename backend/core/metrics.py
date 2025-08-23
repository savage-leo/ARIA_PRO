"""
Prometheus Metrics for ARIA Pro
Production-grade observability with custom metrics
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from typing import Optional
import time
import psutil
import logging

logger = logging.getLogger(__name__)

# Create custom registry to avoid conflicts
registry = CollectorRegistry()

# System metrics
system_info = Info(
    'aria_system_info',
    'System information',
    registry=registry
)

cpu_usage_gauge = Gauge(
    'aria_cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

memory_usage_gauge = Gauge(
    'aria_memory_usage_percent',
    'Memory usage percentage',
    registry=registry
)

disk_usage_gauge = Gauge(
    'aria_disk_usage_percent',
    'Disk usage percentage',
    registry=registry
)

# Trading metrics
orders_submitted = Counter(
    'aria_orders_submitted_total',
    'Total number of orders submitted',
    ['symbol', 'order_type', 'user'],
    registry=registry
)

orders_executed = Counter(
    'aria_orders_executed_total',
    'Total number of orders executed',
    ['symbol', 'order_type'],
    registry=registry
)

orders_rejected = Counter(
    'aria_orders_rejected_total',
    'Total number of orders rejected',
    ['symbol', 'reason'],
    registry=registry
)

order_latency = Histogram(
    'aria_order_latency_seconds',
    'Order execution latency in seconds',
    ['symbol'],
    registry=registry
)

positions_opened = Counter(
    'aria_positions_opened_total',
    'Total positions opened',
    ['symbol'],
    registry=registry
)

positions_closed = Counter(
    'aria_positions_closed_total',
    'Total positions closed',
    ['symbol', 'profit_loss'],
    registry=registry
)

total_profit_gauge = Gauge(
    'aria_total_profit',
    'Total profit/loss',
    registry=registry
)

# Model metrics
model_predictions = Counter(
    'aria_model_predictions_total',
    'Total model predictions made',
    ['model_name', 'symbol'],
    registry=registry
)

model_inference_latency = Histogram(
    'aria_model_inference_seconds',
    'Model inference latency in seconds',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry
)

model_confidence = Histogram(
    'aria_model_confidence',
    'Model prediction confidence distribution',
    ['model_name'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)

# WebSocket metrics
websocket_connections = Gauge(
    'aria_websocket_connections',
    'Current WebSocket connections',
    registry=registry
)

websocket_messages_sent = Counter(
    'aria_websocket_messages_sent_total',
    'Total WebSocket messages sent',
    registry=registry
)

# API metrics
api_requests = Counter(
    'aria_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

api_request_duration = Histogram(
    'aria_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

api_errors = Counter(
    'aria_api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type'],
    registry=registry
)

# Rate limiting metrics
rate_limit_exceeded = Counter(
    'aria_rate_limit_exceeded_total',
    'Total rate limit exceeded events',
    ['endpoint', 'user_type'],
    registry=registry
)

# Authentication metrics
auth_attempts = Counter(
    'aria_auth_attempts_total',
    'Total authentication attempts',
    ['result'],
    registry=registry
)

auth_tokens_issued = Counter(
    'aria_auth_tokens_issued_total',
    'Total auth tokens issued',
    ['token_type'],
    registry=registry
)

# MT5 metrics
mt5_connection_status = Gauge(
    'aria_mt5_connection_status',
    'MT5 connection status (1=connected, 0=disconnected)',
    registry=registry
)

mt5_account_balance = Gauge(
    'aria_mt5_account_balance',
    'MT5 account balance',
    registry=registry
)

mt5_account_equity = Gauge(
    'aria_mt5_account_equity',
    'MT5 account equity',
    registry=registry
)

mt5_account_margin = Gauge(
    'aria_mt5_account_margin',
    'MT5 account margin',
    registry=registry
)

# Auto-trader metrics
auto_trader_status = Gauge(
    'aria_auto_trader_status',
    'Auto-trader status (1=running, 0=stopped)',
    registry=registry
)

auto_trader_signals = Counter(
    'aria_auto_trader_signals_total',
    'Total auto-trader signals generated',
    ['symbol', 'action'],
    registry=registry
)

auto_trader_trades = Counter(
    'aria_auto_trader_trades_total',
    'Total auto-trader trades executed',
    ['symbol', 'action', 'result'],
    registry=registry
)

class MetricsCollector:
    """Collects and updates Prometheus metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self._init_system_info()
    
    def _init_system_info(self):
        """Initialize system information metrics"""
        try:
            import platform
            system_info.info({
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'processor': platform.processor(),
                'hostname': platform.node()
            })
        except Exception as e:
            logger.error(f"Failed to collect system info: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_usage_gauge.set(psutil.cpu_percent(interval=1))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_gauge.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_gauge.set(disk.percent)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def record_order_submitted(
        self,
        symbol: str,
        order_type: str,
        user: str = "system"
    ):
        """Record order submission"""
        orders_submitted.labels(
            symbol=symbol,
            order_type=order_type,
            user=user
        ).inc()
    
    def record_order_executed(
        self,
        symbol: str,
        order_type: str,
        latency: float
    ):
        """Record order execution"""
        orders_executed.labels(
            symbol=symbol,
            order_type=order_type
        ).inc()
        
        order_latency.labels(symbol=symbol).observe(latency)
    
    def record_order_rejected(
        self,
        symbol: str,
        reason: str
    ):
        """Record order rejection"""
        orders_rejected.labels(
            symbol=symbol,
            reason=reason
        ).inc()
    
    def record_model_prediction(
        self,
        model_name: str,
        symbol: str,
        confidence: float,
        inference_time: float
    ):
        """Record model prediction metrics"""
        model_predictions.labels(
            model_name=model_name,
            symbol=symbol
        ).inc()
        
        model_inference_latency.labels(
            model_name=model_name
        ).observe(inference_time)
        
        model_confidence.labels(
            model_name=model_name
        ).observe(confidence)
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Record API request metrics"""
        api_requests.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        if status >= 400:
            error_type = "client_error" if status < 500 else "server_error"
            api_errors.labels(
                endpoint=endpoint,
                error_type=error_type
            ).inc()
    
    def record_rate_limit_exceeded(
        self,
        endpoint: str,
        user_type: str = "anonymous"
    ):
        """Record rate limit exceeded event"""
        rate_limit_exceeded.labels(
            endpoint=endpoint,
            user_type=user_type
        ).inc()
    
    def record_auth_attempt(self, success: bool):
        """Record authentication attempt"""
        result = "success" if success else "failure"
        auth_attempts.labels(result=result).inc()
    
    def record_token_issued(self, token_type: str):
        """Record auth token issuance"""
        auth_tokens_issued.labels(token_type=token_type).inc()
    
    def update_mt5_metrics(
        self,
        connected: bool,
        balance: Optional[float] = None,
        equity: Optional[float] = None,
        margin: Optional[float] = None
    ):
        """Update MT5 connection and account metrics"""
        mt5_connection_status.set(1 if connected else 0)
        
        if balance is not None:
            mt5_account_balance.set(balance)
        if equity is not None:
            mt5_account_equity.set(equity)
        if margin is not None:
            mt5_account_margin.set(margin)
    
    def update_auto_trader_status(self, running: bool):
        """Update auto-trader status"""
        auto_trader_status.set(1 if running else 0)
    
    def record_auto_trader_signal(
        self,
        symbol: str,
        action: str
    ):
        """Record auto-trader signal generation"""
        auto_trader_signals.labels(
            symbol=symbol,
            action=action
        ).inc()
    
    def record_auto_trader_trade(
        self,
        symbol: str,
        action: str,
        success: bool
    ):
        """Record auto-trader trade execution"""
        result = "success" if success else "failure"
        auto_trader_trades.labels(
            symbol=symbol,
            action=action,
            result=result
        ).inc()
    
    def update_websocket_connections(self, count: int):
        """Update WebSocket connection count"""
        websocket_connections.set(count)
    
    def record_websocket_message(self):
        """Record WebSocket message sent"""
        websocket_messages_sent.inc()
    
    def get_metrics(self) -> bytes:
        """Generate Prometheus metrics in text format"""
        # Update system metrics before generating
        self.update_system_metrics()
        
        return generate_latest(registry)

# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create singleton metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
