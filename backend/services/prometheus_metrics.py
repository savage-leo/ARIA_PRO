"""
ARIA PRO Prometheus Metrics Integration
Phase 2 Implementation: Prometheus metrics for production monitoring
"""

import time
import logging
from typing import Dict, Any
from prometheus_client import (
    Counter, Gauge, Histogram, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry
)
from prometheus_client.exposition import start_http_server
import threading

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics collector for ARIA PRO trading system"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.registry = CollectorRegistry()
        
        # Performance Metrics
        self.execution_latency = Histogram(
            'aria_execution_latency_seconds',
            'Execution latency in seconds',
            ['symbol', 'action', 'status'],
            registry=self.registry
        )
        
        self.slippage = Histogram(
            'aria_slippage_pips',
            'Execution slippage in pips',
            ['symbol', 'action'],
            registry=self.registry
        )
        
        self.error_rate = Counter(
            'aria_errors_total',
            'Total number of errors',
            ['error_type', 'symbol'],
            registry=self.registry
        )
        
        # Business Metrics
        self.trade_volume = Counter(
            'aria_trade_volume_total',
            'Total trade volume',
            ['symbol', 'action', 'regime'],
            registry=self.registry
        )
        
        self.pnl = Gauge(
            'aria_pnl_dollars',
            'Current P&L in dollars',
            ['type'],
            registry=self.registry
        )
        
        self.win_rate = Gauge(
            'aria_win_rate_ratio',
            'Current win rate as ratio',
            registry=self.registry
        )
        
        self.drawdown = Gauge(
            'aria_drawdown_percent',
            'Current drawdown as percentage',
            registry=self.registry
        )
        
        # System Health Metrics
        self.mt5_connection = Gauge(
            'aria_mt5_connection_status',
            'MT5 connection status (1=connected, 0=disconnected)',
            registry=self.registry
        )
        
        self.kill_switch_status = Gauge(
            'aria_kill_switch_active',
            'Kill switch status (1=active, 0=inactive)',
            registry=self.registry
        )
        
        # Start metrics server
        self._start_metrics_server()
        
        logger.info(f"Prometheus metrics initialized on port {port}")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(self.port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")
    
    def track_execution(self, symbol: str, action: str, latency_ms: float, 
                       slippage_pips: float, success: bool):
        """Track execution metrics"""
        status = "success" if success else "failed"
        latency_seconds = latency_ms / 1000.0
        
        self.execution_latency.labels(
            symbol=symbol, 
            action=action, 
            status=status
        ).observe(latency_seconds)
        
        if success:
            self.slippage.labels(
                symbol=symbol, 
                action=action
            ).observe(slippage_pips)
    
    def track_error(self, error_type: str, symbol: str = "unknown"):
        """Track error metrics"""
        self.error_rate.labels(
            error_type=error_type, 
            symbol=symbol
        ).inc()
    
    def track_trade(self, symbol: str, action: str, volume: float, 
                   regime: str = "unknown"):
        """Track trade metrics"""
        self.trade_volume.labels(
            symbol=symbol, 
            action=action, 
            regime=regime
        ).inc(volume)
    
    def update_pnl(self, pnl_type: str, value: float):
        """Update P&L metrics"""
        self.pnl.labels(type=pnl_type).set(value)
    
    def update_win_rate(self, win_rate: float):
        """Update win rate metric"""
        self.win_rate.set(win_rate)
    
    def update_drawdown(self, drawdown_percent: float):
        """Update drawdown metric"""
        self.drawdown.set(drawdown_percent)
    
    def update_mt5_connection(self, connected: bool):
        """Update MT5 connection status"""
        self.mt5_connection.set(1 if connected else 0)
    
    def update_kill_switch(self, active: bool):
        """Update kill switch status"""
        self.kill_switch_status.set(1 if active else 0)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics as string"""
        return generate_latest(self.registry)
    
    def get_metrics_content_type(self) -> str:
        """Get content type for metrics"""
        return CONTENT_TYPE_LATEST


# Global Prometheus metrics instance
prometheus_metrics = PrometheusMetrics(port=8000)
