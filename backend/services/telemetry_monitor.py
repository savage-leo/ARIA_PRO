"""
ARIA PRO Real-Time Telemetry Monitor
Phase 1 Implementation: Real-time performance monitoring and business metrics tracking
"""

import time
import threading
import logging
from collections import deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    execution_latency_p50: float = 0.0
    execution_latency_p95: float = 0.0
    execution_latency_p99: float = 0.0
    slippage_average: float = 0.0
    slippage_p95: float = 0.0
    throughput_orders_per_minute: float = 0.0
    error_rate: float = 0.0
    mt5_connection_health: bool = False


@dataclass
class BusinessMetrics:
    """Real-time business metrics"""
    real_time_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    max_equity: float = 0.0


@dataclass
class AlertEvent:
    """Alert event structure"""
    timestamp: float
    alert_type: str
    severity: str
    message: str
    action_required: str


class RealTimePerformanceMonitor:
    """Real-time performance monitoring with alerting"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.lock = threading.Lock()
        
        # Performance metrics storage
        self.execution_latencies = deque(maxlen=max_samples)
        self.slippage_metrics = deque(maxlen=max_samples)
        self.error_counts = deque(maxlen=100)
        
        # Alert thresholds
        self.latency_threshold_ms = 100
        self.slippage_threshold = 0.001
        self.error_rate_threshold = 0.05
        
        # Alert history
        self.alerts = deque(maxlen=100)
        
        # MT5 health tracking
        self.mt5_health = {"connected": False, "last_heartbeat": 0.0}
        
        logger.info("RealTimePerformanceMonitor initialized")

    def track_execution_latency(self, start_time: float, end_time: float) -> float:
        """Track execution latency and trigger alerts if needed"""
        latency_ms = (end_time - start_time) * 1000
        
        with self.lock:
            self.execution_latencies.append(latency_ms)
            
            if len(self.execution_latencies) >= 10:
                sorted_latencies = sorted(self.execution_latencies)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p95 = sorted_latencies[p95_idx]
                
                if p95 > self.latency_threshold_ms:
                    self._trigger_alert(
                        "execution_latency_high",
                        "critical",
                        f"High execution latency: P95={p95:.2f}ms",
                        "immediate_kill_switch"
                    )
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            try:
                prometheus_metrics.track_execution(
                    symbol="unknown",
                    action="unknown", 
                    latency_ms=latency_ms,
                    slippage_pips=0.0,
                    success=True
                )
            except Exception as e:
                logger.error(f"Failed to update Prometheus metrics: {e}")
        
        return latency_ms

    def track_slippage(self, expected_price: float, actual_price: float) -> float:
        """Track execution slippage and trigger alerts if needed"""
        slippage = abs(actual_price - expected_price) / expected_price
        
        with self.lock:
            self.slippage_metrics.append(slippage)
            
            if len(self.slippage_metrics) >= 10:
                sorted_slippage = sorted(self.slippage_metrics)
                p95_idx = int(len(sorted_slippage) * 0.95)
                p95_slippage = sorted_slippage[p95_idx]
                
                if p95_slippage > self.slippage_threshold:
                    self._trigger_alert(
                        "slippage_excessive",
                        "critical",
                        f"Excessive slippage: P95={p95_slippage:.4f}",
                        "pause_trading"
                    )
        
        return slippage

    def track_error(self, error_type: str):
        """Track errors and calculate error rate"""
        with self.lock:
            self.error_counts.append({"timestamp": time.time(), "type": error_type})
            
            if len(self.error_counts) >= 10:
                error_rate = len(list(self.error_counts)[-100:]) / 100.0
                
                if error_rate > self.error_rate_threshold:
                    self._trigger_alert(
                        "error_rate_high",
                        "warning",
                        f"High error rate: {error_rate:.2%}",
                        "reduce_trading_volume"
                    )
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            try:
                prometheus_metrics.track_error(error_type)
            except Exception as e:
                logger.error(f"Failed to update Prometheus error metrics: {e}")

    def update_mt5_health(self, connected: bool):
        """Update MT5 connection health"""
        with self.lock:
            self.mt5_health["connected"] = connected
            self.mt5_health["last_heartbeat"] = time.time()
            
            if not connected:
                self._trigger_alert(
                    "mt5_disconnected",
                    "critical",
                    "MT5 connection lost",
                    "immediate_kill_switch"
                )
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            try:
                prometheus_metrics.update_mt5_connection(connected)
            except Exception as e:
                logger.error(f"Failed to update Prometheus MT5 metrics: {e}")

    def _trigger_alert(self, alert_type: str, severity: str, message: str, action_required: str):
        """Trigger an alert"""
        alert = AlertEvent(
            timestamp=time.time(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            action_required=action_required
        )
        
        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity.upper()}]: {message} - Action: {action_required}")

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self.lock:
            if len(self.execution_latencies) == 0:
                return PerformanceMetrics()
            
            sorted_latencies = sorted(self.execution_latencies)
            p50_idx = int(len(sorted_latencies) * 0.5)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            
            p50 = sorted_latencies[p50_idx]
            p95 = sorted_latencies[p95_idx]
            p99 = sorted_latencies[p99_idx]
            
            slippage_avg = 0.0
            slippage_p95 = 0.0
            if len(self.slippage_metrics) > 0:
                slippage_avg = sum(self.slippage_metrics) / len(self.slippage_metrics)
                sorted_slippage = sorted(self.slippage_metrics)
                p95_idx = int(len(sorted_slippage) * 0.95)
                slippage_p95 = sorted_slippage[p95_idx]
            
            error_rate = 0.0
            if len(self.error_counts) > 0:
                recent_errors = list(self.error_counts)[-100:]
                error_rate = len(recent_errors) / 100.0
            
            return PerformanceMetrics(
                execution_latency_p50=p50,
                execution_latency_p95=p95,
                execution_latency_p99=p99,
                slippage_average=slippage_avg,
                slippage_p95=slippage_p95,
                error_rate=error_rate,
                mt5_connection_health=self.mt5_health["connected"]
            )


class BusinessMetricsTracker:
    """Real-time business metrics tracking"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.trades = deque(maxlen=1000)
        self.real_time_pnl = 0.0
        self.daily_pnl = 0.0
        self.max_equity = 0.0
        self.current_equity = 0.0
        
        logger.info("BusinessMetricsTracker initialized")

    def update_trade(self, trade_data: Dict[str, Any]):
        """Update metrics with new trade data"""
        with self.lock:
            trade_data["timestamp"] = time.time()
            self.trades.append(trade_data)
            
            trade_pnl = trade_data.get("pnl", 0.0)
            self.real_time_pnl += trade_pnl
            self.current_equity += trade_pnl
            self.daily_pnl += trade_pnl
            
            if self.current_equity > self.max_equity:
                self.max_equity = self.current_equity
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            try:
                symbol = trade_data.get("symbol", "unknown")
                action = trade_data.get("action", "unknown")
                volume = trade_data.get("volume", 0.0)
                regime = trade_data.get("regime", "unknown")
                
                prometheus_metrics.track_trade(symbol, action, volume, regime)
                prometheus_metrics.update_pnl("real_time", self.real_time_pnl)
                prometheus_metrics.update_pnl("daily", self.daily_pnl)
            except Exception as e:
                logger.error(f"Failed to update Prometheus trade metrics: {e}")

    def get_business_metrics(self) -> BusinessMetrics:
        """Get current business metrics"""
        with self.lock:
            if len(self.trades) == 0:
                return BusinessMetrics()
            
            winning_trades = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
            total_trades = len(self.trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            total_profit = sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0)
            total_loss = abs(sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
            
            drawdown = 0.0
            if self.max_equity > 0:
                drawdown = (self.max_equity - self.current_equity) / self.max_equity
            
            business_metrics = BusinessMetrics(
                real_time_pnl=self.real_time_pnl,
                daily_pnl=self.daily_pnl,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=drawdown,
                total_trades=total_trades,
                winning_trades=winning_trades,
                max_equity=self.max_equity
            )
            
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                try:
                    prometheus_metrics.update_win_rate(win_rate)
                    prometheus_metrics.update_drawdown(drawdown * 100)  # Convert to percentage
                except Exception as e:
                    logger.error(f"Failed to update Prometheus business metrics: {e}")
            
            return business_metrics


class TelemetryMonitor:
    """Main telemetry monitoring system"""
    
    def __init__(self):
        self.performance_monitor = RealTimePerformanceMonitor()
        self.business_tracker = BusinessMetricsTracker()
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("TelemetryMonitor initialized and started")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_mt5_health()
                self._log_metrics()
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)

    def _check_mt5_health(self):
        """Check MT5 connection health"""
        try:
            from backend.services.mt5_executor import MT5Executor
            mt5 = MT5Executor()
            account_info = mt5.get_account_info()
            connected = account_info is not None
            self.performance_monitor.update_mt5_health(connected)
        except Exception as e:
            logger.error(f"MT5 health check failed: {e}")
            self.performance_monitor.update_mt5_health(False)

    def _log_metrics(self):
        """Log current metrics to file"""
        try:
            perf_metrics = self.performance_monitor.get_performance_metrics()
            business_metrics = self.business_tracker.get_business_metrics()
            
            metrics_data = {
                "timestamp": time.time(),
                "performance": asdict(perf_metrics),
                "business": asdict(business_metrics)
            }
            
            metrics_dir = os.path.join("data", "telemetry")
            os.makedirs(metrics_dir, exist_ok=True)
            
            metrics_file = os.path.join(metrics_dir, f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl")
            
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_data, separators=(",", ":")) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def track_execution(self, start_time: float, end_time: float, 
                       expected_price: float, actual_price: float,
                       trade_data: Dict[str, Any]):
        """Track complete execution with all metrics"""
        latency = self.performance_monitor.track_execution_latency(start_time, end_time)
        slippage = self.performance_monitor.track_slippage(expected_price, actual_price)
        self.business_tracker.update_trade(trade_data)
        
        return {
            "latency_ms": latency,
            "slippage": slippage,
            "timestamp": time.time()
        }

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get complete telemetry summary"""
        perf_metrics = self.performance_monitor.get_performance_metrics()
        business_metrics = self.business_tracker.get_business_metrics()
        alerts = self.performance_monitor.alerts
        
        return {
            "timestamp": time.time(),
            "performance": asdict(perf_metrics),
            "business": asdict(business_metrics),
            "alerts": [asdict(alert) for alert in list(alerts)[-10:]],
            "status": "healthy" if len([a for a in alerts if a.severity == "critical"]) == 0 else "critical"
        }

    def stop(self):
        """Stop the telemetry monitor"""
        self.running = False
        logger.info("TelemetryMonitor stopped")


# Global telemetry monitor instance
telemetry_monitor = TelemetryMonitor()

# Import Prometheus metrics for integration
try:
    from backend.services.prometheus_metrics import prometheus_metrics
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrics integration enabled")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus metrics not available - skipping integration")
