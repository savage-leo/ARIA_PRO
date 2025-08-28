"""
Comprehensive Monitoring and Alerting System
Real-time monitoring with anomaly detection and intelligent alerting
"""

import asyncio
import time
import logging
import json
import smtplib
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum
import statistics
import psutil
import numpy as np
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metric_type: MetricType


@dataclass
class Alert:
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    tags: Dict[str, str]
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class ThresholdRule:
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    severity: AlertSeverity
    window_minutes: int = 5
    min_occurrences: int = 1
    tags_filter: Dict[str, str] = None


class MetricCollector:
    """Collects and stores metrics with time series data"""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, deque] = {}
        self.max_history = max_history
        
    def record(self, name: str, value: float, tags: Dict[str, str] = None, metric_type: MetricType = MetricType.GAUGE):
        """Record a metric value"""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.max_history)
        
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        self.metrics[name].append(metric)
    
    def get_recent_values(self, name: str, minutes: int = 5) -> List[MetricValue]:
        """Get recent values for a metric"""
        if name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def get_latest_value(self, name: str) -> Optional[MetricValue]:
        """Get the latest value for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1]
    
    def calculate_statistics(self, name: str, minutes: int = 5) -> Dict[str, float]:
        """Calculate statistics for a metric over time window"""
        values = self.get_recent_values(name, minutes)
        if not values:
            return {}
        
        numeric_values = [v.value for v in values]
        return {
            "count": len(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "std_dev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0
        }


class AnomalyDetector:
    """Detects anomalies in metric data using statistical methods"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baseline_window = 100  # Number of points for baseline calculation
    
    def detect_anomaly(self, metric_name: str, current_value: float, historical_values: List[float]) -> bool:
        """Detect if current value is anomalous"""
        if len(historical_values) < 10:  # Need minimum data
            return False
        
        # Use recent history for baseline
        baseline_values = historical_values[-self.baseline_window:]
        
        if len(baseline_values) < 5:
            return False
        
        mean = statistics.mean(baseline_values)
        std_dev = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0.0
        
        if std_dev == 0:
            return False
        
        # Z-score based anomaly detection
        z_score = abs(current_value - mean) / std_dev
        return z_score > self.sensitivity
    
    def detect_trend_anomaly(self, values: List[float], window_size: int = 20) -> bool:
        """Detect sudden trend changes"""
        if len(values) < window_size * 2:
            return False
        
        # Compare recent trend with historical trend
        recent_values = values[-window_size:]
        historical_values = values[-window_size*2:-window_size]
        
        recent_trend = self._calculate_trend(recent_values)
        historical_trend = self._calculate_trend(historical_values)
        
        # Detect significant trend change
        trend_change = abs(recent_trend - historical_trend)
        return trend_change > 0.5  # Threshold for trend change
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.threshold_rules: List[ThresholdRule] = []
        self.notification_channels: List[Callable] = []
        
        # Rate limiting for alerts
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=5)
    
    def add_threshold_rule(self, rule: ThresholdRule):
        """Add a threshold-based alert rule"""
        self.threshold_rules.append(rule)
    
    def add_notification_channel(self, channel: Callable):
        """Add a notification channel (function that takes Alert)"""
        self.notification_channels.append(channel)
    
    def check_thresholds(self, metric: MetricValue, collector: MetricCollector) -> List[Alert]:
        """Check metric against threshold rules"""
        alerts = []
        
        for rule in self.threshold_rules:
            if rule.metric_name != metric.name:
                continue
            
            # Check tags filter
            if rule.tags_filter:
                if not all(metric.tags.get(k) == v for k, v in rule.tags_filter.items()):
                    continue
            
            # Check threshold
            if self._evaluate_threshold(metric.value, rule.operator, rule.threshold):
                # Check if we have enough occurrences in window
                recent_values = collector.get_recent_values(metric.name, rule.window_minutes)
                violations = sum(1 for v in recent_values 
                               if self._evaluate_threshold(v.value, rule.operator, rule.threshold))
                
                if violations >= rule.min_occurrences:
                    alert_id = f"{rule.metric_name}_{rule.operator}_{rule.threshold}"
                    
                    # Check cooldown
                    if alert_id in self.alert_cooldowns:
                        if datetime.now() - self.alert_cooldowns[alert_id] < self.cooldown_period:
                            continue
                    
                    alert = Alert(
                        id=alert_id,
                        name=f"{metric.name} threshold violation",
                        severity=rule.severity,
                        message=f"{metric.name} is {metric.value} (threshold: {rule.operator} {rule.threshold})",
                        timestamp=datetime.now(),
                        metric_name=metric.name,
                        current_value=metric.value,
                        threshold=rule.threshold,
                        tags=metric.tags
                    )
                    
                    alerts.append(alert)
                    self.alert_cooldowns[alert_id] = datetime.now()
        
        return alerts
    
    def _evaluate_threshold(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate threshold condition"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        return False
    
    def trigger_alert(self, alert: Alert):
        """Trigger an alert and send notifications"""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                asyncio.create_task(channel(alert))
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.acknowledged = True
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved by {user}")
            return True
        return False


class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self, collector: MetricCollector):
        self.collector = collector
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: int = 30):
        """Start system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.collector.record("system.cpu.usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.collector.record("system.memory.usage_percent", memory.percent)
            self.collector.record("system.memory.available_gb", memory.available / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.collector.record("system.disk.usage_percent", disk.percent)
            self.collector.record("system.disk.free_gb", disk.free / (1024**3))
            
            # Network metrics
            network = psutil.net_io_counters()
            self.collector.record("system.network.bytes_sent", network.bytes_sent, metric_type=MetricType.COUNTER)
            self.collector.record("system.network.bytes_recv", network.bytes_recv, metric_type=MetricType.COUNTER)
            
            # Process metrics
            process = psutil.Process()
            self.collector.record("process.memory.rss_mb", process.memory_info().rss / (1024**2))
            self.collector.record("process.cpu.percent", process.cpu_percent())
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class ComprehensiveMonitor:
    """Main monitoring system that coordinates all components"""
    
    def __init__(self):
        self.collector = MetricCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.system_monitor = SystemMonitor(self.collector)
        
        # Setup default threshold rules
        self._setup_default_rules()
        
        # Setup notification channels
        self._setup_notification_channels()
    
    def _setup_default_rules(self):
        """Setup default monitoring rules"""
        rules = [
            # System resource alerts
            ThresholdRule("system.cpu.usage_percent", ">", 90.0, AlertSeverity.HIGH),
            ThresholdRule("system.memory.usage_percent", ">", 85.0, AlertSeverity.HIGH),
            ThresholdRule("system.disk.usage_percent", ">", 90.0, AlertSeverity.MEDIUM),
            
            # Trading system alerts
            ThresholdRule("trading.execution_latency_ms", ">", 1000.0, AlertSeverity.HIGH),
            ThresholdRule("trading.failed_orders", ">", 5.0, AlertSeverity.CRITICAL, window_minutes=10),
            ThresholdRule("trading.daily_loss_pct", ">", 5.0, AlertSeverity.CRITICAL),
            
            # AI model alerts
            ThresholdRule("model.inference_latency_ms", ">", 500.0, AlertSeverity.MEDIUM),
            ThresholdRule("model.error_rate", ">", 0.1, AlertSeverity.HIGH),
            
            # Market data alerts
            ThresholdRule("market_data.staleness_seconds", ">", 60.0, AlertSeverity.HIGH),
            ThresholdRule("market_data.missing_ticks", ">", 10.0, AlertSeverity.MEDIUM, window_minutes=5),
        ]
        
        for rule in rules:
            self.alert_manager.add_threshold_rule(rule)
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        # Add email notification if configured
        self.alert_manager.add_notification_channel(self._log_notification)
        
        # Add webhook notification if configured
        # self.alert_manager.add_notification_channel(self._webhook_notification)
    
    async def _log_notification(self, alert: Alert):
        """Log notification channel"""
        logger.critical(f"ALERT: {alert.name} - {alert.message}")
    
    async def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, metric_type: MetricType = MetricType.GAUGE):
        """Record a metric and check for alerts"""
        self.collector.record(name, value, tags, metric_type)
        
        # Get the recorded metric
        metric = self.collector.get_latest_value(name)
        if metric:
            # Check threshold alerts
            alerts = self.alert_manager.check_thresholds(metric, self.collector)
            for alert in alerts:
                self.alert_manager.trigger_alert(alert)
            
            # Check for anomalies
            recent_values = self.collector.get_recent_values(name, 60)  # 1 hour
            if len(recent_values) > 10:
                historical_values = [v.value for v in recent_values[:-1]]
                if self.anomaly_detector.detect_anomaly(name, value, historical_values):
                    anomaly_alert = Alert(
                        id=f"anomaly_{name}_{int(time.time())}",
                        name=f"Anomaly detected in {name}",
                        severity=AlertSeverity.MEDIUM,
                        message=f"Anomalous value {value} detected for {name}",
                        timestamp=datetime.now(),
                        metric_name=name,
                        current_value=value,
                        threshold=0.0,
                        tags=tags or {}
                    )
                    self.alert_manager.trigger_alert(anomaly_alert)
    
    async def start(self):
        """Start the monitoring system"""
        await self.system_monitor.start_monitoring()
        logger.info("Comprehensive monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system"""
        await self.system_monitor.stop_monitoring()
        logger.info("Comprehensive monitoring system stopped")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        for name, values in self.collector.metrics.items():
            if values:
                latest = values[-1]
                stats = self.collector.calculate_statistics(name, 60)  # 1 hour stats
                summary[name] = {
                    "latest_value": latest.value,
                    "latest_timestamp": latest.timestamp.isoformat(),
                    "stats": stats
                }
        return summary
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of alerts"""
        return {
            "active_alerts": len(self.alert_manager.active_alerts),
            "total_alerts_today": len([a for a in self.alert_manager.alert_history 
                                     if a.timestamp.date() == datetime.now().date()]),
            "critical_alerts": len([a for a in self.alert_manager.active_alerts.values() 
                                  if a.severity == AlertSeverity.CRITICAL]),
            "unacknowledged_alerts": len([a for a in self.alert_manager.active_alerts.values() 
                                        if not a.acknowledged])
        }


# Global monitoring instance
_comprehensive_monitor: Optional[ComprehensiveMonitor] = None


def get_comprehensive_monitor() -> ComprehensiveMonitor:
    """Get or create singleton comprehensive monitor"""
    global _comprehensive_monitor
    if _comprehensive_monitor is None:
        _comprehensive_monitor = ComprehensiveMonitor()
    return _comprehensive_monitor


# Convenience functions
async def record_trading_metric(name: str, value: float, symbol: str = None):
    """Record a trading-related metric"""
    monitor = get_comprehensive_monitor()
    tags = {"symbol": symbol} if symbol else {}
    await monitor.record_metric(f"trading.{name}", value, tags)


async def record_model_metric(name: str, value: float, model: str = None):
    """Record a model-related metric"""
    monitor = get_comprehensive_monitor()
    tags = {"model": model} if model else {}
    await monitor.record_metric(f"model.{name}", value, tags)


async def record_system_metric(name: str, value: float):
    """Record a system-related metric"""
    monitor = get_comprehensive_monitor()
    await monitor.record_metric(f"system.{name}", value)
