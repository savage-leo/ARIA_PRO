"""
Critical Component Failure Alerting System
Real-time monitoring and alerting for ARIA Pro
"""

import asyncio
import logging
import smtplib
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts"""
    SYSTEM_FAILURE = "system_failure"
    MT5_CONNECTION = "mt5_connection"
    MODEL_FAILURE = "model_failure"
    TRADING_ERROR = "trading_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_BREACH = "security_breach"
    DATA_FEED_FAILURE = "data_feed_failure"
    CACHE_FAILURE = "cache_failure"
    DISK_SPACE = "disk_space"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime
    details: Dict[str, Any] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data

@dataclass
class AlertConfig:
    """Alerting system configuration"""
    enabled: bool = True
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    from_email: str = ""
    to_emails: List[str] = None
    webhook_url: Optional[str] = None
    alert_cooldown: int = 300  # 5 minutes
    max_alerts_per_hour: int = 20
    
    def __post_init__(self):
        if self.to_emails is None:
            self.to_emails = []

class AlertThresholds:
    """System monitoring thresholds"""
    CPU_CRITICAL = 90.0
    CPU_HIGH = 80.0
    MEMORY_CRITICAL = 95.0
    MEMORY_HIGH = 85.0
    DISK_CRITICAL = 95.0
    DISK_HIGH = 85.0
    MT5_CONNECTION_TIMEOUT = 30.0
    MODEL_INFERENCE_TIMEOUT = 10.0
    API_RESPONSE_TIME_HIGH = 5.0
    API_RESPONSE_TIME_CRITICAL = 10.0

class AlertingSystem:
    """Real-time alerting system for critical failures"""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, int] = {}  # Hourly counts
        self.alert_handlers: List[Callable] = []
        self.running = False
        self._lock = threading.Lock()
        
        # Register default handlers
        if self.config.email_enabled:
            self.alert_handlers.append(self._send_email_alert)
        if self.config.webhook_url:
            self.alert_handlers.append(self._send_webhook_alert)
        
        logger.info("Alerting system initialized")
    
    def add_alert_handler(self, handler: Callable):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    async def start(self):
        """Start alerting system"""
        self.running = True
        logger.info("Alerting system started")
    
    async def stop(self):
        """Stop alerting system"""
        self.running = False
        logger.info("Alerting system stopped")
    
    def _generate_alert_id(self, alert_type: AlertType, component: str) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{alert_type.value}_{component}_{timestamp}"
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """Check if alert should be sent based on cooldown and rate limits"""
        now = datetime.now()
        
        # Check cooldown
        if alert_key in self.alert_cooldowns:
            if now - self.alert_cooldowns[alert_key] < timedelta(seconds=self.config.alert_cooldown):
                return False
        
        # Check hourly rate limit
        hour_key = now.strftime("%Y%m%d_%H")
        current_count = self.alert_counts.get(hour_key, 0)
        if current_count >= self.config.max_alerts_per_hour:
            return False
        
        return True
    
    async def send_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        component: str,
        details: Dict[str, Any] = None
    ) -> Optional[Alert]:
        """Send alert if conditions are met"""
        if not self.config.enabled or not self.running:
            return None
        
        alert_key = f"{alert_type.value}_{component}"
        
        # Check if we should send this alert
        if not self._should_send_alert(alert_key):
            logger.debug(f"Alert suppressed due to cooldown/rate limit: {alert_key}")
            return None
        
        # Create alert
        alert_id = self._generate_alert_id(alert_type, component)
        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            component=component,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        with self._lock:
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update cooldown and counts
            self.alert_cooldowns[alert_key] = datetime.now()
            hour_key = datetime.now().strftime("%Y%m%d_%H")
            self.alert_counts[hour_key] = self.alert_counts.get(hour_key, 0) + 1
            
            # Limit history size
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]
        
        # Send alert through all handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"Alert sent: {alert.title} [{alert.severity.value}]")
        return alert
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert.title}")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        with self._lock:
            return self.alert_history[-limit:]
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.config.email_enabled or not self.config.to_emails:
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.from_email
            msg['To'] = ", ".join(self.config.to_emails)
            msg['Subject'] = f"ARIA Pro Alert [{alert.severity.value.upper()}]: {alert.title}"
            
            # Email body
            body = f"""
ARIA Pro System Alert

Severity: {alert.severity.value.upper()}
Component: {alert.component}
Type: {alert.type.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}

Details:
{json.dumps(alert.details, indent=2) if alert.details else 'None'}

Alert ID: {alert.id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        if not self.config.webhook_url:
            return
        
        try:
            import aiohttp
            
            payload = {
                "alert": alert.to_dict(),
                "system": "ARIA Pro",
                "environment": "production"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent: {alert.title}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

class SystemMonitor:
    """System health monitor with alerting"""
    
    def __init__(self, alerting_system: AlertingSystem):
        self.alerting = alerting_system
        self.monitoring = False
        self.monitor_task = None
        self.last_checks = {}
    
    async def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                await self._check_system_health()
                await self._check_mt5_connection()
                await self._check_cache_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _check_system_health(self):
        """Check system resource usage"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > AlertThresholds.CPU_CRITICAL:
                await self.alerting.send_alert(
                    AlertType.CPU_USAGE,
                    AlertSeverity.CRITICAL,
                    "Critical CPU Usage",
                    f"CPU usage is {cpu_percent:.1f}% (threshold: {AlertThresholds.CPU_CRITICAL}%)",
                    "system",
                    {"cpu_percent": cpu_percent}
                )
            elif cpu_percent > AlertThresholds.CPU_HIGH:
                await self.alerting.send_alert(
                    AlertType.CPU_USAGE,
                    AlertSeverity.HIGH,
                    "High CPU Usage",
                    f"CPU usage is {cpu_percent:.1f}% (threshold: {AlertThresholds.CPU_HIGH}%)",
                    "system",
                    {"cpu_percent": cpu_percent}
                )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            if memory_percent > AlertThresholds.MEMORY_CRITICAL:
                await self.alerting.send_alert(
                    AlertType.MEMORY_USAGE,
                    AlertSeverity.CRITICAL,
                    "Critical Memory Usage",
                    f"Memory usage is {memory_percent:.1f}% (threshold: {AlertThresholds.MEMORY_CRITICAL}%)",
                    "system",
                    {"memory_percent": memory_percent, "memory_used_gb": memory.used / (1024**3)}
                )
            elif memory_percent > AlertThresholds.MEMORY_HIGH:
                await self.alerting.send_alert(
                    AlertType.MEMORY_USAGE,
                    AlertSeverity.HIGH,
                    "High Memory Usage",
                    f"Memory usage is {memory_percent:.1f}% (threshold: {AlertThresholds.MEMORY_HIGH}%)",
                    "system",
                    {"memory_percent": memory_percent, "memory_used_gb": memory.used / (1024**3)}
                )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > AlertThresholds.DISK_CRITICAL:
                await self.alerting.send_alert(
                    AlertType.DISK_SPACE,
                    AlertSeverity.CRITICAL,
                    "Critical Disk Space",
                    f"Disk usage is {disk_percent:.1f}% (threshold: {AlertThresholds.DISK_CRITICAL}%)",
                    "system",
                    {"disk_percent": disk_percent, "disk_free_gb": disk.free / (1024**3)}
                )
            elif disk_percent > AlertThresholds.DISK_HIGH:
                await self.alerting.send_alert(
                    AlertType.DISK_SPACE,
                    AlertSeverity.HIGH,
                    "High Disk Usage",
                    f"Disk usage is {disk_percent:.1f}% (threshold: {AlertThresholds.DISK_HIGH}%)",
                    "system",
                    {"disk_percent": disk_percent, "disk_free_gb": disk.free / (1024**3)}
                )
                
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    async def _check_mt5_connection(self):
        """Check MT5 connection health"""
        try:
            from backend.services.mt5_executor import mt5_executor
            
            if not mt5_executor.is_connected():
                await self.alerting.send_alert(
                    AlertType.MT5_CONNECTION,
                    AlertSeverity.CRITICAL,
                    "MT5 Connection Lost",
                    "MT5 connection is not available",
                    "mt5_executor",
                    {"connected": False}
                )
        except Exception as e:
            await self.alerting.send_alert(
                AlertType.MT5_CONNECTION,
                AlertSeverity.HIGH,
                "MT5 Connection Check Failed",
                f"Failed to check MT5 connection: {str(e)}",
                "mt5_executor",
                {"error": str(e)}
            )
    
    async def _check_cache_health(self):
        """Check Redis cache health"""
        try:
            from backend.core.redis_cache import get_redis_cache
            
            redis_cache = get_redis_cache()
            if not redis_cache.connected:
                await self.alerting.send_alert(
                    AlertType.CACHE_FAILURE,
                    AlertSeverity.HIGH,
                    "Redis Cache Disconnected",
                    "Redis cache connection is not available",
                    "redis_cache",
                    {"connected": False}
                )
        except Exception as e:
            await self.alerting.send_alert(
                AlertType.CACHE_FAILURE,
                AlertSeverity.MEDIUM,
                "Cache Health Check Failed",
                f"Failed to check cache health: {str(e)}",
                "redis_cache",
                {"error": str(e)}
            )

# Global alerting instances
_alerting_system: Optional[AlertingSystem] = None
_system_monitor: Optional[SystemMonitor] = None

def get_alerting_system() -> AlertingSystem:
    """Get or create singleton alerting system"""
    global _alerting_system
    if _alerting_system is None:
        from backend.core.config import get_settings
        settings = get_settings()
        
        config = AlertConfig(
            enabled=getattr(settings, 'ALERTING_ENABLED', True),
            email_enabled=getattr(settings, 'ALERT_EMAIL_ENABLED', False),
            smtp_server=getattr(settings, 'SMTP_SERVER', 'smtp.gmail.com'),
            smtp_port=getattr(settings, 'SMTP_PORT', 587),
            smtp_username=getattr(settings, 'SMTP_USERNAME', ''),
            smtp_password=getattr(settings, 'SMTP_PASSWORD', ''),
            from_email=getattr(settings, 'ALERT_FROM_EMAIL', ''),
            to_emails=getattr(settings, 'ALERT_TO_EMAILS', '').split(',') if getattr(settings, 'ALERT_TO_EMAILS', '') else [],
            webhook_url=getattr(settings, 'ALERT_WEBHOOK_URL', None)
        )
        _alerting_system = AlertingSystem(config)
    return _alerting_system

def get_system_monitor() -> SystemMonitor:
    """Get or create singleton system monitor"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor(get_alerting_system())
    return _system_monitor
