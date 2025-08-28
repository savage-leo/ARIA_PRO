"""
ARIA PRO Enhanced Alerting System
Phase 3 Implementation: Multi-channel notifications, escalation, and alert management
"""

import time
import logging
import threading
import json
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert notification channels"""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # Metric condition (e.g., "aria_pnl_dollars < -1000")
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    enabled: bool = True
    description: str = ""


@dataclass
class AlertNotification:
    """Alert notification structure"""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    channels: List[AlertChannel]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    escalated: bool = False
    escalation_level: int = 0


class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = []
        
    def add_recipient(self, email: str, name: str = ""):
        """Add email recipient"""
        self.recipients.append({"email": email, "name": name})
        
    def send_alert(self, alert: AlertNotification) -> bool:
        """Send email alert"""
        if not self.recipients:
            logger.warning("No email recipients configured")
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['Subject'] = f"ARIA PRO Alert: {alert.severity.value.upper()} - {alert.rule_name}"
            
            # Create email body
            body = f"""
ARIA PRO Trading System Alert

Severity: {alert.severity.value.upper()}
Rule: {alert.rule_name}
Time: {datetime.fromtimestamp(alert.timestamp)}
Message: {alert.message}

This is an automated alert from the ARIA PRO trading system.
Please review the system status and take appropriate action.

---
ARIA PRO Trading System
Automated Alert System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send to all recipients
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                
                for recipient in self.recipients:
                    msg['To'] = recipient['email']
                    server.send_message(msg)
                    
            logger.info(f"Email alert sent to {len(self.recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class SlackNotifier:
    """Slack notification handler"""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
        
    def send_alert(self, alert: AlertNotification) -> bool:
        """Send Slack alert"""
        try:
            # Color mapping for severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffa500",
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.EMERGENCY: "#8b0000"
            }
            
            payload = {
                "channel": self.channel,
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#000000"),
                        "title": f"ARIA PRO Alert: {alert.severity.value.upper()}",
                        "fields": [
                            {
                                "title": "Rule",
                                "value": alert.rule_name,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False
                            }
                        ],
                        "footer": "ARIA PRO Trading System",
                        "ts": alert.timestamp
                    }
                ]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent to {self.channel}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class WebhookNotifier:
    """Webhook notification handler"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        
    def send_alert(self, alert: AlertNotification) -> bool:
        """Send webhook alert"""
        try:
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "source": "aria_pro_trading_system"
            }
            
            response = requests.post(
                self.webhook_url, 
                json=payload, 
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent to {self.webhook_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class EnhancedAlertingSystem:
    """Enhanced alerting system with multi-channel notifications"""
    
    def __init__(self):
        self.alerts: List[AlertNotification] = []
        self.rules: List[AlertRule] = []
        self.notifiers: Dict[AlertChannel, Any] = {}
        self.alert_lock = threading.Lock()
        self.running = True
        
        # Alert history storage
        self.alert_dir = os.path.join("data", "alerts")
        os.makedirs(self.alert_dir, exist_ok=True)
        
        # Load default alert rules
        self._load_default_rules()
        
        # Start alert processing thread
        self.alert_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
        self.alert_thread.start()
        
        logger.info("Enhanced alerting system initialized")
    
    def _load_default_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                name="High Execution Latency",
                condition="aria_execution_latency_seconds > 0.1",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK],
                cooldown_minutes=5,
                description="Execution latency exceeds 100ms"
            ),
            AlertRule(
                name="MT5 Connection Lost",
                condition="aria_mt5_connection_status == 0",
                severity=AlertSeverity.EMERGENCY,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
                cooldown_minutes=1,
                description="MT5 connection is down"
            ),
            AlertRule(
                name="High Error Rate",
                condition="rate(aria_errors_total[5m]) > 0.1",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK],
                cooldown_minutes=10,
                description="Error rate exceeds 10%"
            ),
            AlertRule(
                name="Large Drawdown",
                condition="aria_drawdown_percent > 5",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                cooldown_minutes=30,
                description="Drawdown exceeds 5%"
            ),
            AlertRule(
                name="Kill Switch Activated",
                condition="aria_kill_switch_active == 1",
                severity=AlertSeverity.EMERGENCY,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
                cooldown_minutes=1,
                description="Kill switch is active"
            )
        ]
        
        self.rules.extend(default_rules)
        logger.info(f"Loaded {len(default_rules)} default alert rules")
    
    def add_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        with self.alert_lock:
            self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        with self.alert_lock:
            self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, password: str):
        """Configure email notifications"""
        self.notifiers[AlertChannel.EMAIL] = EmailNotifier(smtp_server, smtp_port, username, password)
        logger.info("Email notifications configured")
    
    def configure_slack(self, webhook_url: str, channel: str = "#alerts"):
        """Configure Slack notifications"""
        self.notifiers[AlertChannel.SLACK] = SlackNotifier(webhook_url, channel)
        logger.info("Slack notifications configured")
    
    def configure_webhook(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        """Configure webhook notifications"""
        self.notifiers[AlertChannel.WEBHOOK] = WebhookNotifier(webhook_url, headers)
        logger.info("Webhook notifications configured")
    
    def add_email_recipient(self, email: str, name: str = ""):
        """Add email recipient"""
        if AlertChannel.EMAIL in self.notifiers:
            self.notifiers[AlertChannel.EMAIL].add_recipient(email, name)
            logger.info(f"Added email recipient: {email}")
    
    def evaluate_rules(self, metrics: Dict[str, Any]):
        """Evaluate alert rules against current metrics"""
        with self.alert_lock:
            for rule in self.rules:
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if self._is_in_cooldown(rule):
                    continue
                
                # Evaluate condition (simplified - in production, use a proper expression evaluator)
                if self._evaluate_condition(rule.condition, metrics):
                    self._trigger_alert(rule, metrics)
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition (simplified implementation)"""
        try:
            # This is a simplified evaluator - in production, use a proper expression parser
            if "aria_execution_latency_seconds" in condition:
                latency = metrics.get("aria_execution_latency_seconds", 0)
                if "> 0.1" in condition:
                    return latency > 0.1
            elif "aria_mt5_connection_status" in condition:
                status = metrics.get("aria_mt5_connection_status", 1)
                if "== 0" in condition:
                    return status == 0
            elif "aria_drawdown_percent" in condition:
                drawdown = metrics.get("aria_drawdown_percent", 0)
                if "> 5" in condition:
                    return drawdown > 5
            elif "aria_kill_switch_active" in condition:
                kill_switch = metrics.get("aria_kill_switch_active", 0)
                if "== 1" in condition:
                    return kill_switch == 1
            return False
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period"""
        cooldown_time = time.time() - (rule.cooldown_minutes * 60)
        recent_alerts = [a for a in self.alerts if a.rule_name == rule.name and a.timestamp > cooldown_time]
        return len(recent_alerts) > 0
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        alert = AlertNotification(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=rule.description,
            timestamp=time.time(),
            channels=rule.channels.copy()
        )
        
        with self.alert_lock:
            self.alerts.append(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        # Log alert
        logger.warning(f"ALERT [{rule.severity.value.upper()}]: {rule.name} - {rule.description}")
        
        # Save to file
        self._save_alert(alert)
    
    def _send_notifications(self, alert: AlertNotification):
        """Send notifications through configured channels"""
        for channel in alert.channels:
            if channel == AlertChannel.LOG:
                # Already logged above
                continue
            elif channel in self.notifiers:
                try:
                    success = self.notifiers[channel].send_alert(alert)
                    if success:
                        logger.info(f"Alert sent via {channel.value}")
                    else:
                        logger.error(f"Failed to send alert via {channel.value}")
                except Exception as e:
                    logger.error(f"Error sending alert via {channel.value}: {e}")
            else:
                logger.warning(f"Channel {channel.value} not configured")
    
    def _save_alert(self, alert: AlertNotification):
        """Save alert to file"""
        try:
            alert_file = os.path.join(
                self.alert_dir, 
                f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
            
            with open(alert_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(alert), separators=(",", ":")) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
    
    def _alert_processing_loop(self):
        """Main alert processing loop"""
        while self.running:
            try:
                # Get current metrics from Prometheus
                if hasattr(self, '_get_current_metrics'):
                    metrics = self._get_current_metrics()
                    if metrics:
                        self.evaluate_rules(metrics)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                time.sleep(60)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        with self.alert_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = time.time()
                    logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                    return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[AlertNotification]:
        """Get active (unacknowledged) alerts"""
        with self.alert_lock:
            alerts = [a for a in self.alerts if not a.acknowledged]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts
    
    def get_alert_history(self, hours: int = 24) -> List[AlertNotification]:
        """Get alert history for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        with self.alert_lock:
            return [a for a in self.alerts if a.timestamp > cutoff_time]
    
    def stop(self):
        """Stop the alerting system"""
        self.running = False
        logger.info("Enhanced alerting system stopped")


# Global alerting system instance
enhanced_alerting = EnhancedAlertingSystem()



