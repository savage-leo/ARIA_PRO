"""
Comprehensive Health and Readiness Checks for ARIA Pro
Production-grade health monitoring with detailed component status
"""

import asyncio
import time
import psutil
import os
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentHealth:
    """Health status for a single component"""
    
    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.UNKNOWN
        self.last_check = None
        self.last_success = None
        self.consecutive_failures = 0
        self.error_message = None
        self.metrics = {}
    
    def update(self, status: HealthStatus, error: Optional[str] = None, metrics: Optional[Dict] = None):
        """Update component health status"""
        self.last_check = datetime.utcnow()
        self.status = status
        
        if status == HealthStatus.HEALTHY:
            self.last_success = datetime.utcnow()
            self.consecutive_failures = 0
            self.error_message = None
        else:
            self.consecutive_failures += 1
            self.error_message = error
        
        if metrics:
            self.metrics.update(metrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "name": self.name,
            "status": self.status,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "consecutive_failures": self.consecutive_failures,
            "error": self.error_message,
            "metrics": self.metrics
        }

class HealthChecker:
    """Comprehensive health checker for all system components"""
    
    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self.start_time = datetime.utcnow()
        self._init_components()
    
    def _init_components(self):
        """Initialize component health trackers"""
        component_names = [
            "mt5_connection",
            "database",
            "redis",
            "models",
            "auto_trader",
            "websocket",
            "disk_space",
            "memory",
            "cpu"
        ]
        
        for name in component_names:
            self.components[name] = ComponentHealth(name)
    
    async def check_mt5_connection(self) -> ComponentHealth:
        """Check MT5 connection health"""
        component = self.components["mt5_connection"]
        
        try:
            from backend.core.config import get_settings
            settings = get_settings()
            
            if not settings.mt5_enabled:
                component.update(HealthStatus.HEALTHY, metrics={"enabled": False})
                return component
            
            # Try to import and check MT5
            try:
                import MetaTrader5 as mt5
                
                if not mt5.initialize():
                    raise RuntimeError("Failed to initialize MT5")
                
                account_info = mt5.account_info()
                if account_info is None:
                    raise RuntimeError("Failed to get account info")
                
                component.update(
                    HealthStatus.HEALTHY,
                    metrics={
                        "connected": True,
                        "balance": account_info.balance,
                        "equity": account_info.equity,
                        "margin_free": account_info.margin_free,
                        "server": account_info.server
                    }
                )
                
                mt5.shutdown()
                
            except ImportError:
                if settings.ARIA_ENABLE_MT5:
                    # MT5 required but not available - FAIL FAST
                    raise RuntimeError("MT5 library not available but ARIA_ENABLE_MT5=1")
                component.update(HealthStatus.HEALTHY, metrics={"enabled": False})
                
        except Exception as e:
            component.update(HealthStatus.UNHEALTHY, error=str(e))
            
            # FAIL FAST: If MT5 is required and unhealthy, raise exception
            from backend.core.config import get_settings
            settings = get_settings()
            if settings.ARIA_ENABLE_MT5 and component.consecutive_failures > 3:
                logger.critical(f"MT5 connection critical failure: {e}")
                raise RuntimeError(f"MT5 connection required but unavailable: {e}")
        
        return component
    
    async def check_models(self) -> ComponentHealth:
        """Check AI model availability"""
        component = self.components["models"]
        
        try:
            from backend.core.config import get_settings
            settings = get_settings()
            
            required_models = []
            missing_models = []
            
            # Check required model files
            model_paths = {
                "xgboost": "backend/models/xgboost_forex.onnx",
                "lstm": "backend/models/lstm_forex.onnx",
                "cnn": "backend/models/cnn_patterns.onnx",
                "ppo": "backend/models/ppo_trader.zip"
            }
            
            # Determine which models are required
            if settings.include_xgb:
                required_models.append("xgboost")
            
            if settings.AUTO_TRADE_ENABLED:
                primary_model = settings.AUTO_TRADE_PRIMARY_MODEL
                if primary_model in model_paths:
                    required_models.append(primary_model)
            
            # Check each required model
            for model_name in required_models:
                path = model_paths.get(model_name)
                if path and not os.path.exists(path):
                    missing_models.append(model_name)
            
            if missing_models:
                # FAIL FAST: Required models missing
                error_msg = f"Required models missing: {missing_models}"
                component.update(HealthStatus.UNHEALTHY, error=error_msg)
                
                if settings.AUTO_TRADE_ENABLED:
                    raise RuntimeError(f"Cannot start auto-trading: {error_msg}")
            else:
                component.update(
                    HealthStatus.HEALTHY,
                    metrics={
                        "required_models": required_models,
                        "all_present": True
                    }
                )
                
        except Exception as e:
            component.update(HealthStatus.UNHEALTHY, error=str(e))
            logger.error(f"Model health check failed: {e}")
        
        return component
    
    async def check_auto_trader(self) -> ComponentHealth:
        """Check AutoTrader health"""
        component = self.components["auto_trader"]
        
        try:
            from backend.core.config import get_settings
            settings = get_settings()
            
            if not settings.AUTO_TRADE_ENABLED:
                component.update(HealthStatus.HEALTHY, metrics={"enabled": False})
                return component
            
            # Check if AutoTrader is running
            from backend.services.auto_trader import auto_trader
            
            if auto_trader and auto_trader.running:
                status_dict = auto_trader.get_status()
                
                # Check for critical issues
                if status_dict.get("error_count", 0) > 10:
                    component.update(
                        HealthStatus.DEGRADED,
                        error="High error count",
                        metrics=status_dict
                    )
                else:
                    component.update(HealthStatus.HEALTHY, metrics=status_dict)
            else:
                component.update(
                    HealthStatus.UNHEALTHY,
                    error="AutoTrader not running"
                )
                
        except Exception as e:
            component.update(HealthStatus.UNHEALTHY, error=str(e))
        
        return component
    
    async def check_system_resources(self):
        """Check system resource utilization"""
        
        # CPU check
        cpu_component = self.components["cpu"]
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            cpu_component.update(
                HealthStatus.UNHEALTHY,
                error=f"CPU usage critical: {cpu_percent}%",
                metrics={"usage_percent": cpu_percent}
            )
        elif cpu_percent > 75:
            cpu_component.update(
                HealthStatus.DEGRADED,
                error=f"CPU usage high: {cpu_percent}%",
                metrics={"usage_percent": cpu_percent}
            )
        else:
            cpu_component.update(
                HealthStatus.HEALTHY,
                metrics={"usage_percent": cpu_percent}
            )
        
        # Memory check
        memory_component = self.components["memory"]
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            memory_component.update(
                HealthStatus.UNHEALTHY,
                error=f"Memory usage critical: {memory.percent}%",
                metrics={
                    "usage_percent": memory.percent,
                    "available_mb": memory.available / 1024 / 1024
                }
            )
        elif memory.percent > 80:
            memory_component.update(
                HealthStatus.DEGRADED,
                error=f"Memory usage high: {memory.percent}%",
                metrics={
                    "usage_percent": memory.percent,
                    "available_mb": memory.available / 1024 / 1024
                }
            )
        else:
            memory_component.update(
                HealthStatus.HEALTHY,
                metrics={
                    "usage_percent": memory.percent,
                    "available_mb": memory.available / 1024 / 1024
                }
            )
        
        # Disk space check
        disk_component = self.components["disk_space"]
        disk = psutil.disk_usage("/")
        
        if disk.percent > 95:
            disk_component.update(
                HealthStatus.UNHEALTHY,
                error=f"Disk space critical: {disk.percent}%",
                metrics={
                    "usage_percent": disk.percent,
                    "free_gb": disk.free / 1024 / 1024 / 1024
                }
            )
        elif disk.percent > 85:
            disk_component.update(
                HealthStatus.DEGRADED,
                error=f"Disk space low: {disk.percent}%",
                metrics={
                    "usage_percent": disk.percent,
                    "free_gb": disk.free / 1024 / 1024 / 1024
                }
            )
        else:
            disk_component.update(
                HealthStatus.HEALTHY,
                metrics={
                    "usage_percent": disk.percent,
                    "free_gb": disk.free / 1024 / 1024 / 1024
                }
            )
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        
        # Run checks in parallel where possible
        tasks = [
            self.check_mt5_connection(),
            self.check_models(),
            self.check_auto_trader(),
            self.check_system_resources()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate overall health
        unhealthy_count = sum(
            1 for c in self.components.values()
            if c.status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for c in self.components.values()
            if c.status == HealthStatus.DEGRADED
        )
        
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime,
            "components": {
                name: component.to_dict()
                for name, component in self.components.items()
            }
        }
    
    async def readiness_check(self) -> Dict[str, Any]:
        """Check if system is ready to accept traffic"""
        
        # Critical components that must be healthy
        critical_components = ["mt5_connection", "models", "database"]
        
        from backend.core.config import get_settings
        settings = get_settings()
        
        # Adjust critical components based on configuration
        if not settings.mt5_enabled:
            critical_components.remove("mt5_connection")
        
        # Run health checks
        health_status = await self.check_all()
        
        # Check critical components
        ready = True
        reasons = []
        
        for component_name in critical_components:
            component = self.components.get(component_name)
            if component and component.status != HealthStatus.HEALTHY:
                ready = False
                reasons.append(f"{component_name}: {component.error_message}")
        
        return {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "reasons": reasons if not ready else None,
            "health": health_status
        }

# Global health checker instance
_health_checker: Optional[HealthChecker] = None

def get_health_checker() -> HealthChecker:
    """Get or create singleton health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
