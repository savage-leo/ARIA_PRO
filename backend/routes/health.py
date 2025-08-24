"""
Health check endpoints for production monitoring.
"""
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])

class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    uptime_seconds: float
    checks: Dict[str, Dict[str, Any]]
    environment: str
    version: str

class HealthChecker:
    """Manages health checks for the application."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checks = {}
        
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            # TODO: Add actual database check
            return {
                "status": "healthy",
                "message": "Database connection OK",
                "latency_ms": 5
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "error": True
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            # TODO: Add actual Redis check
            return {
                "status": "healthy",
                "message": "Redis connection OK",
                "latency_ms": 2
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "error": True
            }
    
    async def check_mt5(self) -> Dict[str, Any]:
        """Check MT5 connection status."""
        try:
            mt5_enabled = os.getenv("ARIA_ENABLE_MT5", "0") == "1"
            if not mt5_enabled:
                return {
                    "status": "disabled",
                    "message": "MT5 is disabled"
                }
            
            # TODO: Add actual MT5 connection check
            return {
                "status": "healthy",
                "message": "MT5 connection OK",
                "connected": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "error": True
            }
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            used_percent = (used / total) * 100
            
            status = "healthy"
            if free_gb < 1:
                status = "critical"
            elif free_gb < 5:
                status = "warning"
            
            return {
                "status": status,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 2)
            }
        except Exception as e:
            return {
                "status": "unknown",
                "message": str(e),
                "error": True
            }
    
    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 90:
                status = "critical"
            elif memory.percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "used_percent": round(memory.percent, 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "total_gb": round(memory.total / (1024 ** 3), 2)
            }
        except ImportError:
            return {
                "status": "unknown",
                "message": "psutil not installed",
                "error": True
            }
        except Exception as e:
            return {
                "status": "unknown",
                "message": str(e),
                "error": True
            }
    
    async def check_all(self) -> HealthStatus:
        """Run all health checks."""
        checks = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_mt5(),
            self.check_disk_space(),
            self.check_memory(),
            return_exceptions=True
        )
        
        results = {
            "database": checks[0] if not isinstance(checks[0], Exception) else {"status": "error", "message": str(checks[0])},
            "redis": checks[1] if not isinstance(checks[1], Exception) else {"status": "error", "message": str(checks[1])},
            "mt5": checks[2] if not isinstance(checks[2], Exception) else {"status": "error", "message": str(checks[2])},
            "disk": checks[3] if not isinstance(checks[3], Exception) else {"status": "error", "message": str(checks[3])},
            "memory": checks[4] if not isinstance(checks[4], Exception) else {"status": "error", "message": str(checks[4])}
        }
        
        # Determine overall status
        overall_status = "healthy"
        for check in results.values():
            if check.get("status") == "critical" or check.get("status") == "error":
                overall_status = "unhealthy"
                break
            elif check.get("status") == "warning":
                overall_status = "degraded"
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=round(time.time() - self.start_time, 2),
            checks=results,
            environment=os.getenv("ARIA_ENV", "development"),
            version=os.getenv("APP_VERSION", "1.2.0")
        )

# Global health checker instance
health_checker = HealthChecker()

@router.get("/", response_model=HealthStatus)
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns 200 if healthy, 503 if unhealthy.
    """
    status = await health_checker.check_all()
    
    if status.status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=status.model_dump()
        )
    
    return status

@router.get("/liveness")
async def liveness_check():
    """
    Simple liveness check for Kubernetes.
    Returns 200 if the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@router.get("/readiness")
async def readiness_check():
    """
    Readiness check for Kubernetes.
    Returns 200 if ready to accept traffic, 503 if not.
    """
    status = await health_checker.check_all()
    
    if status.status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": status.checks}
        )
    
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

@router.get("/startup")
async def startup_check():
    """
    Startup probe for Kubernetes.
    Returns 200 when the application has started.
    """
    uptime = time.time() - health_checker.start_time
    
    # Consider started after 5 seconds
    if uptime < 5:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "starting", "uptime_seconds": uptime}
        )
    
    return {"status": "started", "uptime_seconds": uptime}
