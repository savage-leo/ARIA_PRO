"""
CI/CD Pipeline Management API Endpoints
FastAPI routes for managing automated retraining and deployment
"""

from fastapi import APIRouter, HTTPException, Depends, Header, status
from typing import Dict, Any, Optional
import logging
import asyncio

from backend.core.config import get_settings
from backend.automation.cicd_pipeline import get_cicd_pipeline

logger = logging.getLogger(__name__)


def require_admin(
    x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """Simple admin API key validator.

    Accepts either:
    - X-Admin-API-Key: <ADMIN_API_KEY>
    - Authorization: Bearer <ADMIN_API_KEY>
    """
    settings = get_settings()
    expected = (settings.ADMIN_API_KEY or "").strip()
    if not expected:
        # Explicitly deny if no admin key is configured to avoid accidental exposure
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API key is not configured",
        )

    supplied = (x_admin_api_key or "").strip()
    if not supplied and authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            supplied = auth.split(" ", 1)[1].strip()

    if supplied != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin API key")


router = APIRouter(prefix="/api/cicd", tags=["cicd"], dependencies=[Depends(require_admin)])


@router.get("/status")
async def get_pipeline_status():
    """Get current CI/CD pipeline status"""
    try:
        status = get_cicd_pipeline().get_pipeline_status()
        return {"success": True, "data": status}
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_pipeline_scheduler():
    """Start automated pipeline scheduler"""
    try:
        await get_cicd_pipeline().start_scheduler()
        return {"success": True, "message": "Pipeline scheduler started"}
    except Exception as e:
        logger.error(f"Error starting pipeline scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_pipeline_scheduler():
    """Stop automated pipeline scheduler"""
    try:
        await get_cicd_pipeline().stop_scheduler()
        return {"success": True, "message": "Pipeline scheduler stopped"}
    except Exception as e:
        logger.error(f"Error stopping pipeline scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def trigger_pipeline_run(force: bool = False):
    """Trigger manual pipeline run"""
    try:
        # Schedule async pipeline run without blocking request/response cycle
        asyncio.create_task(get_cicd_pipeline().run_pipeline(force))
        return {"success": True, "message": "Pipeline run triggered"}
    except Exception as e:
        logger.error(f"Error triggering pipeline run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
async def get_pipeline_runs(limit: int = 10):
    """Get recent pipeline runs"""
    try:
        pipeline = get_cicd_pipeline()
        runs = pipeline.pipeline_history[-limit:] if pipeline.pipeline_history else []
        
        run_data = []
        for run in runs:
            run_data.append({
                "run_id": run.run_id,
                "status": run.status,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration": run.end_time - run.start_time if run.end_time > 0 else 0,
                "models_updated": run.models_updated,
                "stages": run.stages,
                "error_count": len(run.errors),
                "performance_metrics": run.performance_metrics
            })
        
        return {"success": True, "data": run_data}
    except Exception as e:
        logger.error(f"Error getting pipeline runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
async def get_pipeline_run_details(run_id: str):
    """Get detailed information about a specific pipeline run"""
    try:
        pipeline = get_cicd_pipeline()
        run = next((r for r in pipeline.pipeline_history if r.run_id == run_id), None)
        
        if not run:
            raise HTTPException(status_code=404, detail="Pipeline run not found")
        
        return {
            "success": True,
            "data": {
                "run_id": run.run_id,
                "status": run.status,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration": run.end_time - run.start_time if run.end_time > 0 else 0,
                "models_updated": run.models_updated,
                "stages": run.stages,
                "errors": run.errors,
                "performance_metrics": run.performance_metrics
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline run details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_pipeline_config():
    """Get current pipeline configuration"""
    try:
        config = get_cicd_pipeline().config
        return {
            "success": True,
            "data": {
                "retrain_interval_hours": config.retrain_interval_hours,
                "validation_threshold": config.validation_threshold,
                "stress_test_threshold": config.stress_test_threshold,
                "performance_degradation_limit": config.performance_degradation_limit,
                "backup_retention_days": config.backup_retention_days,
                "enable_auto_deployment": config.enable_auto_deployment,
                "enable_rollback_on_failure": config.enable_rollback_on_failure,
                "notification_webhook": config.notification_webhook,
                "data_prep_timeout": config.data_prep_timeout,
                "training_timeout": config.training_timeout,
                "validation_timeout": config.validation_timeout,
                "stress_timeout": config.stress_timeout,
                "profiling_timeout": config.profiling_timeout,
                "deployment_timeout": config.deployment_timeout,
                "post_validation_timeout": config.post_validation_timeout
            }
        }
    except Exception as e:
        logger.error(f"Error getting pipeline config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config")
async def update_pipeline_config(config_data: Dict[str, Any]):
    """Update pipeline configuration"""
    try:
        # Update configuration
        pipeline = get_cicd_pipeline()
        if "retrain_interval_hours" in config_data:
            pipeline.config.retrain_interval_hours = config_data["retrain_interval_hours"]
        if "validation_threshold" in config_data:
            pipeline.config.validation_threshold = config_data["validation_threshold"]
        if "stress_test_threshold" in config_data:
            pipeline.config.stress_test_threshold = config_data["stress_test_threshold"]
        if "performance_degradation_limit" in config_data:
            pipeline.config.performance_degradation_limit = config_data["performance_degradation_limit"]
        if "backup_retention_days" in config_data:
            pipeline.config.backup_retention_days = config_data["backup_retention_days"]
        if "enable_auto_deployment" in config_data:
            pipeline.config.enable_auto_deployment = config_data["enable_auto_deployment"]
        if "enable_rollback_on_failure" in config_data:
            pipeline.config.enable_rollback_on_failure = config_data["enable_rollback_on_failure"]
        if "notification_webhook" in config_data:
            pipeline.config.notification_webhook = config_data["notification_webhook"]
        if "data_prep_timeout" in config_data:
            pipeline.config.data_prep_timeout = config_data["data_prep_timeout"]
        if "training_timeout" in config_data:
            pipeline.config.training_timeout = config_data["training_timeout"]
        if "validation_timeout" in config_data:
            pipeline.config.validation_timeout = config_data["validation_timeout"]
        if "stress_timeout" in config_data:
            pipeline.config.stress_timeout = config_data["stress_timeout"]
        if "profiling_timeout" in config_data:
            pipeline.config.profiling_timeout = config_data["profiling_timeout"]
        if "deployment_timeout" in config_data:
            pipeline.config.deployment_timeout = config_data["deployment_timeout"]
        if "post_validation_timeout" in config_data:
            pipeline.config.post_validation_timeout = config_data["post_validation_timeout"]
        
        return {"success": True, "message": "Pipeline configuration updated"}
    except Exception as e:
        logger.error(f"Error updating pipeline config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def pipeline_health_check():
    """Health check for CI/CD pipeline system"""
    try:
        status = get_cicd_pipeline().get_pipeline_status()
        
        health_data = {
            "pipeline_scheduler": "running" if status["is_running"] else "stopped",
            "recent_runs": len(status["recent_runs"]),
            "last_run_status": status["recent_runs"][-1]["status"] if status["recent_runs"] else "none",
            "system_status": "healthy"
        }
        
        return {"success": True, "data": health_data}
    except Exception as e:
        logger.error(f"Pipeline health check failed: {e}")
        return {
            "success": False,
            "data": {"system_status": "unhealthy", "error": str(e)}
        }
