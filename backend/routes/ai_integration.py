"""AI Integration API Endpoints"""
from fastapi import APIRouter, Depends
from backend.services.mt5_ai_integration import mt5_ai_integration

router = APIRouter(prefix="/api/ai", tags=["AI Integration"])

@router.get("/status")
async def get_ai_status():
    """Get AI integration status"""
    return {"status": "active" if mt5_ai_integration.running else "inactive"}

@router.post("/start")
async def start_ai():
    """Start AI integration"""
    await mt5_ai_integration.start()
    return {"status": "started"}

@router.post("/stop")
async def stop_ai():
    """Stop AI integration"""
    await mt5_ai_integration.stop()
    return {"status": "stopped"}
