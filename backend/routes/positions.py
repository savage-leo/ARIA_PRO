from fastapi import APIRouter, HTTPException
import logging
from backend.services.mt5_executor import mt5_executor
from backend.core.redis_cache import get_cache_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def get_positions():
    """Get all open positions"""
    cache_manager = get_cache_manager()
    
    # Try cache first
    cached_positions = await cache_manager.get_positions("mt5_account")
    if cached_positions:
        return {"positions": cached_positions, "count": len(cached_positions), "cached": True}
    
    try:
        positions = mt5_executor.get_positions()
        
        # Cache the result
        await cache_manager.cache_positions("mt5_account", positions)
        
        return {"positions": positions, "count": len(positions), "cached": False}
    except Exception as e:
        logger.warning(f"MT5 unavailable for positions: {str(e)}")
        return {
            "positions": [],
            "count": 0,
            "status": "unavailable",
            "message": "MT5 not available",
            "error": str(e),
        }


@router.delete("/{ticket}")
async def close_position(ticket: int):
    """Close a specific position"""
    cache_manager = get_cache_manager()
    
    try:
        result = mt5_executor.close_position(ticket=ticket)
        
        # Invalidate positions cache after closing
        await cache_manager.delete(cache_manager.key_builder.positions("mt5_account"))
        
        logger.info(f"Position {ticket} closed successfully")
        return result
    except Exception as e:
        logger.warning(f"MT5 unavailable or error closing position {ticket}: {str(e)}")
        return {
            "ticket": ticket,
            "status": "unavailable",
            "message": "MT5 not available or position close failed",
            "error": str(e),
        }
