from fastapi import APIRouter, HTTPException
import logging
from backend.services.mt5_executor import mt5_executor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def get_positions():
    """Get all open positions"""
    try:
        positions = mt5_executor.get_positions()
        return {"positions": positions, "count": len(positions)}
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
    try:
        result = mt5_executor.close_position(ticket=ticket)
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
