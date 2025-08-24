from fastapi import APIRouter, HTTPException, Depends
import logging
from backend.services.mt5_executor import mt5_executor
from backend.core.redis_cache import get_cache_manager
from backend.core.auth import get_current_user, require_trader

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/info")
async def get_account_info(current_user=Depends(require_trader)):
    """Get MT5 account information"""
    cache_manager = get_cache_manager()
    
    # Try cache first
    cached_info = await cache_manager.get_account_info("mt5_account")
    if cached_info:
        return {**cached_info, "cached": True}
    
    try:
        account_info = mt5_executor.get_account_info()
        
        # Cache the result
        await cache_manager.cache_account_info("mt5_account", account_info)
        
        return {**account_info, "cached": False}
    except Exception as e:
        logger.warning(f"MT5 unavailable for account info: {str(e)}")
        return {
            "status": "unavailable",
            "message": "MT5 not available",
            "error": str(e),
        }


@router.get("/balance")
async def get_balance(current_user=Depends(require_trader)):
    """Get account balance"""
    cache_manager = get_cache_manager()
    
    # Try cache first
    cached_info = await cache_manager.get_account_info("mt5_account")
    if cached_info and "balance" in cached_info:
        return {
            "balance": cached_info["balance"], 
            "equity": cached_info["equity"],
            "cached": True
        }
    
    try:
        account_info = mt5_executor.get_account_info()
        
        # Cache the result
        await cache_manager.cache_account_info("mt5_account", account_info)
        
        return {
            "balance": account_info["balance"], 
            "equity": account_info["equity"],
            "cached": False
        }
    except Exception as e:
        logger.warning(f"MT5 unavailable for balance: {str(e)}")
        return {
            "balance": None,
            "equity": None,
            "status": "unavailable",
            "message": "MT5 not available",
            "error": str(e),
        }
