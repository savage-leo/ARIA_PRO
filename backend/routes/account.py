from fastapi import APIRouter, HTTPException
import logging
from backend.services.mt5_executor import mt5_executor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/info")
async def get_account_info():
    """Get MT5 account information"""
    try:
        account_info = mt5_executor.get_account_info()
        return account_info
    except Exception as e:
        logger.warning(f"MT5 unavailable for account info: {str(e)}")
        return {
            "status": "unavailable",
            "message": "MT5 not available",
            "error": str(e),
        }


@router.get("/balance")
async def get_balance():
    """Get account balance"""
    try:
        account_info = mt5_executor.get_account_info()
        return {"balance": account_info["balance"], "equity": account_info["equity"]}
    except Exception as e:
        logger.warning(f"MT5 unavailable for balance: {str(e)}")
        return {
            "balance": None,
            "equity": None,
            "status": "unavailable",
            "message": "MT5 not available",
            "error": str(e),
        }
