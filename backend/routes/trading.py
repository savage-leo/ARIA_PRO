from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any
import logging
from backend.services.mt5_executor import mt5_executor
from backend.services.risk_engine import risk_engine
from datetime import datetime
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

router = APIRouter()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)


class OrderRequest(BaseModel):
    symbol: str
    volume: float
    order_type: str  # "buy" or "sell"
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: Optional[str] = "ARIA AI Order"
    
    @validator('volume')
    def validate_volume(cls, v):
        if v <= 0:
            raise ValueError('Volume must be positive')
        if v > 100:
            raise ValueError('Volume cannot exceed 100 lots')
        return v
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 6:
            raise ValueError('Invalid symbol format')
        return v.upper()
    
    @validator('order_type')
    def validate_order_type(cls, v):
        if v.lower() not in ['buy', 'sell']:
            raise ValueError('Order type must be buy or sell')
        return v.lower()


class OrderResponse(BaseModel):
    ticket: int
    status: str
    volume: float
    price: float
    timestamp: str
    slippage: Optional[float] = None


@router.post("/place-order", response_model=OrderResponse)
@limiter.limit("10/minute")
async def place_order(request: Request, order: OrderRequest):
    """Place a real order with risk validation"""
    try:
        # Get account information
        account_info = mt5_executor.get_account_info()

        # Get current positions
        current_positions = mt5_executor.get_positions()

        # Risk validation
        validation = risk_engine.validate_order(
            symbol=order.symbol,
            volume=order.volume,
            order_type=order.order_type,
            account_info=account_info,
            current_positions=current_positions,
        )

        if not validation["approved"]:
            raise HTTPException(
                status_code=400,
                detail=f"Order rejected: {'; '.join(validation['errors'])}",
            )

        # Check emergency stop
        if risk_engine.emergency_stop(account_info):
            raise HTTPException(
                status_code=503,
                detail="Trading halted due to emergency stop conditions",
            )

        # Get current market price for slippage tracking
        symbol_info = mt5_executor.get_symbol_info(order.symbol)
        expected_price = (
            symbol_info["ask"]
            if order.order_type.lower() == "buy"
            else symbol_info["bid"]
        )

        # Place the order
        result = mt5_executor.place_order(
            symbol=order.symbol,
            volume=order.volume,
            order_type=order.order_type,
            sl=order.sl,
            tp=order.tp,
            comment=order.comment,
        )

        # Track slippage
        slippage_record = risk_engine.track_slippage(
            expected_price=expected_price,
            executed_price=result["price"],
            symbol=order.symbol,
        )

        logger.info(f"Order placed successfully: {result['ticket']} for {order.symbol}")

        return OrderResponse(
            ticket=result["ticket"],
            status=result["status"],
            volume=result["volume"],
            price=result["price"],
            timestamp=result["timestamp"],
            slippage=slippage_record["slippage_pips"],
        )

    except RuntimeError as e:
        logger.error(f"MT5 runtime error: {e}")
        raise HTTPException(status_code=503, detail="Trading service unavailable")
    except ValueError as e:
        logger.error(f"Order validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid order parameters")
    except Exception as e:
        logger.error(f"Unexpected error in order placement: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/positions")
async def get_positions():
    """Get all open positions"""
    import os

    # Check if MT5 credentials are configured
    mt5_login = os.getenv("MT5_LOGIN", "").strip()
    mt5_password = os.getenv("MT5_PASSWORD", "").strip()
    mt5_server = os.getenv("MT5_SERVER", "").strip()

    if not mt5_login or not mt5_password or not mt5_server:
        raise HTTPException(
            status_code=503,
            detail="MT5 credentials not configured. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in environment.",
        )

    try:
        # Check if MT5 is properly configured
        if not mt5_executor.connect():
            raise HTTPException(
                status_code=503,
                detail="MT5 not connected. Terminal will be auto-launched if installed. Ensure terminal is running and credentials are valid.",
            )

        positions = mt5_executor.get_positions()
        return {"positions": positions, "count": len(positions)}
    except Exception as e:
        logger.error(f"Failed to get positions: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Positions unavailable: {str(e)}")


@router.delete("/close-position/{ticket}")
@limiter.limit("20/minute")
async def close_position(request: Request, ticket: int, volume: Optional[float] = None):
    """Close a specific position"""
    try:
        result = mt5_executor.close_position(ticket=ticket, volume=volume)
        logger.info(f"Position {ticket} closed successfully")
        return result
    except Exception as e:
        logger.warning(f"MT5 unavailable or error closing position {ticket}: {str(e)}")
        # Return safe fallback instead of 500 to avoid frontend disruption
        return {
            "ticket": ticket,
            "status": "unavailable",
            "message": "MT5 not available or position close failed",
            "error": str(e),
        }


@router.get("/account-info")
async def get_account_info():
    """Get MT5 account information"""
    import os

    # Check if MT5 credentials are configured
    mt5_login = os.getenv("MT5_LOGIN", "").strip()
    mt5_password = os.getenv("MT5_PASSWORD", "").strip()
    mt5_server = os.getenv("MT5_SERVER", "").strip()

    if not mt5_login or not mt5_password or not mt5_server:
        raise HTTPException(
            status_code=503,
            detail="MT5 credentials not configured. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in environment.",
        )

    try:
        # Check if MT5 is properly configured
        if not mt5_executor.connect():
            raise HTTPException(
                status_code=503,
                detail="MT5 not connected. Terminal will be auto-launched if installed. Ensure terminal is running and credentials are valid.",
            )

        account_info = mt5_executor.get_account_info()
        risk_metrics = risk_engine.get_risk_metrics(account_info)

        return {**account_info, "risk_metrics": risk_metrics}
    except Exception as e:
        logger.error(f"Failed to get account info: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Account info unavailable: {str(e)}"
        )


@router.get("/symbol-info/{symbol}")
async def get_symbol_info(symbol: str):
    """Get symbol information"""
    try:
        symbol_info = mt5_executor.get_symbol_info(symbol)
        return symbol_info
    except Exception as e:
        logger.error(f"Failed to get symbol info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-position-size")
async def calculate_position_size(
    symbol: str,
    account_balance: float,
    entry_price: float,
    stop_loss: Optional[float] = None,
):
    """Calculate safe position size based on risk parameters"""
    try:
        position_size = risk_engine.calculate_position_size(
            account_balance=account_balance,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        return {
            "symbol": symbol,
            "position_size": position_size,
            "account_balance": account_balance,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
        }
    except Exception as e:
        logger.error(f"Failed to calculate position size: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-metrics")
async def get_risk_metrics():
    """Get current risk metrics from MT5. No simulation fallback."""
    try:
        if not mt5_executor.connect():
            raise HTTPException(
                status_code=503,
                detail="MT5 not connected. Terminal will be auto-launched if installed. Ensure terminal is running and credentials are valid.",
            )

        account_info = mt5_executor.get_account_info()
        metrics = risk_engine.get_risk_metrics(account_info)
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Risk metrics unavailable: {str(e)}"
        )


@router.post("/set-risk-level/{level}")
async def set_risk_level(level: str):
    """Set risk level (conservative, moderate, aggressive)"""
    try:
        from backend.services.risk_engine import RiskLevel

        if level.lower() not in ["conservative", "moderate", "aggressive"]:
            raise HTTPException(
                status_code=400,
                detail="Risk level must be conservative, moderate, or aggressive",
            )

        risk_level = RiskLevel(level.lower())
        risk_engine.set_risk_level(risk_level)

        return {"message": f"Risk level set to {level}", "risk_level": level}
    except Exception as e:
        logger.error(f"Failed to set risk level: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
