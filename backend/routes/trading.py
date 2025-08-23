from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any
import logging
import time
from backend.services.mt5_executor import mt5_executor
from backend.services.risk_engine import risk_engine
from datetime import datetime
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from backend.core.metrics import get_metrics_collector
from backend.core.structured_logger import get_structured_logger
from backend.core.audit import audit_order_submit, get_audit_logger, AuditEventType
from backend.core.auth import get_current_user, get_current_active_user, require_trader, User

router = APIRouter()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

# Observability singletons
metrics = get_metrics_collector()
slog = get_structured_logger(__name__)


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
@limiter.limit("5/minute")
async def place_order(
    request: Request,
    order: OrderRequest,
    current_user: User = Depends(require_trader),
):
    """Place a real order with risk validation"""
    start_time = time.perf_counter()
    user_id = current_user.username
    ip = request.client.host if request.client else None
    try:
        # Log submit attempt and increment submitted metric
        slog.order_event(
            event="submit_attempt",
            symbol=order.symbol,
            order_type=order.order_type,
            volume=order.volume,
            user_id=user_id,
        )
        metrics.record_order_submitted(
            symbol=order.symbol, order_type=order.order_type, user=user_id
        )
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
            # Metrics + structured log + audit for rejection
            metrics.record_order_rejected(symbol=order.symbol, reason="risk_validation")
            slog.order_event(
                event="rejected",
                symbol=order.symbol,
                order_type=order.order_type,
                volume=order.volume,
                user_id=user_id,
                metrics={"errors": validation["errors"]},
            )
            get_audit_logger().log_event(
                event_type=AuditEventType.ORDER_REJECT,
                action="Risk validation failed",
                username=user_id,
                ip_address=ip,
                symbol=order.symbol,
                details={
                    "order_type": order.order_type,
                    "volume": order.volume,
                    "errors": validation["errors"],
                },
            )
            duration = time.perf_counter() - start_time
            metrics.record_api_request(
                method="POST", endpoint="/trading/place-order", status=400, duration=duration
            )
            slog.api_event(
                method="POST",
                endpoint="/trading/place-order",
                status_code=400,
                response_time=duration,
                user_id=user_id,
                ip_address=ip,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Order rejected: {'; '.join(validation['errors'])}",
            )

        # Check emergency stop
        if risk_engine.emergency_stop(account_info):
            metrics.record_order_rejected(symbol=order.symbol, reason="emergency_stop")
            slog.order_event(
                event="halted_emergency_stop",
                symbol=order.symbol,
                order_type=order.order_type,
                volume=order.volume,
                user_id=user_id,
            )
            get_audit_logger().log_event(
                event_type=AuditEventType.ORDER_REJECT,
                action="Emergency stop active",
                username=user_id,
                ip_address=ip,
                symbol=order.symbol,
                details={
                    "order_type": order.order_type,
                    "volume": order.volume,
                },
            )
            duration = time.perf_counter() - start_time
            metrics.record_api_request(
                method="POST", endpoint="/trading/place-order", status=503, duration=duration
            )
            slog.api_event(
                method="POST",
                endpoint="/trading/place-order",
                status_code=503,
                response_time=duration,
                user_id=user_id,
                ip_address=ip,
            )
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

        latency = time.perf_counter() - start_time
        metrics.record_order_executed(
            symbol=order.symbol, order_type=order.order_type, latency=latency
        )
        slog.order_event(
            event="executed",
            symbol=order.symbol,
            order_type=order.order_type,
            volume=order.volume,
            price=result["price"],
            order_id=str(result["ticket"]),
            user_id=user_id,
            slippage_pips=slippage_record.get("slippage_pips"),
        )
        # Audit successful submission
        try:
            audit_order_submit(
                username=user_id,
                symbol=order.symbol,
                order_id=str(result["ticket"]),
                order_type=order.order_type,
                volume=order.volume,
                price=result["price"],
                sl=order.sl,
                tp=order.tp,
                ip_address=ip,
                risk_metrics=risk_engine.get_risk_metrics(account_info),
            )
        except Exception as _ae:
            logger.warning(f"Audit logging failed for order {result['ticket']}: {_ae}")

        # API metrics/logging for success
        metrics.record_api_request(
            method="POST", endpoint="/trading/place-order", status=200, duration=latency
        )
        slog.api_event(
            method="POST",
            endpoint="/trading/place-order",
            status_code=200,
            response_time=latency,
            user_id=user_id,
            ip_address=ip,
        )

        return OrderResponse(
            ticket=result["ticket"],
            status=result["status"],
            volume=result["volume"],
            price=result["price"],
            timestamp=result["timestamp"],
            slippage=slippage_record["slippage_pips"],
        )

    except RuntimeError as e:
        # Metrics/logging for MT5 runtime error
        metrics.record_order_rejected(symbol=order.symbol, reason="mt5_runtime_error")
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="POST", endpoint="/trading/place-order", status=503, duration=duration
        )
        slog.api_event(
            method="POST",
            endpoint="/trading/place-order",
            status_code=503,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
            error_context="mt5_runtime_error",
        )
        try:
            get_audit_logger().log_event(
                event_type=AuditEventType.ORDER_REJECT,
                action="MT5 runtime error",
                username=user_id,
                ip_address=ip,
                symbol=order.symbol,
                details={
                    "order_type": order.order_type,
                    "volume": order.volume,
                    "error": str(e),
                },
            )
        except Exception:
            pass
        logger.error(f"MT5 runtime error: {e}")
        raise HTTPException(status_code=503, detail="Trading service unavailable")
    except ValueError as e:
        metrics.record_order_rejected(symbol=order.symbol, reason="validation_error")
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="POST", endpoint="/trading/place-order", status=400, duration=duration
        )
        slog.api_event(
            method="POST",
            endpoint="/trading/place-order",
            status_code=400,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
            error_context="validation_error",
        )
        try:
            get_audit_logger().log_event(
                event_type=AuditEventType.ORDER_REJECT,
                action="Order validation error",
                username=user_id,
                ip_address=ip,
                symbol=order.symbol,
                details={
                    "order_type": order.order_type,
                    "volume": order.volume,
                    "error": str(e),
                },
            )
        except Exception:
            pass
        logger.error(f"Order validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid order parameters")
    except Exception as e:
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="POST", endpoint="/trading/place-order", status=500, duration=duration
        )
        slog.api_event(
            method="POST",
            endpoint="/trading/place-order",
            status_code=500,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
            error_context="unexpected_error",
        )
        logger.error(f"Unexpected error in order placement: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/positions")
async def get_positions(
    request: Request, current_user: Optional[User] = Depends(get_current_user)
):
    """Get all open positions"""
    start_time = time.perf_counter()
    user_id = current_user.username if current_user else "anonymous"
    ip = request.client.host if request.client else None
    try:
        # Check MT5 connectivity
        if not mt5_executor.connect():
            duration = time.perf_counter() - start_time
            metrics.record_api_request(
                method="GET", endpoint="/trading/positions", status=503, duration=duration
            )
            slog.api_event(
                method="GET",
                endpoint="/trading/positions",
                status_code=503,
                response_time=duration,
                user_id=user_id,
                ip_address=ip,
            )
            raise HTTPException(
                status_code=503,
                detail="MT5 not connected. Terminal will be auto-launched if installed. Ensure terminal is running and credentials are valid.",
            )

        positions = mt5_executor.get_positions()
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="GET", endpoint="/trading/positions", status=200, duration=duration
        )
        slog.api_event(
            method="GET",
            endpoint="/trading/positions",
            status_code=200,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        return {"positions": positions, "count": len(positions)}
    except Exception as e:
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="GET", endpoint="/trading/positions", status=503, duration=duration
        )
        slog.api_event(
            method="GET",
            endpoint="/trading/positions",
            status_code=503,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        logger.error(f"Failed to get positions: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Positions unavailable: {str(e)}")


@router.delete("/close-position/{ticket}")
@limiter.limit("10/minute")
async def close_position(
    request: Request,
    ticket: int,
    volume: Optional[float] = None,
    current_user: User = Depends(require_trader),
):
    """Close a specific position"""
    start_time = time.perf_counter()
    user_id = current_user.username
    ip = request.client.host if request.client else None
    try:
        result = mt5_executor.close_position(ticket=ticket, volume=volume)
        duration = time.perf_counter() - start_time
        slog.order_event(
            event="position_close_success",
            symbol=str(result.get("ticket", ticket)),
            order_type="close",
            volume=result.get("volume", volume or 0.0),
            price=result.get("price"),
            order_id=str(ticket),
            user_id=user_id,
        )
        metrics.record_api_request(
            method="DELETE", endpoint="/trading/close-position", status=200, duration=duration
        )
        slog.api_event(
            method="DELETE",
            endpoint="/trading/close-position",
            status_code=200,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        return result
    except Exception as e:
        duration = time.perf_counter() - start_time
        slog.order_event(
            event="position_close_error",
            symbol=str(ticket),
            order_type="close",
            volume=volume or 0.0,
            user_id=user_id,
            error=str(e),
        )
        metrics.record_api_request(
            method="DELETE", endpoint="/trading/close-position", status=503, duration=duration
        )
        slog.api_event(
            method="DELETE",
            endpoint="/trading/close-position",
            status_code=503,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        logger.warning(f"MT5 unavailable or error closing position {ticket}: {str(e)}")
        # Return safe fallback instead of 500 to avoid frontend disruption
        return {
            "ticket": ticket,
            "status": "unavailable",
            "message": "MT5 not available or position close failed",
            "error": str(e),
        }


@router.get("/account-info")
async def get_account_info(
    request: Request, current_user: Optional[User] = Depends(get_current_user)
):
    """Get MT5 account information"""
    start_time = time.perf_counter()
    user_id = current_user.username if current_user else "anonymous"
    ip = request.client.host if request.client else None
    try:
        # Check MT5 connectivity
        if not mt5_executor.connect():
            duration = time.perf_counter() - start_time
            metrics.record_api_request(
                method="GET", endpoint="/trading/account-info", status=503, duration=duration
            )
            slog.api_event(
                method="GET",
                endpoint="/trading/account-info",
                status_code=503,
                response_time=duration,
                user_id=user_id,
                ip_address=ip,
            )
            raise HTTPException(
                status_code=503,
                detail="MT5 not connected. Terminal will be auto-launched if installed. Ensure terminal is running and credentials are valid.",
            )

        account_info = mt5_executor.get_account_info()
        risk_metrics = risk_engine.get_risk_metrics(account_info)

        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="GET", endpoint="/trading/account-info", status=200, duration=duration
        )
        slog.api_event(
            method="GET",
            endpoint="/trading/account-info",
            status_code=200,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        return {**account_info, "risk_metrics": risk_metrics}
    except Exception as e:
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="GET", endpoint="/trading/account-info", status=503, duration=duration
        )
        slog.api_event(
            method="GET",
            endpoint="/trading/account-info",
            status_code=503,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        logger.error(f"Failed to get account info: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Account info unavailable: {str(e)}"
        )


@router.get("/symbol-info/{symbol}")
async def get_symbol_info(symbol: str, request: Request, current_user: Optional[User] = Depends(get_current_user)):
    """Get symbol information"""
    start_time = time.perf_counter()
    user_id = current_user.username if current_user else "anonymous"
    ip = request.client.host if request.client else None
    try:
        symbol_info = mt5_executor.get_symbol_info(symbol)
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="GET", endpoint="/trading/symbol-info", status=200, duration=duration
        )
        slog.api_event(
            method="GET",
            endpoint="/trading/symbol-info",
            status_code=200,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        return symbol_info
    except Exception as e:
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="GET", endpoint="/trading/symbol-info", status=500, duration=duration
        )
        slog.api_event(
            method="GET",
            endpoint="/trading/symbol-info",
            status_code=500,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        logger.error(f"Failed to get symbol info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-position-size")
async def calculate_position_size(
    symbol: str,
    account_balance: float,
    entry_price: float,
    stop_loss: Optional[float] = None,
    request: Request = None,
    current_user: Optional[User] = Depends(get_current_user),
):
    """Calculate safe position size based on risk parameters"""
    start_time = time.perf_counter()
    user_id = current_user.username if current_user else "anonymous"
    ip = request.client.host if request.client else None if request else None
    try:
        position_size = risk_engine.calculate_position_size(
            account_balance=account_balance,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="POST", endpoint="/trading/calculate-position-size", status=200, duration=duration
        )
        slog.api_event(
            method="POST",
            endpoint="/trading/calculate-position-size",
            status_code=200,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        return {
            "symbol": symbol,
            "position_size": position_size,
            "account_balance": account_balance,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
        }
    except Exception as e:
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="POST", endpoint="/trading/calculate-position-size", status=500, duration=duration
        )
        slog.api_event(
            method="POST",
            endpoint="/trading/calculate-position-size",
            status_code=500,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        logger.error(f"Failed to calculate position size: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-metrics")
async def get_risk_metrics(request: Request, current_user: Optional[User] = Depends(get_current_user)):
    """Get current risk metrics from MT5. No simulation fallback."""
    start_time = time.perf_counter()
    user_id = current_user.username if current_user else "anonymous"
    ip = request.client.host if request.client else None
    try:
        if not mt5_executor.connect():
            duration = time.perf_counter() - start_time
            metrics.record_api_request(
                method="GET", endpoint="/trading/risk-metrics", status=503, duration=duration
            )
            slog.api_event(
                method="GET",
                endpoint="/trading/risk-metrics",
                status_code=503,
                response_time=duration,
                user_id=user_id,
                ip_address=ip,
            )
            raise HTTPException(
                status_code=503,
                detail="MT5 not connected. Terminal will be auto-launched if installed. Ensure terminal is running and credentials are valid.",
            )

        account_info = mt5_executor.get_account_info()
        metrics_out = risk_engine.get_risk_metrics(account_info)
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="GET", endpoint="/trading/risk-metrics", status=200, duration=duration
        )
        slog.api_event(
            method="GET",
            endpoint="/trading/risk-metrics",
            status_code=200,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        return metrics_out
    except HTTPException:
        raise
    except Exception as e:
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="GET", endpoint="/trading/risk-metrics", status=503, duration=duration
        )
        slog.api_event(
            method="GET",
            endpoint="/trading/risk-metrics",
            status_code=503,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        logger.error(f"Failed to get risk metrics: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Risk metrics unavailable: {str(e)}"
        )


@router.post("/set-risk-level/{level}")
async def set_risk_level(level: str, request: Request, current_user: User = Depends(require_trader)):
    """Set risk level (conservative, moderate, aggressive)"""
    start_time = time.perf_counter()
    user_id = current_user.username
    ip = request.client.host if request.client else None
    try:
        from backend.services.risk_engine import RiskLevel

        if level.lower() not in ["conservative", "moderate", "aggressive"]:
            raise HTTPException(
                status_code=400,
                detail="Risk level must be conservative, moderate, or aggressive",
            )

        risk_level = RiskLevel(level.lower())
        risk_engine.set_risk_level(risk_level)

        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="POST", endpoint="/trading/set-risk-level", status=200, duration=duration
        )
        slog.api_event(
            method="POST",
            endpoint="/trading/set-risk-level",
            status_code=200,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        return {"message": f"Risk level set to {level}", "risk_level": level}
    except Exception as e:
        duration = time.perf_counter() - start_time
        metrics.record_api_request(
            method="POST", endpoint="/trading/set-risk-level", status=500, duration=duration
        )
        slog.api_event(
            method="POST",
            endpoint="/trading/set-risk-level",
            status_code=500,
            response_time=duration,
            user_id=user_id,
            ip_address=ip,
        )
        logger.error(f"Failed to set risk level: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
