"""
Live Guard Middleware for ARIA PRO
Production safety enforcement and trading safeguards
"""

import os
import logging
from typing import Optional, Set, Dict, Any
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LiveGuardMiddleware(BaseHTTPMiddleware):
    """
    Production safety middleware to enforce trading restrictions and safeguards
    """
    
    def __init__(
        self,
        app,
        environment: str = "production",
        auto_trade_enabled: bool = False,
        max_daily_trades: int = 100,
        max_position_size: float = 0.1,  # Max 0.1 lot per position
        max_daily_loss: float = 1000.0,  # Max $1000 daily loss
        blocked_symbols: Optional[Set[str]] = None,
        maintenance_mode: bool = False
    ):
        super().__init__(app)
        self.environment = environment
        self.auto_trade_enabled = auto_trade_enabled
        self.max_daily_trades = max_daily_trades
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.blocked_symbols = blocked_symbols or {"XAUUSD", "XAGUSD"}  # High volatility
        self.maintenance_mode = maintenance_mode
        
        # Trading statistics
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.last_reset = datetime.now()
        self.trade_history = []
        
        # Critical endpoints that require extra validation
        self.critical_endpoints = {
            "/api/trading/execute",
            "/api/trading/close",
            "/api/trading/modify",
            "/api/mt5/place_order",
            "/api/mt5/close_position",
        }
        
        # Read-only endpoints in production
        self.production_readonly = {
            "/api/config/update",
            "/api/system/reset",
            "/api/database/migrate",
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with production safeguards"""
        
        # Check maintenance mode
        if self.maintenance_mode:
            if not request.url.path.startswith("/health"):
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Service Unavailable",
                        "message": "System is under maintenance. Please try again later.",
                        "maintenance": True
                    }
                )
        
        # Reset daily counters if needed
        self._check_daily_reset()
        
        # Check if endpoint is critical
        path = request.url.path
        if any(path.startswith(endpoint) for endpoint in self.critical_endpoints):
            # Validate trading request
            validation_result = await self._validate_trading_request(request)
            if not validation_result["allowed"]:
                logger.warning(f"Trading request blocked: {validation_result['reason']}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Trading Not Allowed",
                        "message": validation_result["reason"],
                        "details": validation_result.get("details", {})
                    }
                )
        
        # Block dangerous operations in production
        if self.environment == "production":
            if any(path.startswith(endpoint) for endpoint in self.production_readonly):
                logger.error(f"Attempted dangerous operation in production: {path}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Operation Not Allowed",
                        "message": "This operation is not allowed in production environment"
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add safety headers
        response.headers["X-Environment"] = self.environment
        response.headers["X-Auto-Trade"] = str(self.auto_trade_enabled).lower()
        response.headers["X-Daily-Trades"] = str(self.daily_trades)
        response.headers["X-Daily-Loss"] = str(self.daily_loss)
        
        return response
    
    def _check_daily_reset(self):
        """Reset daily counters at midnight"""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            logger.info(f"Resetting daily counters. Previous: trades={self.daily_trades}, loss={self.daily_loss}")
            self.daily_trades = 0
            self.daily_loss = 0.0
            self.trade_history = []
            self.last_reset = now
    
    async def _validate_trading_request(self, request: Request) -> Dict[str, Any]:
        """Validate trading request against safety rules"""
        
        # Check if auto trading is enabled
        if not self.auto_trade_enabled:
            auto_trade_env = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
            if not auto_trade_env:
                return {
                    "allowed": False,
                    "reason": "Auto trading is disabled. Enable AUTO_TRADE_ENABLED to allow trading."
                }
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return {
                "allowed": False,
                "reason": f"Daily trade limit reached ({self.max_daily_trades} trades)",
                "details": {"daily_trades": self.daily_trades, "limit": self.max_daily_trades}
            }
        
        # Check daily loss limit
        if self.daily_loss >= self.max_daily_loss:
            return {
                "allowed": False,
                "reason": f"Daily loss limit reached (${self.max_daily_loss})",
                "details": {"daily_loss": self.daily_loss, "limit": self.max_daily_loss}
            }
        
        # Parse request body for additional validation
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    import json
                    data = json.loads(body)
                    
                    # Check symbol restrictions
                    symbol = data.get("symbol", "").upper()
                    if symbol in self.blocked_symbols:
                        return {
                            "allowed": False,
                            "reason": f"Trading blocked for symbol: {symbol}",
                            "details": {"symbol": symbol, "blocked_symbols": list(self.blocked_symbols)}
                        }
                    
                    # Check position size
                    volume = data.get("volume", 0)
                    if volume > self.max_position_size:
                        return {
                            "allowed": False,
                            "reason": f"Position size exceeds limit ({self.max_position_size} lots)",
                            "details": {"requested": volume, "limit": self.max_position_size}
                        }
                    
                    # Weekend trading check
                    now = datetime.now()
                    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                        return {
                            "allowed": False,
                            "reason": "Trading not allowed on weekends",
                            "details": {"day": now.strftime("%A")}
                        }
                    
                    # After-hours check (optional - forex is 24/5)
                    hour = now.hour
                    if hour < 1 or hour >= 22:  # Reduced liquidity hours
                        logger.warning(f"Trading during low liquidity hours: {hour}:00")
        
        except Exception as e:
            logger.error(f"Error validating trading request: {e}")
            # Fail safe - block if we can't validate
            return {
                "allowed": False,
                "reason": "Failed to validate trading request",
                "details": {"error": str(e)}
            }
        
        return {"allowed": True, "reason": "Trading allowed"}
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record trade for daily statistics"""
        self.daily_trades += 1
        
        # Update daily P&L if trade has profit/loss info
        if "profit" in trade_data:
            profit = trade_data["profit"]
            if profit < 0:
                self.daily_loss += abs(profit)
        
        # Keep trade history
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            **trade_data
        })
        
        logger.info(f"Trade recorded: {trade_data}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current live guard status"""
        return {
            "environment": self.environment,
            "auto_trade_enabled": self.auto_trade_enabled,
            "maintenance_mode": self.maintenance_mode,
            "daily_trades": self.daily_trades,
            "daily_loss": self.daily_loss,
            "max_daily_trades": self.max_daily_trades,
            "max_daily_loss": self.max_daily_loss,
            "max_position_size": self.max_position_size,
            "blocked_symbols": list(self.blocked_symbols),
            "last_reset": self.last_reset.isoformat(),
            "trade_count_today": len(self.trade_history)
        }


class KillSwitch:
    """
    Emergency kill switch for trading operations
    """
    
    def __init__(self):
        self.engaged = False
        self.reason = None
        self.engaged_at = None
        self.auto_disengage_after = timedelta(hours=1)  # Auto-disengage after 1 hour
    
    def engage(self, reason: str = "Manual intervention"):
        """Engage the kill switch"""
        self.engaged = True
        self.reason = reason
        self.engaged_at = datetime.now()
        logger.critical(f"KILL SWITCH ENGAGED: {reason}")
    
    def disengage(self):
        """Disengage the kill switch"""
        if self.engaged:
            duration = datetime.now() - self.engaged_at
            logger.warning(f"Kill switch disengaged after {duration}")
        self.engaged = False
        self.reason = None
        self.engaged_at = None
    
    def check(self) -> bool:
        """Check if kill switch is engaged"""
        if self.engaged:
            # Check for auto-disengage
            if self.engaged_at:
                if datetime.now() - self.engaged_at > self.auto_disengage_after:
                    logger.info("Kill switch auto-disengaged after timeout")
                    self.disengage()
                    return False
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status"""
        return {
            "engaged": self.engaged,
            "reason": self.reason,
            "engaged_at": self.engaged_at.isoformat() if self.engaged_at else None,
            "auto_disengage_at": (
                (self.engaged_at + self.auto_disengage_after).isoformat()
                if self.engaged_at else None
            )
        }


# Global kill switch instance
kill_switch = KillSwitch()


def check_trading_allowed() -> tuple[bool, str]:
    """
    Quick check if trading is allowed
    Returns: (allowed: bool, reason: str)
    """
    # Check kill switch first
    if kill_switch.check():
        return False, f"Kill switch engaged: {kill_switch.reason}"
    
    # Check environment variable
    if not os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true":
        return False, "AUTO_TRADE_ENABLED is not set to true"
    
    # Check weekend
    now = datetime.now()
    if now.weekday() >= 5:
        return False, f"Trading not allowed on {now.strftime('%A')}"
    
    return True, "Trading allowed"
